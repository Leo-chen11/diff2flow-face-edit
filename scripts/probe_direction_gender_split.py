"""Is a direction's Male-skewed failure rate a GENERATION problem or a
JUDGE problem?

scripts/dump_attr_failures.py's failure-vs-success audit can only say
"failures are disproportionately Male" (a fraction over a 24-sample
montage cap). It cannot tell you whether those failures are:

  (a) GENERATION, near miss  -- the edit genuinely moved the face older/
      younger/etc., just not by enough to cross the judge's 0.5 decision
      boundary. Points at magnitude/calibration (e.g. a bigger edit_scale
      or per-group residual_scale), not a broken mechanism.
  (b) GENERATION, no effect  -- the edit barely moved the score at all.
      Points at a missing mechanism for that group (the kind of gap
      --hair_color_add_loss_weight / --age_add_hair_prompt_cue were built
      to close for Young-add).
  (c) JUDGE disagreement     -- one judge says fail, an architecturally
      different judge (CLIP zero-shot vs the CelebA supervised classifier)
      says success on the SAME edited image. Points at judge calibration,
      not generation at all -- fixing the model would not move this
      number.

This script runs the SAME edit as dump_attr_failures.py over a much larger,
UNCAPPED sample (dump_attr_failures.py stops at --num_fail successes/
failures for montage saving; this only ever prints numbers, so it can run
over hundreds of samples cheaply), scores every edited image with BOTH
judges, and reports the score DISTRIBUTION (not just pass/fail) split by
a grouping attribute (default: Male=20, since that is the attribute
dump_attr_failures.py's own audits keep finding as the strongest failure
correlate for Young in both directions).

Usage (matches the Young-rm audit already run):
    python -m scripts.probe_direction_gender_split \
        --checkpoint_dir ./output/SDFlow/substyle3_glasses_v18 --step 120000 \
        --attr 39 --direction rm --edit_scale 1.0 \
        --celeba_attr_judge_weights /home/cchen/桌面/SDFlow/data/celeba_attr_resnet18.pth \
        --max_samples 400

Reuse for the Male-rm probe discussed alongside this attribute:
    ... --attr 20 --direction rm --group_attr 39 ...
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'stylegan2'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils import data

from PIL import Image, ImageDraw

from evaluation.evaluate_sdflow import (
    ATTR_NAMES, CLIPAttributeJudge, CelebAAttrClassifierJudge,
    _latest_step, apply_run_config, edit_single_attribute, is_clear, load_models,
    parse_clip_calibration, resolve_controlnet_disable_attrs,
)
# to_pil / save_montage are reused as-is. dump_attr_failures.make_pair is NOT:
# its `watch` row colours any |edit-src| > 0.15 orange as "leakage", which is
# the right semantics for a bystander attribute but the wrong ones here --
# the second judge is scoring the SAME attribute being edited, where a large
# move is the goal, not a leak. _make_judge_pair below colours each judge's
# score by whether THAT judge called it a success instead.
from scripts.dump_attr_failures import save_montage, to_pil
from models.dataset import SDFlowDataset
from models.flows.constant import CELEBA_ATTRIBUTES

# Score bins for the near-miss breakdown. "near miss" is the band adjacent
# to the 0.5 decision boundary on the side that counts as failure -- e.g.
# for direction=rm (success = edit < 0.5), 0.35-0.50 is "moved a lot, just
# short"; anything below 0.20 is "barely moved at all". Mirrors how the
# failure montage looked by eye: 0.34/0.45/0.46 read very differently from
# 0.00/0.01/0.02 even though both count as "judge-failed".
BINS = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]


def _bin_label(i):
    lo, hi = BINS[i], BINS[i + 1]
    return f'[{lo:.2f},{hi:.2f})'


def _histogram(values):
    counts = [0] * (len(BINS) - 1)
    for v in values:
        for i in range(len(BINS) - 1):
            if BINS[i] <= v < BINS[i + 1] or (i == len(BINS) - 2 and v >= BINS[i + 1] - 1e-9):
                counts[i] += 1
                break
    return counts


def _make_judge_pair(src, edited, attr_name, src_celeba, edit_celeba, edit_clip,
                     direction, group_label, cond_src, size=256):
    """src|edited pair captioned with BOTH judges' verdicts on the edit.

    Each judge's edited score is coloured by whether THAT judge called this
    edit a success, so a row where one is green and the other red is exactly
    the disagreement the aggregate table counts.

    cond_src is the conditioner's own reading of the source -- the number
    that actually DECIDES which way the edit goes (see the INSTRUCTION
    section of the report). It is flagged orange when it disagrees with the
    judge's reading, because then the model was told to edit the opposite
    way from the direction this sample was filed under, and the "failure"
    is the instruction, not the edit.
    """
    ok = (lambda s: s < 0.5) if direction == 'rm' else (lambda s: s >= 0.5)
    header_h = 68
    a = to_pil(F.interpolate(src.unsqueeze(0), (size, size))[0])
    b = to_pil(F.interpolate(edited.unsqueeze(0), (size, size))[0])
    canvas = Image.new('RGB', (size * 2, size + header_h), (20, 20, 20))
    canvas.paste(a, (0, header_h)); canvas.paste(b, (size, header_h))
    d = ImageDraw.Draw(canvas)
    green, red, grey = (80, 220, 80), (240, 90, 90), (170, 170, 170)
    orange = (245, 170, 60)
    mismatched = (cond_src >= 0.5) != (src_celeba >= 0.5)
    d.text((4, 6), f'src {attr_name}={src_celeba:.2f}  ({group_label})', fill=grey)
    d.text((size + 4, 6),
           f'cond={cond_src:.2f} -> told to {"REMOVE" if cond_src >= 0.5 else "ADD"}'
           + ('  MISMATCH' if mismatched else ''),
           fill=orange if mismatched else grey)
    d.text((4, 22), 'CelebA:', fill=grey)
    d.text((4, 38), 'CLIP:', fill=grey)
    d.text((size + 4, 22), f'edited {attr_name}={edit_celeba:.2f}',
           fill=green if ok(edit_celeba) else red)
    d.text((size + 4, 38), f'edited {attr_name}={edit_clip:.2f}',
           fill=green if ok(edit_clip) else red)
    return canvas


def _report_group(name, celeba_scores, clip_scores, direction, cond_scores=None):
    n = len(celeba_scores)
    if n == 0:
        print(f'  {name}: (no samples)')
        return
    success = (lambda s: s < 0.5) if direction == 'rm' else (lambda s: s >= 0.5)
    celeba_fail = sum(1 for s in celeba_scores if not success(s)) / n
    clip_fail = sum(1 for s in clip_scores if not success(s)) / n
    disagree = sum(1 for c, k in zip(celeba_scores, clip_scores)
                   if success(c) != success(k)) / n
    mean_celeba = sum(celeba_scores) / n
    mean_clip = sum(clip_scores) / n
    print(f'  {name}  (n={n})')
    print(f'    mean edited score   CelebA={mean_celeba:.3f}   CLIP={mean_clip:.3f}')
    print(f'    fail rate           CelebA={celeba_fail:.1%}   CLIP={clip_fail:.1%}   '
         f'judges disagree on pass/fail: {disagree:.1%}')
    if cond_scores is not None:
        # Was the model even told to edit this way? The edit direction comes
        # from the conditioner's own reading of the source, not from the
        # judge that scores it, so the two can point opposite ways.
        want_has = direction == 'rm'
        mism = [i for i in range(n) if (cond_scores[i] >= 0.5) != want_has]
        if mism:
            mism_fail = sum(1 for i in mism if not success(celeba_scores[i])) / len(mism)
            agree_idx = [i for i in range(n) if i not in set(mism)]
            agree_fail = (sum(1 for i in agree_idx if not success(celeba_scores[i]))
                          / len(agree_idx)) if agree_idx else float('nan')
            print(f'    INSTRUCTION         {len(mism)}/{n} ({len(mism)/n:.1%}) were told the '
                  f'OPPOSITE direction by the conditioner')
            print(f'      CelebA fail rate  told-correctly={agree_fail:.1%} (n={len(agree_idx)})'
                  f'   told-backwards={mism_fail:.1%} (n={len(mism)})')
        else:
            print(f'    INSTRUCTION         0/{n} mismatched -- every sample was told '
                  f'to {"REMOVE" if want_has else "ADD"}')
    hist = _histogram(celeba_scores)
    hist_str = '  '.join(f'{_bin_label(i)}={c}' for i, c in enumerate(hist))
    print(f'    CelebA score distribution: {hist_str}')


@torch.no_grad()
def main(args):
    prior, conditioner, G, id_criterion, attr_teacher, \
        attribute_index, direction_bank, control_encoder = load_models(args)

    if args.attr not in args.attribute_index:
        raise SystemExit(f'attribute_index {args.attribute_index} has no attr {args.attr}.')
    if not args.celeba_attr_judge_weights:
        raise SystemExit('This script always cross-checks against CelebAAttrClassifierJudge '
                         '-- pass --celeba_attr_judge_weights.')
    local_idx = args.attribute_index.index(args.attr)
    attr_name = ATTR_NAMES.get(args.attr, f'attr{args.attr}')
    group_name = ATTR_NAMES.get(args.group_attr, CELEBA_ATTRIBUTES[args.group_attr]
                                if args.group_attr < len(CELEBA_ATTRIBUTES) else f'attr{args.group_attr}')

    celeba_judge = CelebAAttrClassifierJudge(args.celeba_attr_judge_weights, 'cuda')
    clip_judge = CLIPAttributeJudge([args.attr], args.clip_judge_model, 'cuda',
                                    calibration=parse_clip_calibration(args.clip_calibration))
    print(f'[Judge] {attr_name}: CelebAAttrClassifierJudge (grouping + primary) '
         f'and CLIP {args.clip_judge_model} (cross-check) on the same edited images')
    print(f'auditing {attr_name}  direction={args.direction}  grouped by {group_name}\n')

    img_transform = T.Compose([
        T.ToTensor(), T.Resize((args.img_size, args.img_size)),
        T.Normalize(mean=0.5, std=0.5),
    ])
    dataset = SDFlowDataset(
        index_file=args.index_file, image_root=args.image_root,
        latents_file=args.latent_file, preds_file=args.preds_file,
        train=False, transform=img_transform,
    )
    loader = data.DataLoader(dataset, shuffle=False, batch_size=args.batch,
                             num_workers=4, drop_last=False)

    groups = {0: {'celeba': [], 'clip': [], 'cond': []},
              1: {'celeba': [], 'clip': [], 'cond': []}}
    all_celeba, all_clip, all_cond = [], [], []
    # --dump_dir: three montages, each capped at --dump_max, so the aggregate
    # numbers above can be checked against what the images actually show.
    #   fail1/fail0  -- CelebA-judged failures inside each group
    #   disagree     -- the samples the two judges score on opposite sides,
    #                   which is the whole point of the cross-check: eyeball
    #                   these to decide WHICH judge is misreading them.
    dumps = {'fail1': [], 'fail0': [], 'disagree': []}
    n_seen = 0
    for img, latent, pred in loader:
        if n_seen >= args.max_samples:
            break
        img = img.cuda(); latent = latent.cuda()
        _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)

        src_face = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
        src_face_256 = F.interpolate(src_face, (256, 256))
        src_celeba = celeba_judge.scores(src_face_256)   # (B, 40)
        src_target = src_celeba[:, args.attr]

        edited = edit_single_attribute(
            prior, conditioner, G, id_criterion, img, latent, attr_cond, id_cond,
            local_idx, args.edit_scale, direction_bank, attr_global_idx=args.attr,
            control_encoder=control_encoder,
            controlnet_max_norm=getattr(args, 'controlnet_max_norm', 0.0),
            controlnet_disable_attrs=getattr(args, 'controlnet_disable_attrs', None),
        )
        edited_256 = F.interpolate(edited, (256, 256))
        edit_celeba = celeba_judge.scores(edited_256)    # (B, 40)
        edit_target_celeba = edit_celeba[:, args.attr]
        edit_target_clip = clip_judge.scores(edited_256)[:, 0]
        group_val = src_celeba[:, args.group_attr]        # grouping read on the SOURCE
        # The number that actually decides which way edit_single_attribute
        # moves this sample (it flips attr_cond, not the judge's score).
        cond_target = attr_cond[:, local_idx]

        for b in range(img.size(0)):
            if n_seen >= args.max_samples:
                break
            s = src_target[b].item()
            c = cond_target[b].item()
            if not is_clear(s):
                continue
            # --gate_on judge reproduces dump_attr_failures.py and
            # evaluate_sdflow.py's direction split (both bucket by the JUDGE's
            # source score). --gate_on cond instead keeps only the samples the
            # model was actually instructed to edit this way, which is the
            # number that measures the edit rather than the label disagreement.
            gate = s if args.gate_on == 'judge' else c
            if args.direction == 'add':
                if gate >= 0.5:
                    continue
            else:
                if gate < 0.5:
                    continue
            n_seen += 1
            ec = edit_target_celeba[b].item()
            ek = edit_target_clip[b].item()
            g = 1 if group_val[b].item() >= 0.5 else 0
            groups[g]['celeba'].append(ec)
            groups[g]['clip'].append(ek)
            groups[g]['cond'].append(c)
            all_celeba.append(ec)
            all_clip.append(ek)
            all_cond.append(c)

            if args.dump_dir:
                ok = (lambda v: v < 0.5) if args.direction == 'rm' else (lambda v: v >= 0.5)
                buckets = []
                if not ok(ec):
                    buckets.append('fail1' if g == 1 else 'fail0')
                if ok(ec) != ok(ek):
                    buckets.append('disagree')
                for bucket in buckets:
                    if len(dumps[bucket]) >= args.dump_max:
                        continue
                    dumps[bucket].append(_make_judge_pair(
                        src_face[b].detach().cpu(), edited[b].detach().cpu(),
                        attr_name, s, ec, ek, args.direction,
                        f'{group_name}={g}', c))

    print(f'=== Overall ({n_seen} samples, direction={args.direction}, '
          f'gate_on={args.gate_on}) ===')
    _report_group('all', all_celeba, all_clip, args.direction, all_cond)
    print()
    print(f'=== Split by source {group_name} ===')
    _report_group(f'{group_name}=0', groups[0]['celeba'], groups[0]['clip'],
                  args.direction, groups[0]['cond'])
    _report_group(f'{group_name}=1', groups[1]['celeba'], groups[1]['clip'],
                  args.direction, groups[1]['cond'])
    print()
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        tag = f'{attr_name}_{args.direction}'
        print(f'=== Montages -> {args.dump_dir} ===')
        for bucket, label in [
            ('fail1', f'{group_name}1_CELEBA_FAILURES'),
            ('fail0', f'{group_name}0_CELEBA_FAILURES'),
            ('disagree', 'JUDGES_DISAGREE'),
        ]:
            save_montage(dumps[bucket], os.path.join(args.dump_dir, f'{tag}_{label}.png'),
                         cols=2)
        print(f'  In {tag}_JUDGES_DISAGREE.png each row shows both judges\' verdicts on the '
              f'SAME edit (green = that judge called it a success). Whichever colour '
              f'disagrees with your own eyes is the judge that is misreading these.')
        print()
    print('How to read this:')
    print('  - If the weak group\'s CelebA distribution clusters in the near-miss bin')
    print('    ([0.35,0.50) for rm) rather than the far bin ([0.00,0.20)), the edit is')
    print('    moving those faces, just not far enough -- a magnitude/calibration issue.')
    print('  - If "judges disagree on pass/fail" is much higher for the weak group than')
    print('    the other, CelebA and CLIP are reading the SAME images differently for')
    print('    that group -- at least part of the gap is a judge artifact, not a real')
    print('    generation difference, and no amount of retraining will close it.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--step', type=int, default=None)
    p.add_argument('--attr', type=int, default=39, help='Global attribute index being edited.')
    p.add_argument('--direction', default='rm', choices=['add', 'rm'])
    p.add_argument('--edit_scale', type=float, default=1.0)
    p.add_argument('--group_attr', type=int, default=20,
                   help='Global CelebA attribute index to split the report by, read on the '
                        'SOURCE image via the CelebA judge. Default 20 (Male).')
    p.add_argument('--max_samples', type=int, default=400)
    p.add_argument('--gate_on', default='judge', choices=['judge', 'cond'],
                   help='Which source reading decides that a sample belongs to this '
                        'direction. "judge" (default) reproduces dump_attr_failures.py and '
                        'evaluate_sdflow.py, which both bucket add/rm by the JUDGE\'s source '
                        'score -- while the edit direction itself is set by the conditioner '
                        '(edit_single_attribute flips attr_cond). When those two disagree the '
                        'sample is filed under one direction and edited the other way, and '
                        'gets scored as a failure for following the instruction it was given. '
                        '"cond" gates on the conditioner instead, so the pool only holds '
                        'samples the model was actually told to edit this way -- that fail '
                        'rate measures the EDIT, the difference between the two measures the '
                        'label disagreement.')
    p.add_argument('--dump_dir', default=None,
                   help='If set, also save montages: each group\'s CelebA-judged failures, '
                        'and the samples the two judges score on opposite sides. Off by '
                        'default -- the aggregate numbers alone need no image decoding.')
    p.add_argument('--dump_max', type=int, default=16,
                   help='Max pairs per montage saved under --dump_dir.')
    p.add_argument('--controlnet_disable_attrs', nargs='*', type=int, default=None)

    p.add_argument('--index_file',   default='./data/ffhq.txt')
    p.add_argument('--image_root',   default='data/FFHQ')
    p.add_argument('--latent_file',  default='./data/ffhq_e4e_latents.pth')
    p.add_argument('--preds_file',   default='./data/ffhq_e4e_preds.pth')
    p.add_argument('--stygan2_weights', default='./data/stylegan2-ffhq-config-f.pt')
    p.add_argument('--attribute_weights', default='./data/r34_a40_age_256_classifier.pth')
    p.add_argument('--direction_bank_path', default=None)

    p.add_argument('--img_size',         type=int,   default=512)
    p.add_argument('--attribute_index',  nargs='*',  type=int,   default=[15, 20, 39])
    p.add_argument('--flow_modules',     default='512-512-512-512-512')
    p.add_argument('--num_blocks',       type=int,   default=1)
    p.add_argument('--velocity_field',   default='lag_dof')
    p.add_argument('--id_cond_dim',      type=int,   default=32)
    p.add_argument('--id_cond_scale',    type=float, default=0.25)
    p.add_argument('--attr_backbone',    default='resnet50')
    p.add_argument('--conditioner_backbone', default='resnet',
                   choices=['resnet', 'clip', 'resnet_clip'])
    p.add_argument('--clip_model', default='ViT-B/32')
    p.add_argument('--clip_judge_model', default='ViT-L/14')
    p.add_argument('--clip_calibration', default=None,
                    help="Same format as evaluate_sdflow.py: 'attr_idx:thresh:sharpness,...'")
    p.add_argument('--celeba_attr_judge_weights', required=True)
    p.add_argument('--fused_hidden_dim', type=int, default=256)
    p.add_argument('--lag_gate_hidden_dim', type=int,   default=64)
    p.add_argument('--lag_gate_init_bias',  type=float, default=-0.5)
    p.add_argument('--direction_residual_scale', type=float, default=0.05)
    p.add_argument('--glasses_residual_scale',   type=float, default=0.05)
    p.add_argument('--bypass_glasses_direction_bank',
                   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--guided_delta_max_norm', type=float, default=0.0)
    p.add_argument('--override_residual_scale', type=float, default=None)
    p.add_argument('--age_fine_layer_scale', type=float, default=None)
    p.add_argument('--age_fine_layer_start', type=int, default=10)
    p.add_argument('--force_bank_directions', action='store_true')
    p.add_argument('--disable_controlnet', action='store_true')
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--ignore_run_config', action='store_true')

    args = p.parse_args()
    args = apply_run_config(args)
    args = resolve_controlnet_disable_attrs(args)
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
        if args.step is None:
            raise ValueError(f'No checkpoints in {args.checkpoint_dir}/save_models/')
        print(f'Auto-detected latest step: {args.step}')
    main(args)
