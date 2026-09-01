"""
General attribute-edit failure auditor: dump the faces a judge scores as
FAILURES for a chosen attribute + direction, as source|edited montages with
the judge score printed on each, so you can visually decide whether a low
number is a MODEL problem (edit genuinely didn't happen) or a JUDGE problem
(edit happened but the judge under-detects it -- as was the case for
eyeglasses, where CLIP scored real thin frames ~0.4).

direction:
  add  = source LACKS the attribute; edit should ADD it   (success: score crosses up over 0.5)
  rm   = source HAS the attribute;   edit should REMOVE it (success: score crosses down under 0.5)

For Young (attr 39), the aging direction is `rm` (source is young -> make old);
that is the direction dragging the Young metric down, so audit it with:

  python scripts/dump_attr_failures.py \
      --checkpoint_dir /tmp/ema_v16_51k --step 51000 \
      --attr 39 --direction rm --edit_scale 1.0 \
      --num_fail 24 --num_success 8 --out_dir ./young_rm_dump

Eyeglasses (parser judge) add-direction audit:

  python scripts/dump_attr_failures.py \
      --checkpoint_dir /tmp/ema_v16_51k --step 51000 \
      --attr 15 --direction add --glasses_judge parser --edit_scale 1.0 \
      --out_dir ./glasses_add_dump
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
    ATTR_NAMES, CLIPAttributeJudge, CelebAAttrClassifierJudge, GlassesParserJudge,
    _latest_step, apply_run_config, edit_single_attribute, is_clear, load_models,
    parse_clip_calibration, resolve_controlnet_disable_attrs,
)
from common.face_parser import FaceParser
from models.dataset import SDFlowDataset
from models.flows.constant import CELEBA_ATTRIBUTES


def to_pil(img_tensor):
    x = ((img_tensor.clamp(-1, 1) + 1) * 0.5 * 255).byte().cpu()
    return Image.fromarray(x.permute(1, 2, 0).numpy())


def make_pair(src, edited, src_score, edit_score, attr_name, success, size=256, watch=None):
    """watch: optional list of (name, watch_src_score, watch_edit_score) for
    bystander attributes that weren't edited -- flags leakage (any large
    |edit-src| regardless of direction, since there's no "correct" direction
    for something you didn't intend to change)."""
    watch = watch or []
    header_h = 24 + 14 * len(watch)
    a = to_pil(F.interpolate(src.unsqueeze(0), (size, size))[0])
    b = to_pil(F.interpolate(edited.unsqueeze(0), (size, size))[0])
    canvas = Image.new('RGB', (size * 2, size + header_h), (20, 20, 20))
    canvas.paste(a, (0, header_h)); canvas.paste(b, (size, header_h))
    d = ImageDraw.Draw(canvas)
    d.text((4, 6), f'src {attr_name}={src_score:.2f}', fill=(180, 180, 180))
    color = (80, 220, 80) if success else (240, 90, 90)
    d.text((size + 4, 6), f'edited {attr_name}={edit_score:.2f}', fill=color)
    for i, (wname, ws, we) in enumerate(watch):
        y = 20 + (i + 1) * 14
        leaked = abs(we - ws) > 0.15
        wcolor = (240, 180, 60) if leaked else (150, 150, 150)
        d.text((4, y), f'[watch]{wname} src={ws:.2f}', fill=(150, 150, 150))
        d.text((size + 4, y), f'[watch]{wname} edit={we:.2f}', fill=wcolor)
    return canvas


def save_montage(pairs, path, cols=4):
    if not pairs:
        print(f'  (none to save for {path})')
        return
    w, h = pairs[0].size
    rows = (len(pairs) + cols - 1) // cols
    grid = Image.new('RGB', (w * cols, h * rows), (0, 0, 0))
    for i, p in enumerate(pairs):
        grid.paste(p, ((i % cols) * w, (i // cols) * h))
    grid.save(path)
    print(f'  saved {len(pairs)} pairs -> {path}')


@torch.no_grad()
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    prior, conditioner, G, id_criterion, attr_teacher, \
        attribute_index, direction_bank, control_encoder = load_models(args)

    if args.attr not in args.attribute_index:
        raise SystemExit(f'attribute_index {args.attribute_index} has no attr {args.attr}.')
    local_idx = args.attribute_index.index(args.attr)
    attr_name = ATTR_NAMES.get(args.attr, f'attr{args.attr}')

    # Judge: parser for glasses if asked, else CLIP.
    if args.attr == 15 and args.glasses_judge == 'parser':
        pj = GlassesParserJudge(args.face_parser_weights, 'cuda',
                                area_thresh=args.glasses_area_thresh,
                                sharpness=args.glasses_area_sharpness,
                                min_component_frac=getattr(args, 'glasses_min_component_frac', 0.00015))
        print(f'[Judge] {attr_name} = BiSeNet parser (class 6)')
        score_fn = lambda imgs: pj.glasses_prob(imgs)
    else:
        cj = CLIPAttributeJudge(args.attribute_index, args.clip_judge_model, 'cuda',
                                 calibration=parse_clip_calibration(args.clip_calibration))
        print(f'[Judge] {attr_name} = CLIP {args.clip_judge_model} (prompt ensemble)')
        score_fn = lambda imgs: cj.scores(imgs)[:, local_idx]

    print(f'auditing {attr_name}  direction={args.direction}  '
          f'({"source lacks -> add it" if args.direction == "add" else "source has -> remove it"})')

    watch_attrs = list(args.watch_attrs or [])
    watch_score_fn = None
    if watch_attrs:
        if args.celeba_attr_judge_weights:
            wj = CelebAAttrClassifierJudge(args.celeba_attr_judge_weights, 'cuda')
            print(f'[Watch] scoring {watch_attrs} with CelebAAttrClassifierJudge')
            watch_score_fn = lambda imgs: wj.scores(imgs)[:, watch_attrs]
        else:
            wj = CLIPAttributeJudge(watch_attrs, args.clip_judge_model, 'cuda')
            print(f'[Watch] scoring {watch_attrs} with CLIP (no --celeba_attr_judge_weights given)')
            watch_score_fn = lambda imgs: wj.scores(imgs)
    watch_names = [ATTR_NAMES.get(a, f'attr{a}') for a in watch_attrs]

    composite_face_parser = None
    if args.composite_face_region:
        try:
            composite_face_parser = FaceParser(weights_path=args.face_parser_weights).cuda().eval()
            print(f'[Composite] method={args.composite_method} '
                  f'blur_sigma={args.composite_blur_sigma}')
        except (FileNotFoundError, RuntimeError) as exc:
            print(f'[WARN] --composite_face_region requested but face parser unavailable '
                  f'({exc}); compositing disabled.')

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

    fails, succs = [], []
    fail_preds, succ_preds = [], []
    n_dir, n_fail = 0, 0
    watch_leak_abs = {name: [] for name in watch_names}
    for img, latent, pred in loader:
        img = img.cuda(); latent = latent.cuda()
        _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)

        src_face = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
        src_face_256 = F.interpolate(src_face, (256, 256))
        src_scores = score_fn(src_face_256)

        edited = edit_single_attribute(
            prior, conditioner, G, id_criterion, img, latent, attr_cond, id_cond,
            local_idx, args.edit_scale, direction_bank, attr_global_idx=args.attr,
            bypass_glasses_direction_bank=args.bypass_glasses_direction_bank,
            face_parser=composite_face_parser,
            composite_method=args.composite_method,
            composite_blur_sigma=args.composite_blur_sigma,
            control_encoder=control_encoder,
            controlnet_max_norm=getattr(args, 'controlnet_max_norm', 0.0),
            controlnet_disable_attrs=getattr(args, 'controlnet_disable_attrs', None),
        )
        edited_256 = F.interpolate(edited, (256, 256))
        edit_scores = score_fn(edited_256)

        if watch_score_fn is not None:
            watch_src = watch_score_fn(src_face_256)
            watch_edit = watch_score_fn(edited_256)

        for b in range(img.size(0)):
            s = src_scores[b].item(); e = edit_scores[b].item()
            watch = [(watch_names[i], watch_src[b, i].item(), watch_edit[b, i].item())
                     for i in range(len(watch_attrs))] if watch_score_fn is not None else None
            if not is_clear(s):
                continue
            # Direction gate on the SOURCE, and success test on the EDIT.
            if args.direction == 'add':
                if s >= 0.5:
                    continue                 # source already has it; not an add case
                success = e >= 0.5           # added -> crossed up
            else:  # rm
                if s < 0.5:
                    continue                 # source doesn't have it; not a rm case
                success = e < 0.5            # removed -> crossed down
            n_dir += 1
            if watch is not None:
                for wname, ws, we in watch:
                    watch_leak_abs[wname].append(abs(we - ws))
            if not success:
                n_fail += 1
                if len(fails) < args.num_fail:
                    fails.append(make_pair(src_face[b], edited[b], s, e, attr_name, False, watch=watch))
                    fail_preds.append(pred[b, :40].clone())
            elif len(succs) < args.num_success:
                succs.append(make_pair(src_face[b], edited[b], s, e, attr_name, True, watch=watch))
                succ_preds.append(pred[b, :40].clone())
        if len(fails) >= args.num_fail and len(succs) >= args.num_success:
            break

    print(f'\n{attr_name} {args.direction}: samples seen={n_dir}, judge-failed={n_fail} '
          f'({n_fail / max(1, n_dir):.1%})')

    if fail_preds and succ_preds:
        # What does the failure group actually have in common, beyond
        # whatever pattern eyeballing the montage suggests? Compare the
        # r34 binary prediction for every OTHER CelebA attribute between
        # the failure and success groups collected above -- a real,
        # data-driven signal (e.g. "78% of failures are Pale_Skin=1 vs
        # 25% of successes") is worth acting on; a guess from thumbnails
        # (skin tone, hair style, makeup) is not, until it shows up here.
        fail_mat = torch.stack(fail_preds)   # (n_fail_saved, 40)
        succ_mat = torch.stack(succ_preds)   # (n_succ_saved, 40)
        fail_rate = fail_mat.mean(dim=0)
        succ_rate = succ_mat.mean(dim=0)
        diff = (fail_rate - succ_rate).abs()
        order = [i for i in torch.argsort(diff, descending=True).tolist() if i != args.attr]
        print(f'\n=== Failure vs success group: other-attribute fraction "1" '
              f'(fail n={fail_mat.shape[0]}, success n={succ_mat.shape[0]}) ===')
        print('Sorted by |difference| -- top of the list is what the failure group '
              'actually has more/less of, not what it looks like it has:')
        for i in order[:15]:
            name = CELEBA_ATTRIBUTES[i] if i < len(CELEBA_ATTRIBUTES) else f'attr{i}'
            print(f'  {name:<20} fail={fail_rate[i]:.2f}  success={succ_rate[i]:.2f}  '
                  f'|Δ|={diff[i]:.2f}')

    for wname, vals in watch_leak_abs.items():
        if not vals:
            continue
        vals_t = torch.tensor(vals)
        leak_rate = (vals_t > 0.15).float().mean().item()
        print(f'[watch]{wname} leakage over {len(vals)} samples: '
              f'mean|Δ|={vals_t.mean():.3f}  median|Δ|={vals_t.median():.3f}  '
              f'fraction with |Δ|>0.15 = {leak_rate:.1%}')

    tag = f'{attr_name}_{args.direction}'
    save_montage(fails, os.path.join(args.out_dir, f'{tag}_FAILURES.png'))
    save_montage(succs, os.path.join(args.out_dir, f'{tag}_successes.png'))
    print(f'\nOpen {tag}_FAILURES.png. For Young/rm these are faces the judge says '
          f'STILL look young after the aging edit:')
    print('  - they genuinely did NOT get older  -> model/direction problem')
    print('  - they DID get older but the judge missed it -> judge problem (recalibrate)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--step', type=int, default=None)
    p.add_argument('--attr', type=int, default=39, help='Global attribute index (15/20/39).')
    p.add_argument('--direction', default='rm', choices=['add', 'rm'])
    p.add_argument('--edit_scale', type=float, default=1.0)
    p.add_argument('--num_fail', type=int, default=24)
    p.add_argument('--num_success', type=int, default=8)
    p.add_argument('--out_dir', default='./attr_failure_dump')

    p.add_argument('--glasses_judge', default='parser', choices=['clip', 'parser'])
    p.add_argument('--face_parser_weights', default='./data/parsing_bisenet.pth')
    p.add_argument('--glasses_area_thresh', type=float, default=0.0010)
    p.add_argument('--glasses_area_sharpness', type=float, default=0.5)
    p.add_argument('--glasses_min_component_frac', type=float, default=0.00015)
    p.add_argument('--composite_face_region', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--composite_method', default='poisson', choices=['alpha', 'poisson'])
    p.add_argument('--composite_blur_sigma', type=float, default=15)
    p.add_argument('--controlnet_disable_attrs', nargs='*', type=int, default=None,
                    help='Same knob/default as evaluate_sdflow.py: omitted -> auto-resolved '
                         'to gender/age (20, 39) via resolve_controlnet_disable_attrs().')

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
    p.add_argument('--watch_attrs', nargs='*', type=int, default=None,
                    help='Extra attributes to SCORE but not edit, to check for leakage into '
                         'attributes not being edited here (e.g. --attr 39 --watch_attrs 20 '
                         'checks whether editing Young alone shifts Male).')
    p.add_argument('--celeba_attr_judge_weights', default=None,
                    help='Used for --watch_attrs scoring if given (preferred over CLIP).')
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
    p.add_argument('--disable_controlnet', action='store_true',
                   help='ABLATION: skip control_encoder even if the run was trained with it, '
                        'so leakage is measured through the W+ path alone.')
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
