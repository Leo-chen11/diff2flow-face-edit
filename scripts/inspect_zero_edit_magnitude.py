"""
Check whether "the judge score didn't move at all" failures (src score ==
edited score, seen in scripts/dump_attr_failures.py montages for Eyeglasses
add on the female_young-skewed failure group) are caused by the Direction
Bank producing a near-zero guided_delta for those specific faces, versus a
normal-sized edit that just doesn't register on the judge or in the render.

Hooks AttributeDirectionBank.forward() to capture, per sample, the actual
guided_delta norm and gate mixture (alpha) it produced -- the exact
quantities scripts/validate_direction_bank.py and the training-time
"dir_bank_guided_delta_norm" wandb log are built from, but read out
PER-SAMPLE here instead of only the batch mean, and paired with the
before/after judge score for that same sample.

Buckets samples into "near-zero score change" (|edit - src| < --zero_thresh)
vs "normal" and prints guided_delta norm / dir_delta norm / residual norm /
gate-entropy stats for each bucket:
  - near-zero bucket has MUCH smaller guided_delta_norm than normal bucket
      -> the edit genuinely isn't being applied for these faces (training/
         gate problem, not a direction-bank quality problem -- --extreme_min_conf
         can't fix this).
  - both buckets have similar guided_delta_norm
      -> the edit IS being applied at normal magnitude but doesn't register
         (render/judge insensitivity, or the direction itself points somewhere
         that doesn't read as "glasses" to the judge/eye -- a direction-quality
         problem, which --extreme_min_conf-style fixes DO target).

Usage:
  python scripts/inspect_zero_edit_magnitude.py \\
      --checkpoint_dir ./output/SDFlow/substyle3_glasses_v3 --step 60000 \\
      --attr 15 --direction add --zero_thresh 0.02 --num_samples 500
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
import numpy as np

from evaluation.evaluate_sdflow import (
    ATTR_NAMES, CLIPAttributeJudge, GlassesParserJudge,
    edit_single_attribute, is_clear, load_models, parse_clip_calibration,
)
from models.dataset import SDFlowDataset


@torch.no_grad()
def main(args):
    prior, conditioner, G, id_criterion, attr_teacher, \
        attribute_index, direction_bank, control_encoder = load_models(args)
    if direction_bank is None:
        raise SystemExit('This checkpoint has no direction_bank -- nothing to inspect.')

    local_idx = args.attribute_index.index(args.attr)
    attr_name = ATTR_NAMES.get(args.attr, f'attr{args.attr}')

    if args.attr == 15 and args.glasses_judge == 'parser':
        pj = GlassesParserJudge(args.face_parser_weights, 'cuda',
                                area_thresh=args.glasses_area_thresh)
        score_fn = lambda imgs: pj.glasses_prob(imgs)
        print(f'[Judge] {attr_name} = BiSeNet parser')
    else:
        cj = CLIPAttributeJudge(attribute_index, args.clip_judge_model, 'cuda',
                                 calibration=parse_clip_calibration(args.clip_calibration))
        score_fn = lambda imgs: cj.scores(imgs)[:, local_idx]
        print(f'[Judge] {attr_name} = CLIP {args.clip_judge_model}')

    # ---- hook: capture per-sample guided_delta / alpha out of the ONE
    # direction_bank(...) call edit_single_attribute makes per batch. ----
    captured = {}

    def _hook(module, inputs, output):
        guided_delta = output.detach()                      # (B, 18, 512)
        captured['guided_norm'] = guided_delta.reshape(guided_delta.size(0), -1).norm(dim=1).cpu()
        logs = module.last_logs
        captured['dir_norm'] = logs['dir_bank_dir_delta_norm'].item()      # batch mean (scalar)
        captured['residual_norm'] = logs['dir_bank_residual_norm'].item()  # batch mean (scalar)
        alpha = getattr(module, '_last_alpha', None)         # (B, num_attrs, K)
        if alpha is not None:
            a = alpha[:, local_idx, :].detach().cpu()        # (B, K)
            entropy = -(a * (a + 1e-8).log()).sum(dim=-1)     # (B,)
            captured['gate_entropy'] = entropy
            captured['gate_max'] = a.max(dim=-1).values
            captured['gate_argmax'] = a.argmax(dim=-1)        # (B,) which K-slot each sample leans on
        else:
            captured['gate_entropy'] = None
            captured['gate_max'] = None
            captured['gate_argmax'] = None

    handle = direction_bank.register_forward_hook(_hook)

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

    rows = []   # (delta_score, guided_norm, dir_norm, residual_norm, gate_entropy, gate_max)
    seen = 0
    for img, latent, pred in loader:
        if seen >= args.num_samples:
            break
        img = img.cuda(); latent = latent.cuda()
        _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)

        src_face = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
        src_face_256 = F.interpolate(src_face, (256, 256))
        src_scores = score_fn(src_face_256)

        captured.clear()
        edited = edit_single_attribute(
            prior, conditioner, G, id_criterion, img, latent, attr_cond, id_cond,
            local_idx, args.edit_scale, direction_bank, attr_global_idx=args.attr,
            control_encoder=control_encoder,
        )
        edited_256 = F.interpolate(edited, (256, 256))
        edit_scores = score_fn(edited_256)

        guided_norm = captured.get('guided_norm')
        gate_entropy = captured.get('gate_entropy')
        gate_max = captured.get('gate_max')
        gate_argmax = captured.get('gate_argmax')

        for b in range(img.size(0)):
            s = src_scores[b].item(); e = edit_scores[b].item()
            if not is_clear(s):
                continue
            if args.direction == 'add' and s >= 0.5:
                continue
            if args.direction == 'rm' and s < 0.5:
                continue
            rows.append((
                abs(e - s),
                guided_norm[b].item() if guided_norm is not None else float('nan'),
                captured['dir_norm'], captured['residual_norm'],
                gate_entropy[b].item() if gate_entropy is not None else float('nan'),
                gate_max[b].item() if gate_max is not None else float('nan'),
                int(gate_argmax[b].item()) if gate_argmax is not None else -1,
                int(pred[b, 20].item() >= 0.5),   # Male binary (source)
                int(pred[b, 39].item() >= 0.5),   # Young binary (source)
            ))
        seen += img.size(0)

    handle.remove()

    if not rows:
        raise SystemExit('No qualifying samples found -- check --attr/--direction.')

    delta = np.array([r[0] for r in rows])
    guided_norm = np.array([r[1] for r in rows])
    gate_entropy = np.array([r[4] for r in rows])
    gate_max = np.array([r[5] for r in rows])
    gate_argmax = np.array([r[6] for r in rows])

    male_bin = np.array([r[7] for r in rows])
    young_bin = np.array([r[8] for r in rows])

    zero_mask = delta < args.zero_thresh
    moved_mask = ~zero_mask

    print(f'\n{attr_name} {args.direction}: {len(rows)} qualifying samples, '
          f'{zero_mask.sum()} near-zero score change (|Δscore| < {args.zero_thresh}), '
          f'{moved_mask.sum()} moved normally\n')

    def _stats(mask, name):
        if mask.sum() == 0:
            print(f'  {name}: (no samples)')
            return
        print(f'  {name:<22} n={mask.sum():<5} '
              f'guided_delta_norm: mean={guided_norm[mask].mean():.4f} median={np.median(guided_norm[mask]):.4f} '
              f'| gate_entropy: mean={np.nanmean(gate_entropy[mask]):.4f} '
              f'| gate_max_weight: mean={np.nanmean(gate_max[mask]):.4f}')

    _stats(zero_mask, 'near-zero Δscore')
    _stats(moved_mask, 'moved normally')

    # ---- per-K-slot breakdown: which sub-direction is each sample's gate
    # actually leaning on (argmax alpha), and does near-zero-Δscore cluster
    # on one particular slot? A K-slot with a much higher near-zero RATE
    # than the others is a specific bad sub-direction to go recompute/drop,
    # separate from (and more targeted than) the stratum-level
    # --extreme_min_conf fix. ----
    if (gate_argmax >= 0).any():
        print('\nPer K-slot (gate argmax) breakdown:')
        print(f'  {"slot":<6}{"n":<8}{"near-zero":<12}{"near-zero rate":<16}{"mean guided_norm":<18}')
        for k in sorted(set(gate_argmax.tolist())):
            slot_mask = gate_argmax == k
            n = int(slot_mask.sum())
            nz = int((slot_mask & zero_mask).sum())
            print(f'  {k:<6}{n:<8}{nz:<12}{nz / max(1, n):<16.1%}{guided_norm[slot_mask].mean():<18.4f}')
        print(
            '\n  A slot with a near-zero rate much higher than the others (and roughly\n'
            '  the same guided_norm as the rest, per the table above) is a specific bad\n'
            '  sub-direction: that K-slot\'s own direction_units for this attribute is\n'
            '  likely degenerate (built from too few / too noisy samples in whichever\n'
            '  substyle_k sub-cluster produced it) and worth recomputing or dropping,\n'
            '  rather than re-tuning --extreme_min_conf at the whole-stratum level again.'
        )

    # ---- demographic vs slot routing: does the gate actually send a
    # source face to ITS OWN gender x age sub-direction, or does it default
    # to a couple of generic slots regardless of who the face is? Only
    # meaningful for attr 15 (Eyeglasses), whose bank was stratified in this
    # exact order (see compute_glasses_directions): male_young, male_old,
    # female_young, female_old, each contributing num_k//4 consecutive
    # slots -- e.g. num_k=12 -> male_young=[0,1,2], male_old=[3,4,5],
    # female_young=[6,7,8], female_old=[9,10,11]. A different attribute (or
    # a bank built without --substyle_k_attrs 15) uses a different
    # conditioning scheme, so this mapping would be wrong there. ----
    if args.attr == 15 and (gate_argmax >= 0).any():
        num_k = int(gate_argmax.max()) + 1
        if num_k % 4 == 0:
            per_stratum = num_k // 4
            strata = ['male_young', 'male_old', 'female_young', 'female_old']
            demo_group = np.where(male_bin == 1,
                                  np.where(young_bin == 1, 0, 1),   # male: young=0, old=1
                                  np.where(young_bin == 1, 2, 3))   # female: young=2, old=3
            print(f'\nDemographic routing (attr 15 only, num_k={num_k}, '
                  f'{per_stratum} slots/stratum, strata order {strata}):')
            print(f'  {"source demo":<16}{"n":<8}{"own-slot %":<14}{"most-used slot":<18}{"top-slot %":<12}')
            for g, name in enumerate(strata):
                g_mask = demo_group == g
                n = int(g_mask.sum())
                if n == 0:
                    print(f'  {name:<16}{0:<8}(no samples)')
                    continue
                own_lo, own_hi = g * per_stratum, (g + 1) * per_stratum
                own_pct = float(((gate_argmax[g_mask] >= own_lo) & (gate_argmax[g_mask] < own_hi)).mean())
                slots, counts = np.unique(gate_argmax[g_mask], return_counts=True)
                top_slot = int(slots[counts.argmax()])
                top_pct = float(counts.max() / n)
                print(f'  {name:<16}{n:<8}{own_pct:<14.1%}{top_slot:<18}{top_pct:<12.1%}')
            print(
                '\n  "own-slot %" = fraction of that demographic\'s samples routed to THEIR OWN\n'
                '  stratum\'s slots. Low own-slot % (especially for female_young, right after\n'
                '  the --extreme_min_conf fix) means the gate is largely ignoring the source\n'
                '  face\'s actual demographic and defaulting to whichever slot(s) look best on\n'
                '  average across the whole training population -- a gate-routing/training\n'
                '  problem, not a direction-quality problem. If female_young\'s own-slot % is\n'
                '  near 0, the clean direction we fixed is essentially unused.'
            )

    print(
        '\nRead the two-group comparison above as:\n'
        '  guided_delta_norm much SMALLER in the near-zero group -> the edit genuinely\n'
        '    isn\'t being applied for these faces (gate/magnitude collapsed for them) --\n'
        '    a training-dynamics problem, direction-bank fixes like --extreme_min_conf\n'
        '    cannot fix this.\n'
        '  guided_delta_norm SIMILAR between groups -> normal-sized edit is being applied\n'
        '    but doesn\'t register as glasses to the judge/eye -- points at direction\n'
        '    quality/placement or judge sensitivity instead.\n'
        '  gate_entropy much LOWER (closer to 0) in the near-zero group with substyle_k>1\n'
        '    -> gate has collapsed onto one sub-direction for these faces specifically;\n'
        '    see the per-K-slot breakdown above for which one.'
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--step', type=int, default=None)
    p.add_argument('--attr', type=int, default=15)
    p.add_argument('--direction', default='add', choices=['add', 'rm'])
    p.add_argument('--zero_thresh', type=float, default=0.02,
                   help='|edited_score - src_score| below this counts as "near-zero change".')
    p.add_argument('--num_samples', type=int, default=500)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--edit_scale', type=float, default=1.0)
    p.add_argument('--img_size', type=int, default=512)

    p.add_argument('--index_file', default='./data/ffhq.txt')
    p.add_argument('--image_root', default='data/FFHQ')
    p.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    p.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth')
    p.add_argument('--stygan2_weights', default='./data/stylegan2-ffhq-config-f.pt')
    p.add_argument('--attribute_index', nargs='*', type=int, default=[15, 20, 39])

    p.add_argument('--attr_backbone', default='resnet50')
    p.add_argument('--conditioner_backbone', default='resnet')
    p.add_argument('--clip_model', default='ViT-B/32')
    p.add_argument('--fused_hidden_dim', type=int, default=256)
    p.add_argument('--id_cond_dim', type=int, default=32)
    p.add_argument('--id_cond_scale', type=float, default=0.25)
    p.add_argument('--flow_modules', default='512-512-512-512-512')
    p.add_argument('--num_blocks', type=int, default=1)
    p.add_argument('--velocity_field', default='lag_dof',
                   choices=['original', 'lag', 'dof', 'lag_dof'])
    p.add_argument('--lag_gate_hidden_dim', type=int, default=64)
    p.add_argument('--lag_gate_init_bias', type=float, default=-0.5)
    p.add_argument('--attribute_weights', default='./data/r34_a40_age_256_classifier.pth')

    p.add_argument('--direction_bank_path', default=None)
    p.add_argument('--direction_residual_scale', type=float, default=0.15)
    p.add_argument('--glasses_residual_scale', type=float, default=0.35)
    p.add_argument('--override_residual_scale', action='store_true')
    p.add_argument('--force_bank_directions', action='store_true')
    p.add_argument('--guided_delta_max_norm', type=float, default=0.0)

    p.add_argument('--use_controlnet_injection', action='store_true')
    p.add_argument('--controlnet_embed_res', type=int, default=64)
    p.add_argument('--controlnet_channels', type=int, default=512)
    p.add_argument('--controlnet_hidden_dim', type=int, default=256)

    p.add_argument('--clip_judge_model', default='ViT-L/14')
    p.add_argument('--clip_calibration', default=None)
    p.add_argument('--glasses_judge', default='parser', choices=['clip', 'parser'])
    p.add_argument('--face_parser_weights', default='./data/parsing_bisenet.pth')
    p.add_argument('--glasses_area_thresh', type=float, default=0.0010)

    args = p.parse_args()
    if args.step is None:
        from evaluation.evaluate_sdflow import _latest_step
        args.step = _latest_step(args.checkpoint_dir)
        if args.step is None:
            raise SystemExit(f'No checkpoints in {args.checkpoint_dir}/save_models/')
        print(f'Auto-detected latest step: {args.step}')
    main(args)
