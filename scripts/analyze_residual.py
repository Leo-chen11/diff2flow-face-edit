"""What is the learned residual? Principal components of the per-face residual.

The direction bank's edit is  dir_delta (gated bank directions)  +  rs * residual,
where residual = the flow's W+ delta minus its projection on every bank
direction. Ablations put the whole gain over the linear baseline on the
residual. This asks what it is:

  a few global directions the bank is missing   -> low rank: the top components
                                                   carry most of its energy, and
                                                   evaluating with only those
                                                   (evaluate_sdflow.py
                                                   --residual_basis ... --residual_basis_k k)
                                                   keeps the accuracy;
  a correction that differs face by face        -> high rank: it needs many.

Per attribute, from the same edits the eval makes (direction from the R50
judge's reading of the source, --scale), it reports:
  |mean| / mean|r|       1.0 = every face gets the same residual direction (signed
                         by the edit direction, so add and rm along one axis count
                         as the same)
  energy@k               uncentred SVD: share of the residual energy in the top k
  k for 50/80/90%        components needed
  add-vs-rm cos          cosine of the mean residual of add edits vs rm edits
                         (-1 = one axis travelled both ways)
  layers coarse/mid/fine share of the energy in W+ layers 0-3 / 4-9 / 10-17
  |r| / |edit|           residual norm relative to the whole applied delta

and saves the top --keep components per attribute (for --residual_basis), plus
optionally a grid per attribute showing what the top components do.

Usage:
    python -m scripts.analyze_residual \
        --checkpoint_dir ./output/SDFlow/multi_v1_cont20k_ctrl --step 100000 \
        --independent_attr_weights ./data/r50_celebahq_eval.pth \
        --independent_attr_backbone r50 --age_fine_layer_scale 1.0 \
        --num_faces 1000 --scale 1.0 --render_pcs 3
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils import data
from tqdm import tqdm

from evaluation.evaluate_sdflow import (
    ATTR_NAMES, _latest_step, apply_run_config, build_optional_judges, build_parser,
    edited_attr_value, load_models, resolve_controlnet_disable_attrs,
)
from models.dataset import SDFlowDataset

BANDS = (('coarse 0-3', 0, 4), ('mid 4-9', 4, 10), ('fine 10-17', 10, 18))
KS = (1, 2, 4, 8, 16, 32)


def name(g):
    return ATTR_NAMES.get(g, f'attr{g}')


@torch.no_grad()
def bank_edit(prior, direction_bank, latent, attr_cond, id_cond, local_idx, scale, global_idx, direction):
    """edit_single_attribute up to the direction bank (no rendering): returns the
    applied W+ delta and the residual part of it."""
    B = latent.size(0)
    zero_pad = torch.zeros(B, 18, 1, device=latent.device)
    mid, _ = prior(latent, torch.cat([id_cond, attr_cond], 1), zero_pad)
    new_attr = attr_cond.clone()
    new_attr[:, local_idx] = edited_attr_value(attr_cond[:, local_idx], scale, global_idx, direction=direction)
    raw, _ = prior(mid, torch.cat([id_cond, new_attr], 1), zero_pad, reverse=True)
    idx = torch.full((B,), local_idx, device=latent.device, dtype=torch.long)
    delta = direction_bank(raw - latent, new_attr - attr_cond, attr_idx=idx, latent=latent, route_scores=attr_cond)
    delta = delta[0] if isinstance(delta, tuple) else delta
    return delta, direction_bank._last_residual


def main():
    p = build_parser()
    p.add_argument('--num_faces', type=int, default=1000)
    p.add_argument('--scale', type=float, default=1.0)
    p.add_argument('--keep', type=int, default=64, help='Components saved per attribute.')
    p.add_argument('--out', default=None)
    p.add_argument('--render_pcs', type=int, default=3, help='Components to visualise per attribute (0 = none).')
    p.add_argument('--render_faces', type=int, default=6)
    p.add_argument('--render_mult', type=float, default=3.0,
                   help='Rendered step along a component = this x the mean residual norm '
                        '(the residual itself is often too small to see).')
    args = p.parse_args()
    args = apply_run_config(args)
    args = resolve_controlnet_disable_attrs(args)
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
    if not args.independent_attr_weights:
        raise SystemExit('--independent_attr_weights is required (it sets each edit\'s direction, as the eval does).')

    prior, conditioner, G, id_criterion, _, _, direction_bank, _ = load_models(args)
    if direction_bank is None:
        raise SystemExit('this checkpoint has no direction bank, so no residual to analyse')
    _, _, _, indep_teacher, _, _ = build_optional_judges(args, args.attribute_index, id_criterion)

    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)), T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file,
                       train=False, transform=tf)
    loader = data.DataLoader(ds, shuffle=False, batch_size=args.batch, num_workers=4)

    res, edit_norm, dirs = defaultdict(list), defaultdict(list), defaultdict(list)
    keep_lat = []
    seen = 0
    with torch.no_grad():
        for img, latent, _ in tqdm(loader, desc='residuals'):
            if seen >= args.num_faces:
                break
            img, latent = img.cuda(), latent.cuda()
            _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
            src = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
            sp = torch.sigmoid(indep_teacher(F.interpolate(src, (256, 256)))[0])
            for li, g in enumerate(args.attribute_index):
                d = torch.where(sp[:, g] > 0.5, -1.0, 1.0)
                delta, r = bank_edit(prior, direction_bank, latent, attr_cond, id_cond, li, args.scale, g, d)
                res[g].append(r.flatten(1).float().cpu())
                edit_norm[g].append(delta.flatten(1).norm(dim=1).float().cpu())
                dirs[g].append(d.cpu())
            if len(keep_lat) * args.batch < args.render_faces:
                keep_lat.append(latent)
            seen += img.size(0)

    out = {'config': {'checkpoint_dir': args.checkpoint_dir, 'step': args.step, 'scale': args.scale,
                      'num_faces': seen, 'attribute_index': list(args.attribute_index)},
           'attrs': {}}
    print(f'\nResidual structure, {seen} faces per attribute, scale {args.scale}\n')
    hdr = (f'  {"attribute":<11} {"|r|/|edit|":>10} {"|mean|/|r|":>10} '
           + ' '.join(f'{"E@" + str(k):>6}' for k in KS)
           + f' {"k50":>4} {"k80":>4} {"k90":>4} {"add-rm cos":>10}   ' + ' / '.join(b[0] for b in BANDS))
    print(hdr)
    render_lat = torch.cat(keep_lat)[:args.render_faces] if keep_lat else None
    for g in args.attribute_index:
        R = torch.cat(res[g])[:seen]
        D = torch.cat(dirs[g])[:seen]
        En = torch.cat(edit_norm[g])[:seen]
        rn = R.norm(dim=1)
        # sign-aligned by edit direction, so add and rm along one axis do not cancel
        mean_ratio = float((R * D.view(-1, 1)).mean(0).norm() / rn.mean().clamp(min=1e-12))
        _, S, Vh = torch.linalg.svd(R, full_matrices=False)
        energy = (S ** 2).cumsum(0) / (S ** 2).sum().clamp(min=1e-12)

        def k_for(t):
            return int((energy < t).sum().item()) + 1

        cos = None
        if (D > 0).any() and (D < 0).any():
            ma, mr = R[D > 0].mean(0), R[D < 0].mean(0)
            cos = float(F.cosine_similarity(ma, mr, dim=0))
        layer_e = (R.view(-1, 18, 512) ** 2).sum((0, 2))
        layer_e = layer_e / layer_e.sum().clamp(min=1e-12)
        bands = [float(layer_e[a:b].sum()) for _, a, b in BANDS]
        ratio = float((rn / En.clamp(min=1e-12)).mean())
        print(f'  {name(g):<11} {ratio:10.3f} {mean_ratio:10.3f} '
              + ' '.join(f'{float(energy[min(k, len(energy)) - 1]) * 100:5.1f}%' for k in KS)
              + f' {k_for(0.5):4d} {k_for(0.8):4d} {k_for(0.9):4d} '
              + (f'{cos:10.3f}' if cos is not None else f'{"--":>10}')
              + '   ' + ' / '.join(f'{b * 100:4.1f}%' for b in bands))
        out['attrs'][g] = {'basis': Vh[:args.keep].contiguous(), 'energy': energy[:max(args.keep, max(KS))],
                           'mean_ratio': mean_ratio, 'add_rm_cos': cos, 'layer_energy': layer_e,
                           'residual_over_edit': ratio, 'mean_residual_norm': float(rn.mean()),
                           'n': int(R.size(0))}

        if args.render_pcs > 0 and render_lat is not None:
            step = args.render_mult * float(rn.mean())
            rows = []
            with torch.no_grad():
                base = G([render_lat], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
                cols = [base]
                for j in range(min(args.render_pcs, Vh.size(0))):
                    v = Vh[j].view(1, 18, 512).to(render_lat.device, render_lat.dtype)
                    for sgn in (-1.0, 1.0):
                        cols.append(G([render_lat + sgn * step * v], input_is_latent=True,
                                      randomize_noise=False)[0].clamp(-1, 1))
                for i in range(render_lat.size(0)):
                    rows.append(torch.cat([F.interpolate(c[i:i + 1], (256, 256)) for c in cols], 3))
            grid = (torch.cat(rows, 2)[0] + 1) * 0.5
            path = os.path.join(args.checkpoint_dir, f'residual_pcs_{name(g)}_s{args.step}.png')
            torchvision.utils.save_image(grid, path)
            print(f'      -> {path}  (columns: recon, then -/+ PC1, -/+ PC2, ...; step {step:.2f} = '
                  f'{args.render_mult:g} x mean |residual|)')

    print('\n|r|/|edit|: residual norm over the whole applied delta.  |mean|/|r|: 1 = the same '
          'residual for every face.\nE@k: energy in the top k components (uncentred SVD).  '
          'k50/k80/k90: components for 50/80/90% of the energy.\nadd-rm cos: mean residual of '
          'add edits vs rm edits (-1 = the same axis both ways).\n'
          'Next: evaluate_sdflow.py --residual_basis <this file> --residual_basis_k 1 / 4 / 16 and compare '
          'at matched ID with scripts/compare_runs_matched_id.py.')
    path = args.out or os.path.join(args.checkpoint_dir, f'residual_pca_s{args.step}.pth')
    torch.save(out, path)
    print(f'saved {path}')


if __name__ == '__main__':
    main()
