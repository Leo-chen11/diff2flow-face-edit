"""Editing-accuracy curves in the style of the SDFlow paper's Fig. 4.

For each edit strength, every face is edited one attribute at a time and three
numbers are recorded:

  Editing Accuracy        R50 judge: the edited attribute crossed 0.5 in the
                          requested direction (sources the judge reads clearly)
  Identity Preservation   cosine similarity of the independent face-ID model
                          (facenet, casia-webface) between edit and source
                          reconstruction
  Attribute Preservation  share of the OTHER attributes whose R50 label (>0.5)
                          did not change; only attributes the judge reads
                          clearly on the source count (an ambiguous 0.49 that
                          tips to 0.51 is not a side effect). --preserve_set
                          edited = the run's other edited attributes (as the
                          original SDFlow eval script), all40 = all 39 others.

Sweeping the strength traces one curve per method; "All" is the mean over the
attributes. The CSV has the schema evaluation/plot_strength_curves.py reads,
so several methods go on one figure:

    python evaluation/plot_strength_curves.py \
        --run Ours=<ckpt>/curves_model_s100000.csv \
        --run Linear=<ckpt>/curves_linear_s100000.csv \
        --attributes All --output_dir ./output/curves

Methods (--method):
  model    the checkpoint's edits (evaluate_sdflow.edit_single_attribute), with
           any evaluate_sdflow option, e.g. --residual_fixed / --residual_head
  linear   w + sign * alpha * d with d the mean of the attribute's bank slots
           (the InterFaceGAN-style baseline of scripts/linear_baseline.py);
           --scales are then alphas

The absolute numbers depend on the judges (R50, facenet), so only curves made
by this script are comparable with each other; the paper's values used other
models and cannot be overlaid.

Usage:
    python -m scripts.editing_curves \
        --checkpoint_dir ./output/SDFlow/multi_v1_cont20k_ctrl --step 100000 \
        --independent_attr_weights ./data/r50_celebahq_eval.pth \
        --independent_attr_backbone r50 --age_fine_layer_scale 1.0 \
        --method model --scales 0.2 0.4 0.6 0.8 1.0 1.25 1.5 --num_samples 300
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils import data
from tqdm import tqdm

from common.attr_tables import bank_num_k
from evaluation.evaluate_sdflow import (
    ATTR_NAMES, _latest_step, apply_residual_basis, apply_run_config, build_optional_judges,
    build_parser, edit_single_attribute, load_models, resolve_controlnet_disable_attrs,
)
from models.dataset import SDFlowDataset

CLEAR_LO, CLEAR_HI = 0.35, 0.65


def name(g):
    return ATTR_NAMES.get(g, f'attr{g}')


def linear_directions(bank_path, attribute_index):
    """{attr: (18, 512)} mean of the attribute's bank slots, scaled by the
    observed layer norms (scripts/linear_baseline.py, variant 'single')."""
    from scripts.linear_baseline import build_direction_tables
    bank = torch.load(bank_path, map_location='cpu')
    du, ln = bank['direction_units'].float(), bank['layer_norms'].float()
    if du.ndim == 3:
        du, ln = du.unsqueeze(1), ln.unsqueeze(1)
    K = bank_num_k(bank, default=du.shape[1])
    bank_attrs = [int(a) for a in bank['attribute_index']]
    missing = [a for a in attribute_index if a not in bank_attrs]
    if missing:
        raise SystemExit(f'attributes {missing} are not in the bank {bank_path}')
    raw = (ln.unsqueeze(-1) * du)[[bank_attrs.index(a) for a in attribute_index]]
    tables = build_direction_tables(raw, {}, list(attribute_index), K)
    return {a: tables[a]['single'][0] for a in attribute_index}


def main():
    p = build_parser()
    p.add_argument('--method', default='model', choices=['model', 'linear'])
    p.add_argument('--scales', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5],
                   help='Edit strengths (alphas for --method linear).')
    p.add_argument('--preserve_set', default='edited', choices=['edited', 'all40'])
    p.add_argument('--out_csv', default=None,
                   help='Default <checkpoint_dir>/curves_<method>_s<step>.csv (a .json beside it).')
    args = p.parse_args()
    args = apply_run_config(args)
    args = resolve_controlnet_disable_attrs(args)
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
    if not args.independent_attr_weights:
        raise SystemExit('--independent_attr_weights is required (R50 judge).')

    prior, conditioner, G, id_criterion, _, _, direction_bank, control_encoder = load_models(args)
    apply_residual_basis(args, direction_bank)
    _, indep_id, _, indep_teacher, _, _ = build_optional_judges(args, args.attribute_index, id_criterion)
    if indep_id is None:
        raise SystemExit('the independent face-ID model (facenet-pytorch) is required')
    attrs = list(args.attribute_index)
    lin = linear_directions(args.direction_bank_path, attrs) if args.method == 'linear' else None

    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)), T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file,
                       train=False, transform=tf)
    loader = data.DataLoader(ds, shuffle=False, batch_size=args.batch, num_workers=4)

    rec = defaultdict(lambda: defaultdict(list))      # rec[(attr, scale)][metric] -> per-sample values
    seen = 0
    with torch.no_grad():
        for img, latent, _ in tqdm(loader, desc=f'curves ({args.method})'):
            if seen >= args.num_samples:
                break
            img, latent = img.cuda(), latent.cuda()
            _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
            src = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
            src_256 = F.interpolate(src, (256, 256))
            sp = torch.sigmoid(indep_teacher(src_256)[0])[:, :40]
            src_id = indep_id.extract(src_256)
            for li, g in enumerate(attrs):
                d = torch.where(sp[:, g] > 0.5, -1.0, 1.0)
                keep = (sp[:, g] > CLEAR_HI) | (sp[:, g] < CLEAR_LO)
                if not keep.any():
                    continue
                others = [j for j in (attrs if args.preserve_set == 'edited' else range(40)) if j != g]
                for sc in args.scales:
                    if lin is not None:
                        new_lat = latent + (d * sc).view(-1, 1, 1) * lin[g].to(latent)
                        face = G([new_lat], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
                    else:
                        face = edit_single_attribute(
                            prior, conditioner, G, id_criterion, img, latent, attr_cond, id_cond,
                            li, sc, direction_bank, attr_global_idx=g,
                            bypass_glasses_direction_bank=args.bypass_glasses_direction_bank,
                            control_encoder=control_encoder,
                            controlnet_max_norm=getattr(args, 'controlnet_max_norm', 0.0),
                            controlnet_disable_attrs=getattr(args, 'controlnet_disable_attrs', None),
                            controlnet_embed_res=getattr(args, 'controlnet_embed_res', 64),
                            composite=False, direction=d)
                    f256 = F.interpolate(face, (256, 256))
                    ep = torch.sigmoid(indep_teacher(f256)[0])[:, :40]
                    ok = torch.where(sp[:, g] > 0.5, ep[:, g] < 0.5 - args.success_margin,
                                     ep[:, g] > 0.5 + args.success_margin)
                    idv = (src_id * indep_id.extract(f256)).sum(1)
                    so, eo = sp[:, others], ep[:, others]
                    clear_o = (so > CLEAR_HI) | (so < CLEAR_LO)
                    same = ((so > 0.5) == (eo > 0.5)) & clear_o
                    pres = same.float().sum(1) / clear_o.float().sum(1).clamp(min=1)
                    r = rec[(g, sc)]
                    r['acc'] += ok[keep].float().tolist()
                    r['id'] += idv[keep].tolist()
                    r['pres'] += pres[keep].tolist()
            seen += img.size(0)

    rows = []
    for g in attrs:
        for sc in args.scales:
            r = rec.get((g, sc))
            if not r or not r['acc']:
                continue
            rows.append({'attribute': name(g), 'strength': sc, 'target_success': float(np.mean(r['acc'])),
                         'effective_success': float(np.mean(r['acc'])), 'id_sim_real': float(np.mean(r['id'])),
                         'preserve_acc': float(np.mean(r['pres'])), 'n': len(r['acc'])})
    for sc in args.scales:
        per = [x for x in rows if x['strength'] == sc]
        if len(per) == len(attrs):
            rows.append({'attribute': 'All', 'strength': sc,
                         **{k: float(np.mean([x[k] for x in per]))
                            for k in ('target_success', 'effective_success', 'id_sim_real', 'preserve_acc')},
                         'n': sum(x['n'] for x in per)})

    label = f'{args.method}' + ('' if args.method == 'linear' else ' (residual variant)' if any(
        getattr(args, k, None) for k in ('residual_basis', 'residual_fixed', 'residual_head')) else '')
    print(f'\n{label}, {seen} faces, preserve_set={args.preserve_set}')
    print(f'  {"attribute":<11} {"scale":>6} {"EditAcc":>8} {"ID cos":>7} {"AttrPres":>9} {"n":>5}')
    for x in rows:
        print(f'  {x["attribute"]:<11} {x["strength"]:6.2f} {x["target_success"] * 100:7.1f}% '
              f'{x["id_sim_real"]:7.3f} {x["preserve_acc"] * 100:8.1f}% {x["n"]:5d}')

    path = args.out_csv or os.path.join(args.checkpoint_dir, f'curves_{args.method}_s{args.step}.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.splitext(path)[0] + '.json', 'w') as f:
        json.dump({'config': {'checkpoint_dir': args.checkpoint_dir, 'step': args.step, 'method': args.method,
                              'scales': args.scales, 'num_samples': seen, 'preserve_set': args.preserve_set,
                              'residual_basis': getattr(args, 'residual_basis', None),
                              'residual_fixed': getattr(args, 'residual_fixed', None),
                              'residual_head': getattr(args, 'residual_head', None)},
                   'rows': rows}, f, indent=2)
    print(f'saved {path}\nplot: python evaluation/plot_strength_curves.py --run NAME={path} '
          f'[--run OTHER=...] --attributes All --output_dir ./output/curves')


if __name__ == '__main__':
    main()
