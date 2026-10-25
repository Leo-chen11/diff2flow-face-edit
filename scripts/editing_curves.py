"""Editing-accuracy curves in the style of the SDFlow paper's Fig. 4.

For each edit strength, every face is edited one attribute at a time and three
numbers are recorded:

  Editing Accuracy        R50 judge: the edited attribute crossed 0.5 in the
                          requested direction (sources the judge reads clearly)
  Identity Preservation   cosine similarity between edit and source
                          reconstruction: ArcFace (IR-SE50, models/arcface.py,
                          --arcface_weights) as in the SDFlow paper, column
                          id_arc; and the independent facenet (casia-webface),
                          column id_sim_real
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
           any evaluate_sdflow option, e.g. --edit_gain / --preserve_boundaries
  linear   w + sign * alpha * d with d the mean of the attribute's bank slots
           (the InterFaceGAN-style baseline of scripts/linear_baseline.py);
           --scales are then alphas
  sdflow   the released SDFlow (github.com/phil329/SDFlow, Li et al., ICCV 2023) with its own
           checkpoint (--sdflow_ckpt_dir holding labeldist-<step> and prior-<step>, default step 10000):
           its attribute estimator reads the face, the flow encodes w under that reading, the edited
           attribute is set to -scale (remove) or 1+scale (add) as in its models/editor.py, and the
           flow is run backwards. It only knows Eyeglasses, Male and Young (--sdflow_attrs), so give
           every method --only_attrs 15 20 39: "All" then averages the same attributes for all of them.
           --scales are the editor's `scale`, e.g. 0 0.25 0.5 1 1.5 2
  boundary InterFaceGAN: w + sign * alpha * sd * n with n the attribute's unit
           normal of a linear W+ boundary (scripts/fit_attr_boundaries.py,
           --boundaries) and sd the spread of the test faces' latents along n,
           so alpha = 2 moves a face two standard deviations across the
           boundary; uses nothing of the checkpoint (but its data paths), so
           it is an external baseline; --scales are then alphas, e.g. 1 2 3 4 5 6

Judges read G's output downsampled with --eval_resize (default here: area, as
evaluate_sdflow's runs since the resize fix). With the CelebA r18 judge
(--celeba_attr_judge_weights, loaded when the file exists) a second accuracy,
celeb_success, is written too: on faces both judges read clearly and the same
way, did the r18 reading cross 0.5. plot_strength_curves.py --x_metric
celeb_success plots it.

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
import torchvision.transforms as T
from torch.utils import data
from tqdm import tqdm

import evaluation.evaluate_sdflow as E
from common.attr_tables import bank_num_k
from evaluation.evaluate_sdflow import (
    ATTR_NAMES, _latest_step, apply_run_config, build_optional_judges,
    build_parser, edit_single_attribute, load_models, resize_for_judge,
)
from models.arcface import ArcFaceJudge
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


def boundary_directions(path, attribute_index, ds, max_faces=2000):
    """{attr: (18, 512)} unit boundary normal times the std of the dataset's
    latents projected on it (scripts/fit_attr_boundaries.py output)."""
    d = torch.load(path, map_location='cpu')
    normals = d['normals'].float().reshape(40, -1)
    files = list(ds.image_list)[:max_faces]
    X = torch.stack([torch.as_tensor(ds._lookup_precomputed(ds.latents, f)).float().reshape(-1)
                     for f in files])
    out = {}
    for a in attribute_index:
        n = normals[a] / normals[a].norm().clamp(min=1e-12)
        sd = float((X @ n).std())
        print(f'[boundary] {name(a)}: AUC {float(d["auc"][a]):.3f}, latent spread along the normal {sd:.3f}')
        out[a] = (n * sd).view(18, 512)
    return out


class OfficialSDFlow:
    """models/editor.py of the released SDFlow, with the edit direction given. Its flow has the
    same parameter names and shapes as this repo's cnf(512, '512-512-512-512-512', n_attrs, 1),
    so its checkpoint loads strictly; the condition is the attribute estimator's reading only
    (no identity vector)."""

    def __init__(self, ckpt_dir, step, attrs, device='cuda'):
        from models.attribute_estimator import AttributeEstimator
        from models.flows.flow import cnf
        self.attrs = list(attrs)
        self.labeldist = AttributeEstimator(backbone='resnet34', attribute_dim=len(self.attrs)).to(device)
        self.flow = cnf(512, '512-512-512-512-512', len(self.attrs), 1).to(device)
        for module, stem in ((self.labeldist, 'labeldist'), (self.flow, 'prior')):
            path = os.path.join(ckpt_dir, f'{stem}-{step:07d}')
            if not os.path.exists(path):
                raise SystemExit(f'no {path}; the files in {ckpt_dir}: {sorted(os.listdir(ckpt_dir)) if os.path.isdir(ckpt_dir) else "(missing directory)"}'
                                 f' -- pass the right --sdflow_ckpt_dir / --sdflow_step')
            module.load_state_dict(torch.load(path, map_location='cpu'), strict=True)
            module.eval()
        print(f'[SDFlow] released checkpoint {ckpt_dir} step {step}, attributes {self.attrs}')

    @torch.no_grad()
    def edit(self, G, img, latent, g, direction, scale):
        reading = self.labeldist(img, latent)
        z = self.flow(latent, reading, torch.zeros(latent.size(0), latent.size(1), 1).to(latent))[0]
        new = reading.clone()
        new[:, self.attrs.index(g)] = torch.where(direction > 0, 1.0 + scale, -scale)
        w = self.flow(z, new, reverse=True)
        return G([w], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)


def main():
    p = build_parser()
    p.add_argument('--method', default='model', choices=['model', 'linear', 'boundary', 'sdflow'])
    p.add_argument('--sdflow_ckpt_dir', default='./data/ckpt',
                   help='--method sdflow: folder with labeldist-<step> and prior-<step>.')
    p.add_argument('--sdflow_step', type=int, default=10000)
    p.add_argument('--sdflow_attrs', nargs='+', type=int, default=[15, 20, 39],
                   help='CelebA ids the released SDFlow checkpoint was trained on.')
    p.add_argument('--only_attrs', nargs='+', type=int, default=None,
                   help='Evaluate only these CelebA ids (the run\'s own order), so every method\'s "All" '
                        'averages the same attributes; --method sdflow defaults to --sdflow_attrs.')
    p.add_argument('--arcface_weights', default='./data/model_ir_se50.pth',
                   help='ArcFace IR-SE50 weights (model_ir_se50.pth, the pSp / e4e ID-loss model) '
                        'for the paper\'s identity metric; without the file only facenet is measured.')
    p.add_argument('--boundaries', default='./data/attr_boundaries_wplus.pth',
                   help='--method boundary: scripts/fit_attr_boundaries.py output.')
    p.add_argument('--scales', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5],
                   help='Edit strengths (alphas for --method linear).')
    p.add_argument('--preserve_set', default='edited', choices=['edited', 'all40'])
    p.add_argument('--out_csv', default=None,
                   help='Default <checkpoint_dir>/curves_<method>_s<step>.csv (a .json beside it).')
    p.set_defaults(eval_resize='area')
    args = p.parse_args()
    args = apply_run_config(args)
    E._RESIZE_MODE = args.eval_resize
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
    if not args.independent_attr_weights:
        raise SystemExit('--independent_attr_weights is required (R50 judge).')

    prior, conditioner, G, id_criterion, _, _, direction_bank = load_models(args)
    _, indep_id, _, indep_teacher, _, celeb_judge = build_optional_judges(args, args.attribute_index, id_criterion)
    if indep_id is None:
        raise SystemExit('the independent face-ID model (facenet-pytorch) is required')
    arc = None
    if os.path.exists(args.arcface_weights):
        arc = ArcFaceJudge(args.arcface_weights)
        print(f'[Judge] ArcFace IR-SE50: {args.arcface_weights}')
    else:
        print(f'[WARN] no ArcFace weights at {args.arcface_weights}: id_arc is left empty and '
              f'plot_strength_curves.py needs --id_metric indep. Get model_ir_se50.pth from the '
              f'pSp / e4e README (the IR-SE50 of TreB1eN/InsightFace_Pytorch).')
    attrs = list(args.only_attrs or (args.sdflow_attrs if args.method == 'sdflow' else args.attribute_index))
    missing = [a for a in attrs if a not in args.attribute_index]
    if missing:
        raise SystemExit(f'--only_attrs {missing} are not edited attributes of {args.checkpoint_dir} '
                         f'({args.attribute_index})')
    sdf = None
    if args.method == 'sdflow':
        bad = [a for a in attrs if a not in args.sdflow_attrs]
        if bad:
            raise SystemExit(f'the released SDFlow checkpoint only covers {args.sdflow_attrs}, not {bad}: '
                             f'pass --only_attrs {" ".join(map(str, args.sdflow_attrs))}')
        sdf = OfficialSDFlow(args.sdflow_ckpt_dir, args.sdflow_step, args.sdflow_attrs)
    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)), T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file,
                       train=False, transform=tf)
    lin = linear_directions(args.direction_bank_path, attrs) if args.method == 'linear' \
        else boundary_directions(args.boundaries, attrs, ds) if args.method == 'boundary' else None
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
            src_256 = resize_for_judge(src, (256, 256))
            sp = torch.sigmoid(indep_teacher(src_256)[0])[:, :40]
            sc_src = celeb_judge.scores(src_256)[:, :40] if celeb_judge is not None else None
            src_id = indep_id.extract(src_256)
            src_arc = arc.extract(src_256) if arc is not None else None
            for g in attrs:
                li = args.attribute_index.index(g)
                d = torch.where(sp[:, g] > 0.5, -1.0, 1.0)
                keep = (sp[:, g] > CLEAR_HI) | (sp[:, g] < CLEAR_LO)
                if not keep.any():
                    continue
                others = [j for j in (attrs if args.preserve_set == 'edited' else range(40)) if j != g]
                for sc in args.scales:
                    if sdf is not None:
                        face = sdf.edit(G, img, latent, g, d, sc)
                    elif lin is not None:
                        new_lat = latent + (d * sc).view(-1, 1, 1) * lin[g].to(latent)
                        face = G([new_lat], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
                    else:
                        face = edit_single_attribute(
                            prior, conditioner, G, id_criterion, img, latent, attr_cond, id_cond,
                            li, sc, direction_bank, attr_global_idx=g,
                            bypass_glasses_direction_bank=args.bypass_glasses_direction_bank,
                            composite=False, direction=d)
                    f256 = resize_for_judge(face, (256, 256))
                    ep = torch.sigmoid(indep_teacher(f256)[0])[:, :40]
                    ok = torch.where(sp[:, g] > 0.5, ep[:, g] < 0.5 - args.success_margin,
                                     ep[:, g] > 0.5 + args.success_margin)
                    if sc_src is not None:
                        ce = celeb_judge.scores(f256)[:, g]
                        cs = sc_src[:, g]
                        both = keep & ((cs > CLEAR_HI) | (cs < CLEAR_LO)) & ((cs > 0.5) == (sp[:, g] > 0.5))
                        c_ok = torch.where(cs > 0.5, ce < 0.5 - args.success_margin,
                                           ce > 0.5 + args.success_margin)
                    idv = (src_id * indep_id.extract(f256)).sum(1)
                    so, eo = sp[:, others], ep[:, others]
                    clear_o = (so > CLEAR_HI) | (so < CLEAR_LO)
                    same = ((so > 0.5) == (eo > 0.5)) & clear_o
                    pres = same.float().sum(1) / clear_o.float().sum(1).clamp(min=1)
                    r = rec[(g, sc)]
                    r['acc'] += ok[keep].float().tolist()
                    r['id'] += idv[keep].tolist()
                    r['pres'] += pres[keep].tolist()
                    if arc is not None:
                        r['id_arc'] += (src_arc * arc.extract(f256)).sum(1)[keep].tolist()
                    if sc_src is not None:
                        r['celeb'] += c_ok[both].float().tolist()
            seen += img.size(0)

    rows = []
    for g in attrs:
        for sc in args.scales:
            r = rec.get((g, sc))
            if not r or not r['acc']:
                continue
            rows.append({'attribute': name(g), 'strength': sc, 'target_success': float(np.mean(r['acc'])),
                         'effective_success': float(np.mean(r['acc'])), 'id_sim_real': float(np.mean(r['id'])),
                         'preserve_acc': float(np.mean(r['pres'])), 'n': len(r['acc']),
                         'id_arc': float(np.mean(r['id_arc'])) if r['id_arc'] else float('nan'),
                         'celeb_success': float(np.mean(r['celeb'])) if r['celeb'] else float('nan'),
                         'n_celeb': len(r['celeb'])})
    for sc in args.scales:
        per = [x for x in rows if x['strength'] == sc]
        if len(per) == len(attrs):
            rows.append({'attribute': 'All', 'strength': sc,
                         **{k: float(np.mean([x[k] for x in per]))
                            for k in ('target_success', 'effective_success', 'id_sim_real', 'preserve_acc',
                                      'celeb_success', 'id_arc')},
                         'n': sum(x['n'] for x in per), 'n_celeb': sum(x['n_celeb'] for x in per)})

    label = f'{args.method}'
    print(f'\n{label}, {seen} faces, preserve_set={args.preserve_set}')
    print(f'  {"attribute":<11} {"scale":>6} {"EditAcc":>8} {"r18Acc":>7} {"ArcFace":>8} {"facenet":>8} '
          f'{"AttrPres":>9} {"n":>5}')
    for x in rows:
        print(f'  {x["attribute"]:<11} {x["strength"]:6.2f} {x["target_success"] * 100:7.1f}% '
              f'{x["celeb_success"] * 100:6.1f}% {x["id_arc"]:8.3f} {x["id_sim_real"]:8.3f} '
              f'{x["preserve_acc"] * 100:8.1f}% {x["n"]:5d}')

    path = args.out_csv or os.path.join(args.checkpoint_dir, f'curves_{args.method}_s{args.step}.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.splitext(path)[0] + '.json', 'w') as f:
        json.dump({'config': {'checkpoint_dir': args.checkpoint_dir, 'step': args.step, 'method': args.method,
                              'scales': args.scales, 'num_samples': seen, 'preserve_set': args.preserve_set,
                              'eval_resize': args.eval_resize,
                              'arcface_weights': args.arcface_weights if arc is not None else None,
                              'boundaries': args.boundaries if args.method == 'boundary' else None},
                   'rows': rows}, f, indent=2)
    print(f'saved {path}\nplot: python evaluation/plot_strength_curves.py --run NAME={path} '
          f'[--run OTHER=...] --attributes All --output_dir ./output/curves')


if __name__ == '__main__':
    main()
