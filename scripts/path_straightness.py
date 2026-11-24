"""How straight the flow's editing paths are, and how many solver steps an edit needs -- the two things
the transport regulariser (train --transport_kinetic_weight / --transport_jacobian_weight, with
--transport_target full or attr) is meant to change. Compare a run with and without it on the SAME faces.

For each face and attribute, the edit of scripts/editing_curves.py (encode the source under its own
reading, decode under the target reading, strength --scale, direction from the conditioner) is solved
latent-only (no log-det) and measured inside each CNF block, on the trajectory of the ODE itself:

  straightness   path length / distance between the ends, per solve (encode and decode separately),
                 on a --grid-point time grid with a tight adaptive solve: 1.00 = a straight line
  attr share     the attribute part's share of the velocity, mean over the path of ||v_attr|| / ||v||
                 (decomposed fields only): how much of the motion the attributes drive
  few-step error the relative error of the EDIT when solved with a fixed-step solver of K steps instead:
                 ||w'_K - w'_ref|| / ||w'_ref - w||, per solver (euler, midpoint, rk4) and K (1 2 4 8);
                 a straighter field gets a small error with fewer steps
  NFE            velocity evaluations of the reference (adaptive, --ref_tol) edit, encode + decode

The edit measured is the flow's raw output (before the direction bank and the delta cap), the part the
regulariser acts on.

    python -m scripts.path_straightness --checkpoint_dir <run> --step 100000 --num_samples 100

Writes <out_dir>/path_straightness.{csv,md}. Development faces by default (offset 0): a diagnostic.
"""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torchvision.transforms as T
from torch.utils import data
from tqdm import tqdm

from evaluation.evaluate_sdflow import ATTR_NAMES, apply_run_config, build_parser, edited_attr_value, load_models
from models.dataset import SDFlowDataset
from models.flows.cnf import CNF
from models.flows.flow import configure_inference, flow_nfe
from models.flows.odefunc import LAGDOFODEnet


def solve_path(flow, x, cond, reverse, grid):
    """Run the flow's chain (BN, CNF, BN, ...) forward or backward; inside each CNF block return the
    whole trajectory on `grid` time points. -> (output, [trajectory (G, B, 18, 512) per CNF block])."""
    chain = list(flow.chain)
    order = range(len(chain) - 1, -1, -1) if reverse else range(len(chain))
    trajs = []
    for i in order:
        m = chain[i]
        if isinstance(m, CNF):
            t = torch.linspace(0.0, float(m.T), grid, device=x.device)
            tr = m(x, cond, None, integration_times=t, reverse=reverse)     # (G, B, 18, 512)
            trajs.append(tr)
            x = tr[-1]
        else:
            x = m(x, cond, None, None, reverse)
    return x, trajs


def straightness(tr):
    """(B,) path length / end-to-end distance of trajectories (G, B, ...)."""
    seg = (tr[1:] - tr[:-1]).flatten(2).norm(dim=-1).sum(0)
    end = (tr[-1] - tr[0]).flatten(1).norm(dim=-1)
    return seg / end.clamp(min=1e-12)


def attr_share(flow, tr, cond):
    """(B,) mean over the path of ||v_attr|| / ||v|| for a decomposed field (None otherwise)."""
    blocks = [m for m in flow.modules() if isinstance(m, CNF)]
    diffeq = blocks[0].odefunc.diffeq
    if not isinstance(diffeq, LAGDOFODEnet):
        return None
    ratios = []
    G = tr.size(0)
    for k in range(G):
        t = torch.full((tr.size(1), 1), float(k) / (G - 1) * float(blocks[0].T), device=tr.device)
        with torch.no_grad():
            v = diffeq(torch.cat([t, cond], 1), tr[k])
            va = diffeq._last_v_edit
        ratios.append(va.flatten(1).norm(dim=-1) / v.flatten(1).norm(dim=-1).clamp(min=1e-12))
    return torch.stack(ratios).mean(0)


def main():
    p = build_parser()
    p.add_argument('--scale', type=float, default=1.0, help='edit strength (the curves\' units)')
    p.add_argument('--grid', type=int, default=33, help='time points per solve for the path length')
    p.add_argument('--ref_tol', type=float, default=1e-6, help='tolerance of the reference adaptive solve')
    p.add_argument('--steps', nargs='+', type=int, default=[1, 2, 4, 8])
    p.add_argument('--solvers', nargs='+', default=['euler', 'midpoint', 'rk4'])
    p.add_argument('--out_dir', default=None, help='default <checkpoint_dir>/path_straightness_s<step>')
    args = apply_run_config(p.parse_args())
    out_dir = args.out_dir or os.path.join(args.checkpoint_dir, f'path_straightness_s{args.step}')
    os.makedirs(out_dir, exist_ok=True)

    prior, conditioner, G, id_criterion, _, _, _ = load_models(args)
    prior.eval()
    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)), T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root, latents_file=args.latent_file,
                       preds_file=args.preds_file, train=False, transform=tf)
    if args.sample_offset:
        ds = data.Subset(ds, range(args.sample_offset, len(ds)))
    loader = data.DataLoader(ds, shuffle=False, batch_size=args.batch, num_workers=2)
    attrs = list(args.attribute_index)

    acc = defaultdict(list)            # (attr, metric) -> per-face values
    seen = 0
    with torch.no_grad():
        for img, latent, _ in tqdm(loader, desc='paths'):
            if seen >= args.num_samples:
                break
            img, latent = img.cuda(), latent.cuda()
            _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
            src_c = torch.cat([id_cond, attr_cond], 1)
            for li, g in enumerate(attrs):
                new_attr = attr_cond.clone()
                new_attr[:, li] = edited_attr_value(attr_cond[:, li], args.scale, g)
                tgt_c = torch.cat([id_cond, new_attr], 1)
                # reference: tight adaptive, latent only
                configure_inference(prior, solver='dopri5', tol=args.ref_tol, steps=None, skip_divergence=True)
                z, enc = solve_path(prior, latent, src_c, False, args.grid)
                w_ref, dec = solve_path(prior, z, tgt_c, True, args.grid)
                # NFE of a plain reference edit (2 time points, as an edit is solved)
                prior(latent, src_c, None); ne = flow_nfe(prior)
                zz = prior(latent, src_c, None)
                prior(zz, tgt_c, None, reverse=True); nd = flow_nfe(prior)
                edit = (w_ref - latent).flatten(1).norm(dim=-1).clamp(min=1e-12)
                name = ATTR_NAMES.get(g, str(g))
                acc[(name, 'straight_encode')] += torch.stack([straightness(t) for t in enc]).mean(0).tolist()
                acc[(name, 'straight_decode')] += torch.stack([straightness(t) for t in dec]).mean(0).tolist()
                sh = attr_share(prior, dec[0], tgt_c)
                if sh is not None:
                    acc[(name, 'attr_share_decode')] += sh.tolist()
                acc[(name, 'nfe_ref')] += [ne + nd] * latent.size(0)
                for solver in args.solvers:
                    for K in args.steps:
                        configure_inference(prior, solver=solver, steps=K, skip_divergence=True)
                        zk = prior(latent, src_c, None)
                        wk = prior(zk, tgt_c, None, reverse=True)
                        err = (wk - w_ref).flatten(1).norm(dim=-1) / edit
                        acc[(name, f'err_{solver}{K}')] += err.tolist()
            seen += latent.size(0)
    configure_inference(prior, solver='dopri5', tol=1e-5, steps=None, skip_divergence=False)

    names = [ATTR_NAMES.get(g, str(g)) for g in attrs]
    metrics = sorted({m for _, m in acc}, key=lambda m: (not m.startswith('straight'), not m.startswith('attr'),
                                                         not m.startswith('nfe'), m))
    rows = []
    for a in names + ['All']:
        r = {'attribute': a}
        for m in metrics:
            vals = acc.get((a, m)) if a != 'All' else [v for n in names for v in acc.get((n, m), [])]
            r[m] = sum(vals) / len(vals) if vals else float('nan')
        rows.append(r)
    with open(os.path.join(out_dir, 'path_straightness.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['attribute'] + metrics)
        w.writeheader()
        w.writerows(rows)
    md = [f'# Editing paths: {args.checkpoint_dir} step {args.step}, {seen} faces, strength {args.scale:g}', '',
          '| attribute | ' + ' | '.join(metrics) + ' |', '|' + '---|' * (len(metrics) + 1)]
    for r in rows:
        md.append(f'| {r["attribute"]} | ' + ' | '.join(f'{r[m]:.3f}' if not m.startswith('nfe') else f'{r[m]:.0f}'
                                                         for m in metrics) + ' |')
    md += ['', 'straight_*: path length / end-to-end distance inside the CNF (1 = straight). attr_share_decode: '
           'mean ||v_attr|| / ||v|| along the decode. err_<solver><K>: relative error of the edit with K fixed '
           f'steps against a dopri5 tol {args.ref_tol:g} solve. nfe_ref: evaluations of that reference edit.']
    text = '\n'.join(md) + '\n'
    open(os.path.join(out_dir, 'path_straightness.md'), 'w').write(text)
    print(text)
    print(f'saved {out_dir}/path_straightness.csv and .md')


if __name__ == '__main__':
    main()
