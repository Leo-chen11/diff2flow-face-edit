"""Add attributes to an existing direction bank without touching its rows.

A training run picks its rows out of the bank by --attribute_index, so one
bank can serve many attribute sets. Rebuilding it from scratch would need
the exact precompute_directions_stratified.py arguments of the original
(and would change the existing Eyeglasses / Male / Young rows). This keeps
every existing row byte-identical and appends the new attributes, computed
with the same generic gender x age stratified routine precompute uses for
any attribute beyond 15/20/39, with K, extreme_pct and direction_method read
from the bank itself.

Usage:
    python -m scripts.extend_direction_bank \
        --bank ./data/direction_bank_k4_stratified_v3.pth \
        --add 31 5 --out ./data/direction_bank_multi.pth
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F

from common.attr_tables import bank_num_k
from scripts.precompute_directions_stratified import (
    compute_generic_directions, intra_attr_orthogonalize_safe, load_latents, load_paths,
    load_preds, project_out_direction, representative_direction,
    sanitize_non_finite_directions,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bank', required=True, help='Existing bank (.pth).')
    p.add_argument('--add', nargs='+', type=int, required=True, help='CelebA attribute ids to add.')
    p.add_argument('--out', required=True)
    p.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    p.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth')
    p.add_argument('--continuous_preds_file', default='./data/ffhq_e4e_preds_continuous.pth')
    p.add_argument('--min_samples', type=int, default=50)
    p.add_argument('--substyle_k', type=int, default=None,
                   help='Sub-styles per stratum for the new attributes. Default: bank K / 4 '
                        '(the generic routine has 4 gender x age strata).')
    p.add_argument('--decorrelate', action=argparse.BooleanOptionalAction, default=None,
                   help='Project the other attributes\' representative directions out of the NEW '
                        'rows (existing rows are never changed). Default: what the bank used.')
    args = p.parse_args()

    bank = torch.load(args.bank, map_location='cpu')
    attrs = [int(a) for a in bank['attribute_index']]
    du, ln = bank['direction_units'].float(), bank['layer_norms'].float()
    if du.ndim == 3:
        du, ln = du.unsqueeze(1), ln.unsqueeze(1)
    K = bank_num_k(bank, default=du.shape[1])
    pct = float(bank.get('extreme_pct', 20.0))
    method = bank.get('direction_method', 'lda')
    decorrelate = bool(bank.get('decorrelate_cross_attr', False)) if args.decorrelate is None \
        else args.decorrelate
    substyle_k = args.substyle_k or max(1, K // 4)
    new = [a for a in args.add if a not in attrs]
    if not new:
        raise SystemExit(f'all of {args.add} are already in the bank {attrs}')
    print(f'bank {args.bank}: attrs {attrs}, K={K}, method={method}, extreme_pct={pct}, '
          f'decorrelate_cross_attr={bank.get("decorrelate_cross_attr")}')
    print(f'adding {new} with substyle_k={substyle_k} (4 strata x {substyle_k} = '
          f'{4 * substyle_k} directions, fit to K={K})')

    latents = load_latents(args.latent_file)
    preds = load_preds(args.preds_file)
    continuous = load_preds(args.continuous_preds_file)
    pp, cp = load_paths(args.preds_file), load_paths(args.continuous_preds_file)
    if preds.shape[0] != continuous.shape[0] or (pp is not None and cp is not None and pp != cp):
        raise SystemExit('--preds_file and --continuous_preds_file rows do not match')

    kw = dict(pct=pct, method=method, shrinkage=None, strata_margin=0.0, group_center='mean',
              group_trim_frac=0.1, min_conf=0.0)
    new_dirs = {}
    for a in new:
        print(f'\n=== attr {a} (generic, gender x age strata) ===')
        d = compute_generic_directions(a, latents, preds, continuous, K, args.min_samples,
                                       cross_scores=None, substyle_k=substyle_k, **kw)
        if d.shape[0] < K:            # pad like precompute does for short stratifications
            d = torch.cat([d, d[-1:].expand(K - d.shape[0], -1, -1)], dim=0)
        new_dirs[a] = d[:K]

    if decorrelate:
        print('\n=== Removing other attributes\' representative directions from the new rows ===')
        existing_raw = {a: ln[i].unsqueeze(-1) * du[i] for i, a in enumerate(attrs)}
        reps = {a: representative_direction(v) for a, v in {**existing_raw, **new_dirs}.items()}
        for a in new:
            d = new_dirs[a]
            for b, r in reps.items():
                if b != a:
                    d = project_out_direction(d, r)
            new_dirs[a] = d

    stacked = torch.stack([new_dirs[a] for a in new])
    for i in range(stacked.shape[0]):
        stacked[i] = intra_attr_orthogonalize_safe(stacked[i])
    stacked = sanitize_non_finite_directions(stacked, new, latents, continuous, pct=pct,
                                             method=method, shrinkage=None, group_center='mean',
                                             group_trim_frac=0.1)
    new_ln = stacked.norm(dim=-1)
    new_du = F.normalize(stacked, dim=-1, eps=1e-8)

    for i, a in enumerate(new):
        cos = torch.einsum('ild,jld->ijl', new_du[i], new_du[i]).abs().mean(-1)
        worst = (cos - torch.eye(K)).max().item()
        flag = '  <-- near-duplicate slots (gate choice matters less for this attr)' \
            if worst >= 0.99 else ''
        print(f'  attr {a}: worst slot-pair mean|cos| = {worst:.4f}, '
              f'mean layer norm = {new_ln[i].mean():.3f}{flag}')
    for i, a in enumerate(new):
        for j, b in enumerate(attrs):
            c = (new_du[i, 0] * du[j, 0]).sum(-1).mean().item()
            if abs(c) > 0.1:
                print(f'  attr {a} vs existing attr {b}: K0 cos = {c:+.3f}')

    out = dict(bank)
    out['direction_units'] = torch.cat([du, new_du], dim=0)
    out['layer_norms'] = torch.cat([ln, new_ln], dim=0)
    out['attribute_index'] = attrs + new
    strat = dict(bank.get('stratification', {}))
    for a in new:
        strat[a] = ['male_young', 'male_old', 'female_young', 'female_old']
    out['stratification'] = strat
    out['extended_from'] = os.path.abspath(args.bank)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    print(f'\nsaved {args.out}: attrs {out["attribute_index"]}, '
          f'direction_units {tuple(out["direction_units"].shape)}')
    assert torch.equal(out['direction_units'][:len(attrs)], du), 'existing rows changed'


if __name__ == '__main__':
    main()
