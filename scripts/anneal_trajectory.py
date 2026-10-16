"""Scores of one run along its annealing: how do the consensus metrics move as
the bank is handed over to the flow (mix 1 -> 0)?

Give it the `evaluate_sdflow.py --consensus` JSONs of several checkpoints of
ONE run (one JSON per step, each a sweep over >= 2 --eval_scales). For every
attribute it prints, per step:

    mix        bank weight the checkpoint was evaluated with (1 = all bank,
               0 = all flow). From the JSON: --bank_mode prior -> 0, replace -> 1,
               anneal -> --bank_mix, else the schedule in the run's config.json
               (bank_mix_at, the function training and eval share).
    ID         ID_ind at --scale (interpolated between the evaluated scales)
    LPIPS, FID at the same --scale
    AccBoth    consensus accuracy (CLIP and CelebA both say the edit worked)
               at the same ID for every step: --points ID values spread over the
               ID range ALL steps cover for that attribute. Steps differ in how
               large their edits are at one scale, so a same-scale table would
               mix "edits better" with "edits smaller".
    AccCeleb / AccCLIP at the same ID points (--judges)

and an attribute-mean row (the i-th ID point of each attribute, averaged).
Nothing is extrapolated; -- means the step does not reach that ID.

With --labels the rows are labelled files instead of steps, so two runs'
checkpoints (e.g. min 80k vs min_robust 80k) can be compared at matched ID.

Usage:
    python -m scripts.anneal_trajectory \
        $R/eval_anneal_s60000_cons.json $R/eval_anneal_s65000_cons.json ... \
        --scale 0.9 --points 3 --out $R/anneal_trajectory.json
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.direction_bank import bank_mix_at
from scripts.merge_eval_scales import interp, mean

JUDGES = {'both': 'AccBoth', 'celeb': 'AccCeleb', 'clip': 'AccCLIP', 'indep': 'AccInd'}


def run_mix(cfg, args):
    mode = cfg.get('bank_mode', 'replace')
    if mode == 'prior':
        return 0.0
    if mode == 'replace':
        return 1.0
    if mode != 'anneal':
        return None
    if cfg.get('bank_mix') is not None:
        return float(cfg['bank_mix'])
    rc = {}
    path = os.path.join(cfg.get('checkpoint_dir') or '', 'config.json')
    if os.path.exists(path):
        with open(path) as f:
            rc = json.load(f)
    return bank_mix_at(cfg.get('step'),
                       rc.get('bank_mix_start', 1.0), rc.get('bank_mix_end', 0.0),
                       rc.get('bank_anneal_start_step', args.anneal_start),
                       rc.get('bank_anneal_steps', args.anneal_steps))


def load_step(path):
    with open(path) as f:
        r = json.load(f)
    scales = {k: v for k, v in r.items() if isinstance(v, dict) and 'overall' in v}
    if len(scales) < 2:
        raise SystemExit(f'{path}: needs >= 2 evaluated scales, found {len(scales)}')
    return r.get('config', {}), scales


def attr_names(scales):
    first = scales[sorted(scales, key=float)[0]]
    return [a for a, v in first.items()
            if isinstance(v, dict) and 'consensus' in v]


def points(scales, attr, key):
    pts = []
    for s in sorted(scales, key=float):
        c = (scales[s].get(attr) or {}).get('consensus') or {}
        i, v = mean(c, 'id_indep'), mean(c, key)
        if i is not None and v is not None:
            pts.append((i, v))
    return pts


def at_scale(scales, attr, key, scale):
    pts = []
    for s in sorted(scales, key=float):
        c = (scales[s].get(attr) or {}).get('consensus') or {}
        v = mean(c, key)
        if v is not None:
            pts.append((float(s), v))
    return interp(pts, scale)


def bg_at(scales, attr, scale):
    """evaluate_sdflow.py --bg_metric (all faces, not only consensus ones)."""
    pts = [(float(s), v[attr]['bg_mse']['mean']) for s, v in scales.items()
           if isinstance(v.get(attr), dict) and (v[attr].get('bg_mse') or {}).get('mean') is not None]
    return interp(pts, scale)


def fid_at(scales, scale):
    pts = [(float(s), v['fid_edit_vs_recon']) for s, v in scales.items()
           if v.get('fid_edit_vs_recon') is not None]
    return interp(pts, scale)


def fmt(v, pct=False, nd=3):
    if v is None:
        return '   --  '
    return f'{v * 100:6.1f}%' if pct else f'{v:7.{nd}f}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('files', nargs='+', help='--consensus JSONs, one per checkpoint step.')
    p.add_argument('--scale', type=float, default=0.9, help='scale for ID / LPIPS / FID columns')
    p.add_argument('--points', type=int, default=3, help='matched-ID points per attribute')
    p.add_argument('--judges', nargs='+', default=['both', 'celeb', 'clip'], choices=list(JUDGES))
    p.add_argument('--labels', nargs='+', default=None,
                   help='one label per file, in the order given, instead of the checkpoint step '
                        '(and no sorting): use it to compare checkpoints of DIFFERENT runs at the '
                        'same ID points, e.g. --labels min min_robust')
    p.add_argument('--anneal_start', type=int, default=50000,
                   help='schedule fallback when the run has no config.json on this machine')
    p.add_argument('--anneal_steps', type=int, default=20000)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    runs = []
    for f in args.files:
        cfg, scales = load_step(f)
        if not cfg.get('consensus'):
            raise SystemExit(f'{f} was not run with --consensus')
        runs.append((cfg.get('step'), run_mix(cfg, args), cfg, scales, f))
    if args.labels:
        if len(args.labels) != len(runs):
            raise SystemExit('--labels needs one label per file')
        runs = [(lab,) + r[1:] for lab, r in zip(args.labels, runs)]
    else:
        runs.sort(key=lambda r: (r[0] is None, r[0]))
    partner = {str(r[2].get('consensus_second') or 'clip') for r in runs}
    if len(partner) > 1:
        raise SystemExit(f'files pair CelebA with different judges (consensus_second: {sorted(partner)}): '
                         f'AccBoth means something different in each')
    other = {'clip': 'indep', 'indep': 'clip'}[partner.pop()]
    if other in args.judges:
        print(f'[note] these files have no consensus acc_{other}; dropping that column')
        args.judges = [j for j in args.judges if j != other]
    for key in (() if args.labels else ('checkpoint_dir',)) + ('num_samples', 'sample_offset', 'edit_direction',
                'no_train_caps', 'independent_attr_weights'):
        vals = {str(r[2].get(key)) for r in runs}
        if len(vals) > 1:
            print(f'[warn] files differ in {key}: {sorted(vals)} -- not like-for-like')
    modes = {r[2].get('bank_mode') for r in runs}
    if len(modes) > 1:
        print(f'[warn] files use different bank_mode {sorted(map(str, modes))}')

    attrs = [a for a in attr_names(runs[0][3]) if all(a in attr_names(r[3]) for r in runs)]
    print(f'rows: {", ".join(str(r[0]) for r in runs)}   bank_mode: {sorted(map(str, modes))}'
          f'   scale for ID/LPIPS/FID: {args.scale:g}')

    report = {}
    per_pos = {j: [[] for _ in range(args.points)] for j in args.judges}
    for a in attrs:
        # ID range every step covers for this attribute
        ranges = []
        for _, _, _, sc, _ in runs:
            ids = [i for i, _ in points(sc, a, 'acc_both')]
            if ids:
                ranges.append((min(ids), max(ids)))
        if len(ranges) < len(runs):
            continue
        lo, hi = max(r[0] for r in ranges), min(r[1] for r in ranges)
        if lo >= hi:
            print(f'\n{a}: the steps cover no common ID range, skipped')
            continue
        grid = [lo + (hi - lo) * (k + 0.5) / args.points for k in range(args.points)]
        print(f'\n{a}   matched ID_ind points: ' + ', '.join(f'{g:.3f}' for g in grid))
        head = f'   {"step":>10} {"mix":>5} {"ID":>7} {"LPIPS":>7} {"BG":>7} {"FID":>7}'
        for j in args.judges:
            head += '   ' + f'{JUDGES[j]:<8}' + ' ' * (8 * (args.points - 1))
        print(head)
        report[a] = {'id_grid': grid, 'steps': {}}
        for step, mix, cfg, sc, _ in runs:
            row = {'mix': mix, 'id': at_scale(sc, a, 'id_indep', args.scale),
                   'lpips': at_scale(sc, a, 'lpips', args.scale), 'fid': fid_at(sc, args.scale),
                   'bg_mse': bg_at(sc, a, args.scale)}
            line = f'   {str(step):>10} {"--" if mix is None else f"{mix:.2f}":>5} {fmt(row["id"])} ' \
                   f'{fmt(row["lpips"])} {fmt(row["bg_mse"], nd=4)} {fmt(fid_at(sc, args.scale), nd=2)}'
            for j in args.judges:
                vals = [interp(points(sc, a, f'acc_{j}'), g) for g in grid]
                row[f'acc_{j}'] = vals
                for k, v in enumerate(vals):
                    if v is not None:
                        per_pos[j][k].append((step, v))
                line += '   ' + ''.join(fmt(v, pct=True) for v in vals)
            print(line)
            report[a]['steps'][str(step)] = row

    print('\nAttribute mean (i-th ID point of each attribute, averaged)')
    head = f'   {"step":>10} {"mix":>5}'
    for j in args.judges:
        head += '   ' + f'{JUDGES[j]:<8}' + ' ' * (8 * (args.points - 1))
    print(head)
    summary = {}
    for step, mix, _, _, _ in runs:
        line = f'   {str(step):>10} {"--" if mix is None else f"{mix:.2f}":>5}'
        summary[str(step)] = {'mix': mix}
        for j in args.judges:
            cells = []
            for k in range(args.points):
                vs = [v for s, v in per_pos[j][k] if s == step]
                n_attr = sum(1 for a in report if str(step) in report[a]['steps'])
                cells.append(sum(vs) / len(vs) if vs and len(vs) == n_attr else None)
            summary[str(step)][f'acc_{j}'] = cells
            line += '   ' + ''.join(fmt(v, pct=True) for v in cells)
        print(line)

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'files': args.files, 'scale': args.scale, 'per_attr': report,
                       'attr_mean': summary}, f, indent=2)
        print(f'\nsaved {args.out}')


if __name__ == '__main__':
    main()
