"""Combine several evaluate_sdflow.py result files of ONE checkpoint into a single
accuracy-vs-identity table, with accuracy at matched ID_ind over ALL their scales.

evaluate_sdflow.py only interpolates over the scales of its own run, so two runs
(e.g. scales 0.7 1.0 and scales 0.85 1.25) each print a mostly empty matched-ID
table. This pools the scales and interpolates once.

Only files that share the checkpoint, step, sample count and --edit_direction
make sense to pool; mismatches are reported, and a scale that appears twice
keeps the first file's value.

Usage:
    python -m scripts.merge_eval_scales \
        ./output/SDFlow/multi_v1_smile_bangs/eval_v2_step80000_n250_prev*.json \
        ./output/SDFlow/multi_v1_smile_bangs/eval_v2_step80000_n250.json \
        --judge indep --at 0.90 0.85 0.80
"""
import argparse
import json

JUDGES = {'indep': 'AccInd', 'clip': 'AccCLIP', 'celeb': 'AccCeleb'}


def interp(points, x):
    pts = sorted(points)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return None


def mean(row, key):
    v = row.get(key)
    return v['mean'] if isinstance(v, dict) and v.get('mean') is not None else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('files', nargs='+', help='evaluate_sdflow.py result JSONs of one checkpoint.')
    p.add_argument('--judge', default='indep', choices=list(JUDGES))
    p.add_argument('--at', nargs='*', type=float, default=[0.90, 0.85, 0.80],
                   help='ID_ind values to report accuracy at.')
    p.add_argument('--out', default=None, help='Optional JSON output.')
    args = p.parse_args()

    scales, cfgs = {}, []
    for f in args.files:
        with open(f) as fh:
            r = json.load(fh)
        cfg = r.get('config', {})
        cfgs.append((f, cfg))
        for k, v in r.items():
            if isinstance(v, dict) and 'overall' in v:
                if k in scales:
                    print(f'[warn] scale {k} also in {f}; keeping the earlier file\'s value')
                    continue
                scales[k] = v
    if len(scales) < 2:
        raise SystemExit('need at least two distinct scales to interpolate')

    for key in ('checkpoint_dir', 'step', 'num_samples', 'independent_attr_weights',
                'edit_direction'):
        vals = {str(c.get(key)) for _, c in cfgs}
        if len(vals) > 1:
            print(f'[warn] files differ in {key}: {sorted(vals)} -- pooling them is not like-for-like')

    order = sorted(scales, key=float)
    attrs = [a for a in scales[order[0]] if isinstance(scales[order[0]][a], dict)
             and a not in ('overall', 'num_samples')]
    jk, jl = f'acc_{args.judge}', JUDGES[args.judge]
    print(f'scales pooled: {", ".join(order)}   judge: {jl}')

    report = {}
    for a in attrs:
        rows = [(float(s), mean(scales[s][a], 'id_indep'), mean(scales[s][a], jk),
                 mean(scales[s][a], f'{jk}_add'), mean(scales[s][a], f'{jk}_rm'),
                 mean(scales[s][a], 'lpips')) for s in order
                if a in scales[s]]
        rows = [r for r in rows if r[1] is not None and r[2] is not None]
        if len(rows) < 2:
            continue
        print(f'\n=== {a} ===')
        print(f'  {"scale":>6} {"ID_ind":>7} {jl:>8} {"add":>7} {"rm":>7} {"LPIPS":>7}')
        for sc, idv, acc, add, rm, lp in rows:
            fa = f'{add * 100:6.1f}%' if add is not None else '     --'
            fr = f'{rm * 100:6.1f}%' if rm is not None else '     --'
            fl = f'{lp:7.4f}' if lp is not None else '     --'
            print(f'  {sc:>6.2f} {idv:>7.4f} {acc * 100:7.1f}% {fa} {fr} {fl}')
        pts = [(r[1], r[2]) for r in rows]
        lpts = [(r[1], r[5]) for r in rows if r[5] is not None]
        cells, rep, lrep = [], {}, {}
        for t in args.at:
            v = interp(pts, t)
            rep[f'{t:.2f}'] = v
            lv = interp(lpts, t) if len(lpts) > 1 else None
            lrep[f'{t:.2f}'] = lv
            cells.append(f'@ID {t:.2f}: ' + (f'{v * 100:5.1f}%' if v is not None else '   -- ')
                         + (f' (LPIPS {lv:.3f})' if lv is not None and v is not None else ''))
        print('  ' + '   '.join(cells) + f'   (ID range {min(r[1] for r in rows):.3f}'
              f'-{max(r[1] for r in rows):.3f})')
        report[a] = {'at_id': rep, 'lpips_at_id': lrep,
                     'per_scale': [dict(scale=r[0], id=r[1], acc=r[2], lpips=r[5]) for r in rows]}

    ov = [(s, mean(scales[s]['overall'], 'id_indep'), mean(scales[s]['overall'], jk))
          for s in order]
    print(f'\n=== Overall ===\n  {"scale":>6} {"ID_ind":>7} {jl:>8}')
    for s, idv, acc in ov:
        if idv is not None and acc is not None:
            print(f'  {float(s):>6.2f} {idv:>7.4f} {acc * 100:7.1f}%')
    fids = [(s, scales[s].get('fid_edit_vs_recon')) for s in order]
    if any(v is not None for _, v in fids):
        print('\nFID (edited vs source recon, lower = closer to the reconstructions): '
              + '   '.join(f'x{float(s):.2f}: {v:.1f}' for s, v in fids if v is not None))
    print('\nLinear interpolation between neighbouring scales; -- = outside the evaluated ID range '
          '(add a scale rather than extrapolating).')
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(report, fh, indent=2)
        print(f'saved {args.out}')


if __name__ == '__main__':
    main()
