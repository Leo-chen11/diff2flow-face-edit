"""Compare two runs at MATCHED identity: accuracy and side effects (--leak40).

A training change that makes the model edit less (higher ID_ind, lower LPIPS)
lowers accuracy and side effects together at the same edit_scale, so a
same-scale table cannot tell "cleaner edits" from "smaller edits". This
interpolates both runs over their scales and reads them off at the same ID_ind,
inside the ID range BOTH runs cover:

    AccInd                     add / rm success rate
    side effects (successful)  mean |dP| over the other attributes, successful edits
                               only (leak40 'mean_abs_others_success')

Each side is one or more evaluate_sdflow.py JSONs of one checkpoint (pooled like
scripts/merge_eval_scales.py). Run evaluate_sdflow.py with --leak40 for the side
effect columns.

Usage:
    python -m scripts.compare_runs_matched_id \
        --a ./output/SDFlow/multi_v1_cont20k_ctrl/eval_leak40_n500_s90000*.json \
        --b ./output/SDFlow/multi_v1_cont20k_p40recon_w10/eval_leak40_n500_s90000*.json \
        --names ctrl p40
"""
import argparse
import json

from scripts.merge_eval_scales import interp, mean


def load(files):
    scales = {}
    for f in files:
        with open(f) as fh:
            r = json.load(fh)
        for k, v in r.items():
            if isinstance(v, dict) and 'overall' in v and k not in scales:
                scales[k] = v
    return scales


def curves(scales, attr):
    """{metric: [(id, value), ...]} over the scales of one run."""
    out = {'acc': [], 'acc_add': [], 'acc_rm': [], 'leak_add': [], 'leak_rm': []}
    for s in sorted(scales, key=float):
        row = scales[s].get(attr)
        if not isinstance(row, dict):
            continue
        idv = mean(row, 'id_indep')
        if idv is None:
            continue
        for key, m in (('acc', 'acc_indep'), ('acc_add', 'acc_indep_add'), ('acc_rm', 'acc_indep_rm')):
            v = mean(row, m)
            if v is not None:
                out[key].append((idv, v))
        for d in ('add', 'rm'):
            lk = row.get(f'leak40_{d}') or {}
            v = lk.get('mean_abs_others_success')
            if v is not None:
                out[f'leak_{d}'].append((idv, v))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a', nargs='+', required=True, help='Result JSONs of the reference run.')
    p.add_argument('--b', nargs='+', required=True, help='Result JSONs of the changed run.')
    p.add_argument('--names', nargs=2, default=['A', 'B'])
    p.add_argument('--points', type=int, default=3,
                   help='Number of evenly spaced ID values inside the shared range.')
    args = p.parse_args()
    na, nb = args.names
    A, B = load(args.a), load(args.b)
    print(f'{na}: scales {", ".join(sorted(A, key=float))}    {nb}: scales {", ".join(sorted(B, key=float))}')
    first = A[sorted(A, key=float)[0]]
    attrs = [k for k, v in first.items() if isinstance(v, dict) and k not in ('overall', 'num_samples')]

    for attr in attrs:
        ca, cb = curves(A, attr), curves(B, attr)
        ida, idb = [x for x, _ in ca['acc']], [x for x, _ in cb['acc']]
        print(f'\n=== {attr} ===   ID range {na} {min(ida):.3f}-{max(ida):.3f}, '
              f'{nb} {min(idb):.3f}-{max(idb):.3f}' if ida and idb else f'\n=== {attr} === (no data)')
        if len(ida) < 2 or len(idb) < 2:
            print('  need >= 2 scales on each side')
            continue
        lo, hi = max(min(ida), min(idb)), min(max(ida), max(idb))
        if lo >= hi:
            side = nb if min(idb) > max(ida) else na
            print(f'  no shared ID range: {side} edits less at every scale; '
                  f'evaluate {side} at larger edit_scale (or the other at smaller) until they overlap')
            continue
        ids = [hi] if args.points == 1 else [lo + (hi - lo) * i / (args.points - 1) for i in range(args.points)]
        print(f'  {"@ID":>6} | {"AccInd " + na:>11} {nb:>6} {"Δ":>6} | {"add":>5} {"":>5} | {"rm":>5} {"":>5} |'
              f' {"side add " + na:>13} {nb:>6} {"Δ%":>5} | {"side rm " + na:>12} {nb:>6} {"Δ%":>5}')
        for t in ids:
            def g(c, k):
                return interp(c[k], t) if len(c[k]) > 1 else None

            def pc(v):
                return f'{v * 100:5.1f}' if v is not None else '   --'

            def lk(v):
                return f'{v:.3f}' if v is not None else '   --'

            def rel(x, y):
                return f'{(y / x - 1) * 100:+4.0f}%' if x and y is not None else '   --'

            aa, ab = g(ca, 'acc'), g(cb, 'acc')
            d = f'{(ab - aa) * 100:+5.1f}' if aa is not None and ab is not None else '   --'
            la, lb = g(ca, 'leak_add'), g(cb, 'leak_add')
            ra, rb = g(ca, 'leak_rm'), g(cb, 'leak_rm')
            print(f'  {t:6.3f} | {pc(aa):>11} {pc(ab):>6} {d:>6} | {pc(g(ca, "acc_add"))} {pc(g(cb, "acc_add"))} |'
                  f' {pc(g(ca, "acc_rm"))} {pc(g(cb, "acc_rm"))} |'
                  f' {lk(la):>13} {lk(lb):>6} {rel(la, lb):>5} | {lk(ra):>12} {lk(rb):>6} {rel(ra, rb):>5}')
    print('\nAll columns are read off at the same ID_ind (linear interpolation between scales; '
          'ID is per attribute, shared by its add and rm halves).\n'
          'add/rm columns: AccInd of each run. side = mean |dP| over the other attributes, '
          'successful edits only; Δ% < 0 means the changed run leaks less.')


if __name__ == '__main__':
    main()
