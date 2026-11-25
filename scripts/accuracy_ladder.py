"""Every method read at the SAME editing accuracies (default R50 40 50 60 70 80 90 100%): at each one,
the strength in the method's own units, the ArcFace identity, and the other judge's accuracy -- and,
where a method's curve does not reach an accuracy, why.

Each method's curve (scripts/editing_curves.py CSV) is a list of (strength, accuracy, ArcFace) points.
An accuracy is read where the curve FIRST crosses it, interpolated linearly between the two measured
strengths around it. Three outcomes per cell:

  ok          read off the curve (interpolated between two measured strengths, or a measured point)
  below       the WEAKEST measured strength already edits more than that: the curve starts above the
              target. Not a property of the method -- add a smaller strength to its --scales.
  not reached the STRONGEST measured strength edits less than that. The note says how far it got, at
              what strength and identity, and whether the curve was still rising there (then a larger
              strength may reach it: add one to --scales), had flattened (the method saturates below
              the target: a real limit -- e.g. DeltaEdit's Young, which its paper does not cover), or
              had already destroyed the identity (ArcFace < 0.35: not chased further)

100% means every scored face succeeded; a pooled curve rarely gets there exactly, so its cell is
usually "not reached" with the highest accuracy shown.

Pooling: the curve of a method is its "All" row (every attribute it covers, faces pooled), or, with
--pool common, pooled over only the attributes EVERY method covers (GradCtrl has no Bangs, DeltaEdit
no Young), weighting each attribute by its number of scored faces -- the like-for-like comparison.

    python -m scripts.accuracy_ladder --curves output/curves_consensus
    python -m scripts.accuracy_ladder --curves output/curves_consensus --pool common --judge r18
    python -m scripts.accuracy_ladder --run Ours=a.csv --run SDFlow=b.csv --targets 40 50 60 70 80 90 100

Writes <out_dir>/ladder_<judge>_<pool>.{md,csv,png}: the table (strength / ArcFace / other judge per
cell, notes under it) and a figure (ArcFace against accuracy, the 7 points marked on each method's
curve; a missing point is listed in the legend).
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

ATTRS = ['Eyeglasses', 'Male', 'Young', 'Smiling', 'Bangs']
# No method is given the reading rule (CR) by default: the flow methods (ours, SDFlow, StyleFlow) edit
# from their own reading of the source, and every one of them is run as it is. The CR variants are an
# option (--variants cr / both) for the analysis of the rule, not part of the comparison.
BASE_RUNS = [                       # methods that move along the request whatever the source looks like
    ('AdaTrans', 'adatrans_off500.csv'), ('DeltaEdit', 'deltaedit_off500.csv'),
    ('GradCtrl', 'gradctrl_off500.csv'), ('PluGeN', 'plugen_off500.csv'),
    ('LT', 'lt_off500.csv'),
]                                   # InterFaceGAN is not part of the comparison (add it with --run)
NOCR_RUNS = [('Ours', 'ours_nocr_off500.csv'), ('SDFlow', 'sdflow5_off500.csv'),
             ('StyleFlow', 'styleflow_off500.csv')]
CR_RUNS = [('Ours+CR', 'ours_cr0.1_off500.csv'), ('SDFlow+CR', 'sdflow5_cr0.1_off500.csv'),
           ('StyleFlow+CR', 'styleflow_cr0.1_off500.csv')]
DEFAULT_RUNS = NOCR_RUNS + BASE_RUNS


def runs_for(variants='nocr'):
    """The default (name, csv) list: nocr (default, no CR anywhere), cr (the flow methods with the
    reading rule), both."""
    flows = {'nocr': NOCR_RUNS, 'cr': CR_RUNS, 'both': NOCR_RUNS + CR_RUNS}[variants]
    return flows + BASE_RUNS
FLAT = 0.005          # accuracy gain (0-1) between the last two strengths below which a curve has flattened
ID_FLOOR = 0.35       # ArcFace below which the face is no longer the same person: a target not reached by
                      # then is not chased with larger strengths


def load(path):
    """{attribute: {strength: row}} with the numbers as floats."""
    out = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        row = {}
        for k, v in r.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                pass
        out[r['attribute']][row['strength']] = row
    return out


def pooled(data, attrs):
    """[(strength, r50, r18, arcface)] pooled over `attrs` at the strengths every one of them has,
    each attribute weighted by its scored faces (n for R50 / ArcFace, n_celeb for r18)."""
    have = [a for a in attrs if a in data]
    if not have:
        return []
    strengths = sorted(set.intersection(*[set(data[a]) for a in have]))
    pts = []
    for s in strengths:
        rows = [data[a][s] for a in have]
        n = sum(r.get('n', 0) for r in rows)
        nc = sum(r.get('n_celeb', 0) for r in rows)
        if not n:
            continue
        r50 = sum(r['target_success'] * r.get('n', 0) for r in rows) / n
        arc = sum(r['id_arc'] * r.get('n', 0) for r in rows) / n
        r18 = (sum(r['celeb_success'] * r.get('n_celeb', 0) for r in rows) / nc) if nc else float('nan')
        pts.append((s, r50, r18, arc))
    return pts


def all_row(data):
    return sorted((s, r['target_success'], r['celeb_success'], r['id_arc']) for s, r in data.get('All', {}).items())


def ladder(pts, target, key):
    """One cell: (status, strength, r50, r18, arcface, note). key 1 = R50, 2 = r18 decides the accuracy."""
    if not pts:
        return ('missing', None, None, None, None, 'no curve')
    acc = [p[key] for p in pts]
    for i, p in enumerate(pts):
        if p[key] >= target - 1e-9:
            if i == 0:
                if abs(p[key] - target) < 1e-9:
                    return ('ok',) + p + ('',)
                return ('below', *p, f'weakest measured strength {p[0]:g} already gives {p[key] * 100:.1f}%: '
                        f'add a smaller strength')
            q = pts[i - 1]
            t = (target - q[key]) / (p[key] - q[key]) if p[key] != q[key] else 1.0
            mix = tuple(a + t * (b - a) for a, b in zip(q, p))
            return ('ok',) + mix + ('',)
    top = max(range(len(pts)), key=lambda i: acc[i])
    p = pts[top]
    rising = len(pts) >= 2 and pts[-1][key] - pts[-2][key] > FLAT and top == len(pts) - 1
    if target >= 1.0 - 1e-9 and p[key] >= 0.99:
        why = ('100% means every scored face succeeds; the last few faces resist any strength, so a pooled '
               'curve stops just short of it')
    elif rising and pts[-1][3] < ID_FLOOR:
        why = (f'identity already lost at the strongest measured strength (ArcFace {pts[-1][3]:.3f} < {ID_FLOOR}): '
               'not chased further')
    elif rising:
        why = 'still rising at the strongest measured strength: a larger strength may reach it'
    else:
        why = 'flattened: the method saturates below this accuracy'
    return ('not reached', *p, f'highest {p[key] * 100:.1f}% at strength {p[0]:g} (ArcFace {p[3]:.3f}); {why}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', default=None, metavar='NAME=curves.csv')
    p.add_argument('--curves', default='output/curves')
    p.add_argument('--variants', default='nocr', choices=['nocr', 'cr', 'both'],
                   help='which flow-method curves: nocr (default: no method gets the reading rule), cr, both')
    p.add_argument('--targets', nargs='+', type=float, default=[40, 50, 60, 70, 80, 90, 100])
    p.add_argument('--judge', default='r50', choices=['r50', 'r18'], help='whose accuracy is matched')
    p.add_argument('--pool', default='all', choices=['all', 'common'],
                   help='all: each method\'s own "All" curve; common: pooled over the attributes every method covers')
    p.add_argument('--attribute', default=None, help='one attribute instead of a pooled curve')
    p.add_argument('--out_dir', default=None, help='default <curves>/ladder')
    args = p.parse_args()

    runs = []
    if args.run:
        for item in args.run:
            name, _, path = item.partition('=')
            runs.append((name.strip(), path.strip()))
    else:
        runs = [(n, os.path.join(args.curves, f)) for n, f in runs_for(args.variants)]
    data = {}
    for name, path in runs:
        if os.path.exists(path):
            data[name] = load(path)
        else:
            print(f'  - {name}: no {path} (skipped)')
    if not data:
        raise SystemExit('no curve CSV found')

    covered = {n: [a for a in ATTRS if a in d] for n, d in data.items()}
    common = [a for a in ATTRS if all(a in c for c in covered.values())]
    if args.attribute:
        curves = {n: pooled(d, [args.attribute]) for n, d in data.items()}
        scope = args.attribute
    elif args.pool == 'common':
        curves = {n: pooled(d, common) for n, d in data.items()}
        scope = 'pooled over ' + ', '.join(common) + ' (the attributes every method covers)'
    else:
        curves = {n: (all_row(d) or pooled(d, covered[n])) for n, d in data.items()}
        scope = 'each method\'s own attributes pooled'
    key = 1 if args.judge == 'r50' else 2
    other = 'r18' if args.judge == 'r50' else 'R50'

    cells = {n: [ladder(curves[n], t / 100, key) for t in args.targets] for n in curves}
    out_dir = args.out_dir or os.path.join(os.path.dirname(runs[0][1]) or '.', 'ladder')
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f'ladder_{args.judge}_' + (args.attribute or args.pool))

    head = ['Method', 'attributes'] + [f'{args.judge.upper()} {t:g}%' for t in args.targets]
    md = [f'# Every method at the same {args.judge.upper()} editing accuracy ({scope})', '',
          f'Cell: strength (method\'s own units) / ArcFace / {other} accuracy %. '
          '`below` = the weakest measured strength already edits more; `--` = not reached (note below).', '',
          '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    notes = []
    for n, row in cells.items():
        cs = []
        for t, c in zip(args.targets, row):
            st, s, r50, r18, arc, note = c
            if st == 'ok':
                o = r18 if args.judge == 'r50' else r50
                cs.append(f'{s:.3g} / **{arc:.3f}** / {o * 100:.1f}')
            elif st == 'below':
                cs.append('below')
                notes.append(f'- **{n}, {t:g}%**: {note}.')
            elif st == 'not reached':
                cs.append('--')
                notes.append(f'- **{n}, {t:g}%**: {note}.')
            else:
                cs.append('n/a')
        attrs_n = ', '.join(covered[n]) if args.pool == 'all' and not args.attribute else (args.attribute or 'common')
        md.append(f'| {n} | {attrs_n} | ' + ' | '.join(cs) + ' |')
    if notes:
        md += ['', '**Why a cell is empty**', ''] + notes
    md += ['', f'Read at the first crossing of each accuracy, interpolated linearly between the two measured '
           f'strengths around it. Strengths are in each method\'s own units and cannot be compared across '
           f'methods; ArcFace and the {other} accuracy can.']
    text = '\n'.join(md) + '\n'
    open(stem + '.md', 'w').write(text)
    with open(stem + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'target', 'status', 'strength', 'r50', 'r18', 'arcface', 'note'])
        for n, row in cells.items():
            for t, (st, s, r50, r18, arc, note) in zip(args.targets, row):
                w.writerow([n, t, st] + ['' if v is None else f'{v:.4f}' for v in (s, r50, r18, arc)] + [note])
    print(text)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5.8))
        cmap = plt.get_cmap('tab10')
        for k, (n, pts) in enumerate(curves.items()):
            if not pts:
                continue
            col = cmap(k % 10)
            # the method's measured curve as ONE solid line, up to its highest accuracy (a curve that turns
            # back -- e.g. LT, whose accuracy falls again after its peak -- is cut at the peak); the dots
            # are the matched accuracies, read on that same line
            top = max(range(len(pts)), key=lambda i: pts[i][key])
            xs, ys = [q[key] * 100 for q in pts[:top + 1]], [q[3] for q in pts[:top + 1]]
            ax.plot(xs, ys, '-', color=col, lw=1.9)
            got = [(t, c) for t, c in zip(args.targets, cells[n]) if c[0] == 'ok']
            miss = [f'{t:g}' for t, c in zip(args.targets, cells[n]) if c[0] != 'ok']
            lab = n + (f'  (not reached: {", ".join(miss)}%)' if miss else '')
            ax.plot([t for t, _ in got], [c[4] for _, c in got], 'o', color=col, ms=6, mec='white', mew=0.8, label=lab)
        for t in args.targets:
            ax.axvline(t, color='0.88', lw=0.6, zorder=0)
        ax.set_xlabel(f'{args.judge.upper()} editing accuracy (%)')
        ax.set_ylabel('ArcFace identity (higher = better)')
        short = scope.split(' (')[0]
        ax.set_title(f'Identity at matched {args.judge.upper()} accuracy\n{short}', fontsize=10)
        ax.set_xlim(min(args.targets) - 5, max(args.targets) + 2)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc='lower left')
        fig.tight_layout()
        fig.savefig(stem + '.png', dpi=200)
        fig.savefig(stem + '.pdf')
    except Exception as exc:                                                       # noqa: BLE001
        print(f'(no figure: {exc})')
    print(f'saved {stem}.md / .csv / .png / .pdf')


if __name__ == '__main__':
    main()
