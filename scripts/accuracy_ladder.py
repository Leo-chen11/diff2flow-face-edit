"""Every method read at the SAME editing accuracies (default R50 40 50 60 70 80 90 95%): at each one,
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

The 7th point is 95%, not 100%: 100% means every scored face succeeded, which no method's pooled curve
reaches (the last few faces resist any strength), so it compared nothing. --targets ... 100 brings it back.

Pooling: the curve of a method is its "All" row (every attribute it covers, faces pooled), or, with
--pool common, pooled over only the attributes EVERY method covers (GradCtrl has no Bangs, DeltaEdit
no Young), weighting each attribute by its number of scored faces -- the like-for-like comparison.

    python -m scripts.accuracy_ladder --curves output/curves_consensus
    python -m scripts.accuracy_ladder --curves output/curves_consensus --pool common --judge r18
    python -m scripts.accuracy_ladder --run Ours=a.csv --run SDFlow=b.csv --targets 40 50 60 70 80 90 95

Writes <out_dir>/ladder_<judge>_<pool>.{md,csv,png}: the table (strength / ArcFace / other judge per
cell, notes under it) and a figure (ArcFace at each matched accuracy, one line per method through its
points, each line labelled at its end; a missing point is listed in the legend).
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


# categorical colours in a fixed order (checked for colour-blind separation), one marker shape per method
# so that the lines stay apart in grey print; a "+CR" variant takes its method's colour, dashed
PALETTE = ['#257de0', '#f47230', '#00b894', '#f2a600', '#ee7fac', '#008a17', '#6a4ce2', '#ec4647']
MARKERS = 'osD^vPXh<>'
ORDER = ['Ours', 'AdaTrans', 'SDFlow', 'GradCtrl', 'StyleFlow', 'LT', 'PluGeN', 'DeltaEdit', 'InterFaceGAN']


def plot(cells, targets, judge, scope, stem):
    """ArcFace at each matched accuracy: one line per method through the accuracies it reaches, labelled
    at its last point (name and value), the first method (ours) drawn bold. An accuracy no method reaches
    is marked on the axis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, ink2 = '#1f1f1f', '#6b6b6b'
    bases = list(ORDER)                         # a method keeps its colour whichever others are drawn
    for n in cells:
        b = n.replace('+CR', '')
        if b not in bases:
            bases.append(b)
    known = {n: i for i, n in enumerate(ORDER)}
    cells = dict(sorted(cells.items(), key=lambda kv: (known.get(kv[0].replace('+CR', ''), 99), '+CR' in kv[0])))
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    ends, vals = [], []
    for k, (n, row) in enumerate(cells.items()):
        b = bases.index(n.replace('+CR', ''))
        col, mk = PALETTE[b % len(PALETTE)], MARKERS[b % len(MARKERS)]
        got = [(t, c[4]) for t, c in zip(targets, row) if c[0] == 'ok']
        miss = [f'{t:g}' for t, c in zip(targets, row) if c[0] != 'ok']
        if not got:
            continue
        first = n == 'Ours' or (k == 0 and 'Ours' not in cells)
        ax.plot([t for t, _ in got], [v for _, v in got], '--' if '+CR' in n else '-', color=col,
                lw=2.6 if first else 2.0, marker=mk, ms=7, mec='white', mew=1.2, zorder=4 if first else 3,
                label=n + (f'  (not reached: {", ".join(miss)}%)' if miss else ''))
        ends.append([got[-1][1], got[-1][0], f'{n}  {got[-1][1]:.3f}', first])
        vals += [v for _, v in got]
    lo = max(0.0, min(vals) - 0.06) if vals else 0.0
    hi = min(1.0, max(vals) + 0.03) if vals else 1.0
    ax.set_ylim(lo, hi)
    gap = (hi - lo) * 0.035                     # end labels pushed apart so none overlap
    ends.sort(key=lambda e: -e[0])
    ys = []
    for e in ends:
        y = e[0] if not ys or ys[-1] - e[0] >= gap else ys[-1] - gap
        ys.append(y)
    xmax = max(targets)
    for e, y in zip(ends, ys):
        ax.annotate(e[2], (e[1], e[0]), xytext=(xmax + (xmax - min(targets)) * 0.03, y), textcoords='data',
                    va='center', fontsize=9.5, color=ink, fontweight='bold' if e[3] else 'normal',
                    arrowprops=dict(arrowstyle='-', color='#c8c8c8', lw=0.8) if e[1] != xmax or abs(y - e[0]) > 1e-9 else None,
                    annotation_clip=False)
    nobody = [t for i, t in enumerate(targets) if all(row[i][0] != 'ok' for row in cells.values())]
    ax.set_xticks(targets)
    ax.set_xticklabels([f'{t:g}%' + ('\n(not reached\nby any method)' if t in nobody else '') for t in targets])
    span = xmax - min(targets)
    ax.set_xlim(min(targets) - span * 0.05, xmax + span * 0.03)
    ax.set_xlabel(f'Editing accuracy ({judge.upper()} judge)', color=ink)
    ax.set_ylabel('Identity preservation (ArcFace cosine)  \u2191', color=ink)
    ax.set_title(f'Identity at matched editing accuracy\n{scope.split(" (")[0]}', fontsize=11, color=ink, loc='left')
    ax.grid(axis='y', color='#e6e6e6', lw=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('#bdbdbd')
    ax.tick_params(colors=ink2)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='lower left', frameon=False, ncol=2, fontsize=8.5)
    fig.subplots_adjust(right=0.78)
    fig.savefig(stem + '.png', dpi=200, bbox_inches='tight')
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', default=None, metavar='NAME=curves.csv')
    p.add_argument('--curves', default='output/curves')
    p.add_argument('--variants', default='nocr', choices=['nocr', 'cr', 'both'],
                   help='which flow-method curves: nocr (default: no method gets the reading rule), cr, both')
    p.add_argument('--targets', nargs='+', type=float, default=[40, 50, 60, 70, 80, 90, 95])
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
        plot(cells, args.targets, args.judge, scope, stem)
    except Exception as exc:                                                       # noqa: BLE001
        print(f'(no figure: {exc})')
    print(f'saved {stem}.md / .csv / .png / .pdf')


if __name__ == '__main__':
    main()
