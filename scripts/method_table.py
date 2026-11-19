"""The per-attribute method table: one column per method, one row per attribute (and Average), every cell
"ArcFace id / R50 accuracy / r18 accuracy" -- the format of the run-comparison tables, for the methods of
scripts/run_heldout_curves.sh.

Strengths mean different things for different methods (steps, a coefficient, a StyleSpace multiplier), so
a fixed strength cannot be shared; each method is read at the strength where ITS OWN curve reaches the
matched quantity, interpolated linearly between the two measured strengths around it:

  --match acc   (default) the editing accuracy of --judge (r50, or r18: the CelebA judge on the consensus
                faces) reaches --target (percent). That judge's cell is then the target itself; read the
                rest. Best id per row in bold. Report both judges: the curves take the edit direction from
                R50 (scripts/editing_curves.py --direction_judge), so R50 accuracy is not independent of it.
  --match id    the ArcFace similarity falls to --target (e.g. 0.85). The id cell is then the target;
                read the two accuracies. Best R50 accuracy per row in bold.
  --match strength  every method at one FIXED strength (--target, e.g. 0.95), as the table comparing runs
                of one model (ctrl / wsd / long ...). Only meaningful for methods whose strength means
                the same thing -- your own model and its ablations, all with the editor's `scale`. For
                other methods give each its own operating point with NAME=curves.csv@STRENGTH
                (e.g. AdaTrans=...@10 for its 10 trained steps) and say so in the caption. A strength the
                curve did not measure is interpolated between its neighbours and marked ~ (measure it
                exactly with scripts/editing_curves.py --scales 0.95); outside the measured range: --.

The curve is read at its FIRST crossing. A method whose curve never reaches the target for an attribute
shows "--" (match acc) or its strongest measured point marked * (match id: it never loses that much
identity); an acc cell already above the target at the weakest measured strength shows that point
marked *. The Average row is the mean over the attributes a method has a value for, with the count when
it is not all of them.

    python -m scripts.method_table --target 90                 # every method found in output/curves
    python -m scripts.method_table --match id --target 0.85
    python -m scripts.method_table --target 95 --run Ours=output/curves/ours_cr0.1_off500.csv \\
        --run SDFlow=output/curves/sdflow5_off500.csv
    python -m scripts.method_table --match strength --target 0.95 \\
        --run ctrl=<run>/curves_model_s130000_off500_cr0.1.csv --run nomakeup=<run2>/curves_....csv

Writes output/tables/method_table_<match>_<target>.{md,csv}. The curves must be the held-out ones
(scripts/run_heldout_curves.sh: *_off500.csv), one CSV per method with the columns of
scripts/editing_curves.py.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

ATTRS = ['Eyeglasses', 'Male', 'Young', 'Smiling', 'Bangs']
DEFAULT_RUNS = [
    ('Ours', 'ours_cr0.1_off500.csv'), ('SDFlow', 'sdflow5_off500.csv'), ('AdaTrans', 'adatrans_off500.csv'),
    ('DeltaEdit', 'deltaedit_off500.csv'), ('GradCtrl', 'gradctrl_off500.csv'), ('PluGeN', 'plugen_off500.csv'),
    ('StyleFlow', 'styleflow_off500.csv'), ('LT', 'lt_off500.csv'), ('InterFaceGAN', 'interfacegan_off500.csv'),
]


def read_curves(path):
    """{attribute: [(strength, r50 acc, r18 acc, arcface id)] sorted by strength}."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['attribute'] == 'All':
                continue
            row = (float(r['strength']), float(r['target_success']), float(r['celeb_success']), float(r['id_arc']))
            out.setdefault(r['attribute'], []).append(row)
    return {a: sorted(rows) for a, rows in out.items()}


def lerp(p, q, t):
    return tuple(a + t * (b - a) for a, b in zip(p, q))


def read_at(rows, match, target):
    """(strength, r50, r18, id, flag) at the first crossing of `target`, or None.
    acc is in 0-1 and id in 0-1 inside; target for match acc is given in 0-1 too."""
    key = {'acc': 1, 'acc_r18': 2}.get(match, 3)
    rising = match in ('acc', 'acc_r18')
    for i, row in enumerate(rows):
        hit = row[key] >= target if rising else row[key] <= target
        if not hit:
            continue
        if i == 0:
            return row + ('*',) if rows[0][key] != target else row + ('',)
        prev = rows[i - 1]
        span = row[key] - prev[key]
        t = (target - prev[key]) / span if span else 1.0
        return lerp(prev, row, min(max(t, 0.0), 1.0)) + ('',)
    if match == 'id':
        return rows[-1] + ('*',)            # it never loses that much identity: its strongest point
    return None


def read_strength(rows, s):
    """The curve at strength s: the measured row, or the linear interpolation (flag ~), or None outside."""
    for row in rows:
        if abs(row[0] - s) < 1e-9:
            return row + ('',)
    for prev, row in zip(rows, rows[1:]):
        if prev[0] < s < row[0]:
            return lerp(prev, row, (s - prev[0]) / (row[0] - prev[0])) + ('~',)
    return None


def fmt(cell, bold):
    if cell is None:
        return '--'
    _, r50, r18, ident, flag = cell
    s = f'{ident:.3f} / {r50 * 100:.1f} / {r18 * 100:.1f}{flag}'
    return f'**{s}**' if bold else s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', default=None, metavar='NAME=curves.csv',
                   help='a method (repeat); default: every method of scripts/run_heldout_curves.sh found in --curves')
    p.add_argument('--curves', default='output/curves')
    p.add_argument('--match', default='acc', choices=['acc', 'id', 'strength'])
    p.add_argument('--target', type=float, default=None, help='percent for --match acc (default 90), ArcFace for --match id (default 0.85), '
                                                      'the strength for --match strength (default 0.95)')
    p.add_argument('--judge', default='r50', choices=['r50', 'r18'],
                   help='--match acc: whose accuracy reaches the target (default r50).')
    p.add_argument('--attrs', nargs='+', default=ATTRS)
    p.add_argument('--out_dir', default='output/tables')
    args = p.parse_args()

    if args.target is None:
        args.target = {'acc': 90.0, 'id': 0.85, 'strength': 0.95}[args.match]
    target = args.target / 100.0 if args.match == 'acc' else args.target
    runs = []
    if args.run:
        for item in args.run:
            name, _, path = item.partition('=')
            path, _, at = path.partition('@')                    # NAME=curves.csv[@strength]
            runs.append((name.strip(), path.strip(), float(at) if at else None))
    else:
        runs = [(n, os.path.join(args.curves, f), None) for n, f in DEFAULT_RUNS]
    have, at_of = [], {}
    for name, path, at in runs:
        if os.path.exists(path):
            have.append((name, read_curves(path)))
            at_of[name] = at
        else:
            print(f'  - {name}: no {path} (skipped)')
    if not have:
        raise SystemExit('no curve CSV found')

    cells = {}                     # (method, attr) -> cell
    for name, curves in have:
        for a in args.attrs:
            if a not in curves:
                cells[(name, a)] = None
            elif args.match == 'strength':
                cells[(name, a)] = read_strength(curves[a], at_of[name] if at_of[name] is not None else target)
            else:
                m = 'acc_r18' if (args.match == 'acc' and args.judge == 'r18') else args.match
                cells[(name, a)] = read_at(curves[a], m, target)
    names = [n for n, _ in have]

    # Average over the attributes a method has a value for
    avg = {}
    for n in names:
        vals = [cells[(n, a)] for a in args.attrs if cells[(n, a)] is not None]
        if vals:
            m = tuple(sum(v[i] for v in vals) / len(vals) for i in range(4))
            flag = '' if len(vals) == len(args.attrs) else f' ({len(vals)}/{len(args.attrs)})'
            avg[n] = m + (flag + ''.join(sorted({v[4] for v in vals if v[4]})),)
        else:
            avg[n] = None

    # best per row: highest id (match acc) / highest R50 (match id) among the methods that have a value
    best_idx = 3 if args.match == 'acc' else 1
    rows_out, md = [], []
    head = ['Attribute'] + names
    sub = ['', *['id / R50 / r18'] * len(names)]
    md.append('| ' + ' | '.join(head) + ' |')
    md.append('| ' + ' | '.join(['---'] + ['---'] * len(names)) + ' |')
    md.append('| ' + ' | '.join(sub) + ' |')
    for label, getter in [(a, lambda n, a=a: cells[(n, a)]) for a in args.attrs] + [('Average', lambda n: avg[n])]:
        vals = {n: getter(n) for n in names}
        top = max((v[best_idx] for v in vals.values() if v is not None), default=None)
        if args.match == 'strength':                       # as the run-comparison table: only Average in bold
            cs = [fmt(vals[n], label == 'Average') for n in names]
        else:
            cs = [fmt(vals[n], vals[n] is not None and abs(vals[n][best_idx] - top) < 5e-4) for n in names]
        md.append('| ' + ' | '.join([f'**{label}**' if label == 'Average' else label] + cs) + ' |')
        rows_out.append([label] + [('' if vals[n] is None else f'{vals[n][3]:.4f}/{vals[n][1] * 100:.2f}/{vals[n][2] * 100:.2f}') for n in names])

    if args.match == 'strength':
        custom = ', '.join(f'{n} at {a:g}' for n, a in at_of.items() if a is not None)
        note = (f'Every method at strength {args.target:g}' + (f' except {custom}' if custom else '') +
                '; cell = ArcFace id / R50 accuracy % / CelebA-r18 accuracy %. ~ : interpolated between the two '
                'measured strengths around it; -- : outside the measured range; (k/n): k of n attributes available.')
    else:
        unit = (f'{args.judge.upper()} editing accuracy {args.target:g}%' if args.match == 'acc'
                else f'ArcFace {args.target:g}')
        note = (f'Each method read at the strength where its curve reaches {unit} (interpolated, first crossing); '
                'cell = ArcFace id / R50 accuracy % / CelebA-r18 accuracy %. Best ' +
                ('id' if args.match == 'acc' else 'R50 accuracy') + ' per row in bold. -- : the curve never reaches the target; '
                '* : ' + ('the weakest measured strength is already above the target' if args.match == 'acc'
                          else 'the curve never loses that much identity (its strongest point)') +
                '; (k/n): k of n attributes available.')
    text = '\n'.join(md) + '\n\n' + note + '\n'
    print('\n' + text)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f'method_table_{args.match}_{args.target:g}'
                        + (f'_{args.judge}' if args.match == 'acc' and args.judge != 'r50' else ''))
    with open(stem + '.md', 'w') as f:
        f.write(text)
    with open(stem + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['attribute'] + [f'{n} (id/R50/r18)' for n in names])
        w.writerows(rows_out)
    print(f'saved {stem}.md and .csv')


if __name__ == '__main__':
    main()
