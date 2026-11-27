"""Make every method's curve cover R50 40-95% (scripts/accuracy_ladder.py), and check that it does.

Three commands, used by scripts/run_heldout_curves.sh after each method:

  suggest CSV [--min_strength M]   the extra strengths that curve still needs, space-separated (nothing
                                   when it is complete): smaller ones where its weakest strength already
                                   edits past the lowest target ("below"), larger ones where its
                                   strongest strength has not reached a target and the curve is still
                                   rising (x1.5 and x2 the strongest). A target the curve has flattened
                                   below (the method's limit), 100% missed by the last faces, or one not
                                   reached before ArcFace drops under 0.35 is not chased.
  merge A B --out C                the rows of two curve CSVs of the SAME method and faces in one
                                   (B's strengths added to A's), with the JSON beside it
  report --curves DIR              per method: the 7 points at R50 (each method's own attributes, and
                                   the attributes every method covers), PASS when every missing point
                                   is a real limit of the method, FAIL with what is still fixable

    python -m scripts.ladder_check report --curves output/curves_consensus
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.accuracy_ladder import ATTRS, all_row, ladder, load, pooled, runs_for

TARGETS = [40, 50, 60, 70, 80, 90, 95]   # 95, not 100: no pooled curve reaches 100%
# the smallest strength each method can take (run_heldout_curves.sh passes the same to `suggest`):
# a curve that already edits past 40% there cannot go lower -- a limit of the method, not of the grid
MIN_STRENGTH = {'SDFlow': -5.0, 'SDFlow+CR': -5.0}


def _fmt(x):
    return f'{x:.4g}'


def suggest(csv_path, targets, min_strength, judge_key=1):
    data = load(csv_path)
    pts = all_row(data) or pooled(data, [a for a in ATTRS if a in data])
    if len(pts) < 2:
        return []
    have = {round(p[0], 6) for p in pts}
    cells = [ladder(pts, t / 100, judge_key) for t in targets]
    s = [p[0] for p in pts]
    extra = []
    if any(c[0] == 'below' for c in cells):
        s0, s1 = s[0], s[1]
        if s0 > min_strength + 1e-9:
            if s0 > 0 and min_strength >= 0:
                cand = [s0 / 2, s0 / 4]
            else:
                step = s1 - s0
                cand = [s0 - step, s0 - 2 * step]
            extra += [max(min_strength, c) for c in cand]
    rising = [c for c in cells if c[0] == 'not reached' and 'still rising' in c[5]]
    if rising:
        if s[-1] > 0:                                   # geometric: a slow approach is caught in a few rounds
            extra += [s[-1] * 1.5, s[-1] * 2]
        else:
            step = s[-1] - s[-2]
            extra += [s[-1] + 2 * step, s[-1] + 4 * step]
    out = sorted({round(e, 6) for e in extra} - have)
    return [_fmt(e) for e in out]


def merge(a, b, out):
    rows, keys, seen = [], None, set()
    for path in (a, b):
        with open(path) as f:
            r = csv.DictReader(f)
            keys = keys or r.fieldnames
            for row in r:
                k = (row['attribute'], round(float(row['strength']), 6))
                if k not in seen:
                    seen.add(k)
                    rows.append(row)
    order = {n: i for i, n in enumerate(ATTRS + ['All'])}
    rows.sort(key=lambda r: (order.get(r['attribute'], 99), float(r['strength'])))
    tmp = out + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)
    ja, jb = os.path.splitext(a)[0] + '.json', os.path.splitext(b)[0] + '.json'
    if os.path.exists(ja):
        meta = json.load(open(ja))
        cfg = meta.setdefault('config', {})
        scales = set(cfg.get('scales', []))
        if os.path.exists(jb):
            scales |= set(json.load(open(jb)).get('config', {}).get('scales', []))
        cfg['scales'] = sorted(scales)
        cfg['merged_from'] = cfg.get('merged_from', []) + [os.path.basename(b)]
        meta['rows'] = rows
        json.dump(meta, open(os.path.splitext(out)[0] + '.json', 'w'), indent=2)


def classify(cell, min_strength=0.0):
    st, s, *_, note = cell
    if st == 'ok':
        return 'ok', ''
    if st == 'below' and s is not None and s <= min_strength + 1e-9:
        return 'limit', f'already above at its smallest strength ({s:g}): {note.split(":")[0]}'
    if st == 'below':
        return 'fixable', 'weakest strength already above: needs a smaller strength'
    if st == 'not reached' and 'still rising' in note:
        return 'fixable', 'still rising: needs a larger strength'
    if st == 'not reached':
        return 'limit', note
    return 'fixable', 'no curve'


def report(curves_dir, targets, runs=None, variants='nocr'):
    runs = runs or [(n, os.path.join(curves_dir, f)) for n, f in runs_for(variants)]
    data = {n: load(p) for n, p in runs if os.path.exists(p)}
    missing = [n for n, p in runs if not os.path.exists(p)]
    covered = {n: [a for a in ATTRS if a in d] for n, d in data.items()}
    common = [a for a in ATTRS if all(a in c for c in covered.values())]
    lines = [f'# Coverage check: R50 {", ".join(f"{t:g}" for t in targets)}%  ({curves_dir})', '']
    ok_all = True
    for pool_name, get in (('own attributes', lambda n, d: all_row(d) or pooled(d, covered[n])),
                           ('common attributes: ' + ', '.join(common), lambda n, d: pooled(d, common))):
        lines += [f'## {pool_name}', '', '| method | points | verdict | missing points |', '|---|---|---|---|']
        for n, d in data.items():
            cells = [ladder(get(n, d), t / 100, 1) for t in targets]
            kinds = [classify(c, MIN_STRENGTH.get(n, 0.0)) for c in cells]
            got = sum(k == 'ok' for k, _ in kinds)
            fix = [(t, why) for t, (k, why) in zip(targets, kinds) if k == 'fixable']
            lim = [(t, why) for t, (k, why) in zip(targets, kinds) if k == 'limit']
            verdict = 'PASS' if not fix else 'FAIL'
            ok_all &= not fix
            miss = '; '.join([f'{t:g}%: {w}' for t, w in fix] + [f'{t:g}%: {w}' for t, w in lim]) or '-'
            lines.append(f'| {n} | {got}/{len(targets)} | {verdict} | {miss} |')
        lines.append('')
    if missing:
        lines += ['Not run yet: ' + ', '.join(missing), '']
        ok_all = False
    lines += ['PASS: every point is read, or the missing ones are limits of the method (its curve flattened '
              'below the accuracy, or 100% missed by the last faces). FAIL: a point is missing only because '
              'the strengths measured do not reach far enough -- run scripts/run_heldout_curves.sh again: '
              'it adds them.', '', f'**Overall: {"PASS" if ok_all else "FAIL"}**']
    text = '\n'.join(lines) + '\n'
    print(text)
    out = os.path.join(curves_dir, 'coverage_check.md')
    open(out, 'w').write(text)
    print(f'saved {out}')
    return ok_all


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('suggest')
    s.add_argument('csv')
    s.add_argument('--min_strength', type=float, default=0.0)
    s.add_argument('--targets', nargs='+', type=float, default=TARGETS)
    m = sub.add_parser('merge')
    m.add_argument('a')
    m.add_argument('b')
    m.add_argument('--out', required=True)
    r = sub.add_parser('report')
    r.add_argument('--curves', required=True)
    r.add_argument('--targets', nargs='+', type=float, default=TARGETS)
    r.add_argument('--variants', default='nocr', choices=['nocr', 'cr', 'both'])
    args = p.parse_args()
    if args.cmd == 'suggest':
        print(' '.join(suggest(args.csv, args.targets, args.min_strength)))
    elif args.cmd == 'merge':
        merge(args.a, args.b, args.out)
    else:
        sys.exit(0 if report(args.curves, args.targets, variants=args.variants) else 1)


if __name__ == '__main__':
    main()
