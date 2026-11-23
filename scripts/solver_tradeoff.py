"""Speed against quality for each way of solving our flow's ODE (scripts/run_speed_study.sh).

Speed: scripts/speed_benchmark.py's speed.json (ms per face, NFE). Quality: one
scripts/editing_curves.py CSV per solver variant, named <curves_dir>/<variant>.csv, read like
scripts/method_table.py at the strength where the curve reaches the matched accuracy. The table
says, per variant, how much faster it edits than the trained solver and what it costs in identity
and in the other judge's accuracy at the same accuracy -- a variant is only a free speed-up if those
stay within the seed noise (ArcFace 0.003 overall).

    python -m scripts.solver_tradeoff --speed output/speed_study/speed.json --curves_dir output/speed_study/curves
    python -m scripts.solver_tradeoff --flags rk4x4+fast        # the editing_curves flags of a variant
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def variant_flags(spec):
    """The scripts/editing_curves.py flags of a --ode variant of scripts/speed_benchmark.py."""
    fast = spec.endswith('+fast') or spec == 'fast'
    core = spec[:-5] if spec.endswith('+fast') else spec
    flags = []
    if core not in ('trained', 'fast'):
        if '@' in core:
            solver, tol = core.split('@')
            flags += ['--ode_solver', solver, '--ode_tol', tol]
        elif 'x' in core:
            solver, steps = core.split('x')
            flags += ['--ode_solver', solver, '--ode_steps', steps]
        else:
            flags += ['--ode_solver', core]
    if fast:
        flags.append('--fast_edit')
    return flags


def read_all(path):
    """{'All': [(strength, r50, r18, id)]}: the pooled curve (scripts/method_table.read_curves skips it)."""
    import csv
    rows = [(float(r['strength']), float(r['target_success']), float(r['celeb_success']), float(r['id_arc']))
            for r in csv.DictReader(open(path)) if r['attribute'] == 'All']
    return {'All': sorted(rows)}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--flags':
        print(' '.join(variant_flags(sys.argv[2])))
        return
    from scripts.method_table import read_at
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--speed', required=True)
    p.add_argument('--curves_dir', required=True)
    p.add_argument('--batch', type=int, default=1, help='which batch size of speed.json to quote')
    p.add_argument('--targets', nargs='+', type=float, default=[90, 95], help='R50 accuracies to match (percent)')
    p.add_argument('--out', default=None, help='default <dir of --speed>/tradeoff.md')
    args = p.parse_args()

    sp = json.load(open(args.speed))
    ours = {r['variant']: r for r in sp['results'] if r['method'] == 'ours' and r['batch'] == args.batch}
    if not ours:
        raise SystemExit(f'no rows for method ours at batch {args.batch} in {args.speed}')
    base_ms = ours.get('trained', {}).get('total', {}).get('median_ms')

    rows = []
    for v, r in ours.items():
        csvp = os.path.join(args.curves_dir, f'{v}.csv')
        cur = read_all(csvp) if os.path.exists(csvp) else None
        row = {'variant': v, 'solver': r.get('solver', ''), 'total': r.get('total', {}).get('median_ms'),
               'latent': r.get('latent', {}).get('median_ms'),
               'nfe': (r.get('nfe_encode', 0) + r.get('nfe_decode', 0)) if 'nfe_encode' in r else None}
        for t in args.targets:
            for key, m in (('r50', 'acc'), ('r18', 'acc_r18')):
                cell = read_at(cur.get('All', []), m, t / 100) if cur else None
                row[(key, t)] = cell
        rows.append(row)

    ref = next((r for r in rows if r['variant'] == 'trained'), None)
    lines = [f'# Solver variants of our flow: speed vs quality ({sp["meta"].get("gpu", "?")}, batch {args.batch})', '',
             '| variant | ODE | ms / face (total) | latent edit ms | speed-up | NFE enc+dec | '
             + ' | '.join(f'@R50 {t:g}%: ArcFace (Δ) / r18' for t in args.targets)
             + ' | ' + ' | '.join(f'@r18 {t:g}%: ArcFace (Δ)' for t in args.targets) + ' |',
             '|' + '---|' * (6 + 2 * len(args.targets))]
    for r in rows:
        su = f'{base_ms / r["total"]:.2f}x' if base_ms and r['total'] else '--'
        cells = []
        for key in ('r50', 'r18'):
            for t in args.targets:
                c = r[(key, t)]
                c0 = ref[(key, t)] if ref else None
                if c is None:
                    cells.append('-- (no curve)' if not os.path.exists(os.path.join(args.curves_dir, f'{r["variant"]}.csv'))
                                 else '--')
                    continue
                d = f' ({c[3] - c0[3]:+.3f})' if c0 is not None and r is not ref else ''
                cells.append(f'{c[3]:.3f}{d}' + (f' / {c[2] * 100:.1f}' if key == 'r50' else ''))
        tot = f'{r["total"]:.1f}' if r['total'] else '--'
        lat = f'{r["latent"]:.1f}' if r['latent'] else '--'
        lines.append(f'| {r["variant"]} | {r["solver"]} | {tot} | {lat} | {su} | {r["nfe"] if r["nfe"] is not None else "--"} | '
                     + ' | '.join(cells) + ' |')
    lines += ['', 'Quality is read on the "All" curve (all attributes pooled) at the strength where it reaches the '
              'matched accuracy; Δ against the trained solver. Seed noise of ArcFace at matched accuracy: 0.003 '
              'overall (two seeds of nodds_b8), so |Δ| under that is no measurable loss. The curves are on the '
              'development faces: a solver chosen here is then reported on the held-out faces.']
    text = '\n'.join(lines) + '\n'
    out = args.out or os.path.join(os.path.dirname(args.speed), 'tradeoff.md')
    open(out, 'w').write(text)
    print(text)
    print(f'saved {out}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t = args.targets[0]
        pts = [(r['total'], r[('r50', t)][3], r['variant']) for r in rows if r['total'] and r[('r50', t)]]
        if pts:
            fig, ax = plt.subplots(figsize=(6, 4))
            for x, y, n in pts:
                ax.scatter(x, y)
                ax.annotate(n, (x, y), fontsize=8, xytext=(3, 3), textcoords='offset points')
            ax.set_xlabel('ms per edited face (incl. rendering)')
            ax.set_ylabel(f'ArcFace at R50 {t:g}% accuracy')
            ax.set_xscale('log')
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.splitext(out)[0] + '.png', dpi=150)
    except Exception as exc:                                                   # noqa: BLE001
        print(f'(no figure: {exc})')


if __name__ == '__main__':
    main()
