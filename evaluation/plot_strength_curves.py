"""
Plot Identity/Attribute Preservation vs Editing Accuracy curves for one or
more runs on the same axes — reproduces the multi-method comparison style
from the SDFlow paper's Fig. 4 (InterFaceGAN / Latent Transformer / StyleFlow
/ Ours), but for comparing your own runs (e.g. original vs lag_dof, or before
vs after a direction-bank fix) or any external baseline you have numbers for.

Each --run points at a CSV with the schema scripts/editing_curves.py writes
(columns: attribute, strength, target_success, effective_success,
id_sim_real, preserve_acc). Any CSV with
those columns works, so this also accepts a hand-built CSV of baseline
numbers if you reproduce InterFaceGAN/StyleFlow/Latent Transformer yourself
under the same protocol.

Usage:
    python evaluation/plot_strength_curves.py \
        --run Ours=<ckpt>/curves_model_s100000.csv \
        --run Linear=<ckpt>/curves_linear_s100000.csv \
        --output_dir ./output/compare/curves

Do not mix runs measured with different identity models or different
attribute classifiers on the same plot — the X axis (Editing Accuracy) and
the ID cosine similarity are only comparable if every run used the same
classifier and the same identity encoder to produce them.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

READ_AT = (40, 60, 80, 100)     # Editing Accuracy (%) every curve is read at, and the x ticks


def parse_named_path(value):
    """argparse type for NAME=path pairs."""
    if '=' not in value:
        raise argparse.ArgumentTypeError('Expected NAME=path/to/curves.csv')
    name, path = value.split('=', 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise argparse.ArgumentTypeError('Run name cannot be empty.')
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f'File does not exist: {path}')
    return name, path


def read_summary(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'attribute': row['attribute'],
                'strength': float(row['strength']),
                'target_success': float(row['target_success']),
                'effective_success': float(row.get('effective_success', row['target_success'])),
                'celeb_success': float(row.get('celeb_success') or 'nan'),
                'id_sim_real': float(row['id_sim_real']),
                'preserve_acc': float(row['preserve_acc']),
            })
    return rows


def collect_attribute_order(data_by_run, requested):
    if requested:
        return list(requested)
    seen = []
    for rows in data_by_run.values():
        for row in rows:
            if row['attribute'] not in seen:
                seen.append(row['attribute'])
    return seen


def read_at(rows, x_metric, points=READ_AT):
    """{point: (id, preserve %) or None}: the curve's ID and attribute preservation
    at each Editing Accuracy (%), linearly interpolated between the measured
    strengths and never extrapolated -- a curve that does not reach a point
    (strengths too small or too large) gets None there."""
    pts = sorted((r[x_metric] * 100.0, r['id_sim_real'], r['preserve_acc'] * 100.0)
                 for r in rows if r[x_metric] == r[x_metric])
    out = {}
    if not pts:
        return {q: None for q in points}
    x = np.array([t[0] for t in pts])
    for q in points:
        if q < x.min() - 1e-9 or q > x.max() + 1e-9:
            out[q] = None
            continue
        out[q] = (float(np.interp(q, x, [t[1] for t in pts])),
                  float(np.interp(q, x, [t[2] for t in pts])))
    return out


def plot_attribute(attr_name, data_by_run, x_metric, output_dir, xlim=(READ_AT[0], READ_AT[-1])):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = plt.cm.tab10.colors
    plotted_any = False

    for run_idx, (run_name, rows) in enumerate(data_by_run.items()):
        attr_rows = [r for r in rows if r['attribute'] == attr_name]
        if not attr_rows:
            print(f'[warn] run "{run_name}" has no rows for attribute "{attr_name}", skipping.')
            continue
        plotted_any = True

        # Sort by the achieved editing accuracy, not raw strength: strength
        # does not map 1:1 onto the resulting classifier accuracy, and the
        # paper-style figure reads left-to-right by accuracy.
        attr_rows = sorted(attr_rows, key=lambda r: r[x_metric])
        edit_acc = [r[x_metric] * 100.0 for r in attr_rows]
        id_sim = [r['id_sim_real'] for r in attr_rows]
        attr_pres = [r['preserve_acc'] * 100.0 for r in attr_rows]

        color = colors[run_idx % len(colors)]
        axes[0].plot(edit_acc, id_sim, marker='o', color=color, linewidth=2, label=run_name)
        axes[1].plot(edit_acc, attr_pres, marker='o', color=color, linewidth=2, label=run_name)

    if not plotted_any:
        plt.close(fig)
        return None

    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_xticks([q for q in READ_AT if xlim[0] <= q <= xlim[1]])
    axes[0].set_xlabel('Editing Accuracy (%)')
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_title('Identity Preservation')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel('Editing Accuracy (%)')
    axes[1].set_ylabel('Attribute Preservation (%)')
    axes[1].set_title('Attribute Preservation')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(attr_name)
    fig.tight_layout()
    safe_name = attr_name.replace(' ', '_')
    out_path = os.path.join(output_dir, f'curves_{safe_name}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--run', dest='runs', action='append', type=parse_named_path, required=True,
                        help='NAME=path/to/metrics_summary.csv. Repeatable, one per line/method.')
    parser.add_argument('--output_dir', default='./output/curve_comparison')
    parser.add_argument('--x_metric', default='target_success',
                        choices=['target_success', 'effective_success', 'celeb_success'],
                        help='Which success metric to plot as "Editing Accuracy" (celeb_success: '
                             'the CelebA r18 judge, from scripts/editing_curves.py CSVs).')
    parser.add_argument('--xlim', nargs=2, type=float, default=[READ_AT[0], READ_AT[-1]],
                        help='x range of every plot, the same for all attributes (default 40 100).')
    parser.add_argument('--attributes', nargs='*', default=None,
                        help='Restrict to these attribute names. Default: every attribute '
                             'found across the given runs.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data_by_run = {name: read_summary(path) for name, path in args.runs}
    attrs = collect_attribute_order(data_by_run, args.attributes)

    if not attrs:
        raise SystemExit('No attributes found in the provided --run CSV files.')

    written = []
    for attr_name in attrs:
        out_path = plot_attribute(attr_name, data_by_run, args.x_metric, args.output_dir,
                                  xlim=tuple(args.xlim))
        if out_path:
            written.append(out_path)
            print(f'Wrote {out_path}')
        else:
            print(f'[warn] no data for attribute "{attr_name}" in any run, skipped.')

    if not written:
        raise SystemExit('No plots were written — check that --attributes matches the CSV contents.')

    # Numbers read off the curves at the fixed Editing Accuracies.
    table = []
    for attr_name in attrs:
        for run_name, rows in data_by_run.items():
            arows = [r for r in rows if r['attribute'] == attr_name]
            if not arows:
                continue
            for q, v in read_at(arows, args.x_metric).items():
                table.append({'attribute': attr_name, 'run': run_name, 'editing_accuracy': q,
                              'id_sim': '' if v is None else f'{v[0]:.4f}',
                              'attr_preserve': '' if v is None else f'{v[1]:.2f}'})
    path = os.path.join(args.output_dir, f'at_accuracy_{args.x_metric}.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['attribute', 'run', 'editing_accuracy', 'id_sim', 'attr_preserve'])
        w.writeheader()
        w.writerows(table)
    print(f'\nID / attribute preservation at Editing Accuracy ({args.x_metric}); "--" = the curve does not reach it')
    for attr_name in attrs:
        sub = [t for t in table if t['attribute'] == attr_name]
        if not sub:
            continue
        print(f'\n{attr_name}')
        print(f'  {"run":<14}' + ''.join(f'{"acc " + str(q):>16}' for q in READ_AT) + '     (ID / preserve%)')
        for run_name in data_by_run:
            cells = []
            for q in READ_AT:
                t = next((t for t in sub if t['run'] == run_name and t['editing_accuracy'] == q), None)
                cells.append('--' if t is None or t['id_sim'] == '' else f'{t["id_sim"]} / {float(t["attr_preserve"]):4.1f}')
            if any(c != '--' for c in cells):
                print(f'  {run_name:<14}' + ''.join(f'{c:>16}' for c in cells))
    print(f'\nsaved {path}')


if __name__ == '__main__':
    main()
