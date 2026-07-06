"""
Plot Identity/Attribute Preservation vs Editing Accuracy curves for one or
more runs on the same axes — reproduces the multi-method comparison style
from the SDFlow paper's Fig. 4 (InterFaceGAN / Latent Transformer / StyleFlow
/ Ours), but for comparing your own runs (e.g. original vs lag_dof, or before
vs after a direction-bank fix) or any external baseline you have numbers for.

Each --run points at a metrics_summary.csv with the schema produced by
evaluation/eval_strength_sweep.py (columns: attribute, strength,
target_success, effective_success, id_sim_real, preserve_acc). Any CSV with
those columns works, so this also accepts a hand-built CSV of baseline
numbers if you reproduce InterFaceGAN/StyleFlow/Latent Transformer yourself
under the same protocol.

Usage:
    python evaluation/plot_strength_curves.py \
        --run Original=./output/compare/original/metrics_summary.csv \
        --run LAG-DOF=./output/compare/lag_dof/metrics_summary.csv \
        --output_dir ./output/compare/curves

Do not mix runs measured with different identity models or different
attribute classifiers on the same plot — the X axis (Editing Accuracy) and
the ID cosine similarity are only comparable if every run used the same
classifier and the same identity encoder to produce them.
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_named_path(value):
    if '=' not in value:
        raise argparse.ArgumentTypeError('Expected NAME=path/to/metrics_summary.csv')
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


def plot_attribute(attr_name, data_by_run, x_metric, output_dir):
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

    axes[0].set_xlabel('Editing Accuracy (%)')
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_title('Identity Preservation')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel('Editing Accuracy (%)')
    axes[1].set_ylabel('Accuracy (%)')
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
                        choices=['target_success', 'effective_success'],
                        help='Which success metric to plot as "Editing Accuracy".')
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
        out_path = plot_attribute(attr_name, data_by_run, args.x_metric, args.output_dir)
        if out_path:
            written.append(out_path)
            print(f'Wrote {out_path}')
        else:
            print(f'[warn] no data for attribute "{attr_name}" in any run, skipped.')

    if not written:
        raise SystemExit('No plots were written — check that --attributes matches the CSV contents.')


if __name__ == '__main__':
    main()
