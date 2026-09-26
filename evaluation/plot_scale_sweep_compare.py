"""Plot two evaluate_sdflow.py scale-sweep runs against each other.

Reads the JSON each evaluate_sdflow.py run already writes
(<checkpoint_dir>/eval_v2_step<N>_n<num_samples>.json) -- no numbers are
retyped by hand, so the chart can't silently drift from what was actually
measured. Produces three PNGs:

  1. <out>_frontier.png   -- overall Acc vs ID, one line per run, with the
     85% / 0.85 target marked. This is the chart that answers "did the
     Acc/ID frontier actually move, or did this run just slide along the
     same curve".
  2. <out>_directions.png -- per-attribute add/rm AccCeleb vs scale, one
     panel per (attribute, direction), both runs overlaid. Shows whether a
     specific direction actually improved or whether an aggregate number is
     hiding a redistribution.
  3. <out>_identity.png   -- per-attribute ID_arc vs scale, both runs
     overlaid. Pairs with (2): a direction's accuracy can drop while its
     identity budget frees up, which (2) alone won't show.

Usage:
    python -m evaluation.plot_scale_sweep_compare \
        --a ./output/SDFlow/substyle3_glasses_v16/eval_v2_step50000_n500.json --a_label v16 \
        --b ./output/SDFlow/substyle3_glasses_v17/eval_v2_step90000_n500.json --b_label v17 \
        --out_dir ./output/SDFlow/_compare_v16_v17

Works for any two runs, not just v16/v17 -- pass whichever two JSON files
and labels you want compared.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLOR_A = '#3D6B78'
COLOR_B = '#B5652A'
TARGET = '#8B8578'


def _load(path):
    with open(path) as f:
        data = json.load(f)
    scales = sorted((k for k in data if k not in ('config', 'inversion_gap')),
                     key=float)
    return data, scales


def _overall_acc_id(scale_summary):
    ovr = scale_summary['overall']
    id_show = ovr.get('id_indep') or ovr.get('id_arc')
    acc_show = ovr.get('acc_celeb') or ovr.get('acc_clip') or ovr.get('acc_teacher')
    if id_show is None or acc_show is None:
        return None, None
    return acc_show['mean'] * 100, id_show['mean']


def plot_frontier(data_a, scales_a, label_a, data_b, scales_b, label_b, out_path):
    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=150)

    for data, scales, label, color in [
        (data_a, scales_a, label_a, COLOR_A), (data_b, scales_b, label_b, COLOR_B),
    ]:
        xs, ys, tags = [], [], []
        for s in scales:
            acc, idv = _overall_acc_id(data[s])
            if acc is None:
                continue
            xs.append(acc)
            ys.append(idv)
            tags.append(s)
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        tags = [tags[i] for i in order]
        ax.plot(xs, ys, '-o', color=color, label=label, linewidth=2, markersize=5)
        for x, y, t in zip(xs, ys, tags):
            ax.annotate(t, (x, y), textcoords='offset points', xytext=(5, 4),
                        fontsize=7, color=color)

    ax.axvline(85, color=TARGET, linestyle='--', linewidth=1)
    ax.axhline(0.85, color=TARGET, linestyle='--', linewidth=1)
    ax.plot(85, 0.85, 'D', color=TARGET, markersize=7)
    ax.annotate('target 85% / 0.85', (85, 0.85), textcoords='offset points',
                xytext=(6, 6), fontsize=8, color=TARGET)

    ax.set_xlabel('Overall accuracy (%)')
    ax.set_ylabel('Overall identity preservation (ID)')
    ax.set_title('Accuracy / identity frontier')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'wrote {out_path}')


def plot_directions(data_a, scales_a, label_a, data_b, scales_b, label_b, out_path):
    attrs = [k for k in data_a[scales_a[0]] if k not in ('num_samples', 'overall', 'fid_edit_vs_recon')]
    directions = ['add', 'rm']
    fig, axes = plt.subplots(len(attrs), len(directions),
                             figsize=(9, 2.6 * len(attrs)), dpi=150, sharex=True)
    if len(attrs) == 1:
        axes = axes.reshape(1, -1)

    for ai, attr in enumerate(attrs):
        for di, direction in enumerate(directions):
            ax = axes[ai][di]
            for data, scales, label, color in [
                (data_a, scales_a, label_a, COLOR_A), (data_b, scales_b, label_b, COLOR_B),
            ]:
                xs, ys = [], []
                for s in scales:
                    row = data[s].get(attr, {})
                    cell = row.get(f'acc_celeb_{direction}')
                    if cell is None:
                        continue
                    xs.append(float(s))
                    ys.append(cell['mean'] * 100)
                if xs:
                    ax.plot(xs, ys, '-o', color=color, label=label, markersize=3.5)
            ax.set_title(f'{attr} · {direction}', fontsize=10)
            ax.grid(alpha=0.25)
            ax.set_ylim(0, 102)
            if ai == len(attrs) - 1:
                ax.set_xlabel('edit_scale')
            if di == 0:
                ax.set_ylabel('AccCeleb (%)')
    axes[0][0].legend(fontsize=8)
    fig.suptitle('Per-direction accuracy vs scale')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'wrote {out_path}')


def plot_identity(data_a, scales_a, label_a, data_b, scales_b, label_b, out_path):
    attrs = [k for k in data_a[scales_a[0]] if k not in ('num_samples', 'overall', 'fid_edit_vs_recon')]
    fig, axes = plt.subplots(1, len(attrs), figsize=(4 * len(attrs), 4), dpi=150, sharey=True)
    if len(attrs) == 1:
        axes = [axes]

    for ai, attr in enumerate(attrs):
        ax = axes[ai]
        for data, scales, label, color in [
            (data_a, scales_a, label_a, COLOR_A), (data_b, scales_b, label_b, COLOR_B),
        ]:
            xs, ys = [], []
            for s in scales:
                cell = data[s].get(attr, {}).get('id_arc')
                if cell is None:
                    continue
                xs.append(float(s))
                ys.append(cell['mean'])
            if xs:
                ax.plot(xs, ys, '-o', color=color, label=label, markersize=4)
        ax.set_title(attr)
        ax.set_xlabel('edit_scale')
        ax.grid(alpha=0.25)
        if ai == 0:
            ax.set_ylabel('ID_arc')
    axes[0].legend(fontsize=8)
    fig.suptitle('Per-attribute identity (ID_arc) vs scale')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'wrote {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a', required=True, help='eval_v2_*.json for run A')
    p.add_argument('--a_label', required=True)
    p.add_argument('--b', required=True, help='eval_v2_*.json for run B')
    p.add_argument('--b_label', required=True)
    p.add_argument('--out_dir', required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data_a, scales_a = _load(args.a)
    data_b, scales_b = _load(args.b)

    common = sorted(set(scales_a) & set(scales_b), key=float)
    if set(scales_a) != set(scales_b):
        print(f'[WARN] scale sets differ -- A has {scales_a}, B has {scales_b}. '
              f'Frontier plot uses each run\'s own scales; direction/identity plots '
              f'also use each run\'s own scales, so mismatched points just won\'t align on x.')

    plot_frontier(data_a, scales_a, args.a_label, data_b, scales_b, args.b_label,
                  os.path.join(args.out_dir, f'{args.a_label}_vs_{args.b_label}_frontier.png'))
    plot_directions(data_a, scales_a, args.a_label, data_b, scales_b, args.b_label,
                    os.path.join(args.out_dir, f'{args.a_label}_vs_{args.b_label}_directions.png'))
    plot_identity(data_a, scales_a, args.a_label, data_b, scales_b, args.b_label,
                 os.path.join(args.out_dir, f'{args.a_label}_vs_{args.b_label}_identity.png'))


if __name__ == '__main__':
    main()
