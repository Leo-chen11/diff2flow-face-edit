"""Fixed edit scale vs per-face adaptive strength, accuracy against identity
(static PNG + PDF for the paper, light theme).

Six small multiples: the five attributes plus the attribute mean. In each,
x = ID_ind (identity kept, higher = better), y = consensus AccBoth (CLIP or
the glasses parser AND the CelebA classifier accept the edit), so the two
inference procedures are compared at the SAME identity. One curve per
procedure: fixed runs through its edit scales, adaptive through its teacher
margins.

Inputs are evaluate_sdflow.py --consensus JSONs of ONE checkpoint; several
files per procedure are pooled (a scale or margin seen twice keeps the first):
    python scripts/plot_adaptive_vs_fixed.py \
        --fixed  $R/eval80k_sweep_cons.json $R/eval80k_lowscale_cons.json \
        --adaptive $R/eval80k_adaptive_cons.json $R/eval80k_adaptive_hi_cons.json \
        --out figures/fig_adaptive_vs_fixed
An adaptive margin >= 0.5 cannot be reached (the teacher's probability would
have to exceed 1), so every face ends at the top of the ladder: such a point is
moved to the FIXED curve at scale = max(ladder) instead of being plotted as
adaptive.

Writes <out>.png, <out>.pdf and <out>.csv (every plotted point: the table view).
Colors: slots 1-2 of the validated reference palette, checked with
    node validate_palette.js "#2a78d6,#eb6834" --mode light
(all checks pass); series also differ by marker shape and are direct-labeled.
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e1', '#c9c8c3'
SERIES = {
    'fixed':    dict(label='Fixed edit scale', color='#2a78d6', marker='o'),
    'adaptive': dict(label='Per-face adaptive (ours)', color='#eb6834', marker='s'),
}
ATTRS = ['Eyeglasses', 'Male', 'Young', 'Smiling', 'Bangs']
METRICS = {'both': 'AccBoth', 'celeb': 'AccCeleb', 'clip': 'AccCLIP'}


def _mean(d, k):
    v = (d or {}).get(k)
    return v['mean'] if isinstance(v, dict) and v.get('mean') is not None else None


def load(paths, adaptive, metric):
    """{key: {'kind', 'x': {attr|'mean': id}, 'y': {...}}} keyed by scale/margin."""
    pts, moved = {}, {}
    for p in paths:
        r = json.load(open(p))
        cfg = r.get('config', {})
        top = max(cfg.get('adaptive_ladder') or [0.0])
        for k, v in r.items():
            if not (isinstance(v, dict) and 'overall' in v):
                continue
            kind, key = ('adaptive', float(k)) if adaptive else ('fixed', float(k))
            if adaptive and float(k) >= 0.5:      # unreachable margin == fixed at the ladder top
                kind, key = 'fixed', top
            x, y = {}, {}
            for a in ATTRS:
                row = v.get(a) or {}
                cons = row.get('consensus') or {}
                x[a], y[a] = _mean(row, 'id_indep'), _mean(cons, f'acc_{metric}')
            co = v.get('consensus_overall') or {}
            x['mean'] = _mean(v['overall'], 'id_indep')
            y['mean'] = co.get(f'acc_{metric}_attr_mean')
            target = moved if kind != ('adaptive' if adaptive else 'fixed') else pts
            target.setdefault(key, {'kind': kind, 'x': x, 'y': y, 'src': os.path.basename(p)})
    return pts, moved


def panel(ax, series, attr, title, metric_label, show_labels):
    for name, spec in SERIES.items():
        rows = sorted(((v['x'][attr], v['y'][attr], k) for k, v in series[name].items()
                       if v['x'][attr] is not None and v['y'][attr] is not None))
        if not rows:
            continue
        xs, ys = [r[0] for r in rows], [100 * r[1] for r in rows]
        ax.plot(xs, ys, '-', color=spec['color'], lw=2, solid_capstyle='round',
                solid_joinstyle='round', zorder=2)
        ax.plot(xs, ys, linestyle='none', marker=spec['marker'], ms=6.5, color=spec['color'],
                mec=SURFACE, mew=1.6, zorder=3, label=spec['label'])
        if show_labels:   # direct label at the high-identity end of each curve
            ax.annotate(spec['label'].split(' (')[0], (xs[-1], ys[-1]), xytext=(6, 0),
                        textcoords='offset points', fontsize=6.5, color=INK2, va='center')
    ax.set_title(title, loc='left', fontsize=8.5, fontweight='bold', color=INK, pad=5)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=7, length=3, width=0.6)
    ax.set_xlabel('ID$_{ind}$ (identity kept) →', fontsize=7, color=INK2)
    ax.set_ylabel(f'{metric_label} (%)', fontsize=7, color=INK2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fixed', nargs='+', required=True)
    p.add_argument('--adaptive', nargs='+', required=True)
    p.add_argument('--metric', default='both', choices=list(METRICS))
    p.add_argument('--out', default='figures/fig_adaptive_vs_fixed')
    args = p.parse_args()

    fixed, _ = load(args.fixed, False, args.metric)
    adaptive, moved = load(args.adaptive, True, args.metric)
    for k, v in moved.items():
        if k not in fixed:
            print(f'[note] adaptive margin >= 0.5 is unreachable: plotted as fixed scale {k:g}')
            fixed[k] = v
    series = {'fixed': fixed, 'adaptive': adaptive}
    print(f'fixed scales: {sorted(fixed)}   adaptive margins: {sorted(adaptive)}')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9), facecolor=SURFACE)
    titles = [f'({c}) {a}' for c, a in zip('abcde', ATTRS)] + ['(f) Attribute mean']
    for ax, attr, t in zip(axes.flat, ATTRS + ['mean'], titles):
        panel(ax, series, attr, t, METRICS[args.metric], show_labels=False)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.0), labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.4, w_pad=1.0)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{args.out}.{ext}', dpi=220, facecolor=SURFACE)
    with open(f'{args.out}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['procedure', 'scale_or_margin', 'attribute', 'id_ind',
                    f'acc_{args.metric}', 'source'])
        for name in ('fixed', 'adaptive'):
            for k in sorted(series[name]):
                v = series[name][k]
                for a in ATTRS + ['mean']:
                    if v['x'][a] is not None and v['y'][a] is not None:
                        w.writerow([name, k, a, f"{v['x'][a]:.4f}", f"{100 * v['y'][a]:.2f}", v['src']])
    print(f'saved {args.out}.png / .pdf / .csv')


if __name__ == '__main__':
    main()
