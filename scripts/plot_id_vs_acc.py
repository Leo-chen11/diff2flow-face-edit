"""ID-vs-accuracy trade-off curves for the paper (static PNG + PDF, light theme).

x = ID_ind (identity kept after the edit, higher = better)
y = edit accuracy (%), higher = better.  Each model is one curve through its
edit scales (0.80 ... 0.95, right to left), so models are compared at the SAME
identity instead of at whatever point their scale happened to land.
  (a) all sampled faces, CelebA judge       (b) consensus faces, AccBoth
Consensus = CLIP (parser for Eyeglasses) and CelebA both read the source
clearly and the same way; AccBoth = both judges mark the edit as successful.

Data: figures/id_vs_acc_data.csv (values transcribed from the printed eval
tables), or the eval JSONs themselves:
  python scripts/plot_id_vs_acc.py \
      --json flow80k=<run>/eval_v2_step80000_n500_bankprior_cons.json \
      --json A_replace80k=<run>/eval_v2_step80000_n500_cons.json \
      --json base50k=<run>/eval_v2_step50000_n500_cons.json
(JSON values are the pooled 'overall' numbers, the same ones the eval prints.)

Colors: slots 1-3 of the validated reference palette (light), checked with
  node validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light --pairs all
which passes every gate; the only WARN is aqua's 2.74:1 contrast on the light
surface, handled by the relief rule: a legend, a distinct marker shape per
series (so it also survives black-and-white print) and the CSV as table view.
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e1'
# fixed order = categorical slot order; color follows the entity
SERIES = {
    'flow80k':      dict(label='Pure flow, anneal (ours)', color='#2a78d6', marker='o'),
    'A_replace80k': dict(label='replace, +30k steps (control)', color='#eb6834', marker='s'),
    'base50k':      dict(label='replace, 50k (start)', color='#1baf7a', marker='^'),
}


def load_csv(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out.setdefault(r['series'], []).append(dict(
                scale=float(r['scale']), id_ind=float(r['id_ind']),
                acc_celeb=float(r['acc_celeb']),
                acc_both=float(r['acc_both']) if r['acc_both'] else None))
    return out


def load_json(path):
    d = json.load(open(path))
    rows = []
    for k, v in d.items():
        if not (isinstance(v, dict) and 'overall' in v):
            continue
        ov = v['overall']
        cons = (v.get('consensus_overall') or {}).get('acc_both')
        rows.append(dict(scale=float(k), id_ind=ov['id_indep']['mean'],
                         acc_celeb=100 * ov['acc_celeb']['mean'],
                         acc_both=100 * cons['mean'] if cons else None))
    return sorted(rows, key=lambda r: r['scale'])


def at_id(rows, key, x0):
    xs = np.array([r['id_ind'] for r in rows])
    ys = np.array([r[key] for r in rows])
    if not (xs.min() <= x0 <= xs.max()):
        return None
    o = np.argsort(xs)
    return float(np.interp(x0, xs[o], ys[o]))


def panel(ax, data, key, title, ylim, match_id):
    for name, spec in SERIES.items():
        rows = [r for r in data.get(name, []) if r[key] is not None]
        if not rows:
            continue
        ax.plot([r['id_ind'] for r in rows], [r[key] for r in rows], '-',
                color=spec['color'], lw=1.6, marker=spec['marker'], ms=6.5,
                mec=SURFACE, mew=1.2, label=spec['label'], zorder=3)
    flow = sorted([r for r in data.get('flow80k', []) if r[key] is not None], key=lambda r: r['scale'])
    if flow:  # label the ends of the headline curve with their edit scale
        lo, hi = flow[0], flow[-1]
        ax.annotate(f"scale {lo['scale']:.2f}", (lo['id_ind'], lo[key]), xytext=(4, -10),
                    textcoords='offset points', fontsize=6.5, color=INK2)
        ax.annotate(f"scale {hi['scale']:.2f}", (hi['id_ind'], hi[key]), xytext=(-2, 7),
                    textcoords='offset points', fontsize=6.5, color=INK2, ha='center')
    a = at_id(data.get('A_replace80k', []), key, match_id) if data.get('A_replace80k') else None
    f = at_id(data.get('flow80k', []), key, match_id) if data.get('flow80k') else None
    if a is not None and f is not None:
        ax.annotate('', (match_id, a), (match_id, f),
                    arrowprops=dict(arrowstyle='<->', color=INK2, lw=0.9, shrinkA=3, shrinkB=3))
        ax.text(match_id + 0.0006, (a + f) / 2, f'+{f - a:.1f} pp\nat ID {match_id:.3f}',
                fontsize=7, color=INK, va='center')
        print(f'{title}: at ID {match_id}: control {a:.1f}  flow {f:.1f}  gap +{f - a:.1f} pp')
    ax.set_ylim(*ylim)
    ax.set_title(title, loc='left', fontsize=8.5, fontweight='bold', color=INK, pad=6)
    ax.text(0.985, 0.965, 'better ↗', transform=ax.transAxes, ha='right', va='top',
            fontsize=7, color=INK2)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c9c8c3')
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=7, length=3, width=0.6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='figures/id_vs_acc_data.csv')
    p.add_argument('--json', action='append', default=[], metavar='SERIES=PATH',
                   help='read a series from an eval JSON instead of the CSV (repeatable)')
    p.add_argument('--match_id', type=float, default=0.835)
    p.add_argument('--out', default='figures/fig_id_vs_acc')
    args = p.parse_args()

    data = load_csv(args.csv) if not args.json else {}
    for item in args.json:
        name, path = item.split('=', 1)
        data[name] = load_json(path)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), facecolor=SURFACE)
    panel(axes[0], data, 'acc_celeb', '(a) All sampled faces', (70.5, 82.5), args.match_id)
    panel(axes[1], data, 'acc_both', '(b) Consensus faces', (66.0, 80.5), args.match_id)
    for ax in axes:
        style(ax)
        ax.set_xlim(0.816, 0.862)
        ax.set_xticks(np.arange(0.82, 0.861, 0.01))
        ax.set_xlabel('Identity preserved, ID_ind', fontsize=7.5, color=INK2)
    axes[0].set_ylabel('Edit accuracy, CelebA judge (%)', fontsize=7.5, color=INK2)
    axes[1].set_ylabel('Edit accuracy, both judges agree (%)', fontsize=7.5, color=INK2)
    # one legend for both panels, above the plots, so it never covers a curve
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(labels),
               frameon=False, fontsize=7, handlelength=2.2, columnspacing=2.2, labelcolor=INK2)
    fig.text(0.012, 0.012,
             'Each curve runs from edit scale 0.80 (right) to 0.95 (left); 500 test faces, 5 attributes.\n'
             'Consensus faces: CLIP and CelebA read the source clearly and the same way (about 73% of faces).\n'
             'Accuracy in (b): both judges call the edit a success. The 50k checkpoint predates the '
             'consensus metric, so it is only in (a).',
             fontsize=6.3, color=INK2, va='bottom', ha='left', linespacing=1.35)
    fig.tight_layout(rect=(0, 0.11, 1, 0.93), w_pad=2.2)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out + '.png', dpi=300, facecolor=SURFACE)
    fig.savefig(args.out + '.pdf', facecolor=SURFACE)
    print(f'saved {args.out}.png / .pdf')


if __name__ == '__main__':
    main()
