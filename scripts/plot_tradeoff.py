"""Paper-style trade-off curves from evaluate_sdflow.py JSONs, one curve per
run, one point per eval setting (an --eval_scales value, or an
--adaptive_margins value for an --adaptive_ladder eval):

  <out>_id.png / .pdf        x = editing accuracy (%), y = identity kept (ID_ind)
  <out>_preserve.png / .pdf  x = editing accuracy (%), y = attribute preservation
                             = 100 * (1 - mean |dP| of the OTHER attributes)
  <out>.csv                  every plotted point, as a table

Each figure is a grid of small multiples: all attributes pooled, then one panel
per attribute. Up and to the right is better: at the same accuracy the higher
curve keeps more identity (or more of the other attributes). Compare runs along
a vertical line (same accuracy) or a horizontal one (same identity), never by
the eval setting the points came from -- a setting is a different amount of
edit for each run and each --edit_target mode.

    python scripts/plot_tradeoff.py \\
        --run "train mode=<run>/eval_v2_step80000_..._ettrain_learned_clamp.json" \\
        --run "adaptive=<run>/eval_v2_step80000_..._cons_rsarea.json" \\
        --judge indep --out figures/tradeoff_80k

--judge: whose accuracy is on x. indep (the --independent_attr_weights judge,
R50), celeb, clip, cons (consensus subset, both judges agree; ID and
preservation are then taken on the same subset where the eval reports them),
or teacher (the training teacher -- not an independent judge).
--match_acc prints, per attribute, each run's ID at that accuracy (interpolated
along its curve) next to the first run's.

Colors: slots 1-5 of the reference categorical palette (light), validated with
  node validate_palette.js "#2a78d6,#eb6834,#1baf7a,#eda100,#e87ba4" --mode light
which passes every gate; slots 3-5 sit below 3:1 on the surface, so every
series also has its own marker shape, the legend is always shown and the CSV is
the table view.
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e1', '#c9c8c3'
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']   # fixed slot order
MARKERS = ['o', 's', '^', 'D', 'v']
ATTRS = ['Eyeglasses', 'Male', 'Young', 'Smiling', 'Bangs']
JUDGE_LABEL = {'indep': 'R50 judge', 'celeb': 'CelebA judge', 'clip': 'CLIP judge',
               'cons': 'consensus, both judges', 'teacher': 'training teacher'}


def _mean(d, key):
    v = (d or {}).get(key)
    return v['mean'] if isinstance(v, dict) and v.get('mean') is not None else None


def load_points(path, judge, attrs):
    """{panel: [dict(setting, acc, id, keep)]} for 'All attributes' + attrs."""
    d = json.load(open(path))
    cfg = d.get('config', {})
    prefix = 'm' if cfg.get('adaptive_ladder') else 's'
    out = {}
    for key, block in d.items():
        if not (isinstance(block, dict) and 'overall' in block):
            continue
        setting = f'{prefix}{key}'
        leaks = []
        for a in attrs:
            ab = block.get(a)
            if not ab:
                continue
            if judge == 'cons':
                c = ab.get('consensus') or {}
                acc, idv = _mean(c, 'acc_both'), _mean(c, 'id_indep')
            else:
                acc, idv = _mean(ab, f'acc_{judge}'), _mean(ab, 'id_indep')
            leak = _mean(ab, 'leak_clip')
            if leak is not None:
                leaks.append(leak)
            if acc is not None and idv is not None:
                out.setdefault(a, []).append(dict(
                    setting=setting, acc=100 * acc, id=idv,
                    keep=100 * (1 - leak) if leak is not None else None))
        ov = block['overall']
        if judge == 'cons':
            co = block.get('consensus_overall') or {}
            acc, idv = _mean(co, 'acc_both'), _mean(co, 'id_indep')
        else:
            acc, idv = _mean(ov, f'acc_{judge}'), _mean(ov, 'id_indep')
        if acc is not None and idv is not None:
            out.setdefault('All attributes', []).append(dict(
                setting=setting, acc=100 * acc, id=idv,
                keep=100 * (1 - float(np.mean(leaks))) if leaks else None))
    for pts in out.values():
        pts.sort(key=lambda p: p['acc'])
    return out


def y_at_acc(pts, ykey, x0):
    pts = [p for p in pts if p[ykey] is not None]
    if len(pts) < 2:
        return None
    xs = np.array([p['acc'] for p in pts])
    ys = np.array([p[ykey] for p in pts])
    if not (xs.min() <= x0 <= xs.max()):
        return None
    return float(np.interp(x0, xs, ys))


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=7, length=3, width=0.6)


def figure(runs, panels, ykey, ylabel, judge, out, note):
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.55 * nrow + 0.7),
                             facecolor=SURFACE, squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, panel in zip(axes.flat, panels):
        style(ax)
        for i, (name, data) in enumerate(runs):
            pts = [p for p in data.get(panel, []) if p[ykey] is not None]
            if not pts:
                continue
            ax.plot([p['acc'] for p in pts], [p[ykey] for p in pts], '-', color=COLORS[i], lw=2,
                    marker=MARKERS[i], ms=7, mec=SURFACE, mew=2, label=name, zorder=3)
            # label only the two ends of each curve with their eval setting,
            # above for even series and below for odd ones so neighbours don't collide
            dy = 5 if i % 2 == 0 else -11
            for p, (dx, ha) in ((pts[0], (-4, 'right')), (pts[-1], (4, 'left'))):
                ax.annotate(p['setting'], (p['acc'], p[ykey]), xytext=(dx, dy),
                            textcoords='offset points', fontsize=6, color=INK2, ha=ha)
        ax.set_title(panel, loc='left', fontsize=8.5, fontweight='bold', color=INK, pad=5)
        # top right: a trade-off curve runs top-left to bottom-right, so it stays empty
        ax.text(0.98, 0.97, 'better ↗', transform=ax.transAxes, ha='right', va='top',
                fontsize=6.5, color=INK2)
        ax.margins(x=0.12, y=0.15)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=7.5, color=INK2)
    for ax in axes[-1, :]:
        ax.set_xlabel(f'Editing accuracy, {JUDGE_LABEL[judge]} (%)', fontsize=7.5, color=INK2)
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(labels),
               frameon=False, fontsize=7.5, handlelength=2.4, columnspacing=2.4, labelcolor=INK2)
    fig.text(0.01, 0.006, note, fontsize=6.2, color=INK2, va='bottom', ha='left', linespacing=1.35)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95), h_pad=1.6, w_pad=1.6)
    fig.savefig(out + '.png', dpi=250, facecolor=SURFACE)
    fig.savefig(out + '.pdf', facecolor=SURFACE)
    plt.close(fig)
    print(f'saved {out}.png / .pdf')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', action='append', required=True, metavar='NAME=PATH',
                   help='a curve: legend name = evaluate_sdflow.py JSON (repeatable, up to 5)')
    p.add_argument('--judge', default='indep', choices=list(JUDGE_LABEL))
    p.add_argument('--attrs', nargs='*', default=ATTRS)
    p.add_argument('--match_acc', type=float, default=None,
                   help='print each run\'s ID at this accuracy (%%), per panel')
    p.add_argument('--out', default='figures/tradeoff')
    args = p.parse_args()
    if len(args.run) > len(COLORS):
        raise SystemExit(f'at most {len(COLORS)} runs per figure (one fixed color each)')

    runs = []
    for item in args.run:
        name, path = item.split('=', 1)
        runs.append((name, load_points(path, args.judge, args.attrs)))
    panels = ['All attributes'] + [a for a in args.attrs if any(a in d for _, d in runs)]
    if not any(panels[0] in d for _, d in runs):
        raise SystemExit(f'no run has acc_{args.judge}: was the eval run with that judge?')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    who = 'consensus subset' if args.judge == 'cons' else 'all 500 test faces'
    figure(runs, panels, 'id', 'Identity preserved, ID_ind', args.judge, args.out + '_id',
           f'One point per eval setting (s = --eval_scales value, m = --adaptive_margins value), '
           f'{who}.\nID_ind = cosine similarity of edited vs source face, independent face '
           f'recognizer.')
    if any(p['keep'] is not None for _, d in runs for pts in d.values() for p in pts):
        figure(runs, panels, 'keep', 'Attribute preservation (%)', args.judge,
               args.out + '_preserve',
               'Attribute preservation = 100 x (1 - mean |change| in the CLIP judge\'s score of '
               'the attributes NOT being edited).\nOne point per eval setting (s = scale, '
               'm = adaptive margin).')

    with open(args.out + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'panel', 'setting', f'acc_{args.judge}', 'id_ind', 'attr_preservation'])
        for name, data in runs:
            for panel in panels:
                for pt in data.get(panel, []):
                    w.writerow([name, panel, pt['setting'], f"{pt['acc']:.2f}", f"{pt['id']:.4f}",
                                '' if pt['keep'] is None else f"{pt['keep']:.2f}"])
    print(f'saved {args.out}.csv')

    if args.match_acc is not None:
        print(f'\nID at {args.judge} accuracy {args.match_acc:g}% (interpolated; - = outside the curve)')
        print(f'{"panel":<16}' + ''.join(f'{n[:18]:>20}' for n, _ in runs))
        for panel in panels:
            vals = [y_at_acc(d.get(panel, []), 'id', args.match_acc) for _, d in runs]
            print(f'{panel:<16}' + ''.join(f'{"-" if v is None else f"{v:.3f}":>20}' for v in vals))


if __name__ == '__main__':
    main()
