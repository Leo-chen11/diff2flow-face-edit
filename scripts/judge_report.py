"""How trustworthy is each evaluation judge? Score them against CelebA labels.

The edit-success numbers (AccInd / AccCLIP / AccCeleb) are only as good as the
judge behind them, and the judges disagree (Bangs add: CLIP 97% vs ResNet-50
62% on the same edited images). This measures each judge on REAL labelled
faces, where the right answer is known:

    TPR  = P(judge says "has it" | label says it has it)
    TNR  = P(judge says "lacks it" | label says it lacks it)

and uses them as ceilings for the edit-success numbers:

    add ceiling  ~ TPR   (an edit that adds the attribute perfectly is still
                          only accepted when the judge recognises it on a real face)
    rm  ceiling  ~ TNR

A judge whose TPR on Bangs is 70% cannot score a perfect edit above ~70%, so a
62% add score then means the model is close to that ceiling, not that 38% of
the edits failed. A judge with an inflated positive rate (CLIP on Bangs) shows
up as a low TNR.

Data layout is the one scripts/train_attr_classifier.py trains on
(<dir>/images + <dir>/attributes.txt). Use the held-out val split.

Usage:
    python -m scripts.judge_report \
        --data_dir ~/桌面/split_full/val \
        --r50 ./data/r50_celebahq_eval.pth \
        --r18 ./data/celeba_attr_resnet18.pth \
        --clip --attrs 5 15 20 31 39 --out ./judge_report.json
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from scripts.train_attr_classifier import CELEBA_ATTRS, AttrDataset

MALE = 20


@torch.no_grad()
def collect(judges, loader, device, attrs):
    """Returns labels (N, 40) and {judge: scores (N, len(attrs))}."""
    labels, scores = [], {name: [] for name in judges}
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        labels.append(y)
        for name, fn in judges.items():
            scores[name].append(fn(x).float().cpu())
        if (i + 1) % 20 == 0:
            print(f'  {(i + 1) * x.size(0)} images', flush=True)
    return torch.cat(labels), {k: torch.cat(v) for k, v in scores.items()}


def rates(score, label, thresh=0.5):
    pred = score > thresh
    pos, neg = label > 0.5, label <= 0.5
    tpr = (pred & pos).sum().item() / max(1, pos.sum().item())
    tnr = (~pred & neg).sum().item() / max(1, neg.sum().item())
    return tpr, tnr


def best_threshold(score, label):
    """Threshold that maximises balanced accuracy (what a calibrated judge would use)."""
    best = (0.5, -1.0)
    for t in torch.linspace(0.02, 0.98, 49).tolist():
        tpr, tnr = rates(score, label, t)
        if 0.5 * (tpr + tnr) > best[1]:
            best = (t, 0.5 * (tpr + tnr))
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True, help='Folder with images/ and attributes.txt (val split).')
    p.add_argument('--attr_file', default='attributes.txt')
    p.add_argument('--image_subdir', default='images')
    p.add_argument('--attrs', nargs='+', type=int, default=[5, 15, 20, 31, 39])
    p.add_argument('--r50', default=None, help='ResNet-50 judge checkpoint (AccInd).')
    p.add_argument('--r50_backbone', default='r50')
    p.add_argument('--r18', default=None, help='CelebA ResNet-18 judge checkpoint (AccCeleb).')
    p.add_argument('--clip', action='store_true', help='Also score the zero-shot CLIP judge.')
    p.add_argument('--clip_model', default='ViT-L/14')
    p.add_argument('--clip_calibration', default=None, help="'attr:thresh:sharpness,...'")
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max_images', type=int, default=0, help='0 = all.')
    p.add_argument('--out', default=None, help='Optional JSON output.')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    attrs = list(args.attrs)
    names = [CELEBA_ATTRS[a] for a in attrs]

    judges = {}
    if args.r50:
        from common.ops import load_network
        from models.attribute_estimator import AttributeClassifier
        m = AttributeClassifier(backbone=args.r50_backbone)
        m.load_state_dict(load_network(args.r50))
        m.to(device).eval()
        judges['R50 (AccInd)'] = lambda x: torch.sigmoid(m(x)[0])[:, attrs]
    if args.r18:
        from evaluation.evaluate_sdflow import CelebAAttrClassifierJudge
        m18 = CelebAAttrClassifierJudge(args.r18, device)
        judges['R18 (AccCeleb)'] = lambda x: m18.scores(x)[:, attrs]
    if args.clip:
        from evaluation.evaluate_sdflow import CLIPAttributeJudge, parse_clip_calibration
        cj = CLIPAttributeJudge(attrs, args.clip_model, device,
                                calibration=parse_clip_calibration(args.clip_calibration))
        judges['CLIP (AccCLIP)'] = lambda x: cj.scores(x)
    if not judges:
        raise SystemExit('give at least one of --r50 / --r18 / --clip')

    ds = AttrDataset(args.data_dir, args.attr_file, args.image_subdir, args.img_size,
                     train=False, resize_mode='nearest')
    if args.max_images:
        ds.paths, ds.labels = ds.paths[:args.max_images], ds.labels[:args.max_images]
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    print(f'scoring {len(ds)} labelled real images with {list(judges)}')
    labels, scores = collect(judges, loader, device, attrs)
    male = labels[:, MALE] > 0.5

    report = {}
    for ai, (a, name) in enumerate(zip(attrs, names)):
        y = labels[:, a]
        print(f'\n=== {name} (attr {a}): label positive rate {y.mean() * 100:.1f}%  '
              f'(male {y[male].mean() * 100:.1f}%, female {y[~male].mean() * 100:.1f}%) ===')
        print(f'  {"judge":<16} {"TPR":>6} {"TNR":>6} {"bal":>6} {"posRate":>8}   '
              f'{"TPR m/f":>11}   best-thr (bal)')
        for jn, sc in scores.items():
            s = sc[:, ai]
            tpr, tnr = rates(s, y)
            tm, _ = rates(s[male], y[male])
            tf, _ = rates(s[~male], y[~male])
            thr, bal = best_threshold(s, y)
            pos_rate = (s > 0.5).float().mean().item()
            print(f'  {jn:<16} {tpr * 100:5.1f}% {tnr * 100:5.1f}% {50 * (tpr + tnr):5.1f}% '
                  f'{pos_rate * 100:7.1f}%   {tm * 100:4.0f}%/{tf * 100:3.0f}%   '
                  f'{thr:.2f} ({bal * 100:.1f}%)')
            report.setdefault(name, {})[jn] = dict(
                tpr=tpr, tnr=tnr, balanced=0.5 * (tpr + tnr), pos_rate=pos_rate,
                tpr_male=tm, tpr_female=tf, best_thresh=thr, best_balanced=bal)
    print('\nRead: add-edit ceiling ~ TPR, remove-edit ceiling ~ TNR. posRate far above the\n'
          'label positive rate = the judge over-calls the attribute (low TNR). A best-thr far\n'
          'from 0.50 means the 0.5 decision point is mis-calibrated for that judge.')
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(report, f, indent=2)
        print(f'saved {args.out}')


if __name__ == '__main__':
    main()
