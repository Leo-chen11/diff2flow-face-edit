"""Accuracy of attribute classifiers on labelled CelebA-HQ faces, fed exactly
the way the training teacher is fed (the project's [-1, 1] images at 256).

Use it before training with a classifier as a teacher, e.g. to check that
data/convnext_base_a40_512_classifier.pth reads its input the way
make_attr_classifier assumes ('convnext_b512' = [0, 1] before its own
mean/std, 'convnext_b512_pm1' = [-1, 1]): the wrong one shows near-chance
accuracy, the right one should match or beat the r34 teacher. Also prints how
often each classifier agrees with the r34 teacher (a second teacher that
always agrees with it adds nothing).

    python -m scripts.check_attr_classifier --data_dir ~/桌面/split_full/val \
        --model r34:r34:./data/r34_a40_age_256_classifier.pth \
        --model cnx01:convnext_b512:./data/convnext_base_a40_512_classifier.pth \
        --model cnxpm1:convnext_b512_pm1:./data/convnext_base_a40_512_classifier.pth
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader, Subset

from common.ops import load_network
from models.attribute_estimator import make_attr_classifier
from scripts.train_attr_classifier import AttrDataset

KEY = {15: 'Eyeglasses', 20: 'Male', 39: 'Young', 31: 'Smiling', 5: 'Bangs'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True, help='folder with images/ and attributes.txt')
    p.add_argument('--attr_file', default='attributes.txt')
    p.add_argument('--image_subdir', default='images')
    p.add_argument('--model', action='append', required=True,
                   help='name:backbone:weights, repeatable; the first is the reference for agreement')
    p.add_argument('--n', type=int, default=2000, help='faces to score (0 = all)')
    p.add_argument('--batch', type=int, default=16)
    args = p.parse_args()

    ds = AttrDataset(args.data_dir, args.attr_file, args.image_subdir, 256, False, 'nearest')
    if args.n and args.n < len(ds):
        ds = Subset(ds, list(range(args.n)))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4)

    preds, labels = {}, None
    for spec in args.model:
        name, backbone, path = spec.split(':', 2)
        m = make_attr_classifier(backbone, pretrained=False)
        m.load_state_dict(load_network(os.path.expanduser(path)))
        m = m.cuda().eval()
        out, ys = [], []
        with torch.no_grad():
            for x, y in loader:
                out.append(torch.sigmoid(m(x.cuda())[0][:, :40]).cpu())
                ys.append(y)
        preds[name] = torch.cat(out)
        labels = torch.cat(ys)
        del m
        torch.cuda.empty_cache()

    print(f'{len(labels)} faces from {args.data_dir}\n')
    head = f'{"model":<10} {"mean40":>7}  ' + '  '.join(f'{KEY[k]:>17}' for k in KEY)
    print(head + '     (accuracy / balanced accuracy)')
    for name, pr in preds.items():
        hard = (pr > 0.5).float()
        acc = (hard == labels).float().mean(0)
        cells = []
        for k in KEY:
            y, h = labels[:, k], hard[:, k]
            tpr = ((h == 1) & (y == 1)).sum() / (y == 1).sum().clamp(min=1)
            tnr = ((h == 0) & (y == 0)).sum() / (y == 0).sum().clamp(min=1)
            cells.append(f'{acc[k]:6.1%} / {0.5 * (tpr + tnr):6.1%}')
        print(f'{name:<10} {acc.mean():7.1%}  ' + '  '.join(f'{c:>17}' for c in cells))

    ref = next(iter(preds))
    print(f'\nagreement with {ref} (same side of 0.5) on the five edited attributes')
    for name, pr in preds.items():
        if name == ref:
            continue
        ag = ((pr > 0.5) == (preds[ref] > 0.5)).float().mean(0)
        print(f'{name:<10} ' + '  '.join(f'{KEY[k]} {ag[k]:6.1%}' for k in KEY))


if __name__ == '__main__':
    main()
