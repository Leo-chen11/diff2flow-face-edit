"""Train the independent ResNet-50 CelebA-40 attribute classifier used as
the evaluation judge (AccIndep column of evaluation/evaluate_sdflow.py),
matching the SDFlow paper's protocol (ResNet-50 classifier trained on
CelebA labels, used only for evaluation, never for training).

Data: CelebA-HQ split into two folders, each holding the images and an
attribute file:

    split_full/train/images/*.jpg   split_full/train/attributes.txt
    split_full/val/images/*.jpg     split_full/val/attributes.txt

The attribute file may be the CelebAMask-HQ-attribute-anno.txt layout
(optional count line, a line of 40 names, then "file v1 ... v40") or a
CSV with an image_id column; values may be -1/1 or 0/1. When names are
present, columns are reordered to the official CelebA order, which is
what eval indexes directly (15 Eyeglasses, 20 Male, 39 Young).

Preprocessing matches what eval feeds the judge: [-1, 1] images at 256.
Eval downsamples the 1024 StyleGAN output with F.interpolate's default
(nearest), so --resize_mode mixed (default) trains on nearest and
antialiased bilinear downsampling half the time each.

The checkpoint is a full AttributeClassifier(backbone='r50') state dict
(its unused age head keeps its ImageNet weights), so eval loads it with:

    --independent_attr_weights ./data/r50_celebahq_eval.pth --independent_attr_backbone r50

Usage:
    python -m scripts.train_attr_classifier \
        --train_dir ~/桌面/split_full/train --val_dir ~/桌面/split_full/val \
        --out ./data/r50_celebahq_eval.pth
"""
import argparse
import json
import math
import os
import random
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common.attr_tables import CELEBA_ALL_ATTRS
from models.attribute_estimator import AttributeClassifier

CELEBA_ATTRS = CELEBA_ALL_ATTRS
KEY_ATTRS = {15: 'Eyeglasses', 20: 'Male', 39: 'Young'}
IMG_EXTS = ('.jpg', '.jpeg', '.png')


def _is_num(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


def read_attr_file(path):
    """Returns (filenames, labels (N, 40) float 0/1) in official CelebA order."""
    names, files, rows = None, [], []
    with open(path, encoding='utf-8') as f:
        for lineno, raw in enumerate(f, 1):
            toks = raw.replace(',', ' ').split()
            if not toks:
                continue
            if len(toks) == 1 and toks[0].isdigit():
                continue                                   # "30000" count line
            if not any(_is_num(t) for t in toks[1:]):      # header line
                names = toks if len(toks) == 40 else toks[1:]
                continue
            vals = toks[1:]
            if len(vals) != 40:
                raise SystemExit(f'{path}:{lineno}: expected 40 values after the file name, '
                                 f'got {len(vals)}')
            files.append(toks[0])
            rows.append([1.0 if float(v) > 0 else 0.0 for v in vals])
    if not rows:
        raise SystemExit(f'{path}: no data rows found')
    labels = torch.tensor(rows)
    if names is not None:
        if sorted(names) != sorted(CELEBA_ATTRS):
            raise SystemExit(f'{path}: header names are not the 40 CelebA attributes: {names}')
        if names != CELEBA_ATTRS:
            print(f'  {path}: reordering columns to the official CelebA order')
            labels = labels[:, [names.index(a) for a in CELEBA_ATTRS]]
    else:
        print(f'  [WARN] {path}: no header line; assuming the official CelebA column order')
    return files, labels


def resolve_images(image_dir, files):
    """Map each listed file name to an existing path (tolerates a different
    extension or a leading directory in the list)."""
    present = {}
    for fn in os.listdir(image_dir):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in IMG_EXTS:
            present.setdefault(fn, fn)
            present.setdefault(stem, fn)
    paths, keep = [], []
    for i, name in enumerate(files):
        base = os.path.basename(name)
        hit = present.get(base) or present.get(os.path.splitext(base)[0])
        if hit is not None:
            paths.append(os.path.join(image_dir, hit))
            keep.append(i)
    return paths, keep


class AttrDataset(Dataset):
    def __init__(self, root, attr_name, image_subdir, img_size, train, resize_mode):
        root = os.path.expanduser(root)
        files, labels = read_attr_file(os.path.join(root, attr_name))
        paths, keep = resolve_images(os.path.join(root, image_subdir), files)
        missing = len(files) - len(keep)
        print(f'  {root}: {len(keep)} images'
              + (f' ({missing} listed files not found, skipped)' if missing else ''))
        if not keep:
            raise SystemExit(f'{root}: none of the listed images were found')
        self.paths = paths
        self.labels = labels[keep]
        self.img_size = img_size
        self.train = train
        self.resize_mode = resize_mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = TF.to_tensor(Image.open(self.paths[i]).convert('RGB')) * 2.0 - 1.0   # [-1, 1]
        mode = self.resize_mode
        if mode == 'mixed':
            mode = random.choice(('nearest', 'bilinear')) if self.train else 'nearest'
        size = (self.img_size, self.img_size)
        if mode == 'nearest':
            x = F.interpolate(x[None], size)[0]
        else:
            x = F.interpolate(x[None], size, mode='bilinear', align_corners=False,
                              antialias=True)[0]
        if self.train and random.random() < 0.5:
            x = torch.flip(x, dims=[2])
        return x, self.labels[i]


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    correct = torch.zeros(40)
    tp = torch.zeros(40); pos = torch.zeros(40)
    tn = torch.zeros(40); neg = torch.zeros(40)
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast('cuda', enabled=amp):
            logits, _ = model.forward_attr(x)
        pred = (logits.float().sigmoid() > 0.5).float().cpu()
        correct += (pred == y).sum(0)
        tp += (pred * y).sum(0); pos += y.sum(0)
        tn += ((1 - pred) * (1 - y)).sum(0); neg += (1 - y).sum(0)
        n += y.size(0)
    acc = correct / n
    bal = 0.5 * (tp / pos.clamp(min=1) + tn / neg.clamp(min=1))
    return acc, bal


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', required=True)
    p.add_argument('--val_dir', required=True)
    p.add_argument('--attr_file', default='attributes.txt')
    p.add_argument('--image_subdir', default='images')
    p.add_argument('--out', default='./data/r50_celebahq_eval.pth')
    p.add_argument('--backbone', default='r50', choices=['r18', 'r34', 'r50'])
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--resize_mode', default='mixed', choices=['mixed', 'nearest', 'bilinear'])
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda'
    amp = not args.no_amp

    print('loading data')
    train_set = AttrDataset(args.train_dir, args.attr_file, args.image_subdir,
                            args.img_size, True, args.resize_mode)
    val_set = AttrDataset(args.val_dir, args.attr_file, args.image_subdir,
                          args.img_size, False, args.resize_mode)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, drop_last=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    rate = train_set.labels.mean(0)
    print('  train positive rate  ' + '  '.join(f'{KEY_ATTRS[k]} {rate[k]:.1%}' for k in KEY_ATTRS))

    model = AttributeClassifier(backbone=args.backbone).to(device)
    # Only the 40 attribute heads are trained; the age head is never used here.
    params = list(model.extractor.parameters()) + list(model.attr_heads.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    best, best_epoch = -1.0, -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.age_heads.eval()
        t0, run = time.time(), 0.0
        for it, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast('cuda', enabled=amp):
                logits, _ = model.forward_attr(x)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            if (it + 1) % 100 == 0:
                print(f'  epoch {epoch} it {it + 1}/{len(train_loader)} loss {run / (it + 1):.4f}')
        acc, bal = evaluate(model, val_loader, device, amp)
        key = '  '.join(f'{KEY_ATTRS[k]} {acc[k]:.2%} (bal {bal[k]:.2%})' for k in KEY_ATTRS)
        print(f'epoch {epoch}: train loss {run / len(train_loader):.4f}  '
              f'val mean acc {acc.mean():.2%}  |  {key}  [{time.time() - t0:.0f}s]')
        if acc.mean().item() > best:
            best, best_epoch = acc.mean().item(), epoch
            torch.save(model.state_dict(), args.out)
            report = {
                'epoch': epoch, 'backbone': args.backbone, 'img_size': args.img_size,
                'resize_mode': args.resize_mode, 'val_mean_acc': best,
                'val_acc': {a: acc[i].item() for i, a in enumerate(CELEBA_ATTRS)},
                'val_balanced_acc': {a: bal[i].item() for i, a in enumerate(CELEBA_ATTRS)},
                'n_train': len(train_set), 'n_val': len(val_set),
            }
            with open(os.path.splitext(args.out)[0] + '_val.json', 'w') as f:
                json.dump(report, f, indent=2)
            print(f'  saved best -> {args.out}')

    print(f'\nbest val mean acc {best:.2%} at epoch {best_epoch}; weights {args.out}')
    print('per-attribute val accuracy of the saved checkpoint: '
          f'{os.path.splitext(args.out)[0]}_val.json')
    print('use in eval:  --independent_attr_weights ' + args.out
          + f' --independent_attr_backbone {args.backbone}')


if __name__ == '__main__':
    main()
