"""Adversarially fine-tune the r34 TRAINING teacher (--attribute_weights) so the
flow can no longer satisfy it with changes a person (or another classifier)
does not see.

Why: on the anneal/min runs every edited attribute reaches 89-95% on the
training r34, but the independent CelebA ResNet-18 judge agrees much less
often on Young, Male and Bangs (13-23pp lower). The flow is optimized against
the r34's gradients, so whatever the r34 reacts to most cheaply is what the
flow learns. Adversarially trained classifiers have gradients that follow
visible features (Santurkar et al. 2019, "Image Synthesis with a Single Robust
Classifier"; Kawar et al. 2022, "Enhancing Diffusion-Based Image Synthesis
with Robust Classifier Guidance"), so the same loss on a robust teacher should
push toward edits that also convince other judges. One global setting covers
every attribute; nothing here is per attribute.

What it does: starts from the r34 checkpoint and fine-tunes the feature
extractor and the 40 attribute heads (the age head is left as is; training
only reads attribute logits) on
    clean_weight * BCE(f(x),     target)            all 40 attributes
  + adv_weight   * BCE(f(x_adv), target)[:, attrs]  the edited attributes
where x_adv is an L2 PGD attack of radius --eps around x in the teacher's
input space ([-1, 1], 256x256, after the same crop/flip/bilinear resize that
teacher_augment applies in train_sdflow.py). BatchNorm statistics stay
frozen so clean behaviour drifts as little as possible.

Targets (--target):
  teacher  the ORIGINAL r34's own probabilities on the clean image (default).
           The new teacher then keeps the old one's notion of each attribute
           and only becomes robust, so a training run that swaps it in
           differs from its parent in robustness alone.
  label    the CelebA labels. Also moves the teacher's semantics toward the
           labels, which confounds the comparison; use only as a second step.

Never pass the evaluation judges (CelebA ResNet-18, CLIP ViT-L/14) here or
use their outputs as targets: that would turn the reported metric into
self-grading.

Report (before training = the original r34, then after every epoch), on the
val split, for each edited attribute:
  bal    balanced accuracy against the CelebA labels (clean images)
  tpr/tnr  the two halves of bal; pos = share predicted positive, pos_ref
         the original r34's share (a bal drop with high agree is usually a
         threshold shift: pos moving away from pos_ref)
  agree  agreement with the original r34 on clean images
  flip@e share of images whose prediction an L2 attack of radius e flips
         (attack aimed at the model's own clean prediction). This is how easy
         the teacher is to fool; it should drop a lot while bal/agree stay.

Usage (same data layout as scripts/train_attr_classifier.py):
    python -m scripts.train_robust_teacher \
        --train_dir ~/桌面/split_full/train --val_dir ~/桌面/split_full/val \
        --out ./data/r34_robust_e3.pth
Then train with --attribute_weights ./data/r34_robust_e3.pth.
"""
import argparse
import copy
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
from torch.utils.data import DataLoader

from common.ops import load_network
from models.attribute_estimator import AttributeClassifier
from scripts.train_attr_classifier import CELEBA_ATTRS, AttrDataset


def crop_flip_resize(x, out_size, min_scale=0.85):
    """Random crop/flip/bilinear resize with one parameter set for the batch,
    the same family of views teacher_augment feeds the teacher in training."""
    H, W = x.shape[-2:]
    s = random.uniform(min_scale, 1.0)
    ch, cw = int(H * s), int(W * s)
    top, left = random.randint(0, H - ch), random.randint(0, W - cw)
    x = x[:, :, top:top + ch, left:left + cw]
    if random.random() < 0.5:
        x = torch.flip(x, [-1])
    return F.interpolate(x, (out_size, out_size), mode='bilinear', align_corners=False)


def attr_logits(model, x, amp):
    with torch.autocast('cuda', enabled=amp):
        logits, _ = model.forward_attr(x)
    return logits.float()


def pgd_l2(model, x, target, attrs, eps, steps, step_size, amp):
    """L2 PGD that maximizes BCE(f(x + d)[:, attrs], target[:, attrs]) with
    ||d||_2 <= eps per image and x + d kept in [-1, 1]. Only d gets gradients."""
    if eps <= 0 or steps <= 0:
        return x
    d = torch.randn_like(x)
    d = d / d.flatten(1).norm(dim=1).view(-1, 1, 1, 1) * eps * torch.rand(x.size(0), 1, 1, 1,
                                                                          device=x.device)
    d = ((x + d).clamp(-1, 1) - x).detach()
    for _ in range(steps):
        d.requires_grad_(True)
        logits = attr_logits(model, x + d, amp)
        loss = F.binary_cross_entropy_with_logits(logits[:, attrs], target[:, attrs],
                                                  reduction='sum')
        g, = torch.autograd.grad(loss, d)
        g = g / g.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
        d = d.detach() + step_size * g
        n = d.flatten(1).norm(dim=1).clamp_min(1e-12)
        d = d * (eps / n).clamp(max=1.0).view(-1, 1, 1, 1)
        d = ((x + d).clamp(-1, 1) - x).detach()
    return (x + d).detach()


def evaluate(model, ref, loader, attrs, eval_eps, args, device, amp):
    model.eval()
    n_attr = len(attrs)
    tp = torch.zeros(n_attr); pos = torch.zeros(n_attr)
    tn = torch.zeros(n_attr); neg = torch.zeros(n_attr)
    agree = torch.zeros(n_attr)
    pred_pos = torch.zeros(n_attr); ref_pos = torch.zeros(n_attr)
    flips = {e: torch.zeros(n_attr) for e in eval_eps}
    n = 0
    for x, y in loader:
        if args.max_val and n >= args.max_val:
            break
        x = x.to(device, non_blocking=True)
        y = y[:, attrs]
        with torch.no_grad():
            p = attr_logits(model, x, amp).sigmoid()
            p_ref = attr_logits(ref, x, amp).sigmoid()
        pred = (p > 0.5).float()
        pa = pred[:, attrs].cpu()
        tp += (pa * y).sum(0); pos += y.sum(0)
        tn += ((1 - pa) * (1 - y)).sum(0); neg += (1 - y).sum(0)
        ra = (p_ref[:, attrs] > 0.5).float().cpu()
        agree += (pa == ra).sum(0)
        pred_pos += pa.sum(0); ref_pos += ra.sum(0)
        for e in eval_eps:
            xa = pgd_l2(model, x, pred, attrs, e, args.eval_pgd_steps,
                        2.5 * e / args.eval_pgd_steps, amp)
            with torch.no_grad():
                pa_adv = (attr_logits(model, xa, amp).sigmoid() > 0.5).float()
            flips[e] += (pa_adv[:, attrs] != pred[:, attrs]).float().sum(0).cpu()
        n += x.size(0)
    names = [CELEBA_ATTRS[a] for a in attrs]
    bal = 0.5 * (tp / pos.clamp(min=1) + tn / neg.clamp(min=1))
    out = {'n': n}
    for i, name in enumerate(names):
        out[name] = {'bal': bal[i].item(),
                     'tpr': (tp[i] / pos[i].clamp(min=1)).item(),
                     'tnr': (tn[i] / neg[i].clamp(min=1)).item(),
                     'pos': (pred_pos[i] / n).item(), 'pos_ref': (ref_pos[i] / n).item(),
                     'agree': (agree[i] / n).item(),
                     **{f'flip@{e:g}': (flips[e][i] / n).item() for e in eval_eps}}
    out['mean'] = {k: sum(out[nm][k] for nm in names) / n_attr for k in out[names[0]]}
    return out


def print_report(tag, rep, attrs, eval_eps):
    names = [CELEBA_ATTRS[a] for a in attrs] + ['mean']
    cols = ['bal', 'tpr', 'tnr', 'pos', 'pos_ref', 'agree'] + [f'flip@{e:g}' for e in eval_eps]
    print(f'\n[{tag}] val ({rep["n"]} images)')
    print(f'   {"":<20}' + ''.join(f'{c:>11}' for c in cols))
    for nm in names:
        print(f'   {nm:<20}' + ''.join(f'{rep[nm][c]:11.1%}' for c in cols))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', required=True)
    p.add_argument('--val_dir', required=True)
    p.add_argument('--attr_file', default='attributes.txt')
    p.add_argument('--image_subdir', default='images')
    p.add_argument('--weights', default='./data/r34_a40_age_256_classifier.pth',
                   help='the training teacher to start from (train_sdflow --attribute_weights)')
    p.add_argument('--out', default='./data/r34_robust_e3.pth')
    p.add_argument('--attrs', nargs='+', type=int, default=[5, 15, 20, 31, 39],
                   help='attributes the attack targets (the edited ones)')
    p.add_argument('--target', default='teacher', choices=['teacher', 'label'])
    p.add_argument('--eps', type=float, default=3.0,
                   help='L2 radius in the teacher input space ([-1, 1], 256x256x3). '
                        'For scale: the shared noise teacher_augment adds (std 0.02) '
                        'has L2 norm ~8.9.')
    p.add_argument('--pgd_steps', type=int, default=5)
    p.add_argument('--step_size', type=float, default=None, help='default 2.5 * eps / pgd_steps')
    p.add_argument('--eval_eps', nargs='+', type=float, default=None,
                   help='attack radii to report flip rates at (default: eps/2, eps, 2*eps)')
    p.add_argument('--eval_pgd_steps', type=int, default=10)
    p.add_argument('--max_val', type=int, default=2000, help='0 = whole val split')
    p.add_argument('--clean_weight', type=float, default=1.0)
    p.add_argument('--adv_weight', type=float, default=1.0)
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval_ckpt', default=None,
                   help='only report on this saved checkpoint (e.g. an _ep*.pth) against '
                        '--weights, no training; --train_dir is still read but not used')
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda'
    amp = not args.no_amp
    attrs = list(args.attrs)
    step_size = args.step_size or 2.5 * args.eps / args.pgd_steps
    eval_eps = args.eval_eps or [args.eps / 2, args.eps, args.eps * 2]

    print('loading data')
    train_set = AttrDataset(args.train_dir, args.attr_file, args.image_subdir,
                            args.img_size, True, 'bilinear')
    val_set = AttrDataset(args.val_dir, args.attr_file, args.image_subdir,
                          args.img_size, False, 'bilinear')
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, drop_last=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    ref = AttributeClassifier(backbone='r34')
    ref.load_state_dict(load_network(args.weights))
    ref.to(device).eval()
    for q in ref.parameters():
        q.requires_grad_(False)
    model = copy.deepcopy(ref)
    for q in list(model.extractor.parameters()) + list(model.attr_heads.parameters()):
        q.requires_grad_(True)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    if args.eval_ckpt:
        if args.eval_ckpt != args.weights:
            print_report('original teacher', evaluate(ref, ref, val_loader, attrs, eval_eps,
                                                      args, device, amp), attrs, eval_eps)
        model.load_state_dict(load_network(args.eval_ckpt))
        print_report(args.eval_ckpt, evaluate(model, ref, val_loader, attrs, eval_eps,
                                              args, device, amp), attrs, eval_eps)
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    log = {'args': vars(args), 'epochs': {}}
    log_path = os.path.splitext(args.out)[0] + '_val.json'

    rep = evaluate(model, ref, val_loader, attrs, eval_eps, args, device, amp)
    print_report('original teacher', rep, attrs, eval_eps)
    log['epochs']['0'] = rep

    for epoch in range(1, args.epochs + 1):
        model.eval()                       # frozen BN statistics, see docstring
        t0, run_c, run_a = time.time(), 0.0, 0.0
        for it, (x, y) in enumerate(train_loader):
            x = crop_flip_resize(x.to(device, non_blocking=True), args.img_size)
            if args.target == 'teacher':
                with torch.no_grad():
                    target = attr_logits(ref, x, amp).sigmoid()
            else:
                target = y.to(device, non_blocking=True).float()
            x_adv = pgd_l2(model, x, target, attrs, args.eps, args.pgd_steps, step_size, amp)
            loss_c = F.binary_cross_entropy_with_logits(attr_logits(model, x, amp), target)
            loss_a = F.binary_cross_entropy_with_logits(
                attr_logits(model, x_adv, amp)[:, attrs], target[:, attrs])
            loss = args.clean_weight * loss_c + args.adv_weight * loss_a
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_c += loss_c.item()
            run_a += loss_a.item()
            if (it + 1) % 100 == 0:
                print(f'  epoch {epoch} it {it + 1}/{len(train_loader)} '
                      f'clean {run_c / (it + 1):.4f} adv {run_a / (it + 1):.4f}', flush=True)
        rep = evaluate(model, ref, val_loader, attrs, eval_eps, args, device, amp)
        print_report(f'epoch {epoch}  [{time.time() - t0:.0f}s]', rep, attrs, eval_eps)
        log['epochs'][str(epoch)] = rep
        torch.save(model.state_dict(), args.out)
        torch.save(model.state_dict(), os.path.splitext(args.out)[0] + f'_ep{epoch}.pth')
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2)
        print(f'  saved {args.out} (and _ep{epoch}); report {log_path}')


if __name__ == '__main__':
    main()
