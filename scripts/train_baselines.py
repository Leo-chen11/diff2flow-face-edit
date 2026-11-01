"""Train the Latent Transformer and StyleFlow baselines (models/baselines.py) on
this repo's e4e W+ latents of the TRAIN split, labelled by the precomputed r34
predictions -- the same latents and teacher readings our own flow trains on.
No images and no generator are needed: both are trained in latent space.

    # 1. Latent Transformer: the latent classifier, then one transformer per attribute
    python -m scripts.train_baselines lcnet --out ./output/baselines/lt
    python -m scripts.train_baselines lt --out ./output/baselines/lt --attrs 15 20 39 31 5
    # 2. StyleFlow: one conditional flow over the attributes
    python -m scripts.train_baselines styleflow --out ./output/baselines/styleflow --attrs 15 20 39 31 5

Evaluate with scripts/editing_curves.py --method lt --lt_dir ./output/baselines/lt, or
--method styleflow --styleflow_ckpt ./output/baselines/styleflow/styleflow.pth.

Recipes follow the papers' code: Latent Transformer (configs/001.yaml: Adam 1e-3,
weight decay 5e-4, batch 32, 10 epochs, loss = BCE(target) + 10 * latent MSE +
other-attribute logit MSE weighted by 1 - |label correlation|); StyleFlow
(maximum likelihood of W+ under the attribute-conditioned CNF, as the released
SDFlow's flow minus its extra losses).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.baselines import LatentTransformer, LCNet, styleflow_cnf
from models.dataset import SDFlowDataset
from models.flows.utils import standard_normal_logprob


def load_split(args, train=True):
    """(N, 18, 512) latents and (N, 40) r34 probabilities of one split."""
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file, train=train)
    files = list(ds.image_list)
    if args.max_faces and len(files) > args.max_faces:
        files = files[:args.max_faces]
    W = torch.stack([torch.as_tensor(ds._lookup_precomputed(ds.latents, f)).float().reshape(18, 512)
                     for f in files])
    P = torch.stack([torch.as_tensor(ds._lookup_precomputed(ds.preds, f)).float().reshape(-1)[:40]
                     for f in files])
    if P.min() < 0 or P.max() > 1:          # logits, not probabilities
        P = torch.sigmoid(P)
    print(f'{"train" if train else "test"} split: {len(files)} faces')
    return W, P


def batches(n, bs, shuffle=True):
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n - bs + 1 if shuffle else n, bs):
        yield idx[i:i + bs]


def train_lcnet(args):
    W, P = load_split(args, True)
    Wt, Pt = load_split(args, False)
    net = LCNet().cuda()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    labels, labels_t = (P > 0.5).float(), (Pt > 0.5).float()
    for ep in range(args.epochs):
        net.train()
        for b in tqdm(list(batches(len(W), args.batch)), desc=f'lcnet epoch {ep + 1}', leave=False):
            loss = F.binary_cross_entropy_with_logits(net(W[b].cuda()), labels[b].cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pred = torch.cat([net(Wt[b].cuda()).cpu() for b in batches(len(Wt), 512, False)])
        acc = ((pred > 0) == (labels_t > 0.5)).float().mean(0)
        print(f'epoch {ep + 1}: last loss {loss.item():.4f}, test accuracy vs r34 '
              f'mean {acc.mean():.3f}; ' + ', '.join(f'{a}: {acc[a]:.3f}' for a in args.attrs))
    os.makedirs(args.out, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(args.out, 'lcnet.pth'))
    print(f'saved {os.path.join(args.out, "lcnet.pth")}')


def train_lt(args):
    W, P = load_split(args, True)
    lc = LCNet().cuda()
    lc.load_state_dict(torch.load(os.path.join(args.out, 'lcnet.pth'), map_location='cpu'))
    lc.eval()
    for p in lc.parameters():
        p.requires_grad_(False)
    corr = np.nan_to_num(np.corrcoef((P > 0.5).float().numpy().T))
    for a in args.attrs:
        mask = torch.tensor(1 - np.abs(corr[a]), dtype=torch.float32).cuda()
        mask[a] = 0
        T = LatentTransformer().cuda()
        opt = torch.optim.Adam(T.parameters(), lr=args.lr, weight_decay=5e-4)
        for ep in range(args.epochs):
            for b in tqdm(list(batches(len(W), args.batch)), desc=f'lt attr {a} epoch {ep + 1}', leave=False):
                w0 = W[b].cuda()
                with torch.no_grad():
                    logit0 = lc(w0)
                p0 = torch.sigmoid(logit0[:, a])
                coeff = LatentTransformer.train_coeff(p0)
                target = (p0 + coeff).clamp(0, 1).round()
                w1 = T(w0, coeff)
                logit1 = lc(w1)
                pos = target.size(0) / (target.sum() + 1e-8)
                neg = target.size(0) / (target.size(0) - target.sum() + 1e-8)
                loss_pb = (F.binary_cross_entropy_with_logits(logit1[:, a], target, reduction='none')
                           * (pos * target + neg * (1 - target))).mean()
                loss_recon = F.mse_loss(w1, w0)
                loss_reg = F.mse_loss(logit1 * mask, logit0 * mask)
                loss = loss_pb + 10 * loss_recon + loss_reg
                opt.zero_grad()
                loss.backward()
                opt.step()
            print(f'attr {a} epoch {ep + 1}: target {loss_pb.item():.4f}, latent {loss_recon.item():.5f}, '
                  f'others {loss_reg.item():.4f}')
        path = os.path.join(args.out, f'tnet_{a}.pth')
        torch.save(T.state_dict(), path)
        print(f'saved {path}')


def train_styleflow(args):
    W, P = load_split(args, True)
    C = P[:, args.attrs]
    flow = styleflow_cnf(len(args.attrs)).cuda()
    opt = torch.optim.Adam(flow.parameters(), lr=args.lr)
    os.makedirs(args.out, exist_ok=True)
    step = 0
    bar = tqdm(total=args.steps, desc='styleflow')
    while step < args.steps:
        for b in batches(len(W), args.batch):
            w, c = W[b].cuda(), C[b].cuda()
            z, delta = flow(w, c, torch.zeros(w.size(0), 18, 1).to(w))
            logpz = standard_normal_logprob(z).view(w.size(0), -1).sum(1)
            nll = -(logpz - delta.view(w.size(0), -1).sum(1)).mean() / (18 * 512)
            opt.zero_grad()
            nll.backward()
            opt.step()
            step += 1
            bar.update(1)
            if step % 100 == 0:
                bar.set_postfix(nll=f'{nll.item():.4f}')
            if step % 2000 == 0 or step == args.steps:
                torch.save({'state_dict': flow.state_dict(), 'attrs': args.attrs, 'step': step},
                           os.path.join(args.out, 'styleflow.pth'))
            if step >= args.steps:
                break
    print(f'saved {os.path.join(args.out, "styleflow.pth")}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('what', choices=['lcnet', 'lt', 'styleflow'])
    p.add_argument('--out', required=True)
    p.add_argument('--attrs', nargs='+', type=int, default=[15, 20, 39, 31, 5])
    p.add_argument('--index_file', default='./data/ffhq.txt')
    p.add_argument('--image_root', default='data/FFHQ')
    p.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    p.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth')
    p.add_argument('--max_faces', type=int, default=0, help='Use only the first N faces of a split (0 = all).')
    p.add_argument('--epochs', type=int, default=None, help='lcnet default 20, lt default 10.')
    p.add_argument('--batch', type=int, default=None, help='lcnet default 256, lt 32, styleflow 32.')
    p.add_argument('--lr', type=float, default=None, help='lcnet / lt default 1e-3, styleflow 1e-4.')
    p.add_argument('--steps', type=int, default=10000, help='styleflow optimizer steps.')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    defaults = {'lcnet': (20, 256, 1e-3), 'lt': (10, 32, 1e-3), 'styleflow': (1, 32, 1e-4)}[args.what]
    args.epochs = args.epochs or defaults[0]
    args.batch = args.batch or defaults[1]
    args.lr = args.lr or defaults[2]
    {'lcnet': train_lcnet, 'lt': train_lt, 'styleflow': train_styleflow}[args.what](args)


if __name__ == '__main__':
    main()
