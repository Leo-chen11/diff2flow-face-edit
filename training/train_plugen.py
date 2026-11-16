"""Train the PluGeN baseline (models/plugen.py; AAAI 2022) on this repo's data, with the official
StyleGAN training recipe (github.com/gmum/plugen StyleGAN/train.py):

    NICE flow, 4 additive couplings, 4-layer MLPs of width 512
    loss = -[ log N(z[:K]; label, sigma) + log N(z[K:]; 0, 1) ]      (the flow is volume preserving)
    sigma = 0.4 * 0.999 ** epoch, 1000 epochs, batch 32, Adam 1e-4, gradient norm clipped at 1

    python training/train_plugen.py --out ./output/baselines/plugen

Writes <out>/model.pth and <out>/config.json (a checkpoint every --save_freq epochs, resumed with
--resume); scripts/editing_curves.py --method plugen --plugen_dir <out> evaluates it (--scales are
the target value of the attribute coordinate: 1 = the end of the label range).

Deviations from the official setting, all in config.json:
  * data: the e4e W+ latents of this repo's FFHQ train split, read as the mean of their 18 rows
    (the official: StyleFlow's 10000 sampled W latents, 9500 of them for training)
  * labels: this repo's attributes (default 15 20 39 31 5) as the r34 teacher's probabilities p of
    the preds file, on the official [-1, 1] scale as 2p - 1 (the official: 8 attributes of the
    Microsoft Face API -- gender, glasses, yaw, pitch, baldness, beard, age, expression)
  * an epoch is one pass over this train split (about 5 times the official number of samples), so
    the same 1000-epoch sigma schedule is spread over about 5 times as many updates
"""
import argparse
import json
import math
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch


def nll(z, labels, sigma, K):
    """-[log N(z[:K]; labels, sigma) + log N(z[K:]; 0, 1)], mean over the batch."""
    c = 0.5 * math.log(2 * math.pi)
    attr = 0.5 * ((z[:, :K] - labels) / sigma) ** 2 + math.log(sigma) + c
    rest = 0.5 * z[:, K:] ** 2 + c
    return (attr.sum(1) + rest.sum(1)).mean()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='./output/baselines/plugen')
    p.add_argument('--attrs', nargs='+', type=int, default=[15, 20, 39, 31, 5])
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--sigma', type=float, default=0.4)
    p.add_argument('--decay', type=float, default=0.999)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--n_couplings', type=int, default=4)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--save_freq', type=int, default=50)
    p.add_argument('--resume', action='store_true', help='continue from <out>/last.pth')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--index_file', default='./data/ffhq.txt')
    p.add_argument('--image_root', default='data/FFHQ')
    p.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    p.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth')
    args = p.parse_args()

    from models.plugen import PluGeNFlow, to_code
    from training.train_adatrans import load_train_split
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    W, P = load_train_split(args)
    X = W.mean(1).cuda()
    A = to_code(P[:, args.attrs]).cuda()
    K, N = len(args.attrs), len(X)
    print(f'{N} latents, {K} attributes {args.attrs}; label means on [-1, 1]: '
          + '  '.join(f'{a}:{m:+.2f}' for a, m in zip(args.attrs, A.mean(0).tolist())))

    model = PluGeNFlow(512, args.n_layers, args.n_couplings, args.hidden).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = 0
    last = os.path.join(args.out, 'last.pth')
    if args.resume and os.path.exists(last):
        ck = torch.load(last, map_location='cpu')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        start = ck['epoch']
        print(f'resumed from {last} at epoch {start}')

    cfg = {'method': 'PluGeN (AAAI 2022), retrained', 'attrs': list(args.attrs), 'n_layers': args.n_layers,
           'n_couplings': args.n_couplings, 'hidden': args.hidden,
           'input': 'mean of the 18 e4e W+ rows', 'labels': 'r34 preds file, 2p - 1', 'args': vars(args),
           'deviations': ['e4e W+ latents of the FFHQ train split (mean of 18 rows), not 9500 sampled W',
                          'r34 teacher probabilities of this repo\'s attributes as labels, not 8 Face API attributes',
                          'an epoch is one pass over this (about 5x larger) train split']}
    steps = (N + args.batch - 1) // args.batch
    t0 = time.time()
    for epoch in range(start, args.epochs):
        sigma = args.sigma * args.decay ** epoch
        perm = torch.randperm(N, device=X.device)
        tot = 0.0
        model.train()
        for b in range(0, N, args.batch):
            idx = perm[b:b + args.batch]
            loss = nll(model(X[idx]), A[idx], sigma, K)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        done = epoch + 1
        if done % 10 == 0 or done == args.epochs or epoch == start:
            model.eval()
            with torch.no_grad():
                z = model(X[:4096])
                err = (z[:, :K] - A[:4096]).abs().mean(0)
                side = ((z[:, :K] > 0) == (A[:4096] > 0)).float().mean(0)
            el = time.time() - t0
            eta = el / (done - start) * (args.epochs - done)
            print(f'epoch {done}/{args.epochs}  sigma {sigma:.3f}  nll {tot / steps:.2f}  |z_attr - label| '
                  + ' '.join(f'{a}:{e:.2f}' for a, e in zip(args.attrs, err.tolist()))
                  + '  same side % ' + ' '.join(f'{s * 100:.0f}' for s in side.tolist())
                  + f'  rest std {z[:, K:].std().item():.2f}  [{el / 60:.0f} min, about {eta / 60:.0f} left]', flush=True)
        if done % args.save_freq == 0 or done == args.epochs:
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': done}, last)
            torch.save(model.state_dict(), os.path.join(args.out, 'model.pth'))
            cfg.update(epochs_done=done, final_sigma=sigma)
            with open(os.path.join(args.out, 'config.json'), 'w') as f:
                json.dump(cfg, f, indent=2)
    print(f'saved {args.out}/model.pth and config.json')


if __name__ == '__main__':
    main()
