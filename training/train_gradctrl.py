"""The GradCtrl baseline (models/gradctrl.py; Chen et al., ECCV 2022) on this repo's data.

    train            retrain the latent classifier the way the official prepare_data.py + train.py do,
                     on this repo's TRAIN latents and attribute teacher, for this repo's attributes
    import_released  bring the official FFHQ classifier (GradCtrl/model_ffhq/model.pth: gender,
                     smile, eyeglasses, age, pose) into the same layout, for a check that the port
                     reproduces the release; it has no Bangs

    python training/train_gradctrl.py train --out ./output/baselines/gradctrl
    python training/train_gradctrl.py import_released --released_dir <GradCtrl clone>/model_ffhq \\
        --out ./output/baselines/gradctrl_released

Both write <out>/model.pth and <out>/config.json; scripts/editing_curves.py --method gradctrl
--gradctrl_dir <out> evaluates them (--scales are numbers of steps of 4/9; the official demo uses 9).

The official recipe (prepare_data.py, train.py), kept: per attribute the 5% of samples with the
highest and the 5% with the lowest score (--p_perclass), 30% of them held out (--p_val), one binary
branch per attribute, hidden 16, LeakyReLU 0.1, BCE, Adam at its default 1e-3 (train.py builds the
optimizer without its --lr), batch 32, the checkpoint with the best held-out accuracy.
Deviations, all in config.json:
  * samples: this repo's e4e W+ latents of the FFHQ train split, read as the mean of their 18 rows
    (the official samples are 10000 W latents of truncation 0.4); scores: the r34 teacher's
    probabilities of the preds file (the official: an image classifier of the user's choice),
    label 1 = probability > 0.5
  * --n_epochs 10 (the official default is 1; the best epoch on the held-out samples is kept, as
    the official code does)
  * Bangs (5) has no official edit setting: models/gradctrl.py OURS_EXTRA
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F


def save(out, model, cfg):
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, 'model.pth'))
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'saved {out}/model.pth and config.json')


def settings_for(attrs):
    from models.gradctrl import OFFICIAL_FFHQ, OURS_EXTRA
    table = dict(OFFICIAL_FFHQ)
    table.update(OURS_EXTRA)
    return {str(a): {'keep': table[a][0], 'top_channels': table[a][1], 'layers': list(table[a][2]),
                     'source': 'official' if a in OFFICIAL_FFHQ else 'ours'}
            for a in attrs if a in table}


def train(args, W, P):
    from models.gradctrl import MultiLabelClassifier, STEP_SIZE
    torch.manual_seed(args.seed)
    attrs = list(args.attrs)
    X = W.mean(1)
    N = len(X)
    n = int(N * args.p_perclass)
    g = torch.Generator().manual_seed(args.seed)
    tr, va = {}, {}
    for a in attrs:
        order = torch.argsort(P[:, a])
        sel = torch.unique(torch.cat([order[:n], order[-n:]]))
        sel = sel[torch.randperm(len(sel), generator=g)]
        k = int(args.p_val * len(sel))
        va[a], tr[a] = sel[:k], sel[k:]
        pos = (P[tr[a], a] > 0.5).float().mean().item()
        print(f'attr {a}: {len(tr[a])} train / {len(va[a])} held-out samples, {pos * 100:.0f}% positive')
    Y = (P > 0.5).float()

    model = MultiLabelClassifier(len(attrs), n_shared_layers=0, hidden_size=args.hidden_size).cuda()
    opt = torch.optim.Adam(model.parameters())
    Xc, Yc = X.cuda(), Y.cuda()
    L = min(len(tr[a]) for a in attrs)
    best, best_state = -1.0, None

    def held_out():
        model.eval()
        with torch.no_grad():
            acc = [((model.logits(Xc[va[a]])[:, i] > 0).float() == Yc[va[a], a]).float().mean().item()
                   for i, a in enumerate(attrs)]
        model.train()
        return acc

    for ep in range(1, args.n_epochs + 1):
        perm = torch.randperm(L, generator=g)
        tot = 0.0
        for b in range(0, L, args.batch_size):
            j = perm[b:b + args.batch_size]
            loss = 0.0
            for i, a in enumerate(attrs):
                idx = tr[a][j]
                loss = loss + F.binary_cross_entropy_with_logits(model.logits(Xc[idx])[:, i], Yc[idx, a])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        acc = held_out()
        mean = sum(acc) / len(acc)
        print(f'epoch {ep}: loss {tot / max(1, (L + args.batch_size - 1) // args.batch_size):.4f}  held-out acc '
              + '  '.join(f'{a}:{x * 100:.1f}' for a, x in zip(attrs, acc)) + f'  mean {mean * 100:.1f}')
        if mean >= best:
            best = mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    cfg = {'method': 'GradCtrl (ECCV 2022), retrained', 'attrs': attrs, 'hidden_size': args.hidden_size,
           'n_shared_layers': 0, 'signs': {str(a): 1.0 for a in attrs}, 'step_size': STEP_SIZE,
           'settings': settings_for(attrs), 'best_held_out_acc': best,
           'input': 'mean of the 18 e4e W+ rows', 'labels': 'r34 preds file, probability > 0.5',
           'args': vars(args),
           'deviations': ['e4e W+ latents of the FFHQ train split (mean of 18 rows), not sampled W',
                          'r34 teacher scores as labels', f'n_epochs {args.n_epochs} (official default 1)',
                          'Bangs edit setting is ours (models/gradctrl.py OURS_EXTRA)']}
    save(args.out, model, cfg)


def import_released(args, W, P):
    """The official FFHQ classifier, with the sign of each branch read off this repo's teacher:
    the release's labels are 1 for a NEGATIVE score of its own classifier (prepare_data.py), so
    which way a branch points is not known without a check."""
    from models.gradctrl import MultiLabelClassifier, RELEASED_FFHQ_ATTRS, STEP_SIZE
    names = ['gender', 'smile', 'eyeglasses', 'age', 'pose']
    attrs = [RELEASED_FFHQ_ATTRS[n] if RELEASED_FFHQ_ATTRS[n] is not None else -1 for n in names]
    model = MultiLabelClassifier(5, n_shared_layers=0, hidden_size=16).cuda().eval()
    state = torch.load(os.path.join(args.released_dir, 'model.pth'), map_location='cpu')
    model.load_state_dict(state, strict=True)
    with torch.no_grad():
        pr = torch.cat([model.probs(W[i:i + 4096].mean(1).cuda()).cpu() for i in range(0, len(W), 4096)])
    signs = {}
    for i, (nm, a) in enumerate(zip(names, attrs)):
        if a < 0:
            continue
        x, y = pr[:, i], P[:, a]
        r = (((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std() + 1e-12)).item()
        signs[str(a)] = 1.0 if r > 0 else -1.0
        print(f'{nm:>10} -> CelebA {a:2d}: correlation with the r34 teacher {r:+.3f}, sign {signs[str(a)]:+.0f}')
        if abs(r) < 0.3:
            print(f'   [WARN] weak correlation: the released branch and this repo\'s attribute {a} disagree')
    cfg = {'method': 'GradCtrl (ECCV 2022), official FFHQ release', 'attrs': attrs, 'names': names,
           'hidden_size': 16, 'n_shared_layers': 0, 'signs': signs, 'step_size': STEP_SIZE,
           'settings': settings_for([a for a in attrs if a >= 0]),
           'input': 'mean of the 18 e4e W+ rows (the release was trained on W of truncation 0.4)',
           'released_dir': os.path.abspath(args.released_dir)}
    save(args.out, model, cfg)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('what', choices=['train', 'import_released'])
    p.add_argument('--out', default='./output/baselines/gradctrl')
    p.add_argument('--attrs', nargs='+', type=int, default=[15, 20, 39, 31, 5])
    p.add_argument('--released_dir', default=None, help='import_released: the official model_ffhq folder')
    p.add_argument('--p_perclass', type=float, default=0.05)
    p.add_argument('--p_val', type=float, default=0.3)
    p.add_argument('--hidden_size', type=int, default=16)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--index_file', default='./data/ffhq.txt')
    p.add_argument('--image_root', default='data/FFHQ')
    p.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    p.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth')
    args = p.parse_args()
    if args.what == 'import_released' and not args.released_dir:
        raise SystemExit('import_released needs --released_dir <GradCtrl clone>/model_ffhq')
    from training.train_adatrans import load_train_split
    W, P = load_train_split(args)
    (train if args.what == 'train' else import_released)(args, W, P)


if __name__ == '__main__':
    main()
