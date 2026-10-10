"""Linear W+ boundaries for the 40 CelebA attributes, for --preserve_boundaries.

The --leak40 eval shows every edit dragging other attributes along (Male: lipstick
-0.47, big nose +0.31; Young: chubby, double chin). evaluate_sdflow.py
--preserve_boundaries removes, from each edit's W+ delta, its component along
the boundary normals of the attributes that should stay put (conditional
manipulation, as in InterFaceGAN). This fits those normals.

Data: the TRAIN split's precomputed e4e latents and the training teacher's
(r34) 40-attribute predictions (--latent_file / --preds_file); no images are
loaded, and the R50 eval judge is not involved.

Per attribute, a logistic regression on the flattened, standardised W+ (18 x 512)
using only confidently labelled faces (p < --lo or p > --hi), classes weighted
to balance, L2 --l2. All 40 are fitted jointly (one matrix). The saved normal is
the logit's gradient in raw W+ coordinates (w / sigma, unit length): removing a
delta's component along it leaves that attribute's linear score unchanged.

Reported: held-out AUC per attribute (boundaries below --preserve_min_auc in the
eval are not used), and for each edited attribute its most aligned other normals
(|cos|), i.e. what the latent space ties it to.

Usage:
    python -m scripts.fit_attr_boundaries --out ./data/attr_boundaries_wplus.pth
"""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from evaluation.evaluate_sdflow import CELEBA_ALL_ATTRS, build_parser
from models.dataset import SDFlowDataset

EDITED = (15, 20, 39, 31, 5)


def auc(score, label):
    """Mann-Whitney AUC; score, label: (n,)."""
    pos, neg = score[label > 0.5], score[label <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return float('nan')
    ranks = torch.empty_like(score)
    ranks[score.argsort()] = torch.arange(1, score.numel() + 1, dtype=score.dtype, device=score.device)
    rp = ranks[label > 0.5].sum()
    return float((rp - pos.numel() * (pos.numel() + 1) / 2) / (pos.numel() * neg.numel()))


def fit(X, P, lo, hi, l2, steps, lr, device, val_frac=0.1, seed=0):
    """X: (n, d) raw latents, P: (n, A) probabilities. Returns unit normals in raw
    coordinates (A, d), held-out AUC (A,), confident counts (A, 2)."""
    n, d = X.shape
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(n * val_frac))
    va, tr = perm[:n_val], perm[n_val:]
    mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-6)
    Xs = ((X - mu) / sd).to(device)
    Y = (P > 0.5).float().to(device)
    conf = ((P < lo) | (P > hi)).float().to(device)
    trm = torch.zeros(n, device=device)
    trm[tr.to(device)] = 1.0
    M = conf * trm.view(-1, 1)                                     # (n, A) training weight mask
    npos = (M * Y).sum(0).clamp(min=1)
    nneg = (M * (1 - Y)).sum(0).clamp(min=1)
    Wt = M * (Y * (0.5 / npos) + (1 - Y) * (0.5 / nneg))           # each class half the weight
    A = P.size(1)
    W = torch.zeros(d, A, device=device, requires_grad=True)
    b = torch.zeros(A, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        logit = Xs @ W + b
        loss = (F.binary_cross_entropy_with_logits(logit, Y, reduction='none') * Wt).sum() / A \
            + l2 * W.pow(2).sum() / A
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        logit = Xs @ W + b
        vam = torch.zeros(n, dtype=torch.bool, device=device)
        vam[va.to(device)] = True
        aucs = []
        for a in range(A):
            m = vam & (conf[:, a] > 0)
            aucs.append(auc(logit[m, a], Y[m, a]))
        normals = (W / sd.to(device).view(-1, 1)).t()
        normals = normals / normals.norm(dim=1, keepdim=True).clamp(min=1e-12)
    counts = torch.stack([(conf * Y).sum(0), (conf * (1 - Y)).sum(0)], 1).cpu()
    return normals.cpu(), torch.tensor(aucs), counts


def main():
    p = build_parser()
    for act in p._actions:                    # data paths only; no checkpoint needed
        if act.dest == 'checkpoint_dir':
            act.required = False
    p.add_argument('--out', default='./data/attr_boundaries_wplus.pth')
    p.add_argument('--lo', type=float, default=0.3, help='Confident negative below this probability.')
    p.add_argument('--hi', type=float, default=0.7, help='Confident positive above this probability.')
    p.add_argument('--l2', type=float, default=1e-3)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--max_faces', type=int, default=0, help='0 = the whole train split.')
    args = p.parse_args()

    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file,
                       train=True, transform=None)
    files = list(ds.image_list)
    if args.max_faces:
        files = files[:args.max_faces]
    X = torch.stack([torch.as_tensor(ds._lookup_precomputed(ds.latents, f)).float().reshape(-1)
                     for f in files])
    P = torch.stack([torch.as_tensor(ds._lookup_precomputed(ds.preds, f)).float().reshape(-1)[:40]
                     for f in files])
    if X.size(1) != 18 * 512:
        raise SystemExit(f'expected W+ latents of 18 x 512, got {X.size(1)} values per face')
    if P.min() < 0 or P.max() > 1:
        P = torch.sigmoid(P)                                       # logits stored instead of probabilities
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'{len(files)} train faces, W+ {X.size(1)} dims, fitting 40 boundaries on {device}')
    normals, aucs, counts = fit(X, P, args.lo, args.hi, args.l2, args.steps, args.lr, device)

    names = list(CELEBA_ALL_ATTRS)
    print(f'\n  {"attribute":<20} {"AUC":>6} {"n+":>6} {"n-":>6}')
    for a in range(40):
        print(f'  {a:2d} {names[a]:<17} {float(aucs[a]):6.3f} {int(counts[a, 0]):6d} {int(counts[a, 1]):6d}')
    C = normals @ normals.t()
    print('\nMost aligned boundaries of the edited attributes (cos of normals; sign = direction):')
    for g in EDITED:
        c = C[g].clone()
        c[g] = 0
        top = c.abs().argsort(descending=True)[:6]
        print(f'  {names[g]:<11} ' + ', '.join(f'{names[j]} {float(c[j]):+.2f}' for j in top))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({'normals': normals.view(40, 18, 512), 'auc': aucs, 'counts': counts, 'attr_names': names,
                'latent_file': args.latent_file, 'preds_file': args.preds_file, 'n_faces': len(files),
                'lo': args.lo, 'hi': args.hi, 'l2': args.l2}, args.out)
    print(f'\nsaved {args.out}\nnext: evaluate_sdflow.py ... --preserve_boundaries {args.out}')


if __name__ == '__main__':
    main()
