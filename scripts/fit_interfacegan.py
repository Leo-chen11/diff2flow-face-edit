"""InterFaceGAN boundaries (Shen et al., CVPR 2020 / TPAMI 2020) fitted the way the official code
does (github.com/genforce/interfacegan, utils/manipulator.py train_boundary, MIT License): per
attribute, the latent codes are sorted by the attribute score, the top and the bottom 2%
(--chosen_ratio) are the positive and negative samples, 70% of them (--split_ratio) train a
linear SVM (sklearn SVC(kernel='linear')), and the boundary is the SVM's unit normal (no intercept).
An edit moves every W+ layer by distance * normal (the official edit for W+ codes, its
linear_interpolate on 3-D latents; the official demo sweeps distances -3 to 3).

Used here: the e4e W+ latents of this repo's FFHQ train split read as the mean of their 18 rows
(the official boundaries are fitted on sampled W codes) and the r34 teacher's probabilities as the
scores (the official: a classifier of the user's choice on the generated images).

    python -m scripts.fit_interfacegan --out ./data/interfacegan_boundaries.pth

scripts/editing_curves.py --method interfacegan --interfacegan_boundaries <out> evaluates them
(--scales are distances). scripts/fit_attr_boundaries.py is a different thing: logistic regressions
on the flattened W+ for --preserve_boundaries (and --method boundary).
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch


def train_boundary(latent_codes, scores, chosen_ratio=0.02, split_ratio=0.7, rng=None):
    """The official train_boundary (without its invalid_value filter and logger)."""
    from sklearn import svm
    rng = rng or np.random
    order = np.argsort(scores, axis=0)[::-1, 0]
    latent_codes, scores = latent_codes[order], scores[order]
    n = latent_codes.shape[0]
    chosen = min(int(n * chosen_ratio) if 0 < chosen_ratio <= 1 else int(chosen_ratio), n // 2)
    train_num = int(chosen * split_ratio)
    val_num = chosen - train_num
    pos_idx, neg_idx = np.arange(chosen), np.arange(chosen)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    pos, neg = latent_codes[:chosen], latent_codes[-chosen:]
    train = np.concatenate([pos[pos_idx[:train_num]], neg[neg_idx[:train_num]]])
    train_label = np.concatenate([np.ones(train_num, dtype=int), np.zeros(train_num, dtype=int)])
    val = np.concatenate([pos[pos_idx[train_num:]], neg[neg_idx[train_num:]]])
    val_label = np.concatenate([np.ones(val_num, dtype=int), np.zeros(val_num, dtype=int)])
    rest, rest_scores = latent_codes[chosen:-chosen], scores[chosen:-chosen]
    decision = (scores[0] + scores[-1]) / 2
    rest_label = (rest_scores.ravel() >= decision).astype(int)

    clf = svm.SVC(kernel='linear').fit(train, train_label)
    val_acc = float((clf.predict(val) == val_label).mean()) if val_num else float('nan')
    rest_acc = float((clf.predict(rest) == rest_label).mean()) if len(rest) else float('nan')
    a = clf.coef_.reshape(1, -1).astype(np.float32)
    return a / np.linalg.norm(a), dict(chosen=chosen, train=train_num, val=val_num, val_acc=val_acc,
                                       remaining=len(rest), remaining_acc=rest_acc)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='./data/interfacegan_boundaries.pth')
    p.add_argument('--attrs', nargs='+', type=int, default=[15, 20, 39, 31, 5])
    p.add_argument('--chosen_ratio', type=float, default=0.02)
    p.add_argument('--split_ratio', type=float, default=0.7)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--index_file', default='./data/ffhq.txt')
    p.add_argument('--image_root', default='data/FFHQ')
    p.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    p.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth')
    args = p.parse_args()

    from training.train_adatrans import load_train_split
    W, P = load_train_split(args)
    X = W.mean(1).numpy().astype(np.float64)
    rng = np.random.RandomState(args.seed)
    normals, stats = [], {}
    for a in args.attrs:
        n, st = train_boundary(X, P[:, a].numpy()[:, None].astype(np.float64), args.chosen_ratio,
                               args.split_ratio, rng)
        normals.append(torch.from_numpy(n[0]))
        spread = float((torch.from_numpy(X).float() @ normals[-1]).std())
        stats[a] = dict(st, spread=spread)
        print(f'attr {a:2d}: {st["train"]} + {st["train"]} train, {st["val"]} + {st["val"]} held out; '
              f'held-out accuracy {st["val_acc"] * 100:.1f}%, remaining {st["remaining_acc"] * 100:.1f}%; '
              f'std of the latents along the normal {spread:.3f}')
    torch.save({'attrs': list(args.attrs), 'normals': torch.stack(normals), 'stats': stats,
                'config': vars(args), 'input': 'mean of the 18 e4e W+ rows'}, args.out)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
