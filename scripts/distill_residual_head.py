"""Distil the flow's residual into a small network on a few fixed directions.

Evaluating with only the top 4 principal directions of the residual
(evaluate_sdflow.py --residual_basis ... --residual_basis_k 4) kept the full
model's accuracy at matched identity. So per attribute the flow only has to
supply 4 numbers per face. This fits a small MLP (models/residual_head.py)
that predicts those numbers from what the direction bank already sees, using
the flow's own coefficients as targets. With evaluate_sdflow.py
--residual_head, edits then need no ODE solve for the residual.

Data: faces from the TRAIN split, each edited the way the eval does (direction
from the R50 judge's reading of the source) at a random strength in
[--scale_min, --scale_max]. Targets: the residual the bank actually adds,
projected on the top --k directions saved by scripts/analyze_residual.py.

Reported per attribute (held-out 10%):
  basis       share of the residual energy the k directions hold (the ceiling)
  kept fixed  residual energy reproduced by the fixed-direction solution
              (c = attr_delta * constant; what --residual_fixed does, on k dirs)
  kept head   residual energy reproduced by the trained head
              (kept = 1 - |r - r_pred|^2 / |r|^2)
  R2 head     how well the head predicts the flow's k coefficients

Usage:
    python -m scripts.distill_residual_head \
        --checkpoint_dir ./output/SDFlow/multi_v1_cont20k_ctrl --step 100000 \
        --residual_pca ./output/SDFlow/multi_v1_cont20k_ctrl/residual_pca_s100000.pth \
        --independent_attr_weights ./data/r50_celebahq_eval.pth \
        --independent_attr_backbone r50 --age_fine_layer_scale 1.0 --num_faces 4000
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils import data
from tqdm import tqdm

from evaluation.evaluate_sdflow import (
    ATTR_NAMES, _latest_step, apply_run_config, bank_edit, build_optional_judges, build_parser,
    load_models, resolve_controlnet_disable_attrs,
)
from models.dataset import SDFlowDataset
from models.residual_head import ResidualHead, head_features


def name(g):
    return ATTR_NAMES.get(g, f'attr{g}')


def fit_head(X, AD, C, basis, hidden, epochs, lr, batch, device, seed=0):
    """Train one ResidualHead: features X (n, d) and attribute change AD (n,)
    -> coefficients C (n, k). Starts at the fixed-direction solution (base =
    least-squares C ~ AD * base, net = 0) and keeps the epoch with the lowest
    held-out error, so it can only match or improve on that solution."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(X.size(0), generator=g)
    n_val = max(1, X.size(0) // 10)
    va, tr = perm[:n_val], perm[n_val:]
    head = ResidualHead(X.size(1), basis, hidden=hidden)
    head.x_mean.copy_(X[tr].mean(0))
    head.x_std.copy_(X[tr].std(0).clamp(min=1e-6))
    with torch.no_grad():
        a = AD[tr].view(-1, 1)
        head.base.copy_((a * C[tr]).sum(0) / a.pow(2).sum().clamp(min=1e-12))
    head = head.to(device)
    Xd, Ad, Cd = X.to(device), AD.to(device), C.to(device)
    norm = Cd[tr.to(device)].pow(2).mean().clamp(min=1e-12)
    vdev = va.to(device)

    def val_err():
        with torch.no_grad():
            return ((head.coeffs(Xd[vdev], Ad[vdev]) - Cd[vdev]).pow(2).mean() / norm).item()

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    head.eval()
    fixed_err = val_err()
    best, best_state = fixed_err, {k: v.detach().clone() for k, v in head.state_dict().items()}
    for _ in range(epochs):
        head.train()
        order = tr[torch.randperm(tr.numel(), generator=g)].to(device)
        for i in range(0, order.numel(), batch):
            idx = order[i:i + batch]
            loss = (head.coeffs(Xd[idx], Ad[idx]) - Cd[idx]).pow(2).mean() / norm
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        head.eval()
        vl = val_err()
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    head.eval()
    return head, va


def main():
    p = build_parser()
    p.add_argument('--residual_pca', required=True, help='scripts/analyze_residual.py output (.pth).')
    p.add_argument('--k', type=int, default=4, help='Directions per attribute.')
    p.add_argument('--num_faces', type=int, default=4000)
    p.add_argument('--scale_min', type=float, default=0.5)
    p.add_argument('--scale_max', type=float, default=1.25)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--fit_batch', type=int, default=256)
    p.add_argument('--out', default=None)
    args = p.parse_args()
    args = apply_run_config(args)
    args = resolve_controlnet_disable_attrs(args)
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
    if not args.independent_attr_weights:
        raise SystemExit('--independent_attr_weights is required (it sets each edit\'s direction, as the eval does).')

    pca = torch.load(args.residual_pca, map_location='cpu')
    prior, conditioner, G, id_criterion, _, _, direction_bank, _ = load_models(args)
    _, _, _, indep_teacher, _, _ = build_optional_judges(args, args.attribute_index, id_criterion)
    device = direction_bank.residual_scale_raw.device

    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)), T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file,
                       train=True, transform=tf)
    loader = data.DataLoader(ds, shuffle=True, batch_size=args.batch, num_workers=4,
                             generator=torch.Generator().manual_seed(0))
    bases = {g: pca['attrs'][g]['basis'][:args.k].float() for g in args.attribute_index}

    feats, adels, coefs, rnorm2 = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    seen = 0
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for img, latent, _ in tqdm(loader, desc='collect'):
            if seen >= args.num_faces:
                break
            img, latent = img.to(device), latent.to(device)
            B = img.size(0)
            _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
            src = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
            sp = torch.sigmoid(indep_teacher(F.interpolate(src, (256, 256)))[0])
            for li, g in enumerate(args.attribute_index):
                d = torch.where(sp[:, g] > 0.5, -1.0, 1.0)
                sc = (args.scale_min + (args.scale_max - args.scale_min)
                      * torch.rand(B, generator=gen)).to(device)
                _, r, ad = bank_edit(prior, direction_bank, latent, attr_cond, id_cond, li, sc, g, d)
                rf = r.flatten(1).float().cpu()
                feats[g].append(head_features(latent, ad, attr_cond, id_cond).cpu())
                adels[g].append(ad.float().cpu())
                coefs[g].append(rf @ bases[g].t())
                rnorm2[g].append(rf.pow(2).sum(1))
            seen += B

    out = {'config': {'checkpoint_dir': args.checkpoint_dir, 'step': args.step, 'k': args.k,
                      'num_faces': seen, 'scale_range': [args.scale_min, args.scale_max],
                      'residual_pca': args.residual_pca},
           'attrs': {}}
    print(f'\nResidual head, {seen} train faces per attribute, k={args.k} (held-out 10%)')
    print(f'  {"attribute":<11} {"basis":>7} {"kept fixed":>11} {"kept head":>10} {"R2 head":>8}')
    for g in args.attribute_index:
        X, AD = torch.cat(feats[g]), torch.cat(adels[g])
        C, R2n = torch.cat(coefs[g]), torch.cat(rnorm2[g])
        head, va = fit_head(X, AD, C, bases[g], args.hidden, args.epochs, args.lr, args.fit_batch, device)
        Cv, Av = C[va], AD[va]
        with torch.no_grad():
            Cp = head.coeffs(X[va].to(device), Av.to(device)).cpu()
            a = AD.view(-1, 1)
            base_fixed = (a * C).sum(0) / a.pow(2).sum().clamp(min=1e-12)
            Cf = Av.view(-1, 1) * base_fixed

        # |r|^2 = |r_basis|^2 + |r_outside|^2 (orthonormal basis), so a predicted
        # residual's error is |c - c_pred|^2 + |r_outside|^2.
        tot = R2n[va].sum().clamp(min=1e-12)
        outside = R2n[va].sum() - Cv.pow(2).sum()
        basis_share = Cv.pow(2).sum() / tot

        def kept(cp):
            return float(1 - ((cp - Cv).pow(2).sum() + outside) / tot)

        r2 = 1 - (Cp - Cv).pow(2).sum() / (Cv - Cv.mean(0)).pow(2).sum().clamp(min=1e-12)
        print(f'  {name(g):<11} {float(basis_share) * 100:6.1f}% {kept(Cf) * 100:10.1f}% '
              f'{kept(Cp) * 100:9.1f}% {float(r2):8.3f}')
        out['attrs'][g] = {'state_dict': {k: v.cpu() for k, v in head.state_dict().items()},
                           'in_dim': int(X.size(1)), 'basis': bases[g], 'hidden': args.hidden,
                           'val_r2': float(r2), 'basis_share': float(basis_share),
                           'kept': kept(Cp), 'kept_fixed': kept(Cf)}
    print('\nbasis: residual energy inside the k directions (ceiling).  kept fixed / head: residual '
          'energy reproduced by one fixed direction / by the head.  R2: the head\'s prediction of the '
          'flow\'s coefficients.\n'
          'Next: evaluate_sdflow.py --residual_head <this file>, compared at matched ID with '
          'scripts/compare_runs_matched_id.py against the full model.')
    path = args.out or os.path.join(args.checkpoint_dir, f'residual_head_k{args.k}_s{args.step}.pth')
    torch.save(out, path)
    print(f'saved {path}')


if __name__ == '__main__':
    main()
