"""Which W+ rows (StyleGAN2 layers 0-17) a model's edits move -- the check behind
--velocity_layer_embed.

The velocity networks of the flow apply the same function to all 18 rows of a W+ code; without
--velocity_layer_embed they can tell the rows apart only by the rows' own values, and e4e rows sit
close to each other. If that limits the model, its edits move every row about the same way:
  * the share of the edit in each row is about flat across rows, for every attribute alike, and
  * the per-row edits point the same way (mean cosine between rows close to 1),
while the direction bank the flow was taught from (its per-layer directions, e.g. Young restricted to
rows 0-10 by --bank_dir_layers) puts each attribute in its own rows. This script measures both, for
a run and its bank, so the motivation is a number, not an assumption; run it again on a
--velocity_layer_embed run to see whether its edits became layer-specific.

Writes <out>/layer_profile.csv (attribute, source, row 0..17 shares, row cosine, share outside the
bank's rows) and <out>/layer_profile.png (heat maps).

    python -m scripts.layer_profile --checkpoint_dir <run> --step 100000 --num_samples 200
"""
import csv
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torchvision.transforms as T
from torch.utils import data
from tqdm import tqdm

from evaluation.evaluate_sdflow import (
    ATTR_NAMES, apply_run_config, build_parser, edit_multi_attribute, load_models,
)
from models.dataset import SDFlowDataset


def main():
    p = build_parser()
    p.add_argument('--scale', type=float, default=1.0, help='edit strength')
    p.add_argument('--out_dir', default=None, help='default <checkpoint_dir>/layer_profile_s<step>')
    args = apply_run_config(p.parse_args())
    out = args.out_dir or os.path.join(args.checkpoint_dir, f'layer_profile_s{args.step}')
    os.makedirs(out, exist_ok=True)

    prior, conditioner, G, id_criterion, _, _, bank = load_models(args)
    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)), T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root, latents_file=args.latent_file,
                       preds_file=args.preds_file, train=False, transform=tf)
    if args.sample_offset:
        ds = data.Subset(ds, range(args.sample_offset, len(ds)))
    loader = data.DataLoader(ds, shuffle=False, batch_size=args.batch, num_workers=4)
    attrs = list(args.attribute_index)

    sq = {g: torch.zeros(18) for g in attrs}          # sum over faces of each row's share of |delta|^2
    cos = {g: 0.0 for g in attrs}
    n = {g: 0 for g in attrs}
    seen = 0
    with torch.no_grad():
        for img, latent, _ in tqdm(loader, desc='layer profile'):
            if seen >= args.num_samples:
                break
            img, latent = img.cuda(), latent.cuda()
            _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
            for li, g in enumerate(attrs):
                _, new = edit_multi_attribute(prior, conditioner, G, id_criterion, img, latent, attr_cond, id_cond,
                                              [li], args.scale, bank, attr_global_idxs=[g], return_parts=True)
                d = (new - latent).float()                                   # (B, 18, 512)
                e = d.pow(2).sum(-1)                                         # (B, 18)
                ok = e.sum(1) > 1e-12
                if not ok.any():
                    continue
                share = e[ok] / e[ok].sum(1, keepdim=True)
                u = torch.nn.functional.normalize(d[ok], dim=-1)
                c = u @ u.transpose(1, 2)                                    # (k, 18, 18)
                off = ~torch.eye(18, dtype=torch.bool, device=c.device)
                sq[g] += share.sum(0).cpu()
                cos[g] += float(c[:, off].mean(1).sum())
                n[g] += int(ok.sum())
            seen += img.size(0)

    # the bank's own per-layer profile (what stage 2 handed the edit over from). Its direction_units are
    # unit vectors PER ROW (so their norms say nothing); the per-row magnitude is the bank's layer-norm
    # prior (_prior_norms, from the bank file's layer_norms), masked by --bank_dir_layers.
    bank_share = {}
    prior = getattr(bank, '_prior_norms', None) if bank is not None else None
    if prior is not None:
        P = prior.detach().float().cpu()                                     # (A, 18)
        masks = getattr(bank, '_dir_layer_mask', {}) or {}
        for li, g in enumerate(attrs):
            m = P[li].clone()
            if li in masks:
                m = m * masks[li].float().cpu()
            e = m.pow(2)
            bank_share[g] = e / e.sum().clamp(min=1e-12)

    rows = []
    for g in attrs:
        if not n[g]:
            continue
        s = sq[g] / n[g]
        name = ATTR_NAMES.get(g, str(g)) if isinstance(ATTR_NAMES, dict) else str(g)
        r = {'attribute': name, 'source': 'flow edit', 'faces': n[g], 'row_cos': round(cos[g] / n[g], 4),
             'flatness': round(float((s.max() - s.min()) / s.mean()), 3)}
        r.update({f'row{i}': round(float(v), 4) for i, v in enumerate(s)})
        if g in bank_share:
            b = bank_share[g]
            r['outside_bank_rows'] = round(float(s[b < 1e-6].sum()), 4)
            rb = {'attribute': name, 'source': 'bank', 'faces': '', 'row_cos': '',
                  'flatness': round(float((b.max() - b.min()) / b.mean()), 3), 'outside_bank_rows': ''}
            rb.update({f'row{i}': round(float(v), 4) for i, v in enumerate(b)})
            rows += [r, rb]
        else:
            r['outside_bank_rows'] = ''
            rows.append(r)
    keys = ['attribute', 'source', 'faces', 'row_cos', 'flatness', 'outside_bank_rows'] + [f'row{i}' for i in range(18)]
    with open(os.path.join(out, 'layer_profile.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'\n{"attribute":<11} {"source":<9} {"row cos":>7} {"flatness":>8} {"outside bank":>12}   share per row 0..17 (%)')
    for r in rows:
        print(f'{r["attribute"]:<11} {r["source"]:<9} {str(r["row_cos"]):>7} {r["flatness"]:>8} '
              f'{str(r["outside_bank_rows"]):>12}   ' + ' '.join(f'{r[f"row{i}"] * 100:4.1f}' for i in range(18)))
    print('\nrow cos: mean cosine between the edit\'s rows (1 = every row moved the same way). flatness: (max - min) '
          '/ mean of the row shares (0 = perfectly flat). outside bank: the flow edit\'s share in rows the bank does not use.')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        srcs = sorted({r['source'] for r in rows})
        fig, axes = plt.subplots(1, len(srcs), figsize=(7 * len(srcs), 0.5 * len(attrs) + 1.5), squeeze=False)
        for ax, src in zip(axes[0], srcs):
            sub = [r for r in rows if r['source'] == src]
            M = [[r[f'row{i}'] for i in range(18)] for r in sub]
            im = ax.imshow(M, aspect='auto', cmap='viridis')
            ax.set_yticks(range(len(sub)), [r['attribute'] for r in sub])
            ax.set_xticks(range(18))
            ax.set_xlabel('W+ row')
            ax.set_title(f'{src}: share of the edit per row')
            fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'layer_profile.png'), dpi=150)
    except Exception as exc:                                                # noqa: BLE001
        print(f'(no figure: {exc})')
    print(f'saved {out}/layer_profile.csv')


if __name__ == '__main__':
    main()
