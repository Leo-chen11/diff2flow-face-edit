"""
Sample-efficiency diagnostic: how many high/low samples does an attribute's
direction actually need before AccCLIP stops improving?

--min_samples / --extreme_pct in precompute_directions_stratified.py have
always been guessed, then adjusted after the fact when validate_direction_
bank.py showed a weak attribute (eyeglasses reaching only ~1-12% AccCLIP
even at alpha=1.5). This script answers the question directly instead of
guessing again: for a range of sample counts N, recompute the SAME
compute_direction() this project already uses with only N high + N low
examples, apply it to a fixed validation batch, and record AccCLIP as a
function of N.

How to read the curve:
  - Still climbing at the largest N tried (or at the full available pool):
    the direction is genuinely DATA-STARVED. More real samples, or a cheap
    proxy for them (horizontal-flip augmentation -- see the identity-
    preserving image-space augmentation discussion for this project), would
    plausibly help. This is the "should I do data augmentation" question,
    answered empirically instead of by intuition.
  - Flat (or noisy-flat) well before the pool is exhausted: sample count is
    NOT the bottleneck. Throwing more data at it will not fix a direction
    that has already plateaued at N=100-200 -- the ceiling is a geometry/
    architecture problem (wrong latent space, needs whitening/decorrelation,
    needs a ControlNet-style spatial path for a local attribute, etc.),
    exactly what this project's --substyle_k and orthogonalization work
    already investigates.

This never trains anything and touches no loss function -- it just calls
the existing compute_direction() repeatedly at different N and measures the
result with the same CLIP judge validate_direction_bank.py uses, on the
same fixed validation batch every N/trial so results are comparable.

Usage:
  python scripts/sample_efficiency_curve.py \\
      --latent_file ./data/ffhq_e4e_latents.pth \\
      --continuous_preds_file ./data/ffhq_e4e_preds_continuous.pth \\
      --attribute_index 15 20 39 \\
      --sample_sizes 20 50 100 200 500 1000 \\
      --trials 3 --alpha 1.0 --num_samples 100
"""
import argparse
import os
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'stylegan2'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils import data
from tqdm import tqdm
import numpy as np

from evaluation.evaluate_sdflow import (
    ATTR_NAMES, CLIPAttributeJudge, GlassesParserJudge, is_clear, strict_success,
    parse_clip_calibration,
)
from models.dataset import SDFlowDataset
from models.stylegan2.model import Generator
from scripts.precompute_directions_stratified import (
    load_latents, load_preds, extreme_masks, compute_direction,
)


@torch.no_grad()
def score_direction(direction, attr_local_idx, alpha, imgs, latents, src_scores,
                     G, clip_judge, glasses_parser, attr_global_idx, batch_size):
    """Apply `direction` (18, 512 raw displacement) to every latent in the
    fixed validation set at the given alpha, score with clip_judge (and
    glasses_parser for attr 15 if given). Returns mean strict_success over
    samples whose SOURCE score was clearly on one side (same is_clear()
    gating validate_direction_bank.py uses).
    """
    direction = direction.to(latents.device)
    N = latents.shape[0]
    hits = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        lat = latents[start:end]
        s = src_scores[start:end, attr_local_idx]
        sign = torch.where(s < 0.5, torch.ones_like(s), -torch.ones_like(s))
        new_latent = lat + (sign.view(-1, 1, 1) * alpha) * direction.unsqueeze(0)
        edited = G([new_latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
        edited_256 = F.interpolate(edited, (256, 256))
        edit_scores = clip_judge.scores(edited_256)[:, attr_local_idx]
        if glasses_parser is not None and attr_global_idx == 15:
            edit_scores = glasses_parser.glasses_prob(edited_256)
        for b in range(end - start):
            sb = s[b].item()
            if not is_clear(sb):
                continue
            hits.append(float(strict_success(sb, edit_scores[b].item(), 0.0)))
    return float(np.mean(hits)) if hits else float('nan')


@torch.no_grad()
def main(args):
    device = 'cuda'
    print("Loading latents/continuous scores ...")
    latents_all = load_latents(args.latent_file)
    continuous = load_preds(args.continuous_preds_file)
    print(f"  latents={tuple(latents_all.shape)} continuous={tuple(continuous.shape)}")

    ckpt = torch.load(args.stygan2_weights, map_location='cpu')
    G = Generator(size=1024, style_dim=512, n_mlp=8)
    G.load_state_dict(ckpt['g_ema'])
    G.to(device).eval()
    for p in G.parameters():
        p.requires_grad_(False)

    clip_judge = CLIPAttributeJudge(args.attribute_index, args.clip_judge_model, device,
                                     calibration=parse_clip_calibration(args.clip_calibration))
    glasses_parser = None
    if args.glasses_judge == 'parser' and 15 in args.attribute_index:
        glasses_parser = GlassesParserJudge(args.face_parser_weights, device,
                                            area_thresh=args.glasses_area_thresh)
        print('[Judge] Eyeglasses scored by BiSeNet parser (matches validate_direction_bank.py default).')

    # Fixed validation batch, loaded once -- every N/trial is scored against
    # the SAME images so AccCLIP differences are attributable to the
    # direction, not to which images happened to be sampled.
    img_transform = T.Compose([
        T.ToTensor(), T.Resize((args.img_size, args.img_size)),
        T.Normalize(mean=0.5, std=0.5),
    ])
    dataset = SDFlowDataset(
        index_file=args.index_file, image_root=args.image_root,
        latents_file=args.latent_file, preds_file=args.preds_file,
        train=False, transform=img_transform,
    )
    loader = data.DataLoader(dataset, shuffle=False, batch_size=args.batch, num_workers=4)
    val_imgs, val_latents = [], []
    seen = 0
    for img, latent, _pred in loader:
        if seen >= args.num_samples:
            break
        val_imgs.append(img)
        val_latents.append(latent)
        seen += img.size(0)
    val_imgs = torch.cat(val_imgs)[:args.num_samples].to(device)
    val_latents = torch.cat(val_latents)[:args.num_samples].to(device)
    print(f"Validation batch: {val_latents.shape[0]} samples (fixed across every N/trial below)")

    local_idx = {a: i for i, a in enumerate(args.attribute_index)}
    src_256 = F.interpolate(val_imgs, (256, 256))
    src_scores = clip_judge.scores(src_256)
    if glasses_parser is not None and 15 in local_idx:
        src_scores[:, local_idx[15]] = glasses_parser.glasses_prob(src_256)

    results = {}
    for attr in args.attribute_index:
        name = ATTR_NAMES.get(attr, f'attr{attr}')
        print(f"\n{'='*70}\nAttr {attr} ({name})\n{'='*70}")
        cont_scores = continuous[:, attr]
        mask_high, mask_low = extreme_masks(cont_scores, pct=args.extreme_pct)
        high_idx = mask_high.nonzero(as_tuple=True)[0]
        low_idx = mask_low.nonzero(as_tuple=True)[0]
        pool_n = min(high_idx.numel(), low_idx.numel())
        print(f"  extreme_pct={args.extreme_pct}% pool: high={high_idx.numel()} low={low_idx.numel()}")

        row = []
        for N in args.sample_sizes:
            if N > pool_n:
                print(f"  N={N:>5}: SKIPPED (pool only has {pool_n} on the smaller side)")
                row.append((N, float('nan'), 0))
                continue
            accs = []
            n_trials_ok = 0
            for trial in range(args.trials):
                g = torch.Generator().manual_seed(trial)
                sub_high = high_idx[torch.randperm(high_idx.numel(), generator=g)[:N]]
                sub_low = low_idx[torch.randperm(low_idx.numel(), generator=g)[:N]]
                mh = torch.zeros(latents_all.shape[0], dtype=torch.bool)
                ml = torch.zeros(latents_all.shape[0], dtype=torch.bool)
                mh[sub_high] = True
                ml[sub_low] = True
                direction, nh, nl = compute_direction(
                    latents_all, mh, ml, min_samples=min(N, args.min_samples_floor),
                    method=args.direction_method, shrinkage=args.shrinkage,
                )
                if direction is None:
                    continue
                acc = score_direction(direction, local_idx[attr], args.alpha, val_imgs, val_latents,
                                      src_scores, G, clip_judge, glasses_parser, attr, args.score_batch)
                if not np.isnan(acc):
                    accs.append(acc)
                    n_trials_ok += 1
            mean_acc = float(np.mean(accs)) if accs else float('nan')
            std_acc = float(np.std(accs)) if len(accs) > 1 else 0.0
            row.append((N, mean_acc, n_trials_ok))
            print(f"  N={N:>5}: AccCLIP={mean_acc*100:5.1f}% (+/-{std_acc*100:4.1f}%, "
                  f"{n_trials_ok}/{args.trials} trials produced a valid direction)")
        results[attr] = row

    print(f"\n{'='*70}\nSummary (AccCLIP % vs N, alpha={args.alpha})\n{'='*70}")
    header = "N".rjust(8) + "".join(
        f"  {ATTR_NAMES.get(a, f'attr{a}'):>12}" for a in args.attribute_index)
    print(header)
    for i, N in enumerate(args.sample_sizes):
        line = str(N).rjust(8)
        for a in args.attribute_index:
            _, acc, _ = results[a][i]
            line += f"  {acc*100:11.1f}%" if not np.isnan(acc) else f"  {'n/a':>12}"
        print(line)
    print("\nRead this as: still rising at the largest N (or pool-limited before it plateaus) means "
          "the direction is genuinely data-starved -- more samples/augmentation would plausibly help. "
          "Flat well before the pool runs out means N is not the bottleneck -- more data will not fix "
          "this attribute's direction, the geometry/architecture needs work instead "
          "(--substyle_k, decorrelation, or a ControlNet-style local path).")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--attribute_index', nargs='*', type=int, default=[15, 20, 39])
    p.add_argument('--sample_sizes', nargs='*', type=int, default=[20, 50, 100, 200, 500, 1000])
    p.add_argument('--trials', type=int, default=3,
                   help='Repeat each N this many times with different random subsamples, average '
                        'AccCLIP over them -- otherwise a single unlucky/lucky subsample at small N '
                        'looks like signal instead of noise.')
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--extreme_pct', type=float, default=20.0,
                   help='Same meaning as in precompute_directions_stratified.py: the high/low pool '
                        'this sweep subsamples N from is the top/bottom this-percent of the '
                        'continuous score, not the whole dataset.')
    p.add_argument('--direction_method', choices=['lda', 'mean_diff'], default='lda')
    p.add_argument('--shrinkage', type=float, default=None)
    p.add_argument('--min_samples_floor', type=int, default=8,
                   help='Passed as compute_direction\'s own min_samples floor, capped at min(N, this) '
                        'so a deliberately small N is not itself rejected by that check.')
    p.add_argument('--num_samples', type=int, default=100,
                   help='Size of the FIXED validation batch every N/trial is scored against. Kept '
                        'smaller than validate_direction_bank.py\'s default (200) because this script '
                        'runs the generator len(attribute_index)*len(sample_sizes)*trials times.')
    p.add_argument('--batch', type=int, default=8, help='Dataloader batch size when building the '
                   'fixed validation set.')
    p.add_argument('--score_batch', type=int, default=8, help='Batch size when running G/judges over '
                   'the fixed validation set for one (attr, N, trial) direction.')
    p.add_argument('--img_size', type=int, default=512)

    p.add_argument('--index_file', default='./data/ffhq.txt')
    p.add_argument('--image_root', default='data/FFHQ')
    p.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    p.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth')
    p.add_argument('--continuous_preds_file', default='./data/ffhq_e4e_preds_continuous.pth')
    p.add_argument('--stygan2_weights', default='./data/stylegan2-ffhq-config-f.pt')

    p.add_argument('--clip_judge_model', default='ViT-L/14')
    p.add_argument('--clip_calibration', default=None)
    p.add_argument('--glasses_judge', default='parser', choices=['clip', 'parser'])
    p.add_argument('--face_parser_weights', default='./data/parsing_bisenet.pth')
    p.add_argument('--glasses_area_thresh', type=float, default=0.0010)

    args = p.parse_args()
    main(args)
