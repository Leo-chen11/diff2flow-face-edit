"""Recalibrate the CLIPAttributeJudge decision boundary for one attribute.

CLIPAttributeJudge.scores() is a raw softmax(pos_prompt, neg_prompt) score.
strict_success() (evaluation/evaluate_sdflow.py) assumes 0.5 on that raw
score is the true "attribute present vs absent" boundary. For glasses this
was wrong (CLIP under-detects thin frames) and got fixed with a dedicated
pixel-level judge (GlassesParserJudge). No_Beard/Smiling/Wavy_Hair have no
such pixel judge available, so instead this script fits a per-attribute
(thresh, sharpness) remap -- sigmoid((raw - thresh) / sharpness) -- so 0.5
on the *recalibrated* score lines up with where the attribute actually is,
verified against images a human sorted by eye. Feed the fitted numbers into
--clip_calibration on evaluate_sdflow.py / dump_attr_failures.py.

Usage, two passes:

  1) Dump a spread of edited images spanning the raw-score range, for a
     human to sort by eye:

     python scripts/calibrate_clip_thresh.py dump \
         --checkpoint_dir /tmp/ema_v23 --attr 24 \
         --out_dir ./calib_attr24 --num_per_bucket 20

     This writes ./calib_attr24/unsorted/*.png. Open them and drag/move
     each file into two folders you create yourself:
         ./calib_attr24/has_attr/     (visually: prompt's POSITIVE is true)
         ./calib_attr24/no_attr/      (visually: prompt's NEGATIVE is true)
     Skip/delete ambiguous ones instead of guessing.

  2) Fit the threshold from your sorted folders:

     python scripts/calibrate_clip_thresh.py fit \
         --attr 24 --out_dir ./calib_attr24 --clip_judge_model ViT-L/14

     Prints the best thresh/sharpness and a ready-to-paste
     --clip_calibration fragment.
"""
import argparse
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils import data

from evaluation.evaluate_sdflow import (
    ATTR_NAMES, CLIPAttributeJudge, apply_run_config, edit_single_attribute,
    resolve_controlnet_disable_attrs,
    load_models,
)
from models.dataset import SDFlowDataset


def _stratified_indices(continuous_path, attr, image_list, n_per_bin, n_bins=5, seed=0):
    """DATASET indices (0..len(image_list)-1, i.e. into the already train/
    test-split-filtered SDFlowDataset) spanning the full continuous-score
    range for `attr`, evenly drawn across `n_bins` equal-width score bins,
    instead of relying on a plain shuffle to happen to cover rare classes.

    WHY BY FILENAME, NOT ROW POSITION: SDFlowDataset filters its image list
    by the index CSV's 'split' column BEFORE building self.image_list (see
    models/dataset.py), so dataset[i] is NOT row i of the continuous-preds
    file -- that file covers the whole unfiltered image list. Same reason
    models/dataset.py's own _lookup_precomputed() looks scores up by
    filename via a path->index dict instead of assuming positional
    alignment; this mirrors that pattern rather than reintroducing the bug.

    WHY STRATIFY AT ALL: a plain `shuffle=True` draw of the first few
    hundred images samples the SOURCE population's natural class balance.
    For an attribute with a rare class (e.g. Young: only ~25% of FFHQ
    scores <0.5 "old" by r34, and an even smaller ~13% by CLIP's own
    judgment -- see extract_continuous_attr.py --cross_judge clip), a small
    random draw can easily under-represent that class enough that the human
    sorting has too few real examples of it to fit a reliable threshold
    against. Binning the actual continuous score and drawing evenly from
    every bin guarantees the human gets a real spread to look at,
    independent of how imbalanced the underlying population is.
    """
    preds = torch.load(continuous_path, map_location='cpu')
    if not (isinstance(preds, dict) and 'paths' in preds and 'values' in preds):
        raise SystemExit(f'--continuous_preds_file {continuous_path} is missing paths/values '
                         '(expected the output of extract_continuous_attr.py).')
    path_to_row = {p: i for i, p in enumerate(preds['paths'])}
    missing = [f for f in image_list if f not in path_to_row]
    if missing:
        raise SystemExit(
            f'{len(missing)}/{len(image_list)} dataset images have no row in '
            f'--continuous_preds_file (e.g. {missing[0]!r}) -- it must cover this dataset\'s '
            'full image list, not just a subset.'
        )
    # dataset-local index -> that image's continuous score for `attr`.
    scores = torch.tensor([preds['values'][path_to_row[f], attr] for f in image_list])

    g = torch.Generator().manual_seed(seed)
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    indices = []
    for i in range(n_bins):
        lo, hi = bin_edges[i].item(), bin_edges[i + 1].item()
        in_bin = ((scores >= lo) & (scores < hi if i < n_bins - 1 else scores <= hi)).nonzero(as_tuple=True)[0]
        if in_bin.numel() == 0:
            print(f'  [stratify] bin [{lo:.2f},{hi:.2f}): 0 samples, skipping')
            continue
        take = min(n_per_bin, in_bin.numel())
        picked = in_bin[torch.randperm(in_bin.numel(), generator=g)[:take]]
        print(f'  [stratify] bin [{lo:.2f},{hi:.2f}): {in_bin.numel()} available, took {take}')
        indices.extend(picked.tolist())
    return indices


@torch.no_grad()
def do_dump(args):
    os.makedirs(os.path.join(args.out_dir, 'unsorted'), exist_ok=True)
    prior, conditioner, G, id_criterion, attr_teacher, \
        attribute_index, direction_bank, control_encoder = load_models(args)
    if args.attr not in args.attribute_index:
        raise SystemExit(f'attribute_index {args.attribute_index} has no attr {args.attr}.')
    local_idx = args.attribute_index.index(args.attr)
    attr_name = ATTR_NAMES.get(args.attr, f'attr{args.attr}')

    img_transform = T.Compose([
        T.ToTensor(), T.Resize((args.img_size, args.img_size)),
        T.Normalize(mean=0.5, std=0.5),
    ])
    dataset = SDFlowDataset(
        index_file=args.index_file, image_root=args.image_root,
        latents_file=args.latent_file, preds_file=args.preds_file,
        train=False, transform=img_transform,
    )
    src_target = args.num_per_bucket
    if args.stratify_src_by_score:
        print(f'Stratifying source sampling by attr {args.attr}\'s continuous score '
              f'({args.continuous_preds_file}) so rare classes are not just left to shuffle luck:')
        strat_idx = _stratified_indices(
            args.continuous_preds_file, args.attr, list(dataset.image_list),
            n_per_bin=max(args.num_per_bucket, 1), n_bins=args.stratify_bins,
        )
        dataset = data.Subset(dataset, strat_idx)
        # Save the WHOLE balanced pool, not just num_per_bucket of it -- a
        # further random truncation back down to num_per_bucket would
        # undo the balancing this just did.
        src_target = len(strat_idx)
        print(f'  -> saving all {src_target} stratified source crops (not truncated back '
              f'down to --num_per_bucket={args.num_per_bucket}).')
    loader = data.DataLoader(dataset, shuffle=True, batch_size=args.batch,
                             num_workers=4, drop_last=False)

    cj = CLIPAttributeJudge(args.attribute_index, args.clip_judge_model, 'cuda')

    scales = [0.6, 0.8, 1.0, 1.2]
    saved_src, saved_edit = 0, 0
    n = 0
    for img, latent, _pred in loader:
        img = img.cuda(); latent = latent.cuda()
        _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
        src_face = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
        src_face_256 = F.interpolate(src_face, (256, 256))
        src_scores = cj.scores(src_face_256)[:, local_idx]

        # Save some untouched source crops too -- real, unedited faces are
        # part of the score distribution the judge has to handle.
        for b in range(img.size(0)):
            if saved_src >= src_target:
                break
            _save(src_face[b], src_scores[b].item(), args.out_dir, 'src', saved_src)
            saved_src += 1

        for scale in scales:
            edited = edit_single_attribute(
                prior, conditioner, G, id_criterion, img, latent, attr_cond, id_cond,
                local_idx, scale, direction_bank, attr_global_idx=args.attr,
                control_encoder=control_encoder,
                controlnet_max_norm=getattr(args, 'controlnet_max_norm', 0.0),
                controlnet_disable_attrs=getattr(args, 'controlnet_disable_attrs', None),
            )
            edited_256 = F.interpolate(edited, (256, 256))
            edit_scores = cj.scores(edited_256)[:, local_idx]
            for b in range(img.size(0)):
                if saved_edit >= args.num_per_bucket * len(scales):
                    break
                _save(edited[b], edit_scores[b].item(), args.out_dir,
                      f'edit_s{scale}', saved_edit)
                saved_edit += 1

        n += img.size(0)
        if saved_src >= src_target and \
           saved_edit >= args.num_per_bucket * len(scales):
            break

    print(f'\nSaved {saved_src} source + {saved_edit} edited crops -> '
          f'{os.path.join(args.out_dir, "unsorted")}')
    print(f'Attribute: {attr_name}  (prompt positive = '
          f'{CLIPAttributeJudge.PROMPTS.get(args.attr, ("?", "?"))[0]!r})')
    print('Now sort each file by eye into:')
    print(f'  {args.out_dir}/has_attr/   (prompt positive is visually true)')
    print(f'  {args.out_dir}/no_attr/    (prompt negative is visually true)')
    print('Skip ambiguous ones. Then re-run with `fit`.')


def _save(img_tensor, raw_score, out_dir, tag, idx):
    x = ((img_tensor.clamp(-1, 1) + 1) * 0.5 * 255).byte().cpu()
    Image.fromarray(x.permute(1, 2, 0).numpy()).resize((256, 256)).save(
        os.path.join(out_dir, 'unsorted', f'{tag}_{idx:04d}_raw{raw_score:.3f}.png'))


@torch.no_grad()
def do_fit(args):
    cj = CLIPAttributeJudge([args.attr], args.clip_judge_model, 'cuda')
    tfm = T.Compose([T.ToTensor(), T.Resize((224, 224)), T.Normalize(mean=0.5, std=0.5)])

    def score_folder(name):
        folder = os.path.join(args.out_dir, name)
        if not os.path.isdir(folder):
            raise SystemExit(f'missing folder: {folder}')
        scores = []
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img = Image.open(os.path.join(folder, fname)).convert('RGB')
            x = tfm(img).unsqueeze(0).cuda()  # Normalize(0.5,0.5) already maps [0,1]->[-1,1]
            s = cj.scores(x)[0, 0].item()
            scores.append(s)
        return scores

    pos = score_folder('has_attr')
    neg = score_folder('no_attr')
    if not pos or not neg:
        raise SystemExit('need at least one file in both has_attr/ and no_attr/')
    print(f'has_attr: n={len(pos)}  raw score mean={sum(pos)/len(pos):.3f} '
          f'min={min(pos):.3f} max={max(pos):.3f}')
    print(f'no_attr:  n={len(neg)}  raw score mean={sum(neg)/len(neg):.3f} '
          f'min={min(neg):.3f} max={max(neg):.3f}')

    # Best threshold = value maximizing balanced accuracy (pos should score
    # above thresh, neg below). Scan midpoints between consecutive sorted
    # scores -- the optimum for a monotonic pass/fail rule always sits at
    # one of these midpoints.
    candidates = sorted(set(pos + neg))
    mids = [(candidates[i] + candidates[i + 1]) / 2 for i in range(len(candidates) - 1)]
    if not mids:
        mids = [0.5]
    best_thresh, best_acc = 0.5, -1.0
    for t in mids:
        tp = sum(1 for s in pos if s >= t)
        tn = sum(1 for s in neg if s < t)
        acc = 0.5 * (tp / len(pos) + tn / len(neg))
        if acc > best_acc:
            best_acc, best_thresh = acc, t

    # Sharpness: spread of scores right around the boundary, so the
    # recalibrated score isn't a near-step-function on real data. Falls
    # back to a small default if everything is far from the boundary.
    near = [s for s in pos + neg if abs(s - best_thresh) < 0.2]
    sharpness = max(0.05, (sum((s - best_thresh) ** 2 for s in near) / len(near)) ** 0.5
                     ) if near else 0.1

    print(f'\nBest thresh={best_thresh:.3f}  balanced accuracy={best_acc:.1%}')
    print(f'Sharpness={sharpness:.3f} (only matters if you use --success_margin > 0)')
    print(f'\nOld raw-0.5 boundary would have scored: '
          f'balanced accuracy={0.5*(sum(1 for s in pos if s>=0.5)/len(pos) + sum(1 for s in neg if s<0.5)/len(neg)):.1%}')
    print(f'\nPaste into --clip_calibration:  {args.attr}:{best_thresh:.3f}:{sharpness:.3f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='mode', required=True)

    d = sub.add_parser('dump')
    d.add_argument('--checkpoint_dir', required=True)
    d.add_argument('--step', type=int, default=None)
    d.add_argument('--attr', type=int, required=True)
    d.add_argument('--out_dir', required=True)
    d.add_argument('--num_per_bucket', type=int, default=20)
    d.add_argument('--batch', type=int, default=4)
    d.add_argument('--index_file', default='./data/ffhq.txt')
    d.add_argument('--image_root', default='data/FFHQ')
    d.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    d.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth')
    d.add_argument('--continuous_preds_file', default='./data/ffhq_e4e_preds_continuous.pth',
                    help='Output of extract_continuous_attr.py, used by --stratify_src_by_score '
                         'to bin the SOURCE sampling by this attribute\'s actual score instead '
                         'of leaving rare-class coverage to shuffle luck.')
    d.add_argument('--stratify_src_by_score',
                    action=argparse.BooleanOptionalAction, default=True,
                    help='Bin source-face sampling by --continuous_preds_file\'s score for '
                         '--attr into --stratify_bins equal-width bins and draw evenly from '
                         'each, instead of a plain shuffle. Default on: a plain shuffle draw '
                         'undersamples any attribute whose classes are imbalanced in the real '
                         'population (confirmed for Young: ~25%% "old" by r34, ~13%% by CLIP\'s '
                         'own judgment), leaving too few real examples of the rare class for a '
                         'reliable fit. --no-stratify_src_by_score reproduces the old plain-'
                         'shuffle behavior.')
    d.add_argument('--stratify_bins', type=int, default=5,
                    help='Number of equal-width score bins for --stratify_src_by_score.')
    d.add_argument('--stygan2_weights', default='./data/stylegan2-ffhq-config-f.pt')
    d.add_argument('--attribute_weights', default='./data/r34_a40_age_256_classifier.pth')
    d.add_argument('--direction_bank_path', default=None)
    d.add_argument('--img_size', type=int, default=512)
    d.add_argument('--attribute_index', nargs='*', type=int, default=[15, 20, 39])
    d.add_argument('--flow_modules', default='512-512-512-512-512')
    d.add_argument('--num_blocks', type=int, default=1)
    d.add_argument('--velocity_field', default='lag_dof')
    d.add_argument('--id_cond_dim', type=int, default=32)
    d.add_argument('--id_cond_scale', type=float, default=0.25)
    d.add_argument('--attr_backbone', default='resnet50')
    d.add_argument('--conditioner_backbone', default='resnet')
    d.add_argument('--clip_model', default='ViT-B/32')
    d.add_argument('--clip_judge_model', default='ViT-L/14')
    d.add_argument('--fused_hidden_dim', type=int, default=256)
    d.add_argument('--lag_gate_hidden_dim', type=int, default=64)
    d.add_argument('--lag_gate_init_bias', type=float, default=-0.5)
    d.add_argument('--direction_residual_scale', type=float, default=0.05)
    d.add_argument('--glasses_residual_scale', type=float, default=0.05)
    d.add_argument('--bypass_glasses_direction_bank', action='store_true')
    d.add_argument('--guided_delta_max_norm', type=float, default=0.0)
    d.add_argument('--override_residual_scale', type=float, default=None)
    d.add_argument('--age_fine_layer_scale', type=float, default=None)
    d.add_argument('--age_fine_layer_start', type=int, default=10)
    d.add_argument('--force_bank_directions', action='store_true')
    d.add_argument('--controlnet_disable_attrs', nargs='*', type=int, default=None,
                    help='Same knob/default as evaluate_sdflow.py: omitted -> auto-resolved '
                         'to gender/age (20, 39) via resolve_controlnet_disable_attrs().')
    d.add_argument('--ignore_run_config', action='store_true')

    f = sub.add_parser('fit')
    f.add_argument('--attr', type=int, required=True)
    f.add_argument('--out_dir', required=True)
    f.add_argument('--clip_judge_model', default='ViT-L/14')

    args = p.parse_args()
    if args.mode == 'dump':
        args = apply_run_config(args)
        args = resolve_controlnet_disable_attrs(args)
        do_dump(args)
    else:
        do_fit(args)
