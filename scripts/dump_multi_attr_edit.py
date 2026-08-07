"""Visual test for editing 2-3 attributes on the same face at once, using
evaluate_sdflow.edit_multi_attribute() (sums independently-calibrated
single-attribute deltas -- see that function's docstring for why).

Usage:
    python scripts/dump_multi_attr_edit.py \
        --checkpoint_dir /tmp/ema_v21 \
        --attrs 15 20 39 --edit_scale 1.0 \
        --num_samples 16 --out_dir ./multi_attr_test
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torch.utils import data

from evaluation.evaluate_sdflow import (
    ATTR_NAMES, CLIPAttributeJudge, CelebAAttrClassifierJudge, _latest_step,
    apply_run_config, edit_multi_attribute, load_models, parse_clip_calibration,
)
from models.dataset import SDFlowDataset


def to_pil(img_tensor, size=256):
    x = F.interpolate(img_tensor.unsqueeze(0), (size, size))[0]
    x = ((x.clamp(-1, 1) + 1) * 0.5 * 255).byte().cpu()
    return Image.fromarray(x.permute(1, 2, 0).numpy())


def build_score_fn(args, attrs):
    """Prefer CelebAAttrClassifierJudge (see evaluate_sdflow.py -- doesn't need
    per-attribute prompt/threshold tuning, more reliable than CLIP zero-shot,
    confirmed empirically for beard/hair on this project). Falls back to CLIP,
    or None (no score overlay) if neither is configured/available."""
    if getattr(args, 'celeba_attr_judge_weights', None):
        judge = CelebAAttrClassifierJudge(args.celeba_attr_judge_weights, 'cuda')
        print('[Judge] scoring with CelebAAttrClassifierJudge')
        return lambda imgs: judge.scores(imgs)[:, attrs], 'Celeb'
    try:
        judge = CLIPAttributeJudge(attrs, args.clip_judge_model, 'cuda',
                                   calibration=parse_clip_calibration(
                                       getattr(args, 'clip_calibration', None)))
        print('[Judge] scoring with CLIPAttributeJudge (no --celeba_attr_judge_weights given)')
        return lambda imgs: judge.scores(imgs), 'CLIP'
    except ImportError:
        print('[WARN] no judge available -- pass --celeba_attr_judge_weights or install CLIP. '
              'Score overlay disabled.')
        return None, None


@torch.no_grad()
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    prior, conditioner, G, id_criterion, attr_teacher, \
        attribute_index, direction_bank, control_encoder = load_models(args)

    local_idxs = []
    for a in args.attrs:
        if a not in args.attribute_index:
            raise SystemExit(f'attribute_index {args.attribute_index} has no attr {a}.')
        local_idxs.append(args.attribute_index.index(a))

    img_transform = T.Compose([
        T.ToTensor(), T.Resize((args.img_size, args.img_size)),
        T.Normalize(mean=0.5, std=0.5),
    ])
    dataset = SDFlowDataset(
        index_file=args.index_file, image_root=args.image_root,
        latents_file=args.latent_file, preds_file=args.preds_file,
        train=False, transform=img_transform,
    )
    loader = data.DataLoader(dataset, shuffle=True, batch_size=args.batch,
                             num_workers=4, drop_last=False)

    attr_names = [ATTR_NAMES.get(a, f'attr{a}') for a in args.attrs]
    tag = '+'.join(attr_names)

    watch_attrs = list(args.watch_attrs or [])
    all_scored_attrs = args.attrs + [a for a in watch_attrs if a not in args.attrs]
    all_scored_names = [ATTR_NAMES.get(a, f'attr{a}') for a in all_scored_attrs]
    is_watched = [a in watch_attrs and a not in args.attrs for a in all_scored_attrs]

    score_fn, judge_name = build_score_fn(args, all_scored_attrs)
    n_score_lines = len(all_scored_attrs) if score_fn is not None else 0
    header_h = 24 + 14 * n_score_lines

    saved, pairs = 0, []
    for img, latent, _pred in loader:
        img = img.cuda(); latent = latent.cuda()
        _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)

        src_face = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)

        # Direction per attribute: 'add' pushes toward the source's OWN
        # opposite polarity per sample (same convention as edit_single_attribute
        # / generate_test_image), 'rm' the same expression just kept for
        # readability -- both directions use the identical formula, this
        # script doesn't gate by source value like dump_attr_failures.py does.
        edited = edit_multi_attribute(
            prior, conditioner, G, id_criterion, img, latent, attr_cond, id_cond,
            local_idxs, args.edit_scale, direction_bank,
            control_encoder=control_encoder,
            controlnet_max_norm=getattr(args, 'controlnet_max_norm', 0.0),
        )

        src_face_256 = F.interpolate(src_face, (256, 256))
        edited_256 = F.interpolate(edited, (256, 256))
        if score_fn is not None:
            src_scores = score_fn(src_face_256)
            edit_scores = score_fn(edited_256)

        for b in range(img.size(0)):
            if saved >= args.num_samples:
                break
            canvas = Image.new('RGB', (256 * 2, 256 + header_h), (20, 20, 20))
            canvas.paste(to_pil(src_face[b]), (0, header_h))
            canvas.paste(to_pil(edited[b]), (256, header_h))
            if score_fn is not None:
                d = ImageDraw.Draw(canvas)
                for i, (name, watched) in enumerate(zip(all_scored_names, is_watched)):
                    s = src_scores[b, i].item()
                    e = edit_scores[b, i].item()
                    y = 4 + i * 14
                    label = f'[watch]{name}' if watched else name
                    if watched:
                        # No "correct direction" for a bystander attribute --
                        # any large |e-s| is leakage, flag it regardless of sign.
                        color = (240, 180, 60) if abs(e - s) > 0.15 else (150, 150, 150)
                    else:
                        moved = (e > s) if s < 0.5 else (e < s)
                        color = (80, 220, 80) if moved else (240, 90, 90)
                    d.text((4, y), f'{label} src={s:.2f}', fill=(180, 180, 180))
                    d.text((260, y), f'{label} edit={e:.2f}', fill=color)
            pairs.append(canvas)
            saved += 1
        if saved >= args.num_samples:
            break

    cols = 4
    rows = (len(pairs) + cols - 1) // cols
    w, h = pairs[0].size
    grid = Image.new('RGB', (w * cols, h * rows), (0, 0, 0))
    for i, p in enumerate(pairs):
        grid.paste(p, ((i % cols) * w, (i // cols) * h))
    out_path = os.path.join(args.out_dir, f'multi_{tag}.png')
    grid.save(out_path)
    print(f'Editing: {attr_names} simultaneously, scale={args.edit_scale}'
          + (f' (scored by {judge_name})' if judge_name else ''))
    print(f'Saved {len(pairs)} pairs -> {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--step', type=int, default=None)
    p.add_argument('--attrs', nargs='+', type=int, required=True,
                    help='2-3 absolute CelebA attribute indices to edit at once.')
    p.add_argument('--watch_attrs', nargs='*', type=int, default=None,
                    help='Extra attributes to SCORE but not edit, to check for leakage '
                         'into attributes you deliberately left out of --attrs.')
    p.add_argument('--edit_scale', type=float, default=1.0)
    p.add_argument('--num_samples', type=int, default=16)
    p.add_argument('--out_dir', default='./multi_attr_test')
    p.add_argument('--batch', type=int, default=4)

    p.add_argument('--index_file', default='./data/ffhq.txt')
    p.add_argument('--image_root', default='data/FFHQ')
    p.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    p.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth')
    p.add_argument('--stygan2_weights', default='./data/stylegan2-ffhq-config-f.pt')
    p.add_argument('--attribute_weights', default='./data/r34_a40_age_256_classifier.pth')
    p.add_argument('--direction_bank_path', default=None)
    p.add_argument('--img_size', type=int, default=512)
    p.add_argument('--attribute_index', nargs='*', type=int, default=[15, 20, 39])
    p.add_argument('--flow_modules', default='512-512-512-512-512')
    p.add_argument('--num_blocks', type=int, default=1)
    p.add_argument('--velocity_field', default='lag_dof')
    p.add_argument('--id_cond_dim', type=int, default=32)
    p.add_argument('--id_cond_scale', type=float, default=0.25)
    p.add_argument('--attr_backbone', default='resnet50')
    p.add_argument('--conditioner_backbone', default='resnet')
    p.add_argument('--clip_model', default='ViT-B/32')
    p.add_argument('--fused_hidden_dim', type=int, default=256)
    p.add_argument('--lag_gate_hidden_dim', type=int, default=64)
    p.add_argument('--lag_gate_init_bias', type=float, default=-0.5)
    p.add_argument('--direction_residual_scale', type=float, default=0.05)
    p.add_argument('--glasses_residual_scale', type=float, default=0.05)
    p.add_argument('--bypass_glasses_direction_bank', action='store_true')
    p.add_argument('--guided_delta_max_norm', type=float, default=0.0)
    p.add_argument('--override_residual_scale', type=float, default=None)
    p.add_argument('--age_fine_layer_scale', type=float, default=None)
    p.add_argument('--age_fine_layer_start', type=int, default=4)
    p.add_argument('--force_bank_directions', action='store_true')
    p.add_argument('--ignore_run_config', action='store_true')
    p.add_argument('--celeba_attr_judge_weights', default=None,
                    help='Preferred score source (see evaluate_sdflow.py CelebAAttrClassifierJudge). '
                         'Falls back to CLIP if omitted.')
    p.add_argument('--clip_judge_model', default='ViT-L/14')
    p.add_argument('--clip_calibration', default=None)

    args = p.parse_args()
    args = apply_run_config(args)
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
        if args.step is None:
            raise ValueError(f'No checkpoints in {args.checkpoint_dir}/save_models/')
        print(f'Auto-detected latest step: {args.step}')
    main(args)
