"""
Render an attribute-edit preview grid from ANY checkpoint at ANY edit scale.

Purpose: training previews are rendered with whatever --preview_scale the run
used, so grids from different runs are often not comparable (v10 rendered at
0.5, v12 at 1.25). This script renders the SAME faces at the SAME scale from
any checkpoint (live or EMA), so visual A/B comparisons between runs are fair.

Usage:
  python scripts/render_preview.py \
      --checkpoint_dir /tmp/ema_v10 \
      --direction_bank_path ./data/direction_bank_k4_age4_stratified.pth \
      --scale 1.25 --step 79000

Output grid: one row per face, columns = [reconstruction, one edit per
attribute]. Reads config.json from the checkpoint dir (same auto-alignment as
evaluate_sdflow.py).
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'stylegan2'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from evaluation.evaluate_sdflow import (
    ATTR_NAMES,
    _latest_step,
    apply_run_config,
    edit_single_attribute,
    load_models,
)
from models.dataset import SDFlowDataset


def _lookup_pred(dataset, index):
    file = dataset.image_list[index]
    return dataset._lookup_precomputed(dataset.preds, file)


def pick_balanced_faces(dataset, attribute_index, num_faces):
    """One low-score and one high-score source per attribute, then fill."""
    attr_ids = [int(i) for i in attribute_index]
    scores = torch.stack(
        [_lookup_pred(dataset, i)[attr_ids].float() for i in range(len(dataset))]
    )
    selected = []

    def add(idx):
        if idx not in selected and len(selected) < num_faces:
            selected.append(idx)

    for local in range(len(attr_ids)):
        add(int(scores[:, local].argmin()))
        add(int(scores[:, local].argmax()))
    for idx in range(len(dataset)):
        if len(selected) >= num_faces:
            break
        add(idx)
    return selected


@torch.no_grad()
def main(args):
    prior, conditioner, G, id_criterion, attr_teacher, attribute_index, \
        direction_bank = load_models(args)

    img_transform = T.Compose([
        T.ToTensor(),
        T.Resize((args.img_size, args.img_size)),
        T.Normalize(mean=0.5, std=0.5),
    ])
    dataset = SDFlowDataset(
        index_file=args.index_file,
        image_root=args.image_root,
        latents_file=args.latent_file,
        preds_file=args.preds_file,
        train=False,
        transform=img_transform,
    )
    face_ids = pick_balanced_faces(dataset, args.attribute_index, args.num_faces)
    print(f'faces: {face_ids}')

    rows = []
    for idx in face_ids:
        img, latent, _pred = dataset[idx]
        img = img.unsqueeze(0).cuda()
        latent = latent.unsqueeze(0).cuda()
        _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)

        recon = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
        cells = [F.interpolate(recon, (args.cell_size, args.cell_size))]

        for local_idx in range(len(args.attribute_index)):
            edited = edit_single_attribute(
                prior, conditioner, G, id_criterion,
                img, latent, attr_cond, id_cond,
                local_idx, args.scale, direction_bank,
                attr_global_idx=args.attribute_index[local_idx],
                bypass_glasses_direction_bank=args.bypass_glasses_direction_bank,
            )
            cells.append(F.interpolate(edited, (args.cell_size, args.cell_size)))

        rows.append(torch.cat(cells, dim=3))  # concat horizontally

    grid = torch.cat(rows, dim=2)             # stack rows vertically
    grid = (grid.squeeze(0) + 1.0) * 0.5

    out = args.out
    if out is None:
        names = '-'.join(ATTR_NAMES.get(i, str(i)) for i in args.attribute_index)
        out = os.path.join(
            args.checkpoint_dir,
            f'preview_step{args.step}_scale{args.scale}_{names}.png',
        )
    torchvision.utils.save_image(grid, out)
    print(f'saved -> {out}')
    print(f'columns: [recon] + edits at scale {args.scale} for '
          f'{[ATTR_NAMES.get(i, i) for i in args.attribute_index]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--scale', type=float, default=1.25)
    parser.add_argument('--num_faces', type=int, default=8)
    parser.add_argument('--cell_size', type=int, default=256)
    parser.add_argument('--out', default=None)

    # Data (same defaults as evaluate_sdflow.py)
    parser.add_argument('--index_file',   default='./data/ffhq.txt')
    parser.add_argument('--image_root',   default='data/FFHQ')
    parser.add_argument('--latent_file',  default='./data/ffhq_e4e_latents.pth')
    parser.add_argument('--preds_file',   default='./data/ffhq_e4e_preds.pth')
    parser.add_argument('--stygan2_weights', default='./data/stylegan2-ffhq-config-f.pt')
    parser.add_argument('--attribute_weights', default='./data/r34_a40_age_256_classifier.pth')
    parser.add_argument('--direction_bank_path', default=None)

    # Model structure (auto-aligned from config.json when present)
    parser.add_argument('--img_size',         type=int,   default=512)
    parser.add_argument('--attribute_index',  nargs='*',  type=int,   default=[15, 20, 39])
    parser.add_argument('--flow_modules',     default='512-512-512-512-512')
    parser.add_argument('--num_blocks',       type=int,   default=1)
    parser.add_argument('--velocity_field',   default='lag_dof')
    parser.add_argument('--id_cond_dim',      type=int,   default=32)
    parser.add_argument('--id_cond_scale',    type=float, default=0.25)
    parser.add_argument('--attr_backbone',    default='resnet50')
    parser.add_argument('--conditioner_backbone', default='resnet',
                        choices=['resnet', 'clip', 'resnet_clip'])
    parser.add_argument('--clip_model', default='ViT-B/32')
    parser.add_argument('--fused_hidden_dim', type=int, default=256)
    parser.add_argument('--lag_gate_hidden_dim', type=int,   default=64)
    parser.add_argument('--lag_gate_init_bias',  type=float, default=-0.5)
    parser.add_argument('--direction_residual_scale', type=float, default=0.05)
    parser.add_argument('--glasses_residual_scale',   type=float, default=0.05)
    parser.add_argument('--bypass_glasses_direction_bank',
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--guided_delta_max_norm', type=float, default=0.0)
    parser.add_argument('--ignore_run_config', action='store_true')

    args = parser.parse_args()
    args = apply_run_config(args)
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
        if args.step is None:
            raise ValueError(f'No checkpoints in {args.checkpoint_dir}/save_models/')
        print(f'Auto-detected latest step: {args.step}')
    main(args)
