"""
Inspect the LAG-DOF layer-wise attribute gate g_attr's ACTUAL per-layer
values on a trained checkpoint.

Why: the method doc claims g_attr learns which of the 18 W+ layers each
attribute edit should touch (layer-wise selectivity). Training only ever
logged the gate's smoothness regularizer (lag_gate_smooth), never the gate
values themselves -- so it has never been checked whether the gate actually
specializes per layer, or has degenerated toward a near-uniform ~1 (which
would make it functionally a global scale, not a selective gate, and would
explain why the raw flow output alone -- bank off -- edits diffusely across
the whole face instead of locally).

This calls LayerGate directly at a handful of (t, target_condition) points
with a FIXED state y = the source W+ latent, rather than intercepting the
adaptive ODE solver mid-integration. This is a simplification (the real
trajectory's y evolves during integration) but isolates exactly the
question that matters here: does the gate's CONTEXT-conditioning (time +
target condition) produce meaningfully different values across the 18
layer positions and across different attributes, or does it just output a
near-constant value everywhere.

Usage:
  python scripts/inspect_gate.py \
      --checkpoint_dir output/SDFlow/lda_v14_diag_no_age_aux \
      --step 62000 --num_faces 3
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'stylegan2'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from evaluation.evaluate_sdflow import ATTR_NAMES, _latest_step, apply_run_config, load_models
from models.dataset import SDFlowDataset
from models.flows.odefunc import LAGDOFODEnet
from scripts.render_preview import pick_balanced_faces


def find_layer_gate(prior):
    for module in prior.modules():
        if isinstance(module, LAGDOFODEnet):
            return module.layer_gate, module.mode
    return None, None


@torch.no_grad()
def main(args):
    prior, conditioner, G, id_criterion, attr_teacher, attribute_index, \
        direction_bank, control_encoder = load_models(args)

    layer_gate, mode = find_layer_gate(prior)
    if layer_gate is None:
        raise SystemExit(
            f"No LAGDOFODEnet found in this checkpoint's prior "
            f"(velocity_field={getattr(args, 'velocity_field', '?')}). Gate "
            f"inspection only applies to lag / dof / lag_dof checkpoints."
        )
    print(f"Found LAGDOFODEnet (mode={mode})\n")

    img_transform = T.Compose([
        T.ToTensor(),
        T.Resize((args.img_size, args.img_size)),
        T.Normalize(mean=0.5, std=0.5),
    ])
    dataset = SDFlowDataset(
        index_file=args.index_file, image_root=args.image_root,
        latents_file=args.latent_file, preds_file=args.preds_file,
        train=False, transform=img_transform,
    )
    face_ids = pick_balanced_faces(dataset, args.attribute_index, args.num_faces)
    print(f"faces: {face_ids}\n")

    device = 'cuda'
    t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    collected = {}  # attr_name -> list of (18,) tensors, one per (face, t)

    for idx in face_ids:
        img, latent, _pred = dataset[idx]
        img = img.unsqueeze(0).to(device)
        latent = latent.unsqueeze(0).to(device)
        _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)

        for local_idx in range(len(args.attribute_index)):
            attr_name = ATTR_NAMES.get(args.attribute_index[local_idx],
                                       str(args.attribute_index[local_idx]))
            new_attr_cond = attr_cond.clone()
            src = attr_cond[:, local_idx]
            new_attr_cond[:, local_idx] = src * (1.0 - args.scale) + (1.0 - src) * args.scale
            new_cond = torch.cat([id_cond, new_attr_cond], dim=1)

            print(f"--- face {idx}, editing {attr_name} ---")
            per_t = []
            for t_val in t_values:
                t = torch.full((1, 1), t_val, device=device, dtype=latent.dtype)
                tc = torch.cat([t, new_cond], dim=1)
                gate = layer_gate(tc, latent)          # (1, 18, 1)
                g = gate.squeeze(0).squeeze(-1).cpu()  # (18,)
                per_t.append(g)
                vals = " ".join(f"{v:.2f}" for v in g.tolist())
                print(f"  t={t_val:.2f}: {vals}   [mean={g.mean():.3f} std={g.std():.3f}]")

            avg = torch.stack(per_t).mean(dim=0)  # (18,), averaged over t for this face+attr
            collected.setdefault(attr_name, []).append(avg)
            print()

    print("=== Summary (per attribute, averaged over faces and t) ===")
    print("std_over_layers near 0 => gate is ~uniform, i.e. NOT selectively "
          "gating layers despite the method's layer-wise-gate design.\n")
    mean_vecs = {}
    for attr_name, arrs in collected.items():
        stacked = torch.stack(arrs)              # (num_faces, 18)
        mean_per_layer = stacked.mean(dim=0)      # (18,)
        mean_vecs[attr_name] = mean_per_layer
        vals = " ".join(f"{v:.2f}" for v in mean_per_layer.tolist())
        print(f"{attr_name:<12} layers: {vals}")
        print(f"{'':<12} mean={mean_per_layer.mean():.3f}  "
              f"std_over_layers={mean_per_layer.std():.3f}  "
              f"min={mean_per_layer.min():.3f}  max={mean_per_layer.max():.3f}")

    attr_names = list(mean_vecs.keys())
    if len(attr_names) >= 2:
        print("\n=== Cross-attribute gate pattern similarity ===")
        print("cos near 1.0 => different attributes produce the SAME layer "
              "pattern, i.e. the gate is not attribute-specific either.\n")
        for i in range(len(attr_names)):
            for j in range(i + 1, len(attr_names)):
                a, b = attr_names[i], attr_names[j]
                cos = F.cosine_similarity(
                    mean_vecs[a].unsqueeze(0), mean_vecs[b].unsqueeze(0)
                ).item()
                print(f"  {a} vs {b}: cos={cos:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--scale', type=float, default=1.25,
                        help='Target-condition scale used to build the (t, target_cond) '
                             'probe points. Matches the deployment scale of interest.')
    parser.add_argument('--num_faces', type=int, default=3)

    parser.add_argument('--index_file',   default='./data/ffhq.txt')
    parser.add_argument('--image_root',   default='data/FFHQ')
    parser.add_argument('--latent_file',  default='./data/ffhq_e4e_latents.pth')
    parser.add_argument('--preds_file',   default='./data/ffhq_e4e_preds.pth')
    parser.add_argument('--stygan2_weights', default='./data/stylegan2-ffhq-config-f.pt')
    parser.add_argument('--attribute_weights', default='./data/r34_a40_age_256_classifier.pth')
    parser.add_argument('--direction_bank_path', default=None)

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
    parser.add_argument('--override_residual_scale', type=float, default=None)
    parser.add_argument('--age_fine_layer_scale', type=float, default=None)
    parser.add_argument('--age_fine_layer_start', type=int, default=10)
    parser.add_argument('--ignore_run_config', action='store_true')

    args = parser.parse_args()
    args = apply_run_config(args)
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
        if args.step is None:
            raise ValueError(f'No checkpoints in {args.checkpoint_dir}/save_models/')
        print(f'Auto-detected latest step: {args.step}')
    main(args)
