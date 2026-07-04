"""
SDFlow Evaluation Script

Computes:
  1. Identity Preservation (ArcFace cosine similarity)
  2. Editing Accuracy (attribute classifier score change)
  3. Attribute Preservation (non-target attributes shouldn't change)

Usage:
  python evaluation/evaluate_sdflow.py \
      --checkpoint_dir ./output/SDFlow/v13_stratified_k4 \
      --step 20000 \
      --num_samples 500 \
      --eval_scales 0.80 0.85 0.90 0.95
"""

import argparse
import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'stylegan2'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils import data
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from common.id_loss import IDLoss
from common.ops import load_network
from models.dataset import SDFlowDataset
from models.flows.flow import cnf
from models.attribute_estimator import AttributeClassifier
from models.conditioner import IdentityAttributeConditioner
from models.stylegan2.model import Generator


ATTR_NAMES = {15: 'Eyeglasses', 20: 'Male', 39: 'Young'}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_path(checkpoint_dir, module_name, step):
    """Return path like save_models/prior-0017000 (no .pth extension)."""
    return os.path.join(checkpoint_dir, 'save_models',
                        f'{module_name}-{str(step).zfill(7)}')


def _latest_step(checkpoint_dir, module_name='prior'):
    """Auto-detect the highest saved step for a given module."""
    d = os.path.join(checkpoint_dir, 'save_models')
    if not os.path.isdir(d):
        return None
    steps = []
    for f in os.listdir(d):
        if f.startswith(f'{module_name}-'):
            try:
                steps.append(int(f.split('-')[1]))
            except ValueError:
                pass
    return max(steps) if steps else None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(args):
    device = 'cuda'
    attribute_index = torch.tensor(args.attribute_index, dtype=torch.long)
    num_attrs = len(args.attribute_index)
    condition_dim = args.id_cond_dim + num_attrs

    # ── Flow ──────────────────────────────────────────────────────────────
    prior = cnf(
        512, args.flow_modules, condition_dim, args.num_blocks,
        velocity_field=args.velocity_field, num_layers=18,
        gate_hidden_dim=args.lag_gate_hidden_dim,
        gate_init_bias=args.lag_gate_init_bias,
        attr_context_dim=num_attrs,
        train_T=False,
    ).to(device).eval()

    prior_ckpt = _ckpt_path(args.checkpoint_dir, 'prior', args.step)
    print(f'Loading prior  ← {prior_ckpt}')
    prior.load_state_dict(load_network(prior_ckpt), strict=False)

    # ── Conditioner ───────────────────────────────────────────────────────
    conditioner = IdentityAttributeConditioner(
        attr_dim=num_attrs,
        id_dim=args.id_cond_dim,
        id_scale=args.id_cond_scale,
        attr_backbone=args.attr_backbone,
        conditioner_backbone=args.conditioner_backbone,
        clip_model=args.clip_model,
        fused_hidden_dim=args.fused_hidden_dim,
    ).to(device).eval()

    cond_ckpt = _ckpt_path(args.checkpoint_dir, 'conditioner', args.step)
    print(f'Loading cond   ← {cond_ckpt}')
    conditioner.load_state_dict(load_network(cond_ckpt), strict=False)

    # ── Direction Bank (optional, loads trained weights if saved) ─────────
    direction_bank = None
    db_ckpt_path = _ckpt_path(args.checkpoint_dir, 'direction_bank', args.step)
    bank_path = args.direction_bank_path
    if bank_path:
        from models.direction_bank import AttributeDirectionBank
        bank_meta = torch.load(bank_path, map_location='cpu')
        num_k = int(bank_meta.get('num_k', bank_meta.get('K', 1))) \
            if isinstance(bank_meta, dict) else 1
        _per_attr_rs = [
            args.glasses_residual_scale if idx == 15 else args.direction_residual_scale
            for idx in args.attribute_index
        ]
        # No per-attribute direction_scale/layer_scale/delta_max_norm: every
        # attribute is treated the same, and the only magnitude safety net is
        # the shared guided_delta_max_norm below.
        direction_bank = AttributeDirectionBank(
            num_attrs=num_attrs,
            num_layers=18,
            latent_dim=512,
            num_k=num_k,
            bank_path=bank_path,
            attribute_index=args.attribute_index,
            residual_scale=args.direction_residual_scale,
            per_attr_residual_scale=_per_attr_rs,
            freeze_directions=True,
            guided_delta_max_norm=(
                args.guided_delta_max_norm if args.guided_delta_max_norm > 0 else None
            ),
        ).to(device).eval()
        if os.path.exists(db_ckpt_path):
            result = direction_bank.load_state_dict(load_network(db_ckpt_path), strict=False)
            if result.missing_keys:
                print(f'[WARN] Direction bank missing keys: {result.missing_keys[:8]}')
            if result.unexpected_keys:
                print(f'[WARN] Direction bank unexpected keys: {result.unexpected_keys[:8]}')
            print(f'Loading bank   ← {db_ckpt_path}')
        else:
            print(f'Direction bank init (no trained weights at {db_ckpt_path})')

    # ── StyleGAN2 ─────────────────────────────────────────────────────────
    ckpt = torch.load(args.stygan2_weights, map_location='cpu')
    G = Generator(size=1024, style_dim=512, n_mlp=8)
    G.load_state_dict(ckpt['g_ema'])
    G.to(device).eval()
    for p in G.parameters():
        p.requires_grad_(False)

    # ── IDLoss ────────────────────────────────────────────────────────────
    id_criterion = IDLoss(crop=True).to(device).eval()
    for p in id_criterion.parameters():
        p.requires_grad_(False)

    # ── Attribute teacher ─────────────────────────────────────────────────
    attr_teacher = AttributeClassifier(backbone='r34')
    attr_teacher.load_state_dict(load_network(args.attribute_weights))
    attr_teacher.to(device).eval()
    for p in attr_teacher.parameters():
        p.requires_grad_(False)

    return prior, conditioner, G, id_criterion, attr_teacher, \
           attribute_index, direction_bank


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------

@torch.no_grad()
def edit_single_attribute(prior, conditioner, G, id_criterion,
                          img, latent, attr_cond, id_cond,
                          attr_local_idx, edit_scale, direction_bank=None,
                          attr_global_idx=None, bypass_glasses_direction_bank=True):
    B = img.size(0)
    device = img.device
    zero_pad = torch.zeros(B, 18, 1, device=device)

    src_cond = torch.cat([id_cond, attr_cond], dim=1)
    mid_latent, _ = prior(latent, src_cond, zero_pad)

    new_attr_cond = attr_cond.clone()
    src = attr_cond[:, attr_local_idx]
    new_attr_cond[:, attr_local_idx] = src * (1.0 - edit_scale) + (1.0 - src) * edit_scale
    new_cond = torch.cat([id_cond, new_attr_cond], dim=1)

    new_latents_raw, _ = prior(mid_latent, new_cond, zero_pad, reverse=True)

    bypass_bank = (
        bypass_glasses_direction_bank
        and attr_global_idx == 15
    )
    if direction_bank is not None and not bypass_bank:
        flow_delta = new_latents_raw - latent
        attr_delta = new_attr_cond - attr_cond
        batch_attr_idx = torch.full((B,), attr_local_idx, device=device, dtype=torch.long)
        guided_delta = direction_bank(flow_delta, attr_delta,
                                      attr_idx=batch_attr_idx, latent=latent)
        new_latents = latent + guided_delta
    else:
        new_latents = new_latents_raw

    edited_face = G([new_latents], input_is_latent=True,
                    randomize_noise=False)[0].clamp(-1, 1)
    return edited_face


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(args):
    prior, conditioner, G, id_criterion, attr_teacher, \
        attribute_index, direction_bank = load_models(args)

    img_transform = T.Compose([
        T.ToTensor(),
        T.Resize((args.img_size, args.img_size)),
        T.Normalize(mean=0.5, std=0.5),
    ])
    test_dataset = SDFlowDataset(
        index_file=args.index_file,
        image_root=args.image_root,
        latents_file=args.latent_file,
        preds_file=args.preds_file,
        train=False,
        transform=img_transform,
    )
    test_loader = data.DataLoader(
        test_dataset, shuffle=False, batch_size=args.batch,
        num_workers=4, drop_last=False,
    )
    print(f'Test set: {len(test_dataset)} images')

    all_results = {}

    for edit_scale in args.eval_scales:
        print(f'\n{"="*60}')
        print(f'edit_scale = {edit_scale}')
        print(f'{"="*60}')

        metrics = defaultdict(list)
        sample_count = 0

        for img, latent, pred in tqdm(test_loader, desc=f'scale={edit_scale}'):
            if sample_count >= args.num_samples:
                break
            img    = img.cuda()
            latent = latent.cuda()
            pred   = pred.cuda()
            B      = img.size(0)

            _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)

            # Source face predictions from teacher
            src_face_256 = F.interpolate(
                G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1),
                (256, 256),
            )
            src_id_feat  = F.normalize(id_criterion.extract_features(src_face_256), dim=1)
            src_probs    = torch.sigmoid(attr_teacher(src_face_256)[0])[:, attribute_index]

            for local_idx in range(len(args.attribute_index)):
                attr_name = ATTR_NAMES.get(args.attribute_index[local_idx],
                                           f'attr{args.attribute_index[local_idx]}')
                src_score = src_probs[:, local_idx]
                clear_mask = (src_score > 0.65) | (src_score < 0.35)
                if not clear_mask.any():
                    continue

                edited_face = edit_single_attribute(
                    prior, conditioner, G, id_criterion,
                    img, latent, attr_cond, id_cond,
                    local_idx, edit_scale, direction_bank,
                    attr_global_idx=args.attribute_index[local_idx],
                    bypass_glasses_direction_bank=args.bypass_glasses_direction_bank,
                )
                edited_256 = F.interpolate(edited_face, (256, 256))
                edit_id_feat = F.normalize(id_criterion.extract_features(edited_256), dim=1)
                edit_probs   = torch.sigmoid(attr_teacher(edited_256)[0])[:, attribute_index]
                edit_score   = edit_probs[:, local_idx]

                for b in range(B):
                    if not clear_mask[b]:
                        continue
                    s = src_score[b].item()
                    e = edit_score[b].item()

                    # Identity preservation
                    id_cos = F.cosine_similarity(src_id_feat[b:b+1], edit_id_feat[b:b+1]).item()
                    metrics[f'id_{attr_name}'].append(id_cos)

                    # Editing accuracy
                    success = (e < s - 0.05) if s > 0.5 else (e > s + 0.05)
                    metrics[f'acc_{attr_name}'].append(float(success))
                    metrics[f'delta_{attr_name}'].append(abs(e - s))

                    # Attribute preservation (leakage on non-target attrs)
                    for other_idx in range(len(args.attribute_index)):
                        if other_idx == local_idx:
                            continue
                        other_name = ATTR_NAMES.get(args.attribute_index[other_idx],
                                                    f'attr{args.attribute_index[other_idx]}')
                        leakage = abs(edit_probs[b, other_idx].item()
                                      - src_probs[b, other_idx].item())
                        metrics[f'leak_{attr_name}_on_{other_name}'].append(leakage)

            sample_count += B

        # ── Print & collect ────────────────────────────────────────────────
        scale_summary = {'num_samples': sample_count}
        print(f'\n  {sample_count} samples evaluated')
        print(f'  {"Attribute":<12} {"ID↑":>8} {"Acc↑":>8} {"Δscore↑":>10}  Leakage↓')
        print(f'  {"-"*55}')
        for attr_name in ['Eyeglasses', 'Male', 'Young']:
            id_m   = np.mean(metrics[f'id_{attr_name}'])   if metrics[f'id_{attr_name}']   else float('nan')
            acc_m  = np.mean(metrics[f'acc_{attr_name}'])  if metrics[f'acc_{attr_name}']  else float('nan')
            dlt_m  = np.mean(metrics[f'delta_{attr_name}'])if metrics[f'delta_{attr_name}']else float('nan')
            leaks  = [v for k, v_list in metrics.items()
                      if k.startswith(f'leak_{attr_name}_on_')
                      for v in v_list]
            leak_m = np.mean(leaks) if leaks else float('nan')
            print(f'  {attr_name:<12} {id_m:>8.4f} {acc_m*100:>7.1f}% {dlt_m:>10.4f}  {leak_m:.4f}')
            scale_summary[attr_name] = {
                'id_cosine':      float(id_m),
                'edit_acc_pct':   float(acc_m * 100),
                'avg_delta':      float(dlt_m),
                'avg_leakage':    float(leak_m),
            }

        # Overall
        all_id  = [v for k, vl in metrics.items() if k.startswith('id_')  for v in vl]
        all_acc = [v for k, vl in metrics.items() if k.startswith('acc_') for v in vl]
        ovr_id  = float(np.mean(all_id))  if all_id  else float('nan')
        ovr_acc = float(np.mean(all_acc)) if all_acc else float('nan')
        print(f'  {"-"*55}')
        print(f'  {"Overall":<12} {ovr_id:>8.4f} {ovr_acc*100:>7.1f}%')
        scale_summary['overall'] = {'id_cosine': ovr_id, 'edit_acc_pct': ovr_acc * 100}
        all_results[str(edit_scale)] = scale_summary

    # ── Save JSON ──────────────────────────────────────────────────────────
    out_path = os.path.join(
        args.checkpoint_dir,
        f'eval_step{args.step}_n{args.num_samples}.json',
    )
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved → {out_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('--checkpoint_dir', required=True,
                        help='e.g. ./output/SDFlow/v13_stratified_k4')
    parser.add_argument('--step', type=int, default=None,
                        help='Checkpoint step (default: auto-detect latest)')

    # Data
    parser.add_argument('--index_file',   default='./data/ffhq.txt')
    parser.add_argument('--image_root',   default='data/FFHQ')
    parser.add_argument('--latent_file',  default='./data/ffhq_e4e_latents.pth')
    parser.add_argument('--preds_file',   default='./data/ffhq_e4e_preds.pth')
    parser.add_argument('--stygan2_weights', default='./data/stylegan2-ffhq-config-f.pt')
    parser.add_argument('--attribute_weights', default='./data/r34_a40_age_256_classifier.pth')
    parser.add_argument('--direction_bank_path', default=None)

    # Model config (must match training)
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
    parser.add_argument('--glasses_residual_scale',   type=float, default=0.05,
                        help='Residual scale for eyeglasses (must match training value).')
    parser.add_argument('--bypass_glasses_direction_bank',
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Bypass Direction Bank for Eyeglasses during evaluation.')
    parser.add_argument('--guided_delta_max_norm', type=float, default=0.0,
                        help='Shared max norm for the final guided W+ delta, applied uniformly to '
                             'every attribute. Set <=0 to disable (no cap).')

    # Eval config
    parser.add_argument('--batch',        type=int,   default=4)
    parser.add_argument('--num_samples',  type=int,   default=500)
    parser.add_argument('--eval_scales',  nargs='*',  type=float,
                        default=[0.80, 0.85, 0.90, 0.95])

    args = parser.parse_args()

    # Auto-detect latest step if not specified
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
        if args.step is None:
            raise ValueError(f'No checkpoints found in {args.checkpoint_dir}/save_models/')
        print(f'Auto-detected latest step: {args.step}')

    evaluate(args)
