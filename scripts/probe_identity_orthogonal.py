"""Does projecting the edit direction away from its OWN worst identity
direction recover ID cosine, and how much accuracy does it cost?

This is the cheapest test of "orthogonalize the edit against identity"
that does not require estimating a subspace ahead of time (via random
perturbations + SVD of a Jacobian, the textbook version). Instead it uses
a fact that was already true every training step and just never exploited
this way: id_loss's own gradient w.r.t. the applied edit, AT the edit
currently being made, already points along the (locally) steepest
identity-damaging direction. That is the same quantity id_loss's backward
pass computes to push guided_delta around during training -- this script
just reads it out once, per sample, without touching training at all, and
asks what happens if that ONE direction is surgically removed from the
edit instead of being fought with a scalar loss weight.

Procedure per sample:
  1. Compute guided_delta exactly as edit_single_attribute does (mirrored
     inline below so the baseline number matches eval/training exactly),
     then detach it -- everything from here on treats it as a free
     variable, not something direction_bank produced.
  2. Repeat --num_directions times:
       a. render G(latent + guided_delta), score ID cosine against the
          source reconstruction
       b. backward to get d(id_loss)/d(guided_delta) -- one direction
       c. project that direction out of guided_delta (Gram-Schmidt against
          the directions already removed, so each step removes something
          new rather than re-removing the same one)
  3. Report, per sample and averaged: how much ID cosine came back, how
     much the attribute score moved, both vs BASELINE (unprojected).
  4. CONTROL: repeat with the SAME NUMBER of RANDOM unit directions of
     matched norm instead of the identity-derived ones, so "removing any
     K directions of similar size" can be told apart from "removing K
     directions specifically chosen for hurting identity".

ControlNet injection is deliberately left out of this probe -- the
projection operates on guided_delta, the W+ edit; ControlNet's skip
tensors are a separate additive path into the generator's feature maps
and aren't touched by a W+-space subspace argument, so including them
would mix two different mechanisms into one number.

Usage:
    python -m scripts.probe_identity_orthogonal \
        --checkpoint_dir ./output/SDFlow/substyle3_glasses_v18 --step 120000 \
        --attr 39 --direction rm --edit_scale 1.0 \
        --num_directions 3 --num_samples 24 \
        --celeba_attr_judge_weights /home/cchen/桌面/SDFlow/data/celeba_attr_resnet18.pth
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
from torch.utils import data

from evaluation.evaluate_sdflow import (
    ATTR_NAMES, CelebAAttrClassifierJudge, _latest_step, apply_run_config,
    is_clear, load_models,
)
from models.dataset import SDFlowDataset


def compute_guided_delta(prior, direction_bank, latent, attr_cond, id_cond,
                         attr_local_idx, edit_scale):
    """Mirrors evaluate_sdflow.edit_single_attribute's guided_delta
    computation exactly (same formulas, same call order), so the baseline
    this script starts from is the same edit eval/training would produce.
    Returns guided_delta DETACHED -- this script treats it as the object
    under test, not as something to keep differentiating through
    direction_bank/prior/conditioner.
    """
    B = latent.size(0)
    device = latent.device
    zero_pad = torch.zeros(B, 18, 1, device=device)

    src_cond = torch.cat([id_cond, attr_cond], dim=1)
    mid_latent, _ = prior(latent, src_cond, zero_pad)

    new_attr_cond = attr_cond.clone()
    src = attr_cond[:, attr_local_idx]
    new_attr_cond[:, attr_local_idx] = src * (1.0 - edit_scale) + (1.0 - src) * edit_scale
    new_cond = torch.cat([id_cond, new_attr_cond], dim=1)

    new_latents_raw, _ = prior(mid_latent, new_cond, zero_pad, reverse=True)
    flow_delta = new_latents_raw - latent
    attr_delta = new_attr_cond - attr_cond
    batch_attr_idx = torch.full((B,), attr_local_idx, device=device, dtype=torch.long)
    guided_delta = direction_bank(flow_delta, attr_delta, attr_idx=batch_attr_idx, latent=latent)
    return guided_delta.detach()


def id_cosine(G, id_criterion, latent, delta, src_feat):
    """cos_sim(ArcFace(G(latent+delta)), src_feat). Renders at 256 --
    matches evaluate_sdflow's own resolution for identity scoring."""
    edited = G([latent + delta], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
    edited_256 = F.interpolate(edited, (256, 256))
    feat = id_criterion.extract_features(edited_256)
    feat = F.normalize(feat, dim=1)
    return F.cosine_similarity(feat, src_feat, dim=1), edited


def gram_schmidt_project_out(vec, removed_units):
    """vec: (18,512) flattened as needed. removed_units: list of already-
    orthonormalized (18,512) directions. Returns vec with all of them
    projected out."""
    v = vec.clone()
    for u in removed_units:
        v = v - (v * u).sum() * u
    return v


def unit(vec):
    n = vec.norm()
    return vec / n.clamp(min=1e-8)


def project_out_identity_directions(G, id_criterion, latent, guided_delta,
                                    src_feat, num_directions):
    """Iteratively: measure id_loss at the current point, backprop to get
    the locally steepest identity-damaging direction, project it out
    (Gram-Schmidt against previously removed ones), repeat.

    Returns (final_delta, removed_units) -- removed_units is reused by the
    caller to build a matched-norm random control.
    """
    delta = guided_delta.clone()
    removed_units = []
    for _ in range(num_directions):
        d = delta.clone().requires_grad_(True)
        with torch.enable_grad():
            cos, _ = id_cosine(G, id_criterion, latent, d, src_feat)
            id_loss = (1.0 - cos).mean()
            grad = torch.autograd.grad(id_loss, d)[0]
        u = unit(gram_schmidt_project_out(grad[0], removed_units))
        removed_units.append(u)
        delta = (delta[0] - (delta[0] * u).sum() * u).unsqueeze(0)
    return delta, removed_units


def random_control_delta(guided_delta, removed_units, seed):
    """Same number of directions, same total norm removed, but random
    instead of identity-derived -- isolates "removing K directions of this
    size" from "removing K directions CHOSEN for hurting identity"."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    delta = guided_delta.clone()
    for u_real in removed_units:
        removed_norm = (guided_delta[0] * u_real).sum().abs()   # match magnitude
        r = torch.randn(delta[0].shape, generator=g).to(delta.device)
        r = unit(r)
        delta = (delta[0] - torch.sign((delta[0] * r).sum()) * removed_norm * r).unsqueeze(0)
    return delta


@torch.no_grad()
def main(args):
    prior, conditioner, G, id_criterion, attr_teacher, \
        attribute_index, direction_bank, control_encoder = load_models(args)
    if args.attr not in args.attribute_index:
        raise SystemExit(f'attribute_index {args.attribute_index} has no attr {args.attr}.')
    local_idx = args.attribute_index.index(args.attr)
    attr_name = ATTR_NAMES.get(args.attr, f'attr{args.attr}')

    judge = CelebAAttrClassifierJudge(args.celeba_attr_judge_weights, 'cuda') \
        if args.celeba_attr_judge_weights else None

    print(f'auditing {attr_name} direction={args.direction} scale={args.edit_scale}')
    print(f'removing top {args.num_directions} identity-damaging direction(s) per sample, '
         f'gated on the conditioner (the reading that decides the edit -- see this '
         f'project\'s own probe_direction_gender_split.py for why judge-based gating '
         f'disagrees with what the model was actually told to do)\n')

    img_transform = T.Compose([
        T.ToTensor(), T.Resize((args.img_size, args.img_size)),
        T.Normalize(mean=0.5, std=0.5),
    ])
    dataset = SDFlowDataset(
        index_file=args.index_file, image_root=args.image_root,
        latents_file=args.latent_file, preds_file=args.preds_file,
        train=False, transform=img_transform,
    )
    loader = data.DataLoader(dataset, shuffle=False, batch_size=1,
                             num_workers=2, drop_last=False)

    base_ids, proj_ids, ctrl_ids = [], [], []
    base_scores, proj_scores, ctrl_scores = [], [], []
    seen = 0
    for img, latent, pred in loader:
        if seen >= args.num_samples:
            break
        img = img.cuda(); latent = latent.cuda()
        _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
        c = attr_cond[0, local_idx].item()
        if args.direction == 'rm' and c < 0.5:
            continue
        if args.direction == 'add' and c >= 0.5:
            continue
        if not is_clear(c):
            continue
        seen += 1

        src_recon = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
        src_feat = F.normalize(
            id_criterion.extract_features(F.interpolate(src_recon, (256, 256))), dim=1)

        guided_delta = compute_guided_delta(prior, direction_bank, latent, attr_cond, id_cond,
                                            local_idx, args.edit_scale)

        base_cos, base_img = id_cosine(G, id_criterion, latent, guided_delta, src_feat)
        proj_delta, removed_units = project_out_identity_directions(
            G, id_criterion, latent, guided_delta, src_feat, args.num_directions)
        proj_cos, proj_img = id_cosine(G, id_criterion, latent, proj_delta, src_feat)
        ctrl_delta = random_control_delta(guided_delta, removed_units, args.control_seed + seen)
        ctrl_cos, ctrl_img = id_cosine(G, id_criterion, latent, ctrl_delta, src_feat)

        base_ids.append(base_cos.item()); proj_ids.append(proj_cos.item())
        ctrl_ids.append(ctrl_cos.item())

        line = (f'  #{seen:<2} ID  base={base_cos.item():.4f}  '
               f'projected={proj_cos.item():.4f} ({proj_cos.item()-base_cos.item():+.4f})  '
               f'random_ctrl={ctrl_cos.item():.4f} ({ctrl_cos.item()-base_cos.item():+.4f})')
        if judge is not None:
            bs = judge.scores(F.interpolate(base_img, (256, 256)))[0, args.attr].item()
            ps = judge.scores(F.interpolate(proj_img, (256, 256)))[0, args.attr].item()
            cs = judge.scores(F.interpolate(ctrl_img, (256, 256)))[0, args.attr].item()
            base_scores.append(bs); proj_scores.append(ps); ctrl_scores.append(cs)
            line += (f'\n       {attr_name}  base={bs:.2f}  projected={ps:.2f} '
                    f'({ps-bs:+.2f})  random_ctrl={cs:.2f} ({cs-bs:+.2f})')
        print(line)

    if not base_ids:
        raise SystemExit('no samples matched that attribute/direction.')

    n = len(base_ids)
    def mean(xs): return sum(xs) / len(xs)
    print(f'\n=== over {n} samples, {args.num_directions} direction(s) removed ===')
    print(f'  ID cosine        base={mean(base_ids):.4f}  '
         f'projected={mean(proj_ids):.4f} ({mean(proj_ids)-mean(base_ids):+.4f})  '
         f'random_ctrl={mean(ctrl_ids):.4f} ({mean(ctrl_ids)-mean(base_ids):+.4f})')
    if base_scores:
        print(f'  {attr_name} score      base={mean(base_scores):.3f}  '
             f'projected={mean(proj_scores):.3f} ({mean(proj_scores)-mean(base_scores):+.3f})  '
             f'random_ctrl={mean(ctrl_scores):.3f} ({mean(ctrl_scores)-mean(base_scores):+.3f})')
    print()
    print('How to read this:')
    print('  - ID recovered by "projected" should be >= what "random_ctrl" recovers, by a')
    print('    meaningful margin -- if they are similar, the identity-derived direction is')
    print('    not doing anything a random cut of the same size would not also do, and the')
    print('    orthogonalization idea is not earning its complexity here.')
    print('  - The attribute score under "projected" should drop only a LITTLE next to how')
    print('    much ID it recovers. A big score drop for a small ID gain means this attribute\'s')
    print('    edit and its identity cost are not separable in W+ -- the subspaces overlap too')
    print('    much for a rank-K cut to help, and the fix has to be elsewhere (magnitude/')
    print('    calibration, not direction).')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--step', type=int, default=None)
    p.add_argument('--attr', type=int, default=39)
    p.add_argument('--direction', default='rm', choices=['add', 'rm'])
    p.add_argument('--edit_scale', type=float, default=1.0)
    p.add_argument('--num_directions', type=int, default=1,
                   help='How many identity-damaging directions to remove, greedily.')
    p.add_argument('--num_samples', type=int, default=24)
    p.add_argument('--control_seed', type=int, default=777)
    p.add_argument('--celeba_attr_judge_weights', default=None,
                   help='Optional but recommended -- without it only ID cosine is reported, '
                        'not whether the attribute edit itself survived the projection.')
    p.add_argument('--controlnet_disable_attrs', nargs='*', type=int, default=None)

    p.add_argument('--index_file',   default='./data/ffhq.txt')
    p.add_argument('--image_root',   default='data/FFHQ')
    p.add_argument('--latent_file',  default='./data/ffhq_e4e_latents.pth')
    p.add_argument('--preds_file',   default='./data/ffhq_e4e_preds.pth')
    p.add_argument('--stygan2_weights', default='./data/stylegan2-ffhq-config-f.pt')
    p.add_argument('--attribute_weights', default='./data/r34_a40_age_256_classifier.pth')
    p.add_argument('--direction_bank_path', default=None)

    p.add_argument('--img_size',         type=int,   default=512)
    p.add_argument('--attribute_index',  nargs='*',  type=int,   default=[15, 20, 39])
    p.add_argument('--flow_modules',     default='512-512-512-512-512')
    p.add_argument('--num_blocks',       type=int,   default=1)
    p.add_argument('--velocity_field',   default='lag_dof')
    p.add_argument('--id_cond_dim',      type=int,   default=32)
    p.add_argument('--id_cond_scale',    type=float, default=0.25)
    p.add_argument('--attr_backbone',    default='resnet50')
    p.add_argument('--conditioner_backbone', default='resnet',
                   choices=['resnet', 'clip', 'resnet_clip'])
    p.add_argument('--clip_model', default='ViT-B/32')
    p.add_argument('--fused_hidden_dim', type=int, default=256)
    p.add_argument('--lag_gate_hidden_dim', type=int,   default=64)
    p.add_argument('--lag_gate_init_bias',  type=float, default=-0.5)
    p.add_argument('--direction_residual_scale', type=float, default=0.05)
    p.add_argument('--glasses_residual_scale',   type=float, default=0.05)
    p.add_argument('--guided_delta_max_norm', type=float, default=0.0)
    p.add_argument('--override_residual_scale', type=float, default=None)
    p.add_argument('--force_bank_directions', action='store_true')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--ignore_run_config', action='store_true')

    args = p.parse_args()
    args = apply_run_config(args)
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
        if args.step is None:
            raise ValueError(f'No checkpoints in {args.checkpoint_dir}/save_models/')
        print(f'Auto-detected latest step: {args.step}')
    main(args)
