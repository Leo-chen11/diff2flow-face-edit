"""
Check whether a precomputed Attribute Direction Bank's directions carry
identity-correlated signal, independent of the SDFlow training/inference
pipeline.

Motivation: if id_sim drops when editing with a given attribute direction,
that could mean either (a) the direction is doing exactly what it should --
producing a real visual change that a face-recognition model is sensitive to
-- or (b) the direction itself is contaminated with an identity-relevant
component (e.g. from dataset sampling imbalance between the high/low strata
used to compute it), in which case no amount of loss reweighting during
training will fix it; the direction needs to be recomputed with the
identity-correlated component removed.

Method: for each attribute, perturb a batch of real W+ latents by a fixed-
norm step along (1) the attribute's own direction and (2) many random unit
directions with the *same per-layer magnitude profile*. Measure the ArcFace
identity cosine drift caused by each. If the attribute direction's drift is
far outside the spread of the random-direction baseline, the direction is
likely carrying identity signal beyond what the visual change requires.

This does NOT use SDFlow.transform() or the trained magnitude_net/gate_net --
it is a raw geometric probe of the direction vectors stored in the bank file,
so it isolates "is this direction itself contaminated" from "did training
learn to use it well."

Usage:
    python evaluation/probe_direction_identity_contamination.py \
        --direction_bank_path ./data/direction_bank_k4_stratified.pth \
        --attribute_index 15 20 39 \
        --attribute_names Eyeglasses Gender Young \
        --target_delta_norm 15.0 \
        --num_random_directions 20 \
        --max_samples 32
"""

import argparse
import csv
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'stylegan2'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils import data
from tqdm import tqdm

from common.id_loss import IDLoss
from models.dataset import SDFlowDataset
from models.stylegan2.model import Generator


def attr_name_map(indices, names):
    if names and len(names) != len(indices):
        raise ValueError('--attribute_names must have the same length as --attribute_index.')
    if names:
        return {int(idx): name for idx, name in zip(indices, names)}
    return {int(idx): f'attr_{idx}' for idx in indices}


def load_bank_directions(bank_path, attribute_index):
    """Returns (unit_directions, layer_scale) both shaped (num_attrs, 18, 512) / (num_attrs, 18),
    averaged over K mixture components (with near-uniform gate init, the mean over K is a
    reasonable single representative direction per attribute for this geometric probe)."""
    bank = torch.load(bank_path, map_location='cpu')
    du = bank['direction_units'].float()
    ln = bank['layer_norms'].float()
    if du.ndim == 3:
        du = du.unsqueeze(1)
    if ln.ndim == 2:
        ln = ln.unsqueeze(1)

    bank_attrs = [int(x) for x in bank.get('attribute_index', [])]
    if bank_attrs:
        order = [bank_attrs.index(x) for x in attribute_index]
        du = du[order]
        ln = ln[order]

    du = F.normalize(du, dim=-1, eps=1e-8)          # (A, K, 18, 512) unit vectors per layer
    mean_dir = du.mean(dim=1)                        # (A, 18, 512)
    mean_dir = F.normalize(mean_dir, dim=-1, eps=1e-8)
    mean_layer_scale = ln.mean(dim=1)                 # (A, 18) relative per-layer magnitude
    return mean_dir, mean_layer_scale


def scaled_direction(unit_dir_per_layer, layer_scale, target_norm):
    """unit_dir_per_layer: (18, 512) unit vectors. layer_scale: (18,) relative weights.
    Returns a (18, 512) delta whose total norm equals target_norm, distributed across
    layers proportionally to layer_scale."""
    weighted = unit_dir_per_layer * layer_scale.view(-1, 1).clamp(min=1e-6)
    total_norm = weighted.reshape(-1).norm().clamp(min=1e-8)
    return weighted * (target_norm / total_norm)


def random_unit_directions(layer_scale, target_norm, num_layers, latent_dim, count, device, generator):
    """count random (18, 512) deltas, each with the same per-layer magnitude profile
    (layer_scale) and total norm as the attribute direction, but random orientation."""
    raw = torch.randn(count, num_layers, latent_dim, generator=generator, device=device)
    raw = F.normalize(raw, dim=-1, eps=1e-8)
    weighted = raw * layer_scale.view(1, -1, 1).clamp(min=1e-6)
    total_norm = weighted.reshape(count, -1).norm(dim=1).clamp(min=1e-8)
    return weighted * (target_norm / total_norm).view(count, 1, 1)


@torch.no_grad()
def id_embed(generator, id_model, latents, img_size):
    faces = generator([latents], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
    faces = F.interpolate(faces, (img_size, img_size), mode='bilinear', align_corners=False)
    return F.normalize(id_model.extract_features(faces), dim=1)


def main():
    parser = argparse.ArgumentParser(
        description="Probe whether Direction Bank directions carry identity-correlated signal."
    )
    parser.add_argument('--direction_bank_path', required=True, type=str)
    parser.add_argument('--attribute_index', nargs='*', default=[15, 20, 39], type=int)
    parser.add_argument('--attribute_names', nargs='*', default=None)
    parser.add_argument('--target_delta_norm', type=float, default=15.0,
                        help='Total W+ delta norm to test at, matching typical operating '
                             'magnitude (see dir_bank_guided_delta_norm in training logs).')
    parser.add_argument('--num_random_directions', type=int, default=20,
                        help='Random-direction draws per sample, used as the null-hypothesis baseline.')

    parser.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    parser.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth')
    parser.add_argument('--index_file', default='./data/ffhq.txt')
    parser.add_argument('--image_root', default='data/FFHQ')
    parser.add_argument('--stylegan2_weights', default='./data/stylegan2-ffhq-config-f.pt')

    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--id_img_size', type=int, default=256)
    parser.add_argument('--max_samples', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', default='./output/probe_direction_contamination')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    name_by_attr = attr_name_map(args.attribute_index, args.attribute_names)
    rng = torch.Generator(device=device).manual_seed(args.seed)

    transform = T.Compose([
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
        transform=transform,
    )
    if args.max_samples > 0:
        dataset = data.Subset(dataset, list(range(min(args.max_samples, len(dataset)))))
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    print('Loading StyleGAN2...')
    ckpt = torch.load(args.stylegan2_weights, map_location='cpu')
    generator = Generator(size=1024, style_dim=512, n_mlp=8)
    generator.load_state_dict(ckpt['g_ema'])
    generator = generator.to(device).eval()

    print('Loading ID model...')
    id_model = IDLoss(crop=True).to(device).eval()

    print(f'Loading direction bank from {args.direction_bank_path} ...')
    mean_dir, mean_layer_scale = load_bank_directions(args.direction_bank_path, args.attribute_index)
    num_attrs, num_layers, latent_dim = mean_dir.shape
    mean_dir = mean_dir.to(device)
    mean_layer_scale = mean_layer_scale.to(device)

    rows = []
    for a_idx, attr_global_idx in enumerate(args.attribute_index):
        attr_name = name_by_attr[int(attr_global_idx)]
        attr_delta = scaled_direction(mean_dir[a_idx], mean_layer_scale[a_idx], args.target_delta_norm)

        attr_drifts = []
        random_drifts = []

        for _, latent, _ in tqdm(loader, desc=attr_name):
            latent = latent.to(device, non_blocking=True)   # (1, 18, 512), matches other eval scripts

            base_embed = id_embed(generator, id_model, latent, args.id_img_size)

            attr_embed = id_embed(generator, id_model, latent + attr_delta.unsqueeze(0), args.id_img_size)
            attr_drift = (1.0 - F.cosine_similarity(base_embed, attr_embed, dim=1)).item()
            attr_drifts.append(attr_drift)

            rand_deltas = random_unit_directions(
                mean_layer_scale[a_idx], args.target_delta_norm, num_layers, latent_dim,
                args.num_random_directions, device, rng,
            )
            for r in range(args.num_random_directions):
                rand_embed = id_embed(generator, id_model, latent + rand_deltas[r].unsqueeze(0), args.id_img_size)
                random_drifts.append((1.0 - F.cosine_similarity(base_embed, rand_embed, dim=1)).item())

        attr_mean = sum(attr_drifts) / len(attr_drifts)
        rand_mean = sum(random_drifts) / len(random_drifts)
        rand_var = sum((x - rand_mean) ** 2 for x in random_drifts) / max(1, len(random_drifts) - 1)
        rand_std = rand_var ** 0.5
        z_score = (attr_mean - rand_mean) / rand_std if rand_std > 1e-8 else float('nan')

        print(f'\n{attr_name} (attr {attr_global_idx}):')
        print(f'  attribute-direction id drift : mean={attr_mean:.4f}')
        print(f'  random-direction id drift    : mean={rand_mean:.4f}  std={rand_std:.4f}')
        print(f'  z-score (attr vs random null) : {z_score:.2f}')
        verdict = (
            'LIKELY CONTAMINATED (direction moves identity far more than a random direction of the same size)'
            if z_score > 2.0 else
            'looks clean (identity cost is in line with a random perturbation of the same magnitude)'
        )
        print(f'  verdict: {verdict}')

        rows.append({
            'attribute': attr_name,
            'attribute_index': int(attr_global_idx),
            'target_delta_norm': args.target_delta_norm,
            'attr_direction_id_drift_mean': attr_mean,
            'random_direction_id_drift_mean': rand_mean,
            'random_direction_id_drift_std': rand_std,
            'z_score': z_score,
            'n_samples': len(attr_drifts),
            'n_random_draws': len(random_drifts),
        })

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'contamination_probe.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
