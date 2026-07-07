"""
Adversarial-robustness sanity check for the frozen attribute teacher.

The exact same checkpoint (--attribute_weights, default
data/r34_a40_age_256_classifier.pth) is used BOTH as the training-time
counterfactual-teacher loss in training/train_sdflow.py AND as the success
metric in evaluation/eval_strength_sweep.py. That means loss_target during
training and target_success at eval time cannot tell you whether the model
produced a genuine, human/CLIP-visible attribute edit versus an adversarial
pixel-level perturbation that only fools this one frozen classifier on the
exact clean pixels it was optimized against.

This script re-scores already-edited faces after mild, semantically
irrelevant perturbations (light Gaussian noise, blur, JPEG re-encoding). If
target_success collapses under these trivial perturbations, the "success"
was very likely adversarial, not a real semantic change -- and every
training/eval number derived from this teacher should be treated as
unreliable until that's addressed (e.g. by scoring with a classifier never
used in the training loop, such as a separately trained ConvNeXt-Base).

Usage:
    python evaluation/eval_adversarial_robustness.py \
        --ckpt_dir ./output/SDFlow/lda_dds_clip_v2_fresh/save_models \
        --velocity_field lag_dof \
        --direction_bank_path ./data/direction_bank_k4_stratified.pth \
        --attribute_index 15 20 39 \
        --attribute_names Eyeglasses Gender Young \
        --strength 1.0 \
        --max_samples 64

Outputs:
    output/eval_adversarial_robustness/robustness_summary.csv
"""

import argparse
import csv
import io
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'stylegan2'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils import data
from tqdm import tqdm

from common.ops import load_network
from models.attribute_estimator import AttributeClassifier
from models.dataset import SDFlowDataset
from models.editor import SDFlow
from models.stylegan2.model import Generator


def attr_name_map(indices, names):
    if names and len(names) != len(indices):
        raise ValueError('--attribute_names must have the same length as --attribute_index.')
    if names:
        return {int(idx): name for idx, name in zip(indices, names)}
    return {int(idx): f'attr_{idx}' for idx in indices}


@torch.no_grad()
def generate_faces(generator, latents, img_size):
    faces = generator([latents], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
    if faces.size(-1) != img_size:
        faces = F.interpolate(faces, (img_size, img_size), mode='bilinear', align_corners=False)
    return faces


@torch.no_grad()
def classifier_prob(attr_teacher, faces, attr_global_idx):
    logits, _ = attr_teacher(F.interpolate(faces, (256, 256), mode='bilinear', align_corners=False))
    probs = torch.sigmoid(logits)
    return probs[:, attr_global_idx]


def add_gaussian_noise(faces, std):
    if std <= 0:
        return faces
    noise = torch.randn_like(faces) * std
    return (faces + noise).clamp(-1, 1)


def gaussian_blur(faces, kernel_size):
    if kernel_size <= 1:
        return faces
    return TF.gaussian_blur(faces, kernel_size=[kernel_size, kernel_size])


def jpeg_roundtrip(faces, quality):
    """faces: [-1,1] BCHW tensor -> JPEG-recompressed [-1,1] BCHW tensor."""
    if quality >= 100:
        return faces
    out = torch.empty_like(faces)
    imgs01 = (faces.clamp(-1, 1) + 1.0) * 0.5
    for b in range(faces.shape[0]):
        pil_img = TF.to_pil_image(imgs01[b].cpu())
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=int(quality))
        buf.seek(0)
        recon = Image.open(buf).convert('RGB')
        recon_t = TF.to_tensor(recon).to(faces.device)
        out[b] = recon_t * 2.0 - 1.0
    return out


PERTURBATIONS = {
    'clean': lambda x: x,
    'noise_0.02': lambda x: add_gaussian_noise(x, 0.02),
    'noise_0.05': lambda x: add_gaussian_noise(x, 0.05),
    'blur_3': lambda x: gaussian_blur(x, 3),
    'blur_5': lambda x: gaussian_blur(x, 5),
    'jpeg_70': lambda x: jpeg_roundtrip(x, 70),
    'jpeg_40': lambda x: jpeg_roundtrip(x, 40),
}


def main():
    parser = argparse.ArgumentParser(description='Adversarial-robustness check for the frozen attribute teacher.')
    parser.add_argument('--ckpt_dir', required=True, help='Directory containing conditioner-* and prior-* checkpoints.')
    parser.add_argument('--ckpt_step', type=int, default=None)
    parser.add_argument('--output_dir', default='./output/eval_adversarial_robustness')
    parser.add_argument('--velocity_field', default='lag_dof', choices=['original', 'lag', 'dof', 'lag_dof'])
    parser.add_argument('--flow_modules', type=str, default='512-512-512-512-512')
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--lag_gate_hidden_dim', type=int, default=64)
    parser.add_argument('--lag_gate_init_bias', type=float, default=-0.5)
    parser.add_argument('--id_cond_dim', type=int, default=32)
    parser.add_argument('--id_cond_scale', type=float, default=0.25)
    parser.add_argument('--attr_backbone', default='resnet50')
    parser.add_argument('--conditioner_backbone', default='resnet', choices=['resnet', 'clip', 'resnet_clip'])
    parser.add_argument('--clip_model', default='ViT-B/32')
    parser.add_argument('--fused_hidden_dim', type=int, default=256)
    parser.add_argument('--direction_bank_path', default=None, type=str,
                        help='Path to precomputed Attribute Direction Bank (.pth).')
    parser.add_argument('--direction_residual_scale', type=float, default=0.05)
    parser.add_argument('--direction_freeze', '--direction-freeze',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--bypass_glasses_direction_bank',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--guided_delta_max_norm', type=float, default=0.0)

    parser.add_argument('--attribute_index', nargs='*', default=[15, 20, 39], type=int)
    parser.add_argument('--attribute_names', nargs='*', default=None)
    parser.add_argument('--strength', type=float, default=1.0,
                        help='Single edit strength to test robustness at (use the strength you actually deploy with).')

    parser.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    parser.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth')
    parser.add_argument('--index_file', default='./data/ffhq.txt')
    parser.add_argument('--image_root', default='data/FFHQ')
    parser.add_argument('--stylegan2_weights', default='./data/stylegan2-ffhq-config-f.pt')
    parser.add_argument('--attribute_weights', default='./data/r34_a40_age_256_classifier.pth',
                        help='Same checkpoint used as the training-time counterfactual teacher. '
                             'This script exists specifically to stress-test that model.')

    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=64)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    invalid = [i for i in args.attribute_index if not (0 <= i < 40)]
    if invalid:
        parser.error(f"--attribute_index values must be in [0, 39]; got: {invalid}")

    device = torch.device(args.device)
    name_by_attr = attr_name_map(args.attribute_index, args.attribute_names)

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
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith('cuda'),
    )

    print('Loading StyleGAN2...')
    ckpt = torch.load(args.stylegan2_weights, map_location='cpu')
    generator = Generator(size=1024, style_dim=512, n_mlp=8)
    generator.load_state_dict(ckpt['g_ema'])
    generator = generator.to(device).eval()

    print('Loading attribute teacher (same checkpoint used as the training-time teacher)...')
    attr_teacher = AttributeClassifier(backbone='r34')
    attr_teacher.load_state_dict(load_network(args.attribute_weights))
    attr_teacher = attr_teacher.to(device).eval()

    rows = []
    for attr_local_idx, attr_global_idx in enumerate(args.attribute_index):
        attr_name = name_by_attr[int(attr_global_idx)]
        print(f'\nEvaluating {attr_name} ({attr_global_idx}) @ strength={args.strength}...')
        transformer = SDFlow(
            ckpt_dir=args.ckpt_dir,
            attr_num=int(attr_global_idx),
            attr_list=args.attribute_index,
            scale=args.strength,
            device=str(device),
            id_cond_dim=args.id_cond_dim,
            id_cond_scale=args.id_cond_scale,
            attr_backbone=args.attr_backbone,
            conditioner_backbone=args.conditioner_backbone,
            clip_model=args.clip_model,
            fused_hidden_dim=args.fused_hidden_dim,
            flow_modules=args.flow_modules,
            num_blocks=args.num_blocks,
            velocity_field=args.velocity_field,
            lag_gate_hidden_dim=args.lag_gate_hidden_dim,
            lag_gate_init_bias=args.lag_gate_init_bias,
            direction_bank_path=args.direction_bank_path,
            direction_residual_scale=args.direction_residual_scale,
            direction_freeze=args.direction_freeze,
            ckpt_step=args.ckpt_step,
            bypass_glasses_direction_bank=args.bypass_glasses_direction_bank,
            guided_delta_max_norm=(args.guided_delta_max_norm if args.guided_delta_max_norm > 0 else None),
        )

        counts = {name: {'success': 0, 'total': 0} for name in PERTURBATIONS}

        for images, latents, source_preds in tqdm(loader, desc=attr_name):
            images = images.to(device, non_blocking=True)
            latents = latents.to(device, non_blocking=True)
            source_preds = source_preds.to(device, non_blocking=True)

            with torch.no_grad():
                src_target_prob = classifier_prob(
                    attr_teacher, generate_faces(generator, latents, args.img_size), attr_global_idx
                )
                src_target_binary = (src_target_prob > 0.5).float()
                final_target_binary = 1.0 - src_target_binary

                edited_latents = transformer.transform(latents, source_preds, images)
                edited_faces = generate_faces(generator, edited_latents, args.img_size)

                for pname, pfn in PERTURBATIONS.items():
                    perturbed = pfn(edited_faces)
                    edited_target_prob = classifier_prob(attr_teacher, perturbed, attr_global_idx)
                    edited_binary = (edited_target_prob > 0.5).float()
                    success = (edited_binary == final_target_binary).float()
                    counts[pname]['success'] += int(success.sum().item())
                    counts[pname]['total'] += int(success.numel())

        clean_rate = counts['clean']['success'] / max(1, counts['clean']['total'])
        print(f'  clean target_success = {clean_rate:.3f}')
        for pname in PERTURBATIONS:
            rate = counts[pname]['success'] / max(1, counts[pname]['total'])
            drop = clean_rate - rate
            rows.append({
                'attribute': attr_name,
                'attribute_index': int(attr_global_idx),
                'perturbation': pname,
                'target_success': rate,
                'drop_vs_clean': drop,
                'n': counts[pname]['total'],
            })
            flag = '  <-- big drop, likely adversarial' if (pname != 'clean' and drop > 0.15) else ''
            print(f'  {pname:>10}: success={rate:.3f}  drop={drop:+.3f}{flag}')

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'robustness_summary.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nWrote {out_path}')

    print('\nVerdict guide: if any non-clean perturbation drops target_success by')
    print('more than ~0.15-0.20 relative to clean, the classifier is very likely being')
    print('fooled by pixel-level detail rather than a genuine attribute change -- treat')
    print('loss_target from training and target_success from eval_strength_sweep.py as')
    print('unreliable for judging real editing quality until this is addressed (e.g. by')
    print('scoring with a classifier that was never used in the training loop).')


if __name__ == '__main__':
    main()
