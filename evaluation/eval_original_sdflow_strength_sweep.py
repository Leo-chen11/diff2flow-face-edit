"""
Evaluate an original SDFlow checkpoint that saves labeldist-* and prior-*.

This script is for comparing the original 5-attribute SDFlow run against the
newer LAG-DOF / direction-bank runs with the same style of strength sweep.

Example:
    python evaluation/eval_original_sdflow_strength_sweep.py \
        --ckpt_dir /home/cchen/桌面/SDFlow/output/SDFlow/default/save_models \
        --ckpt_step 13000 \
        --attribute_index 15 20 31 33 39 \
        --attribute_names Eyeglasses Male Smiling Wavy_Hair Young \
        --eval_attribute_index 15 20 39 \
        --max_samples 64 \
        --save_images 8
"""

import argparse
import csv
import glob
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
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from common.id_loss import IDLoss
from common.ops import load_network
from models.attribute_estimator import AttributeClassifier, AttributeEstimator
from models.dataset import SDFlowDataset
from models.flows.flow import cnf
from models.stylegan2.model import Generator


DEFAULT_ATTR_NAMES = {
    15: 'Eyeglasses',
    20: 'Male',
    31: 'Smiling',
    33: 'Wavy_Hair',
    39: 'Young',
}


def parse_strengths(value):
    if value:
        return [float(v) for v in value.split(',')]
    return [round(i / 10.0, 1) for i in range(1, 11)]


def mean_or_zero(values):
    return sum(values) / len(values) if values else 0.0


def format_step(step):
    return str(int(step)).zfill(7)


def find_ckpt(ckpt_dir, prefix, step=None):
    if step is not None:
        path = os.path.join(ckpt_dir, f'{prefix}-{format_step(step)}')
        if not os.path.exists(path):
            raise FileNotFoundError(f'No checkpoint found: {path}')
        return path
    files = sorted(glob.glob(os.path.join(ckpt_dir, f'{prefix}-*')))
    if not files:
        raise FileNotFoundError(f'No {prefix}-* checkpoints found in {ckpt_dir}')
    return files[-1]


def infer_labeldist_backbone(state):
    fc_weight = state.get('backbone.fc.weight')
    if fc_weight is None:
        return 'resnet34'
    in_dim = int(fc_weight.shape[1])
    if in_dim == 2048:
        # ResNet-50/101 both use 2048-dim fc input. The original SDFlow runs
        # here use ResNet-50; detect ResNet-101 only when the deeper layer exists.
        if any(k.startswith('backbone.layer3.22.') for k in state):
            return 'resnet101'
        return 'resnet50'
    if in_dim == 512:
        if any(k.startswith('backbone.layer3.5.') for k in state):
            return 'resnet34'
        return 'resnet18'
    raise ValueError(f'Cannot infer AttributeEstimator backbone from fc input dim {in_dim}')


def build_name_map(indices, names):
    if names is not None:
        if len(names) != len(indices):
            raise ValueError('--attribute_names must match --attribute_index length.')
        return {int(idx): name for idx, name in zip(indices, names)}
    return {int(idx): DEFAULT_ATTR_NAMES.get(int(idx), f'attr_{idx}') for idx in indices}


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def generate_faces(generator, latents, img_size):
    faces = generator([latents], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
    if faces.size(-1) != img_size:
        faces = F.interpolate(faces, (img_size, img_size), mode='bilinear', align_corners=False)
    return faces


class OriginalSDFlowEditor:
    def __init__(self, args, attr_num, device):
        self.device = device
        self.attr_num = int(attr_num)
        self.class_indices = [int(v) for v in args.attribute_index]
        if self.attr_num not in self.class_indices:
            raise ValueError(f'attr_num {self.attr_num} is not in --attribute_index {self.class_indices}')

        labeldist_path = find_ckpt(args.ckpt_dir, 'labeldist', args.ckpt_step)
        labeldist_state = torch.load(labeldist_path, map_location='cpu')
        labeldist_backbone = args.labeldist_backbone
        if labeldist_backbone == 'auto':
            labeldist_backbone = infer_labeldist_backbone(labeldist_state)
        self.labeldist = AttributeEstimator(
            backbone=labeldist_backbone,
            attribute_dim=len(self.class_indices),
        ).to(device)
        self.labeldist.load_state_dict(labeldist_state, strict=True)
        self.labeldist.eval()

        self.flow = cnf(
            512,
            args.flow_modules,
            len(self.class_indices),
            args.num_blocks,
            velocity_field='original',
            train_T=True,
        ).to(device)
        prior_path = find_ckpt(args.ckpt_dir, 'prior', args.ckpt_step)
        self.flow.load_state_dict(torch.load(prior_path, map_location='cpu'), strict=True)
        self.flow.eval()

        print(f'Loaded labeldist from {labeldist_path} ({labeldist_backbone})')
        print(f'Loaded prior from {prior_path}')

    @torch.no_grad()
    def transform(self, latents, source_preds, images, strength):
        attr_binary = (source_preds[:, self.attr_num] > 0.5).float()
        target_value = torch.where(
            attr_binary > 0.5,
            torch.full_like(attr_binary, -float(strength)),
            torch.full_like(attr_binary, 1.0 + float(strength)),
        )

        label_dist = self.labeldist(images, latents)
        zeros = torch.zeros(latents.size(0), latents.size(1), 1, device=latents.device)
        z, _ = self.flow(latents, label_dist, zeros)

        new_label_dist = label_dist.clone()
        local_idx = self.class_indices.index(self.attr_num)
        new_label_dist[:, local_idx] = target_value
        return self.flow(z, new_label_dist, reverse=True)


@torch.no_grad()
def evaluate_batch(args, editor, generator, attr_teacher, id_model, images, latents,
                   source_preds, attr_global_idx, eval_indices, attr_local_eval_idx,
                   strengths, image_rows):
    device = images.device
    batch = images.size(0)
    src_faces = generate_faces(generator, latents, args.img_size)

    src_logits, _ = attr_teacher(F.interpolate(src_faces, (256, 256), mode='bilinear', align_corners=False))
    src_probs_all = torch.sigmoid(src_logits)
    src_selected_probs = src_probs_all[:, eval_indices]
    src_target_prob = src_probs_all[:, attr_global_idx]
    src_target_binary = (src_target_prob > 0.5).float()
    final_target_binary = 1.0 - src_target_binary
    edit_direction = final_target_binary * 2.0 - 1.0

    src_id_feat = F.normalize(id_model.extract_features(src_faces), dim=1)

    save_start = 0
    save_count = 0
    if image_rows is not None:
        save_start = len(image_rows)
        save_count = min(args.save_images - save_start, batch)
        for b in range(save_count):
            image_rows.append([images[b].detach().cpu(), src_faces[b].detach().cpu()])

    rows = []
    for strength in strengths:
        edited_latents = editor.transform(latents, source_preds, images, strength)
        edited_faces = generate_faces(generator, edited_latents, args.img_size)

        edited_logits, _ = attr_teacher(
            F.interpolate(edited_faces, (256, 256), mode='bilinear', align_corners=False)
        )
        edited_probs_all = torch.sigmoid(edited_logits)
        edited_selected_probs = edited_probs_all[:, eval_indices]
        edited_target_prob = edited_probs_all[:, attr_global_idx]

        target_prob = src_target_prob * (1.0 - strength) + final_target_binary * strength
        edited_binary = (edited_target_prob > 0.5).float()
        target_success = (edited_binary == final_target_binary).float()
        target_gain = edit_direction * (edited_target_prob - src_target_prob)
        target_change = target_gain.clamp(min=0.0)
        max_possible_change = (final_target_binary - src_target_prob).abs().clamp(min=1e-6)
        target_gain_norm = (target_change / max_possible_change).clamp(min=0.0, max=1.0)
        target_abs_error = (edited_target_prob - final_target_binary).abs()
        effective_success = target_success * (target_gain >= args.min_target_gain).float()

        non_target = [i for i in range(len(eval_indices)) if i != attr_local_eval_idx]
        if non_target:
            leakage = (edited_selected_probs[:, non_target] - src_selected_probs[:, non_target]).abs().mean(dim=1)
            preserve_acc = (
                (edited_selected_probs[:, non_target] > 0.5).float()
                == (src_selected_probs[:, non_target] > 0.5).float()
            ).float().mean(dim=1)
        else:
            leakage = torch.zeros(batch, device=device)
            preserve_acc = torch.ones(batch, device=device)

        edited_id_feat = F.normalize(id_model.extract_features(edited_faces), dim=1)
        id_sim = (src_id_feat * edited_id_feat).sum(dim=1)

        delta = edited_latents - latents
        delta_rms = delta.pow(2).mean(dim=(1, 2)).sqrt()
        delta_coarse = delta[:, :4, :].pow(2).mean(dim=(1, 2)).sqrt()
        delta_middle = delta[:, 4:12, :].pow(2).mean(dim=(1, 2)).sqrt()
        delta_fine = delta[:, 12:, :].pow(2).mean(dim=(1, 2)).sqrt()

        balanced_score = (
            args.score_gain_weight * target_gain_norm
            + args.score_success_weight * effective_success
            + args.score_id_weight * id_sim
            + args.score_preserve_weight * preserve_acc
            - args.score_leak_weight * leakage
            - args.score_delta_weight * delta_rms
            - args.score_coarse_weight * delta_coarse
        )

        for b in range(batch):
            rows.append({
                'strength': strength,
                'target_src_prob': float(src_target_prob[b].detach().cpu()),
                'target_prob': float(target_prob[b].detach().cpu()),
                'target_final_binary': float(final_target_binary[b].detach().cpu()),
                'target_edit_prob': float(edited_target_prob[b].detach().cpu()),
                'target_success': float(target_success[b].detach().cpu()),
                'effective_success': float(effective_success[b].detach().cpu()),
                'target_abs_error': float(target_abs_error[b].detach().cpu()),
                'target_gain': float(target_gain[b].detach().cpu()),
                'target_change': float(target_change[b].detach().cpu()),
                'target_gain_norm': float(target_gain_norm[b].detach().cpu()),
                'leakage_l1': float(leakage[b].detach().cpu()),
                'preserve_acc': float(preserve_acc[b].detach().cpu()),
                'id_sim_real': float(id_sim[b].detach().cpu()),
                'delta_rms': float(delta_rms[b].detach().cpu()),
                'delta_coarse_rms': float(delta_coarse[b].detach().cpu()),
                'delta_middle_rms': float(delta_middle[b].detach().cpu()),
                'delta_fine_rms': float(delta_fine[b].detach().cpu()),
                'balanced_score': float(balanced_score[b].detach().cpu()),
            })

        if image_rows is not None:
            for b in range(save_count):
                image_rows[save_start + b].append(edited_faces[b].detach().cpu())

    return rows


def summarize(detail_rows):
    buckets = defaultdict(list)
    for row in detail_rows:
        buckets[(row['attribute'], row['strength'])].append(row)

    summary_rows = []
    for (attribute, strength), rows in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1])):
        summary_rows.append({
            'attribute': attribute,
            'strength': strength,
            'n': len(rows),
            'target_success': mean_or_zero([r['target_success'] for r in rows]),
            'effective_success': mean_or_zero([r['effective_success'] for r in rows]),
            'target_abs_error': mean_or_zero([r['target_abs_error'] for r in rows]),
            'target_gain': mean_or_zero([r['target_gain'] for r in rows]),
            'target_change': mean_or_zero([r['target_change'] for r in rows]),
            'target_gain_norm': mean_or_zero([r['target_gain_norm'] for r in rows]),
            'leakage_l1': mean_or_zero([r['leakage_l1'] for r in rows]),
            'preserve_acc': mean_or_zero([r['preserve_acc'] for r in rows]),
            'id_sim_real': mean_or_zero([r['id_sim_real'] for r in rows]),
            'delta_rms': mean_or_zero([r['delta_rms'] for r in rows]),
            'delta_coarse_rms': mean_or_zero([r['delta_coarse_rms'] for r in rows]),
            'delta_middle_rms': mean_or_zero([r['delta_middle_rms'] for r in rows]),
            'delta_fine_rms': mean_or_zero([r['delta_fine_rms'] for r in rows]),
            'balanced_score': mean_or_zero([r['balanced_score'] for r in rows]),
        })
    return summary_rows


def practical_score(row, args, id_floor=None):
    if id_floor is None:
        id_floor = args.best_min_id
    id_violation = max(0.0, float(id_floor) - float(row['id_sim_real']))
    leak_excess = max(0.0, float(row['leakage_l1']) - float(args.best_max_leakage))
    delta_excess = max(0.0, float(row['delta_rms']) - float(args.best_max_delta))
    return (
        args.practical_gain_weight * float(row['target_gain_norm'])
        + args.practical_success_weight * float(row['effective_success'])
        + args.practical_id_weight * float(row['id_sim_real'])
        + args.practical_preserve_weight * float(row['preserve_acc'])
        - args.practical_leak_weight * float(row['leakage_l1'])
        - args.practical_delta_weight * float(row['delta_rms'])
        - args.practical_coarse_weight * float(row['delta_coarse_rms'])
        - args.practical_id_violation_weight * id_violation
        - args.practical_leak_violation_weight * leak_excess
        - args.practical_delta_violation_weight * delta_excess
    )


def passes_filter(row, args, min_id):
    return (
        float(row['id_sim_real']) >= float(min_id)
        and float(row['effective_success']) >= float(args.best_min_effective_success)
        and float(row['target_gain_norm']) >= float(args.best_min_gain_norm)
        and float(row['leakage_l1']) <= float(args.best_max_leakage)
        and float(row['delta_rms']) <= float(args.best_max_delta)
    )


def annotate_and_select(summary_rows, args):
    for row in summary_rows:
        row['practical_score'] = practical_score(row, args)
        row['passes_strict_best_filter'] = float(passes_filter(row, args, args.best_min_id))
        row['passes_fallback_best_filter'] = float(passes_filter(row, args, args.best_fallback_min_id))

    by_attr = defaultdict(list)
    for row in summary_rows:
        by_attr[row['attribute']].append(row)

    selected = []
    for _, rows in by_attr.items():
        strict = [r for r in rows if passes_filter(r, args, args.best_min_id)]
        fallback = [r for r in rows if passes_filter(r, args, args.best_fallback_min_id)]
        if strict:
            pool = strict
            mode = f'strict(id>={args.best_min_id:.2f})'
            id_floor = args.best_min_id
        elif fallback:
            pool = fallback
            mode = f'fallback(id>={args.best_fallback_min_id:.2f})'
            id_floor = args.best_fallback_min_id
        else:
            pool = rows
            mode = 'unconstrained_practical_score'
            id_floor = args.best_fallback_min_id
        for row in pool:
            row['_selection_score'] = practical_score(row, args, id_floor=id_floor)
        best = max(pool, key=lambda r: r['_selection_score'])
        best['selection_mode'] = mode
        best['selection_score'] = best['_selection_score']
        selected.append(best)

    return summary_rows, selected


def save_strength_grids(output_dir, attr_name, image_rows, strengths):
    if not image_rows:
        return
    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    safe_name = attr_name.replace(' ', '_')
    for row_idx, row in enumerate(image_rows):
        grid = make_grid(torch.stack(row, dim=0), nrow=len(row), normalize=True, value_range=(-1, 1))
        save_image(grid, os.path.join(image_dir, f'{safe_name}_sample{row_idx:03d}.png'))


def main():
    parser = argparse.ArgumentParser(description='Evaluate original SDFlow labeldist/prior checkpoints.')
    parser.add_argument('--ckpt_dir', required=True, help='Directory with labeldist-* and prior-* checkpoints.')
    parser.add_argument('--ckpt_step', type=int, default=None, help='Checkpoint step, e.g. 13000. Default: latest.')
    parser.add_argument('--output_dir', default='./output/eval_original_sdflow_strength_sweep')

    parser.add_argument('--attribute_index', nargs='*', type=int, default=[15, 20, 31, 33, 39],
                        help='Attributes used to train the original checkpoint.')
    parser.add_argument('--attribute_names', nargs='*', default=None,
                        help='Names for --attribute_index.')
    parser.add_argument('--eval_attribute_index', nargs='*', type=int, default=None,
                        help='Subset to evaluate. Default: all --attribute_index.')
    parser.add_argument('--strengths', default='', help='Comma-separated strengths. Default: 0.1,...,1.0')

    parser.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    parser.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth')
    parser.add_argument('--index_file', default='./data/ffhq.txt')
    parser.add_argument('--image_root', default='./data/FFHQ')
    parser.add_argument('--stylegan2_weights', default='./data/stylegan2-ffhq-config-f.pt')
    parser.add_argument('--attribute_weights', default='./data/r34_a40_age_256_classifier.pth')

    parser.add_argument('--flow_modules', default='512-512-512-512-512')
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--labeldist_backbone', default='auto',
                        help='AttributeEstimator backbone. Use auto, resnet34, resnet50, etc.')

    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=64)
    parser.add_argument('--save_images', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--min_target_gain', type=float, default=0.08)
    parser.add_argument('--score_gain_weight', type=float, default=1.0)
    parser.add_argument('--score_success_weight', type=float, default=0.5)
    parser.add_argument('--score_id_weight', type=float, default=1.0)
    parser.add_argument('--score_preserve_weight', type=float, default=0.2)
    parser.add_argument('--score_leak_weight', type=float, default=2.0)
    parser.add_argument('--score_delta_weight', type=float, default=1.0)
    parser.add_argument('--score_coarse_weight', type=float, default=3.0)

    parser.add_argument('--best_min_id', type=float, default=0.75)
    parser.add_argument('--best_fallback_min_id', type=float, default=0.70)
    parser.add_argument('--best_min_effective_success', type=float, default=0.30)
    parser.add_argument('--best_min_gain_norm', type=float, default=0.25)
    parser.add_argument('--best_max_leakage', type=float, default=0.12)
    parser.add_argument('--best_max_delta', type=float, default=0.20)

    parser.add_argument('--practical_gain_weight', type=float, default=0.70)
    parser.add_argument('--practical_success_weight', type=float, default=0.40)
    parser.add_argument('--practical_id_weight', type=float, default=1.50)
    parser.add_argument('--practical_preserve_weight', type=float, default=0.30)
    parser.add_argument('--practical_leak_weight', type=float, default=3.00)
    parser.add_argument('--practical_delta_weight', type=float, default=2.00)
    parser.add_argument('--practical_coarse_weight', type=float, default=6.00)
    parser.add_argument('--practical_id_violation_weight', type=float, default=5.00)
    parser.add_argument('--practical_leak_violation_weight', type=float, default=2.00)
    parser.add_argument('--practical_delta_violation_weight', type=float, default=2.00)
    args = parser.parse_args()

    for idx in args.attribute_index:
        if idx < 0 or idx >= 40:
            parser.error(f'Invalid --attribute_index value {idx}; expected 0..39.')
    eval_indices = args.eval_attribute_index or args.attribute_index
    for idx in eval_indices:
        if idx not in args.attribute_index:
            parser.error(f'--eval_attribute_index {idx} is not in trained --attribute_index {args.attribute_index}.')

    strengths = parse_strengths(args.strengths)
    name_by_attr = build_name_map(args.attribute_index, args.attribute_names)
    device = torch.device(args.device)

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

    print('Loading attribute teacher...')
    attr_teacher = AttributeClassifier(backbone='r34')
    attr_teacher.load_state_dict(load_network(args.attribute_weights))
    attr_teacher = attr_teacher.to(device).eval()

    print('Loading ID model...')
    id_model = IDLoss(crop=True).to(device).eval()

    detail_rows = []
    for attr_global_idx in eval_indices:
        attr_name = name_by_attr[int(attr_global_idx)]
        print(f'Evaluating {attr_name} ({attr_global_idx})...')
        editor = OriginalSDFlowEditor(args, int(attr_global_idx), device)

        image_rows = [] if args.save_images > 0 else None
        sample_offset = 0
        for images, latents, source_preds in tqdm(loader, desc=attr_name):
            images = images.to(device, non_blocking=True)
            latents = latents.to(device, non_blocking=True)
            source_preds = source_preds.to(device, non_blocking=True)

            batch_rows = evaluate_batch(
                args,
                editor,
                generator,
                attr_teacher,
                id_model,
                images,
                latents,
                source_preds,
                int(attr_global_idx),
                [int(v) for v in eval_indices],
                [int(v) for v in eval_indices].index(int(attr_global_idx)),
                strengths,
                image_rows,
            )
            for local_i, row in enumerate(batch_rows):
                row['attribute'] = attr_name
                row['attribute_index'] = int(attr_global_idx)
                row['sample_index'] = sample_offset + (local_i % images.size(0))
                detail_rows.append(row)
            sample_offset += images.size(0)

        save_strength_grids(args.output_dir, attr_name, image_rows, strengths)

    summary_rows = summarize(detail_rows)
    summary_rows, selected = annotate_and_select(summary_rows, args)
    detail_path = os.path.join(args.output_dir, 'metrics_detail.csv')
    summary_path = os.path.join(args.output_dir, 'metrics_summary.csv')
    write_csv(detail_path, detail_rows)
    write_csv(summary_path, summary_rows)

    print(f'Wrote {detail_path}')
    print(f'Wrote {summary_path}')
    print('\nSummary:')
    for row in summary_rows:
        print(
            f"{row['attribute']:>12} s={row['strength']:.1f} "
            f"succ={row['target_success']:.3f} "
            f"eff={row['effective_success']:.3f} "
            f"gain={row['target_gain']:.4f} "
            f"gainN={row['target_gain_norm']:.3f} "
            f"leak={row['leakage_l1']:.4f} "
            f"id={row['id_sim_real']:.4f} "
            f"delta={row['delta_rms']:.4f} "
            f"score={row['balanced_score']:.4f} "
            f"pScore={row['practical_score']:.4f} "
            f"passS={int(row['passes_strict_best_filter'])}"
        )

    print('\nBest strength by raw balanced_score (edit-strength-biased):')
    by_attr = defaultdict(list)
    for row in summary_rows:
        by_attr[row['attribute']].append(row)
    for attr, rows in by_attr.items():
        best = max(rows, key=lambda r: r['balanced_score'])
        print(
            f"{attr:>12}: s={best['strength']:.1f} "
            f"score={best['balanced_score']:.4f} "
            f"eff={best['effective_success']:.3f} "
            f"gain={best['target_gain']:.4f} "
            f"id={best['id_sim_real']:.4f} "
            f"leak={best['leakage_l1']:.4f}"
        )

    print('\nRecommended strength by identity-constrained practical_score:')
    for best in selected:
        print(
            f"{best['attribute']:>12}: s={best['strength']:.1f} "
            f"mode={best['selection_mode']} "
            f"pScore={best['selection_score']:.4f} "
            f"eff={best['effective_success']:.3f} "
            f"gain={best['target_gain']:.4f} "
            f"gainN={best['target_gain_norm']:.3f} "
            f"id={best['id_sim_real']:.4f} "
            f"leak={best['leakage_l1']:.4f} "
            f"delta={best['delta_rms']:.4f}"
        )


if __name__ == '__main__':
    main()
