"""
Evaluate SDFlow/LAG-DOF checkpoints across edit strengths 0.1 ... 1.0.

Example:
    python evaluation/eval_strength_sweep.py \
        --ckpt_dir ./output/SDFlow/default/save_models \
        --velocity_field lag_dof \
        --attribute_index 15 20 39 \
        --attribute_names Eyeglasses Gender Young \
        --max_samples 64 \
        --save_images 8

Outputs:
    output/eval_strength_sweep/metrics_detail.csv
    output/eval_strength_sweep/metrics_summary.csv
    output/eval_strength_sweep/images/*.png
    output/eval_strength_sweep/plots/curves_<attribute>.png       (per attribute)
    output/eval_strength_sweep/plots/curves_all_attributes.png    (overview)

The curve plots reproduce the Identity/Attribute Preservation vs Editing
Accuracy figure from the SDFlow paper, computed from this script's own
metrics so they reflect the real SDFlow.transform() path (direction bank /
layer mask / LAG-DOF gate) and the same ArcFace identity model used
elsewhere in this repo. Use --no-plot to skip, or --run_name to change the
legend label. To compare several runs (e.g. original vs lag_dof, or against
external baselines) on the same axes, see plot_strength_curves.py.
"""

import argparse
import csv
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

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from common.ops import load_network
from common.id_loss import IDLoss
from models.attribute_estimator import AttributeClassifier
from models.dataset import SDFlowDataset
from models.editor import SDFlow
from models.stylegan2.model import Generator


def parse_strengths(value):
    if value:
        return [float(v) for v in value.split(',')]
    return [round(i / 10.0, 1) for i in range(1, 11)]


def mean_or_zero(values):
    return sum(values) / len(values) if values else 0.0


def attr_name_map(indices, names):
    if names and len(names) != len(indices):
        raise ValueError('--attribute_names must have the same length as --attribute_index.')
    if names:
        return {int(idx): name for idx, name in zip(indices, names)}
    return {int(idx): f'attr_{idx}' for idx in indices}


def tensor_to_float_list(x):
    return [float(v) for v in x.detach().cpu().view(-1)]


@torch.no_grad()
def generate_faces(generator, latents, img_size):
    faces = generator([latents], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
    if faces.size(-1) != img_size:
        faces = F.interpolate(faces, (img_size, img_size), mode='bilinear', align_corners=False)
    return faces


@torch.no_grad()
def evaluate_batch(args, transformer, generator, attr_teacher, id_model,
                   images, latents, source_preds, attr_global_idx, attr_local_idx, strengths,
                   image_rows):
    device = images.device
    batch = images.size(0)
    src_faces = generate_faces(generator, latents, args.img_size)

    # Use src_faces (generator reconstruction of original latents) as the source
    # reference so that source and edited measurements are in the same domain
    # (both are StyleGAN2 outputs). Using raw `images` here would introduce a
    # real-vs-generated domain gap that systematically lowers id_sim_real.
    src_teacher_logits, _ = attr_teacher(F.interpolate(src_faces, (256, 256), mode='bilinear', align_corners=False))
    src_probs_all = torch.sigmoid(src_teacher_logits)
    src_selected_probs = src_probs_all[:, args.attribute_index]
    src_target_prob = src_probs_all[:, attr_global_idx]
    src_target_binary = (src_target_prob > 0.5).float()

    src_id_feat = F.normalize(id_model.extract_features(src_faces), dim=1)
    src_recon_id_feat = src_id_feat  # same reference; kept for metric naming consistency
    rows = []
    save_start = 0
    save_count = 0
    if image_rows is not None:
        save_start = len(image_rows)
        save_count = min(args.save_images - save_start, batch)
        for b in range(save_count):
            image_rows.append([images[b].detach().cpu(), src_faces[b].detach().cpu()])

    for strength in strengths:
        transformer.scale = strength
        edited_latents = transformer.transform(latents, source_preds, images)
        edited_faces = generate_faces(generator, edited_latents, args.img_size)

        edited_logits, _ = attr_teacher(
            F.interpolate(edited_faces, (256, 256), mode='bilinear', align_corners=False)
        )
        edited_probs_all = torch.sigmoid(edited_logits)
        edited_selected_probs = edited_probs_all[:, args.attribute_index]
        edited_target_prob = edited_probs_all[:, attr_global_idx]

        # ------------------------------------------------------------
        # Improved evaluation:
        #   1) Final target is the flipped attribute side, independent of strength.
        #   2) Strength only controls how far the edit moves toward that target.
        #   3) We measure both classifier success and actual directional progress.
        #
        # This avoids the misleading case where a tiny edit is counted as
        # "successful" simply because it is already near the 0.5 threshold.
        # ------------------------------------------------------------
        final_target_binary = 1.0 - src_target_binary
        edit_direction = final_target_binary * 2.0 - 1.0  # +1: add / young-side, -1: remove / old-side

        # Optional interpolated target for reference only.
        target_prob = src_target_prob * (1.0 - strength) + final_target_binary * strength

        edited_binary = (edited_target_prob > 0.5).float()
        target_success = (edited_binary == final_target_binary).float()

        # How much the edited probability moved in the desired direction.
        target_gain = edit_direction * (edited_target_prob - src_target_prob)
        target_change = target_gain.clamp(min=0.0)

        # Normalize by the maximum possible movement toward the final binary target.
        max_possible_change = (final_target_binary - src_target_prob).abs().clamp(min=1e-6)
        target_gain_norm = (target_change / max_possible_change).clamp(min=0.0, max=1.0)

        # Final target error, not the interpolated target error.
        target_abs_error = (edited_target_prob - final_target_binary).abs()

        # A stricter success: it must cross to the target side AND visibly move enough.
        effective_success = target_success * (target_gain >= args.min_target_gain).float()

        non_target = [i for i in range(len(args.attribute_index)) if i != attr_local_idx]
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
        id_sim_real = (src_id_feat * edited_id_feat).sum(dim=1)
        id_sim_recon = (src_recon_id_feat * edited_id_feat).sum(dim=1)

        delta = edited_latents - latents
        assert delta.shape[1] == 18, (
            f"Expected 18 W+ layers (StyleGAN2-FFHQ-1024), got {delta.shape[1]}. "
            "Update the coarse/middle/fine split indices if using a different config."
        )
        delta_rms = delta.pow(2).mean(dim=(1, 2)).sqrt()
        delta_coarse = delta[:, :4, :].pow(2).mean(dim=(1, 2)).sqrt()
        delta_middle = delta[:, 4:12, :].pow(2).mean(dim=(1, 2)).sqrt()
        delta_fine = delta[:, 12:, :].pow(2).mean(dim=(1, 2)).sqrt()

        # Composite score for choosing a practical edit strength.
        # Higher is better. It rewards actual directional target movement and ID preservation,
        # while penalizing leakage and large latent movement, especially coarse-layer movement.
        balanced_score = (
            args.score_gain_weight * target_gain_norm
            + args.score_success_weight * effective_success
            + args.score_id_weight * id_sim_real
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
                'id_sim_real': float(id_sim_real[b].detach().cpu()),
                'id_sim_recon': float(id_sim_recon[b].detach().cpu()),
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


def summarize(detail_rows, attr_names):
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
            'id_sim_recon': mean_or_zero([r['id_sim_recon'] for r in rows]),
            'delta_rms': mean_or_zero([r['delta_rms'] for r in rows]),
            'delta_coarse_rms': mean_or_zero([r['delta_coarse_rms'] for r in rows]),
            'delta_middle_rms': mean_or_zero([r['delta_middle_rms'] for r in rows]),
            'delta_fine_rms': mean_or_zero([r['delta_fine_rms'] for r in rows]),
            'balanced_score': mean_or_zero([r['balanced_score'] for r in rows]),
        })
    return summary_rows


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_strength_grids(output_dir, attr_name, image_rows, strengths):
    if not image_rows:
        return
    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    # Column order: real | reconstructed | s=0.1 | s=0.2 | ...
    strength_labels = ['real', 'recon'] + [f's={s:.1f}' for s in strengths]
    safe_name = attr_name.replace(' ', '_')
    for row_idx, row in enumerate(image_rows):
        grid = make_grid(torch.stack(row, dim=0), nrow=len(row), normalize=True, value_range=(-1, 1))
        label = '_'.join(strength_labels[:len(row)])
        save_image(grid, os.path.join(image_dir, f'{safe_name}_sample{row_idx:03d}_{label}.png'))


def plot_editing_curves(summary_rows, output_dir, run_name='Ours', x_metric='target_success'):
    """Identity/Attribute Preservation vs Editing Accuracy curves (reproduces
    the Fig. 4 style plot from the SDFlow paper), computed directly from this
    script's own summary rows.

    Unlike evaluation/eval_curves.py, this plots numbers that come from the
    real SDFlow.transform() path (direction bank / layer mask / LAG-DOF gate
    included) and from the same ArcFace identity model used everywhere else
    in this repo, so it is safe to use for internal ablation comparisons.
    """
    if plt is None:
        print('[plot] matplotlib not installed; skipping curve plots (pip install matplotlib).')
        return

    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    by_attr = defaultdict(list)
    for row in summary_rows:
        by_attr[row['attribute']].append(row)

    colors = plt.cm.tab10.colors
    overview_fig, overview_axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for color_idx, (attr_name, rows) in enumerate(sorted(by_attr.items())):
        rows = sorted(rows, key=lambda r: r['strength'])
        # Sort by the achieved editing accuracy (not raw strength) so the line
        # reads left-to-right the same way the paper figure does, since
        # strength does not map 1:1 onto the resulting classifier accuracy.
        rows = sorted(rows, key=lambda r: r[x_metric])
        edit_acc = [r[x_metric] * 100.0 for r in rows]
        id_sim = [r['id_sim_real'] for r in rows]
        attr_pres = [r['preserve_acc'] * 100.0 for r in rows]

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].plot(edit_acc, id_sim, marker='o', color='#c0392b', linewidth=2, label=run_name)
        axes[0].set_xlabel('Editing Accuracy (%)')
        axes[0].set_ylabel('Cosine Similarity')
        axes[0].set_title('Identity Preservation')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(edit_acc, attr_pres, marker='o', color='#c0392b', linewidth=2, label=run_name)
        axes[1].set_xlabel('Editing Accuracy (%)')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Attribute Preservation')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.suptitle(attr_name)
        fig.tight_layout()
        safe_name = attr_name.replace(' ', '_')
        fig_path = os.path.join(plot_dir, f'curves_{safe_name}.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[plot] wrote {fig_path}')

        color = colors[color_idx % len(colors)]
        overview_axes[0].plot(edit_acc, id_sim, marker='o', color=color, linewidth=2, label=attr_name)
        overview_axes[1].plot(edit_acc, attr_pres, marker='o', color=color, linewidth=2, label=attr_name)

    overview_axes[0].set_xlabel('Editing Accuracy (%)')
    overview_axes[0].set_ylabel('Cosine Similarity')
    overview_axes[0].set_title('Identity Preservation')
    overview_axes[0].grid(True, alpha=0.3)
    overview_axes[0].legend()

    overview_axes[1].set_xlabel('Editing Accuracy (%)')
    overview_axes[1].set_ylabel('Accuracy (%)')
    overview_axes[1].set_title('Attribute Preservation')
    overview_axes[1].grid(True, alpha=0.3)
    overview_axes[1].legend()

    overview_fig.suptitle(f'{run_name}: all attributes')
    overview_fig.tight_layout()
    overview_path = os.path.join(plot_dir, 'curves_all_attributes.png')
    overview_fig.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close(overview_fig)
    print(f'[plot] wrote {overview_path}')


def compute_practical_score(row, args, id_floor=None):
    """Score used for selecting a visually useful edit strength.

    balanced_score rewards edit strength. practical_score additionally penalizes
    identity drift and excessive latent movement so high-strength edits do not
    win only because the classifier confidence is high.
    """
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


def passes_best_filter(row, args, min_id):
    """Hard constraints for recommended scale selection."""
    return (
        float(row['id_sim_real']) >= float(min_id)
        and float(row['effective_success']) >= float(args.best_min_effective_success)
        and float(row['target_gain_norm']) >= float(args.best_min_gain_norm)
        and float(row['leakage_l1']) <= float(args.best_max_leakage)
        and float(row['delta_rms']) <= float(args.best_max_delta)
    )


def annotate_summary_for_selection(summary_rows, args):
    for row in summary_rows:
        row['practical_score'] = compute_practical_score(row, args)
        row['passes_strict_best_filter'] = float(passes_best_filter(row, args, args.best_min_id))
        row['passes_fallback_best_filter'] = float(passes_best_filter(row, args, args.best_fallback_min_id))
    return summary_rows


def select_recommended_strengths(summary_rows, args):
    """Select one recommended strength per attribute.

    Priority:
      1. Use strict identity-preserving candidates.
      2. If none exist, use fallback identity threshold.
      3. If still none exist, use all candidates but rank by practical_score.
    """
    by_attr = defaultdict(list)
    for row in summary_rows:
        by_attr[row['attribute']].append(row)

    selected = []
    for _, rows in by_attr.items():
        strict = [r for r in rows if passes_best_filter(r, args, args.best_min_id)]
        fallback = [r for r in rows if passes_best_filter(r, args, args.best_fallback_min_id)]

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

        for r in pool:
            r['_selection_score'] = compute_practical_score(r, args, id_floor=id_floor)
        best = max(pool, key=lambda r: r['_selection_score'])
        best['selection_mode'] = mode
        best['selection_score'] = best['_selection_score']
        selected.append(best)

    return selected


def main():
    parser = argparse.ArgumentParser(description='Evaluate SDFlow edit strengths and save sweep images.')
    parser.add_argument('--ckpt_dir', required=True, help='Directory containing conditioner-* and prior-* checkpoints.')
    parser.add_argument('--ckpt_step', type=int, default=None,
                        help='Checkpoint step to evaluate, e.g. 20000. Default: latest checkpoint.')
    parser.add_argument('--output_dir', default='./output/eval_strength_sweep')
    parser.add_argument('--velocity_field', default='lag_dof', choices=['original', 'lag', 'dof', 'lag_dof'])
    parser.add_argument('--flow_modules', type=str, default='512-512-512-512-512')
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--lag_gate_hidden_dim', type=int, default=64)
    parser.add_argument('--lag_gate_init_bias', type=float, default=-0.5)
    parser.add_argument('--id_cond_dim', type=int, default=32)
    parser.add_argument('--id_cond_scale', type=float, default=0.25)
    parser.add_argument('--attr_backbone', default='resnet50')
    parser.add_argument('--conditioner_backbone', default='resnet',
                        choices=['resnet', 'clip', 'resnet_clip'])
    parser.add_argument('--clip_model', default='ViT-B/32')
    parser.add_argument('--fused_hidden_dim', type=int, default=256)
    parser.add_argument('--direction_bank_path', default=None, type=str,
                        help='Path to precomputed Attribute Direction Bank (.pth).')
    parser.add_argument('--direction_residual_scale', type=float, default=0.05)
    parser.add_argument('--direction_freeze', '--direction-freeze',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--bypass_glasses_direction_bank',
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Bypass Direction Bank for Eyeglasses during evaluation/inference.')
    parser.add_argument('--age_direction_scale', type=float, default=0.55,
                        help='Age/Young-only multiplier for Direction Bank guided delta.')
    parser.add_argument('--age_coarse_layer_scale', type=float, default=0.75,
                        help='Age/Young-only scale for W+ coarse layers 0:4.')
    parser.add_argument('--age_middle_layer_scale', type=float, default=1.0,
                        help='Age/Young-only scale for W+ middle layers 4:9.')
    parser.add_argument('--age_fine_layer_scale', type=float, default=0.45,
                        help='Age/Young-only scale for W+ fine layers 9:18.')
    parser.add_argument('--age_delta_max_norm', type=float, default=10.0,
                        help='Age/Young-only max norm for final guided W+ delta. Set <=0 to disable.')

    parser.add_argument('--attribute_index', nargs='*', default=[15, 20, 39], type=int)
    parser.add_argument('--attribute_names', nargs='*', default=None)
    parser.add_argument('--strengths', default='', help='Comma-separated strengths. Default: 0.1,0.2,...,1.0')

    parser.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    parser.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth')
    parser.add_argument('--index_file', default='./data/ffhq.txt')
    parser.add_argument('--image_root', default='data/FFHQ')
    parser.add_argument('--stylegan2_weights', default='./data/stylegan2-ffhq-config-f.pt')
    parser.add_argument('--attribute_weights', default='./data/r34_a40_age_256_classifier.pth')

    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=64)
    parser.add_argument('--save_images', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    # Paper-style curve plots (Identity/Attribute Preservation vs Editing Accuracy).
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=True,
                        help='Write Fig.4-style curve plots to <output_dir>/plots/.')
    parser.add_argument('--run_name', default='Ours',
                        help='Legend label for this run in the plotted curves.')
    parser.add_argument('--plot_x_metric', default='target_success',
                        choices=['target_success', 'effective_success'],
                        help='Which success metric to treat as "Editing Accuracy" on the X axis.')

    # Scoring options.
    # min_target_gain prevents tiny edits near the classifier boundary from being counted as useful.
    parser.add_argument('--min_target_gain', type=float, default=0.08)
    parser.add_argument('--score_gain_weight', type=float, default=1.0)
    parser.add_argument('--score_success_weight', type=float, default=0.5)
    parser.add_argument('--score_id_weight', type=float, default=1.0)
    parser.add_argument('--score_preserve_weight', type=float, default=0.2)
    parser.add_argument('--score_leak_weight', type=float, default=2.0)
    parser.add_argument('--score_delta_weight', type=float, default=1.0)
    parser.add_argument('--score_coarse_weight', type=float, default=3.0)

    # Recommended-scale selection. These constraints prevent high-strength edits
    # from being selected when they only win because target gain is high but
    # identity preservation is poor.
    parser.add_argument('--best_min_id', type=float, default=0.75)
    parser.add_argument('--best_fallback_min_id', type=float, default=0.70)
    parser.add_argument('--best_min_effective_success', type=float, default=0.30)
    parser.add_argument('--best_min_gain_norm', type=float, default=0.25)
    parser.add_argument('--best_max_leakage', type=float, default=0.12)
    parser.add_argument('--best_max_delta', type=float, default=0.20)

    # Practical score used inside the constrained candidate set.
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

    invalid = [i for i in args.attribute_index if not (0 <= i < 40)]
    if invalid:
        parser.error(f"--attribute_index values must be in [0, 39]; got: {invalid}")

    device = torch.device(args.device)
    strengths = parse_strengths(args.strengths)
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

    print('Loading attribute teacher...')
    attr_teacher = AttributeClassifier(backbone='r34')
    attr_teacher.load_state_dict(load_network(args.attribute_weights))
    attr_teacher = attr_teacher.to(device).eval()

    print('Loading ID model...')
    id_model = IDLoss(crop=True).to(device).eval()

    detail_rows = []
    for attr_local_idx, attr_global_idx in enumerate(args.attribute_index):
        attr_name = name_by_attr[int(attr_global_idx)]
        print(f'Evaluating {attr_name} ({attr_global_idx})...')
        transformer = SDFlow(
            ckpt_dir=args.ckpt_dir,
            attr_num=int(attr_global_idx),
            attr_list=args.attribute_index,
            scale=0.0,
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
            age_direction_scale=args.age_direction_scale,
            age_coarse_layer_scale=args.age_coarse_layer_scale,
            age_middle_layer_scale=args.age_middle_layer_scale,
            age_fine_layer_scale=args.age_fine_layer_scale,
            age_delta_max_norm=args.age_delta_max_norm,
        )

        image_rows = [] if args.save_images > 0 else None
        sample_offset = 0
        for images, latents, source_preds in tqdm(loader, desc=attr_name):
            images = images.to(device, non_blocking=True)
            latents = latents.to(device, non_blocking=True)
            source_preds = source_preds.to(device, non_blocking=True)

            batch_rows = evaluate_batch(
                args,
                transformer,
                generator,
                attr_teacher,
                id_model,
                images,
                latents,
                source_preds,
                int(attr_global_idx),
                attr_local_idx,
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

    summary_rows = summarize(detail_rows, name_by_attr)
    summary_rows = annotate_summary_for_selection(summary_rows, args)
    detail_path = os.path.join(args.output_dir, 'metrics_detail.csv')
    summary_path = os.path.join(args.output_dir, 'metrics_summary.csv')
    write_csv(detail_path, detail_rows)
    write_csv(summary_path, summary_rows)

    print(f'Wrote {detail_path}')
    print(f'Wrote {summary_path}')

    if args.plot:
        plot_editing_curves(
            summary_rows,
            args.output_dir,
            run_name=args.run_name,
            x_metric=args.plot_x_metric,
        )
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
    for best in select_recommended_strengths(summary_rows, args):
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
