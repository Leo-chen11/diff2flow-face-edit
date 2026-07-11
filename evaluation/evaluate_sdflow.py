"""
SDFlow Evaluation Script (v2 — independent judges)

Fixes over v1:
  1. Independent judges. The training-time attribute teacher (r34 classifier)
     and training-time ArcFace are still reported, but only as *reference*
     columns. The headline metrics come from models never used in training:
       - Attribute:  CLIP zero-shot judge (default ViT-L/14, different from the
         ViT-B/32 used by --use_clip_prompt_loss / clip conditioner), or an
         optional second classifier checkpoint via --independent_attr_weights.
       - Identity:   facenet-pytorch InceptionResnetV1 (VGGFace2), different
         from the insightface ArcFace used by id_loss during training.
  2. Strict success definition: the edited score must cross the 0.5 decision
     boundary (optionally with --success_margin), instead of merely moving
     0.05 in the right direction. The old lenient accuracy is still logged
     as acc_lenient for comparison with previous runs.
  3. --bypass_glasses_direction_bank now defaults to False, so all attributes
     are evaluated through the SAME pipeline. Pass the flag explicitly if you
     want the old behavior.
  4. Perceptual quality: LPIPS between the source reconstruction and the
     edited image, plus an "inversion gap" reference (real image vs e4e
     reconstruction, both LPIPS and independent-ID) so you can see how much
     of the quality ceiling is eaten by inversion before any editing happens.
     Optional FID via --compute_fid (needs torchmetrics + torch-fidelity).

Optional extra dependencies (script degrades gracefully without them):
  pip install lpips facenet-pytorch
  pip install git+https://github.com/openai/CLIP.git
  pip install torchmetrics torch-fidelity        # only for --compute_fid

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
import torch.nn as nn
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
# Independent judges (never used as training losses)
# ---------------------------------------------------------------------------

class CLIPAttributeJudge(nn.Module):
    """Zero-shot CLIP attribute scorer, independent from the training teacher.

    Deliberately uses prompt wording different from models/clip_prompt_loss.py
    and (by default) a different CLIP architecture, to reduce judge/teacher
    overlap when the run was trained with --use_clip_prompt_loss.
    """

    # (positive = "attribute present" in CelebA polarity, negative)
    PROMPTS = {
        15: ("a headshot of a person who is wearing glasses",
             "a headshot of a person who is not wearing glasses"),
        20: ("a headshot of a man",
             "a headshot of a woman"),
        39: ("a headshot of a young adult",
             "a headshot of an elderly senior person"),
    }
    DEFAULT_PROMPTS = ("a headshot of a person",
                       "a headshot of a person")

    def __init__(self, attribute_index, model_name='ViT-L/14', device='cuda'):
        super().__init__()
        import clip  # raises ImportError -> caller decides how to degrade
        self.model, _ = clip.load(model_name, device=device, jit=False)
        self.model = self.model.float().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.attribute_index = [int(i) for i in attribute_index]

        with torch.no_grad():
            text_feats = []
            for idx in self.attribute_index:
                pos, neg = self.PROMPTS.get(idx, self.DEFAULT_PROMPTS)
                tokens = clip.tokenize([pos, neg]).to(device)
                feats = self.model.encode_text(tokens).float()
                text_feats.append(F.normalize(feats, dim=-1))
            # (A, 2, D)
            self.register_buffer('text_feats', torch.stack(text_feats, dim=0))

    @torch.no_grad()
    def scores(self, images):
        """images: [-1, 1] tensor (B, 3, H, W) -> (B, A) prob attribute present."""
        x = (images + 1.0) * 0.5
        x = F.interpolate(x, (224, 224), mode='bicubic', align_corners=False)
        mean = x.new_tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1)
        std = x.new_tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1)
        x = (x - mean) / std
        img_feat = F.normalize(self.model.encode_image(x).float(), dim=-1)  # (B, D)
        # (B, A, 2) similarity to pos/neg prompt of each attribute
        logits = 100.0 * torch.einsum('bd,akd->bak', img_feat, self.text_feats)
        return torch.softmax(logits, dim=-1)[:, :, 0]


class IndependentIDJudge(nn.Module):
    """FaceNet (InceptionResnetV1, VGGFace2) identity embedder — a different
    architecture and training set from the insightface ArcFace used in id_loss."""

    def __init__(self, device='cuda'):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1  # ImportError handled by caller
        self.net = InceptionResnetV1(pretrained='vggface2').to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def extract(self, x):
        """x: [-1, 1] tensor. Same center-face crop convention as IDLoss."""
        w = x.size(-1)
        scale = lambda v: int(v * w / 256)
        crop_h, x1, x2 = scale(188), scale(35), scale(32)
        x = x[:, :, x1:x1 + crop_h, x2:x2 + crop_h]
        x = F.interpolate(x, size=160, mode='bilinear', align_corners=False)
        return F.normalize(self.net(x), dim=1)


def build_optional_judges(args, attribute_index, id_criterion):
    """Instantiate independent judges; degrade gracefully with clear warnings."""
    device = 'cuda'
    clip_judge, indep_id, lpips_fn, indep_teacher = None, None, None, None

    try:
        clip_judge = CLIPAttributeJudge(attribute_index, args.clip_judge_model, device)
        print(f'[Judge] CLIP attribute judge: {args.clip_judge_model}')
    except ImportError:
        print('[WARN] OpenAI CLIP not installed -> no independent attribute judge. '
              'pip install git+https://github.com/openai/CLIP.git')

    try:
        indep_id = IndependentIDJudge(device)
        # If the training IDLoss itself fell back to facenet (input_size 160),
        # this judge is NOT independent -- say so loudly instead of silently
        # reporting a rigged number.
        if getattr(id_criterion, 'input_size', 112) == 160:
            print('[WARN] Training IDLoss is ALSO facenet (insightface unavailable). '
                  'id_indep below is NOT independent from the training identity loss.')
        else:
            print('[Judge] Independent ID judge: facenet InceptionResnetV1 (VGGFace2)')
    except ImportError:
        print('[WARN] facenet-pytorch not installed -> no independent ID metric. '
              'pip install facenet-pytorch')

    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
        for p in lpips_fn.parameters():
            p.requires_grad_(False)
        print('[Judge] LPIPS (alex) enabled')
    except ImportError:
        print('[WARN] lpips not installed -> no perceptual metric. pip install lpips')

    if args.independent_attr_weights:
        indep_teacher = AttributeClassifier(backbone=args.independent_attr_backbone)
        indep_teacher.load_state_dict(load_network(args.independent_attr_weights))
        indep_teacher.to(device).eval()
        for p in indep_teacher.parameters():
            p.requires_grad_(False)
        print(f'[Judge] Independent classifier: {args.independent_attr_weights}')

    return clip_judge, indep_id, lpips_fn, indep_teacher


def build_fid(args):
    if not args.compute_fid:
        return None
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid = FrechetInceptionDistance(feature=2048, normalize=True).cuda()
        print('[Judge] FID enabled (source recon = real set, edited = fake set)')
        return fid
    except ImportError:
        print('[WARN] --compute_fid requires torchmetrics + torch-fidelity; skipping FID.')
        return None


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

        if args.override_residual_scale is not None:
            # Diagnostic knob: the trained residual_scale is typically frozen
            # near its 0.05 init (low-lr param group + softplus reparam), which
            # makes the final delta ~95% dataset-level direction. This override
            # lets you probe at eval time how much per-sample flow contribution
            # helps, without retraining or editing checkpoints.
            ov = torch.full_like(direction_bank.residual_scale_raw,
                                 float(args.override_residual_scale))
            with torch.no_grad():
                direction_bank.residual_scale_raw.copy_(torch.log(torch.expm1(ov.clamp(min=1e-6))))
            print(f'[Override] direction bank residual_scale forced to '
                  f'{args.override_residual_scale} for ALL attributes '
                  f'(trained value ignored)')

    # ── StyleGAN2 ─────────────────────────────────────────────────────────
    ckpt = torch.load(args.stygan2_weights, map_location='cpu')
    G = Generator(size=1024, style_dim=512, n_mlp=8)
    G.load_state_dict(ckpt['g_ema'])
    G.to(device).eval()
    for p in G.parameters():
        p.requires_grad_(False)

    # ── IDLoss (training judge, kept as reference) ────────────────────────
    id_criterion = IDLoss(crop=True).to(device).eval()
    for p in id_criterion.parameters():
        p.requires_grad_(False)

    # ── Attribute teacher (training judge, kept as reference) ─────────────
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
                          attr_global_idx=None, bypass_glasses_direction_bank=False):
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
# Metric helpers
# ---------------------------------------------------------------------------

def strict_success(src_score, edit_score, margin):
    """Edited score must cross the 0.5 decision boundary (by `margin`)."""
    if src_score > 0.5:
        return edit_score < (0.5 - margin)
    return edit_score > (0.5 + margin)


def lenient_success(src_score, edit_score):
    """Old v1 definition, kept for comparison with previous runs."""
    return (edit_score < src_score - 0.05) if src_score > 0.5 \
        else (edit_score > src_score + 0.05)


def is_clear(score, low=0.35, high=0.65):
    return score > high or score < low


def _summ(values):
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        'mean': float(arr.mean()),
        'p10': float(np.percentile(arr, 10)),
        'p50': float(np.percentile(arr, 50)),
        'p90': float(np.percentile(arr, 90)),
        'n': int(arr.size),
    }


def _fmt(summary, pct=False):
    if summary is None:
        return '     -- '
    v = summary['mean'] * (100.0 if pct else 1.0)
    return f'{v:>7.1f}%' if pct else f'{v:>7.4f} '


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(args):
    prior, conditioner, G, id_criterion, attr_teacher, \
        attribute_index, direction_bank = load_models(args)

    clip_judge, indep_id, lpips_fn, indep_teacher = \
        build_optional_judges(args, args.attribute_index, id_criterion)

    if args.bypass_glasses_direction_bank:
        print('[WARN] --bypass_glasses_direction_bank is ON: Eyeglasses is evaluated '
              'WITHOUT the direction bank while other attributes use it. The numbers '
              'below come from two different pipelines — do not report them as one system.')

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

    all_results = {
        'config': {
            'checkpoint_dir': args.checkpoint_dir,
            'step': args.step,
            'num_samples': args.num_samples,
            'eval_scales': args.eval_scales,
            'success_margin': args.success_margin,
            'bypass_glasses_direction_bank': args.bypass_glasses_direction_bank,
            'clip_judge_model': args.clip_judge_model if clip_judge is not None else None,
            'independent_id': indep_id is not None,
            'lpips': lpips_fn is not None,
            'independent_attr_weights': args.independent_attr_weights,
        },
    }

    # ── Inversion-gap reference (scale independent, computed once) ─────────
    # Real image vs e4e/StyleGAN reconstruction: this is the quality ceiling
    # every edit inherits before the flow touches anything.
    inv_metrics = defaultdict(list)

    for edit_scale in args.eval_scales:
        print(f'\n{"="*60}')
        print(f'edit_scale = {edit_scale}')
        print(f'{"="*60}')

        metrics = defaultdict(list)
        sample_count = 0
        fid = build_fid(args)
        first_scale = str(edit_scale) == str(args.eval_scales[0])

        for img, latent, pred in tqdm(test_loader, desc=f'scale={edit_scale}'):
            if sample_count >= args.num_samples:
                break
            img = img.cuda()
            latent = latent.cuda()
            B = img.size(0)

            _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)

            # Source reconstruction (this is what edits are compared against)
            src_face = G([latent], input_is_latent=True,
                         randomize_noise=False)[0].clamp(-1, 1)
            src_face_256 = F.interpolate(src_face, (256, 256))
            real_256 = F.interpolate(img, (256, 256))

            # Judges on the source
            src_id_arc = F.normalize(id_criterion.extract_features(src_face_256), dim=1)
            src_probs_teacher = torch.sigmoid(attr_teacher(src_face_256)[0])[:, attribute_index]
            src_probs_clip = clip_judge.scores(src_face_256) if clip_judge is not None else None
            src_probs_indep = torch.sigmoid(indep_teacher(src_face_256)[0])[:, attribute_index] \
                if indep_teacher is not None else None
            src_id_indep = indep_id.extract(src_face_256) if indep_id is not None else None

            # Inversion gap (once, on the first scale pass only)
            if first_scale:
                if indep_id is not None:
                    real_feat = indep_id.extract(real_256)
                    inv_metrics['inversion_id_indep'].extend(
                        (real_feat * src_id_indep).sum(dim=1).cpu().tolist())
                if lpips_fn is not None:
                    d = lpips_fn(real_256, src_face_256).flatten()
                    inv_metrics['inversion_lpips'].extend(d.cpu().tolist())

            if fid is not None:
                fid.update(((src_face_256 + 1) * 0.5).clamp(0, 1), real=True)

            for local_idx in range(len(args.attribute_index)):
                attr_name = ATTR_NAMES.get(args.attribute_index[local_idx],
                                           f'attr{args.attribute_index[local_idx]}')

                edited_face = edit_single_attribute(
                    prior, conditioner, G, id_criterion,
                    img, latent, attr_cond, id_cond,
                    local_idx, edit_scale, direction_bank,
                    attr_global_idx=args.attribute_index[local_idx],
                    bypass_glasses_direction_bank=args.bypass_glasses_direction_bank,
                )
                edited_256 = F.interpolate(edited_face, (256, 256))

                edit_id_arc = F.normalize(id_criterion.extract_features(edited_256), dim=1)
                edit_probs_teacher = torch.sigmoid(attr_teacher(edited_256)[0])[:, attribute_index]
                edit_probs_clip = clip_judge.scores(edited_256) if clip_judge is not None else None
                edit_probs_indep = torch.sigmoid(indep_teacher(edited_256)[0])[:, attribute_index] \
                    if indep_teacher is not None else None
                edit_id_indep = indep_id.extract(edited_256) if indep_id is not None else None
                lpips_d = lpips_fn(src_face_256, edited_256).flatten() \
                    if lpips_fn is not None else None

                if fid is not None:
                    fid.update(((edited_256 + 1) * 0.5).clamp(0, 1), real=False)

                for b in range(B):
                    # ── Attribute accuracy per judge (each judge defines its own
                    #    clear-source mask from its own source score) ───────────
                    judge_sets = [('teacher', src_probs_teacher, edit_probs_teacher)]
                    if src_probs_clip is not None:
                        judge_sets.append(('clip', src_probs_clip, edit_probs_clip))
                    if src_probs_indep is not None:
                        judge_sets.append(('indep', src_probs_indep, edit_probs_indep))

                    any_clear = False
                    for jname, sp, ep in judge_sets:
                        s = sp[b, local_idx].item()
                        e = ep[b, local_idx].item()
                        if not is_clear(s):
                            continue
                        any_clear = True
                        metrics[f'acc_{jname}_{attr_name}'].append(
                            float(strict_success(s, e, args.success_margin)))
                        if jname == 'teacher':
                            metrics[f'acc_lenient_{attr_name}'].append(
                                float(lenient_success(s, e)))
                        metrics[f'delta_{jname}_{attr_name}'].append(
                            (e - s) if s < 0.5 else (s - e))  # signed toward target
                        # Leakage on non-target attributes, same judge
                        for other_idx in range(len(args.attribute_index)):
                            if other_idx == local_idx:
                                continue
                            metrics[f'leak_{jname}_{attr_name}'].append(
                                abs(ep[b, other_idx].item() - sp[b, other_idx].item()))

                    if not any_clear:
                        continue

                    # ── Identity ──────────────────────────────────────────────
                    metrics[f'id_arc_{attr_name}'].append(
                        F.cosine_similarity(src_id_arc[b:b+1], edit_id_arc[b:b+1]).item())
                    if edit_id_indep is not None:
                        metrics[f'id_indep_{attr_name}'].append(
                            (src_id_indep[b] * edit_id_indep[b]).sum().item())

                    # ── Perceptual distance recon -> edit ─────────────────────
                    if lpips_d is not None:
                        metrics[f'lpips_{attr_name}'].append(lpips_d[b].item())

            sample_count += B

        # ── Print & collect ────────────────────────────────────────────────
        scale_summary = {'num_samples': sample_count}
        print(f'\n  {sample_count} samples evaluated  '
              f'(strict success = cross 0.5±{args.success_margin})')
        header = (f'  {"Attribute":<12} {"ID_arc*":>8} {"ID_ind↑":>8} '
                  f'{"AccT*":>8} {"AccCLIP↑":>9} {"AccInd↑":>8} '
                  f'{"LPIPS↓":>8} {"LeakCLIP↓":>10}')
        print(header)
        print(f'  {"-" * (len(header) - 2)}')

        attr_names = [ATTR_NAMES.get(i, f'attr{i}') for i in args.attribute_index]
        for attr_name in attr_names:
            row = {}
            for key, pct in [
                (f'id_arc_{attr_name}', False),
                (f'id_indep_{attr_name}', False),
                (f'acc_teacher_{attr_name}', True),
                (f'acc_lenient_{attr_name}', True),
                (f'acc_clip_{attr_name}', True),
                (f'acc_indep_{attr_name}', True),
                (f'delta_teacher_{attr_name}', False),
                (f'delta_clip_{attr_name}', False),
                (f'lpips_{attr_name}', False),
                (f'leak_teacher_{attr_name}', False),
                (f'leak_clip_{attr_name}', False),
            ]:
                row[key.replace(f'_{attr_name}', '')] = _summ(metrics[key])
            scale_summary[attr_name] = row
            print(f'  {attr_name:<12} '
                  f'{_fmt(row["id_arc"])}'
                  f'{_fmt(row["id_indep"])} '
                  f'{_fmt(row["acc_teacher"], pct=True)} '
                  f'{_fmt(row["acc_clip"], pct=True)}  '
                  f'{_fmt(row["acc_indep"], pct=True)} '
                  f'{_fmt(row["lpips"])} '
                  f'{_fmt(row["leak_clip"])}')

        print(f'  (* = same model as the training loss; reference only, '
              f'inflated by construction)')

        # Overall (independent judges only, so the headline number is honest)
        ovr = {}
        for prefix, label in [('id_indep_', 'id_indep'),
                              ('acc_clip_', 'acc_clip'),
                              ('acc_indep_', 'acc_indep'),
                              ('lpips_', 'lpips'),
                              ('id_arc_', 'id_arc'),
                              ('acc_teacher_', 'acc_teacher')]:
            vals = [v for k, vl in metrics.items() if k.startswith(prefix) for v in vl]
            ovr[label] = _summ(vals)
        scale_summary['overall'] = ovr
        id_show = ovr['id_indep'] if ovr['id_indep'] else ovr['id_arc']
        acc_show = ovr['acc_clip'] if ovr['acc_clip'] else ovr['acc_teacher']
        print(f'  {"-" * 55}')
        print(f'  {"Overall":<12} ID(ind or arc): {_fmt(id_show)}  '
              f'Acc(CLIP or teacher): {_fmt(acc_show, pct=True)}')

        if fid is not None:
            fid_val = float(fid.compute().item())
            scale_summary['fid_edit_vs_recon'] = fid_val
            print(f'  FID (edited vs source recon): {fid_val:.2f}')

        all_results[str(edit_scale)] = scale_summary

    # ── Inversion-gap reference ────────────────────────────────────────────
    if inv_metrics:
        inv_summary = {k: _summ(v) for k, v in inv_metrics.items()}
        all_results['inversion_gap'] = inv_summary
        print(f'\n  Inversion gap (real image vs reconstruction, before any edit):')
        for k, s in inv_summary.items():
            if s is not None:
                print(f'    {k}: mean={s["mean"]:.4f}  p10={s["p10"]:.4f}  p90={s["p90"]:.4f}')
        print('  -> edited-image identity/LPIPS can never beat this ceiling; '
              'compare edit metrics against it, not against 1.0/0.0.')

    # ── Save JSON ──────────────────────────────────────────────────────────
    out_path = os.path.join(
        args.checkpoint_dir,
        f'eval_v2_step{args.step}_n{args.num_samples}.json',
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
                        default=False,
                        help='OLD BEHAVIOR (was default True): skip the Direction Bank for '
                             'Eyeglasses only. Off by default so every attribute goes through '
                             'the same pipeline; turning it on mixes two different systems '
                             'into one results table.')
    parser.add_argument('--guided_delta_max_norm', type=float, default=0.0,
                        help='Shared max norm for the final guided W+ delta, applied uniformly to '
                             'every attribute. Set <=0 to disable (no cap).')

    # Independent judges
    parser.add_argument('--clip_judge_model', default='ViT-L/14',
                        help='CLIP model for the zero-shot attribute judge. Keep it different '
                             'from --clip_prompt_model used in training (default ViT-B/32) so '
                             'the judge is not the teacher.')
    parser.add_argument('--independent_attr_weights', default=None,
                        help='Optional second attribute-classifier checkpoint NOT used during '
                             'training. Strongest form of independent attribute judging.')
    parser.add_argument('--independent_attr_backbone', default='r34',
                        help='Backbone for --independent_attr_weights.')
    parser.add_argument('--override_residual_scale', type=float, default=None,
                        help='Force the direction-bank residual_scale to this value for all '
                             'attributes at eval time (e.g. 0.15 or 0.3), overriding the trained '
                             'value (which tends to be frozen near its 0.05 init). Diagnostic only.')
    parser.add_argument('--success_margin', type=float, default=0.0,
                        help='Strict success requires the edited score to cross 0.5 by this '
                             'margin. 0.0 = just cross the decision boundary.')
    parser.add_argument('--compute_fid', action='store_true',
                        help='Compute FID (edited vs source reconstructions). Needs '
                             'torchmetrics + torch-fidelity.')

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
