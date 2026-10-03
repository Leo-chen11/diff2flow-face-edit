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
import time

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
from models.control_encoder import add_skips, clip_skips
from models.stylegan2.model import Generator


ATTR_NAMES = {15: 'Eyeglasses', 20: 'Male', 24: 'No_Beard', 31: 'Smiling',
              33: 'Wavy_Hair', 39: 'Young'}

# BiSeNet skin class -- MUST match training/train_sdflow.py's
# AGE_TEXTURE_REGION_CLASS (and scripts/probe_noise_texture.py's SKIN_CLASS)
# exactly: this is the region a --controlnet_region_cond checkpoint was
# trained to read for age(39) edits, so evaluating it with a different
# region definition would silently feed it something it never learned to use.
AGE_TEXTURE_REGION_CLASS = [1]

# Official CelebA list_attr_celeba.txt column order (0-indexed). Index 15 =
# Eyeglasses, 20 = Male, 24 = No_Beard, 31 = Smiling, 33 = Wavy_Hair, 39 =
# Young -- matches this project's attribute indices directly.
CELEBA_ALL_ATTRS = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
    'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
    'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young',
]
# Every CelebA attribute gets a readable name (e.g. Bangs, not attr5).
for _i, _n in enumerate(CELEBA_ALL_ATTRS):
    ATTR_NAMES.setdefault(_i, _n)


# ---------------------------------------------------------------------------
# Independent judges (never used as training losses)
# ---------------------------------------------------------------------------

def parse_clip_calibration(spec):
    """Parse '--clip_calibration 24:0.42:0.15,33:0.60:0.20' into
    {24: (0.42, 0.15), 33: (0.60, 0.20)}. Returns {} for None/empty."""
    if not spec:
        return {}
    out = {}
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(':')
        if len(parts) != 3:
            raise ValueError(
                f"--clip_calibration entry {chunk!r} must be 'attr_idx:thresh:sharpness'")
        idx, thresh, sharp = parts
        out[int(idx)] = (float(thresh), float(sharp))
    return out


class CLIPAttributeJudge(nn.Module):
    """Zero-shot CLIP attribute scorer, independent from the training teacher.

    Deliberately uses prompt wording different from models/clip_prompt_loss.py
    and (by default) a different CLIP architecture, to reduce judge/teacher
    overlap when the run was trained with --use_clip_prompt_loss.
    """

    # (positive = "attribute present" in CelebA polarity, negative)
    PROMPTS = {
        5: ("a headshot of a person with bangs covering the forehead",
            "a headshot of a person with no bangs and a visible forehead"),
        15: ("a headshot of a person who is wearing glasses",
             "a headshot of a person who is not wearing glasses"),
        20: ("a headshot of a man",
             "a headshot of a woman"),
        24: ("a headshot of a clean-shaven person with no beard",
             "a headshot of a person with a full beard"),
        31: ("a headshot of a smiling person",
             "a headshot of a person with a neutral expression"),
        33: ("a headshot of a person with wavy hair",
             "a headshot of a person with straight hair"),
        39: ("a headshot of a young adult",
             "a headshot of an elderly senior person"),
    }
    DEFAULT_PROMPTS = ("a headshot of a person",
                       "a headshot of a person")

    def __init__(self, attribute_index, model_name='ViT-L/14', device='cuda', calibration=None):
        """
        calibration: optional {attr_idx: (thresh, sharpness)}. The raw
            softmax(pos vs neg) score is remapped through
            sigmoid((raw - thresh) / sharpness) so the *recalibrated* score's
            0.5 boundary lines up with the attribute's real visual decision
            boundary, instead of assuming CLIP's raw pos/neg tie point (0.5
            on the un-remapped score) is already correct. Mirrors how
            GlassesParserJudge calibrates its own area_thresh/sharpness.
            Attributes not in the dict pass through unchanged
            (thresh=0.5, sharpness->0 i.e. identity).
        """
        super().__init__()
        import clip  # raises ImportError -> caller decides how to degrade
        self.model, _ = clip.load(model_name, device=device, jit=False)
        self.model = self.model.float().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.attribute_index = [int(i) for i in attribute_index]
        self.calibration = {int(k): (float(v[0]), float(v[1]))
                             for k, v in (calibration or {}).items()}

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
        raw = torch.softmax(logits, dim=-1)[:, :, 0]   # (B, A)
        if not self.calibration:
            return raw
        out = raw.clone()
        for a, idx in enumerate(self.attribute_index):
            if idx not in self.calibration:
                continue
            thresh, sharpness = self.calibration[idx]
            out[:, a] = torch.sigmoid((raw[:, a] - thresh) / max(sharpness, 1e-4))
        return out


class CelebAAttrClassifierJudge(nn.Module):
    """Independent supervised CelebA-40-attribute classifier (ResNet18 +
    MLP head, trained with BCE loss directly on CelebA labels), used as a
    generic replacement for CLIPAttributeJudge that doesn't need a
    per-attribute prompt or a hand-tuned decision threshold.

    Unlike CLIP zero-shot (a raw pos/neg similarity with no guarantee that
    0.5 means anything), this network's sigmoid output is directly trained
    to cross 0.5 at "attribute present" by construction, so it generalizes
    to new attributes without the prompt-wording / threshold-calibration
    fights CLIP required for glasses (thin frames) and beard (only
    recognized "full beard", missing stubble/partial growth).

    Architecture and CelebA-order output match
    https://github.com/Hawaii0821/FaceAttr-Analysis (ResNet18 backbone,
    512->512->128->40 MLP head, sigmoid). Weights are a different
    architecture/training run than this project's r34 training teacher, so
    it stays a genuinely independent judge.
    """

    def __init__(self, weights_path, device='cuda'):
        super().__init__()
        from torchvision import models as tv_models

        backbone = tv_models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(p=0.5),
            nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(p=0.5),
            nn.Linear(128, len(CELEBA_ALL_ATTRS)),
        )
        self.sigmoid = nn.Sigmoid()

        # Re-key the upstream checkpoint's featureExtractor.model.* /
        # featureClassfier.fc.* names onto this module's names.
        raw = torch.load(weights_path, map_location='cpu')
        state = raw.get('state_dict', raw) if isinstance(raw, dict) else raw
        state = {k[len('module.'):] if k.startswith('module.') else k: v
                 for k, v in state.items()}
        remapped = {}
        for k, v in state.items():
            if k.startswith('featureExtractor.model.'):
                remapped['feature_extractor.' + k[len('featureExtractor.model.'):]] = v
            elif k.startswith('featureClassfier.fc.'):
                remapped['classifier.' + k[len('featureClassfier.fc.'):]] = v
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        print(f'[Judge] CelebAAttrClassifierJudge loaded {weights_path}: '
              f'{len(missing)} missing, {len(unexpected)} unexpected keys'
              + ('' if not missing else f' (missing e.g. {missing[:3]})'))

        self.to(device).eval()
        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    @torch.no_grad()
    def scores(self, images):
        """images: [-1, 1] tensor (B, 3, H, W) -> (B, 40) prob attribute
        present, in official CelebA column order (index directly with the
        project's own attribute indices, e.g. [:, 15] for Eyeglasses)."""
        x = (images + 1.0) * 0.5
        x = F.interpolate(x, (224, 224), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        feat = self.feature_extractor(x)
        feat = feat.view(feat.size(0), -1)
        return self.sigmoid(self.classifier(feat))


class GlassesParserJudge(nn.Module):
    """Purpose-built eyeglasses detector using the BiSeNet face parser's
    dedicated glasses class (label 6), to REPLACE CLIP zero-shot scoring for
    eyeglasses.

    WHY: a visual audit (scripts/dump_glasses_failures.py) showed the CLIP
    zero-shot judge scores genuine THIN / rimless eyeglasses around 0.4 --
    just under a 0.5 threshold -- so ~44% of glasses-add edits that VISIBLY
    had glasses were counted as failures. CLIP zero-shot is weak on thin,
    high-frequency structures like eyeglass frames. BiSeNet has a dedicated
    'eyeglasses' segmentation class and detects the frames at the pixel level,
    which matches the eye far better than a text-prompt similarity. The
    presence score is a smooth function of the glasses-pixel AREA fraction,
    crossing 0.5 at --glasses_area_thresh, so it plugs into the same
    strict_success / add-rm-split machinery as every other judge.

    INDEPENDENCE: BiSeNet segmentation is a different model and task from both
    the r34 attribute teacher and the facenet/ArcFace id judge, so it is an
    independent glasses judge -- UNLESS the run trained with
    --local_region_loss_weight > 0 (which uses this same parser). v16 trained
    with it at 0, so it is independent there; note the caveat for runs that
    used the locality loss.

    CALIBRATION: after changing --glasses_area_thresh, re-run the dumper to
    confirm the new successes really have glasses and the new failures really
    do not. This is calibration grounded in the image audit, not number
    inflation.
    """
    GLASSES_CLASS = 6

    def __init__(self, weights_path, device='cuda', area_thresh=0.0010, sharpness=0.5,
                 min_component_frac=0.00015):
        super().__init__()
        from common.face_parser import FaceParser
        self.parser = FaceParser(weights_path=weights_path).to(device).eval()
        for p in self.parser.parameters():
            p.requires_grad_(False)
        self.area_thresh = float(area_thresh)
        self.sharpness = float(sharpness)
        # Drop connected components smaller than this fraction of the 512x512
        # mask before summing the glasses area. WHY: with the raw whole-mask
        # area fraction, denom = sharpness*area_thresh ~= 5e-4, so a dozen
        # stray pixels BiSeNet mislabels as class 6 (reflections, hair strands,
        # a compression artifact near the eye) alone saturate the sigmoid to
        # ~1.0 -- the score reads "confidently has glasses" even when nothing
        # glasses-shaped was ever drawn, which is exactly the "score is fake"
        # failure mode reported from visually auditing high-scoring samples.
        # A real frame (even thin/rimless) forms a connected blob far larger
        # than sensor noise, so filtering by component size removes the noise
        # without needing to blunt area_thresh/sharpness (which would also
        # suppress genuine thin-frame detections).
        self.min_component_px = max(1, int(round(min_component_frac * 512 * 512)))

    @torch.no_grad()
    def glasses_prob(self, images):
        """images: [-1,1] (B,3,H,W) -> (B,) probability eyeglasses are present."""
        import cv2
        import numpy as np
        inp = F.interpolate(images, 512, mode='bilinear', align_corners=False)
        inp = (inp * 0.5 + 0.5 - self.parser.mean) / self.parser.std
        seg = self.parser.net(inp).argmax(dim=1)                     # (B,512,512)
        glasses_mask = (seg == self.GLASSES_CLASS)

        fracs = []
        for b in range(glasses_mask.size(0)):
            m = glasses_mask[b].byte().cpu().numpy()
            if m.sum() == 0:
                fracs.append(0.0)
                continue
            num, _labels, stats, _cent = cv2.connectedComponentsWithStats(m, connectivity=8)
            areas = stats[1:, cv2.CC_STAT_AREA]   # component 0 is background
            kept = areas[areas >= self.min_component_px].sum() if num > 1 else 0
            fracs.append(float(kept) / (512 * 512))
        frac = glasses_mask.new_tensor(fracs, dtype=torch.float32)   # (B,) filtered area fraction

        denom = self.sharpness * self.area_thresh + 1e-8
        return torch.sigmoid((frac - self.area_thresh) / denom)      # 0.5 at frac==thresh


class IndependentIDJudge(nn.Module):
    """FaceNet (InceptionResnetV1) identity embedder.

    NOTE ON INDEPENDENCE: common/id_loss.py tries insightface ArcFace first,
    but common/nn/insightface.py does not exist in this repo, so IDLoss
    ALWAYS falls back to facenet-pytorch (InceptionResnetV1, vggface2
    weights) on every machine -- id_arc is never actually ArcFace here.
    To keep this judge from being a literal duplicate of the training-time
    identity loss (same architecture AND same weights AND same training
    set), default to the 'casia-webface' pretrained weights instead of
    'vggface2': same architecture as the training-time fallback, but a
    different training dataset, so embeddings are not identical. This is
    a partial-independence compromise, not true architectural independence
    (that would need a real ArcFace/insightface install) -- chosen because
    it keeps the batched-GPU-tensor extract() interface eval relies on,
    instead of switching to a CPU/PIL-based library like deepface that
    would require rewriting the calling convention and be much slower
    over 500 samples x multiple scales x multiple attributes.
    """

    def __init__(self, device='cuda', pretrained='casia-webface'):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1  # ImportError handled by caller
        self.net = InceptionResnetV1(pretrained=pretrained).to(device).eval()
        self.pretrained = pretrained
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
    clip_judge, indep_id, lpips_fn, indep_teacher, glasses_parser, celeb_judge = \
        None, None, None, None, None, None

    try:
        clip_calibration = parse_clip_calibration(getattr(args, 'clip_calibration', None))
        clip_judge = CLIPAttributeJudge(attribute_index, args.clip_judge_model, device,
                                         calibration=clip_calibration)
        print(f'[Judge] CLIP attribute judge: {args.clip_judge_model}')
        if clip_calibration:
            print(f'[Judge] CLIP score recalibration active: {clip_calibration}')
    except ImportError:
        print('[WARN] OpenAI CLIP not installed -> no independent attribute judge. '
              'pip install git+https://github.com/openai/CLIP.git')

    try:
        indep_id = IndependentIDJudge(device, pretrained=args.id_indep_pretrained)
        # If the training IDLoss itself fell back to facenet (input_size 160),
        # this judge shares the SAME architecture as id_arc -- say so loudly
        # instead of silently reporting a number that looks independent.
        if getattr(id_criterion, 'input_size', 112) == 160:
            same_weights = (args.id_indep_pretrained == 'vggface2')
            if same_weights:
                print('[WARN] Training IDLoss is ALSO facenet/vggface2 (insightface '
                      'unavailable) and id_indep uses the SAME weights -- id_indep is '
                      'a literal duplicate of id_arc, not independent at all.')
            else:
                print(f'[WARN] Training IDLoss is ALSO facenet (insightface unavailable). '
                      f'id_indep uses facenet/{args.id_indep_pretrained} (different training '
                      f'set) so it is PARTIALLY independent -- same architecture as id_arc, '
                      f'different weights. Not a substitute for a true architectural '
                      f'independent judge (e.g. real ArcFace/insightface).')
        else:
            print(f'[Judge] Independent ID judge: facenet InceptionResnetV1 '
                  f'({args.id_indep_pretrained}) -- architecturally independent from '
                  f'training id_arc (insightface ArcFace).')
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

    if getattr(args, 'glasses_judge', 'clip') == 'parser' \
            and 15 in [int(i) for i in attribute_index]:
        try:
            glasses_parser = GlassesParserJudge(
                args.face_parser_weights, device,
                area_thresh=args.glasses_area_thresh,
                sharpness=args.glasses_area_sharpness,
                min_component_frac=getattr(args, 'glasses_min_component_frac', 0.00015),
            )
            print(f'[Judge] Eyeglasses scored by BiSeNet parser (class 6, '
                  f'area_thresh={args.glasses_area_thresh}) -- REPLACES CLIP for '
                  f'glasses only. CLIP zero-shot under-detects thin frames; the '
                  f'AccCLIP glasses cell is now parser-scored (label it as such).')
        except (FileNotFoundError, RuntimeError) as exc:
            print(f'[WARN] --glasses_judge parser requested but face parser '
                  f'unavailable ({exc}); falling back to CLIP for glasses.')

    if getattr(args, 'celeba_attr_judge_weights', None):
        try:
            celeb_judge = CelebAAttrClassifierJudge(args.celeba_attr_judge_weights, device)
            print('[Judge] CelebAAttrClassifierJudge active -- independent supervised '
                  '40-attribute classifier, generalizes to any attribute without '
                  'per-attribute prompt/threshold tuning.')
        except (FileNotFoundError, RuntimeError) as exc:
            print(f'[WARN] --celeba_attr_judge_weights given but failed to load ({exc}); '
                  f'continuing without it.')

    return clip_judge, indep_id, lpips_fn, indep_teacher, glasses_parser, celeb_judge


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
# Run-config auto-load
# ---------------------------------------------------------------------------

# Model-structure keys that MUST match training; anything here silently loaded
# wrong under strict=False used to produce plausible-looking garbage metrics.
RUN_CONFIG_KEYS = [
    'attribute_index', 'flow_modules', 'num_blocks', 'velocity_field',
    'lag_gate_hidden_dim', 'lag_gate_init_bias', 'id_cond_dim', 'id_cond_scale',
    'attr_backbone', 'conditioner_backbone', 'clip_model', 'fused_hidden_dim',
    'img_size', 'direction_residual_scale', 'direction_bank_path',
    'use_attr_lora', 'attr_lora_rank', 'signed_magnitude_input',
    'magnitude_latent_cond',
    'use_controlnet_injection', 'controlnet_embed_res', 'controlnet_channels',
    'controlnet_hidden_dim', 'controlnet_max_norm', 'controlnet_init_gain',
    'controlnet_per_direction', 'controlnet_latent_cond', 'controlnet_res',
    'controlnet_region_cond', 'content_bank_path', 'content_film', 'region_saliency_path',
    'age_gate_by_strata', 'residual_scale_cap', 'bank_dir_layers', 'gate_uniform_attrs',
]


def apply_run_config(args):
    """If the run saved a config.json (train_sdflow.py writes one), use it as
    the source of truth for model-structure flags. CLI flags you pass
    explicitly still win; --ignore_run_config disables the whole mechanism."""
    cfg_path = os.path.join(args.checkpoint_dir, 'config.json')
    if getattr(args, 'ignore_run_config', False):
        return args
    if not os.path.exists(cfg_path):
        print(f'[RunConfig] no config.json in {args.checkpoint_dir} — falling back to CLI '
              f'flags. Make sure they match training exactly (strict-load checks below '
              f'will catch structural mismatches, but not value-only ones like '
              f'lag_gate_init_bias).')
        return args
    with open(cfg_path) as f:
        cfg = json.load(f)
    explicit = set()
    for tok in sys.argv[1:]:
        if tok.startswith('--'):
            explicit.add(tok.split('=')[0].lstrip('-').replace('-', '_'))
    for key in RUN_CONFIG_KEYS:
        if key not in cfg or key in explicit:
            continue
        old = getattr(args, key, None)
        new = cfg[key]
        if old != new:
            print(f'[RunConfig] {key}: {old} -> {new} (from config.json)')
            setattr(args, key, new)
    return args


def resolve_controlnet_disable_attrs(args):
    """Auto-default for --controlnet_disable_attrs: when ControlNet injection
    is active and the caller did not explicitly choose which attributes get
    it (the flag is left at its argparse default of None), skip gender/age
    (20, 39) and keep the injection only for eyeglasses.

    WHY as a default and not just a documented flag: the finding (see
    edit_single_attribute) is that ControlNet is load-bearing for eyeglasses
    structure (AccCeleb add 93%->12% without it) but measured NO accuracy
    benefit for gender/age (~70.7% either way) while it DOES introduce a
    hairline/collar sparkle artifact that scale sweeps don't remove. Leaving
    that as an opt-in flag means every eval/render run pays the artifact cost
    for attributes that get nothing from it, unless the user remembers to
    pass the flag by hand. Call this once right after argument parsing (and
    after apply_run_config, so config.json's use_controlnet_injection has
    already been applied) in every entry point that edits attributes.
    """
    using_controlnet = (getattr(args, 'use_controlnet_injection', False)
                         and not getattr(args, 'disable_controlnet', False))
    if using_controlnet and getattr(args, 'controlnet_disable_attrs', None) is None:
        # The "no benefit for age/gender" measurement that motivates this
        # default was taken with SINGLE-resolution (64x64) injection, and
        # 64x64 is the band where mid-level structure lives, not the 128-512
        # bands that carry wrinkles and skin texture. A multi-resolution run
        # reaches a band the old finding never tested, so applying the old
        # default there would silently switch off exactly the thing the run
        # was built to measure. Keep the default only for single-resolution.
        multi_res = len(getattr(args, 'controlnet_res', None) or []) > 1
        auto = [i for i in (20, 39) if i in args.attribute_index]
        if multi_res:
            print('[Default] ControlNet injection kept for ALL attributes: this checkpoint '
                  f'injects at {sorted(args.controlnet_res)}, and the gender/age auto-disable '
                  'was measured on 64x64-only injection (which cannot reach fine texture). '
                  'Pass --controlnet_disable_attrs 20 39 to restore the old behaviour.')
        elif auto:
            args.controlnet_disable_attrs = auto
            print(f'[Default] ControlNet injection auto-disabled for {auto} (gender/age) -- '
                  f'no measured accuracy benefit at 64x64, causes a hairline/collar sparkle '
                  f'artifact; kept for eyeglasses. Pass --controlnet_disable_attrs explicitly '
                  f'to override.')
    return args


def _check_load(result, name):
    """strict=False load, but missing keys are a hard error: they mean the eval
    model structure does not match the checkpoint and every metric would be
    garbage. Unexpected keys (extra buffers from newer training code) only warn."""
    if result.missing_keys:
        raise RuntimeError(
            f'{name} checkpoint does not match the eval model structure. '
            f'Missing keys: {result.missing_keys[:8]} ... '
            f'Fix the model-structure flags (or rely on config.json auto-load).'
        )
    if result.unexpected_keys:
        print(f'[WARN] {name} unexpected keys (ignored): {result.unexpected_keys[:8]}')


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

def _attach_region_parser(args, control_encoder, device):
    """A --controlnet_region_cond encoder was trained with a BiSeNet skin mask
    as its region input for EVERY age(39) edit. edit_single_attribute /
    edit_multi_attribute only built that mask when the CALLER passed a
    face_parser, and several scripts never do (probe_direction_gender_split,
    calibrate_clip_thresh, inspect_zero_edit_magnitude, dump_multi_attr_edit).
    Their age edits silently got the all-ones mask instead: a different input
    from training, so a different edit from the one evaluate_sdflow.py scores.
    Attaching the parser here gives every caller the training-time mask.

    Stored with object.__setattr__ so it is NOT registered as a submodule of
    control_encoder (it must stay out of its parameters() / state_dict())."""
    if not getattr(control_encoder, 'region_cond', False):
        return
    if getattr(args, 'region_saliency_path', None):
        print(f'[WARN] this run trained with --region_saliency_path '
              f'({args.region_saliency_path}): non-age attributes saw per-sample saliency '
              f'maps as their region input, but eval has no saliency lookup and feeds them '
              f'the all-ones mask. Their numbers do NOT reflect what was trained.')
    from common.face_parser import FaceParser
    weights = getattr(args, 'face_parser_weights', None) or './data/parsing_bisenet.pth'
    try:
        parser = FaceParser(weights_path=weights).to(device).eval()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f'[WARN] region_cond checkpoint but face parser unavailable ({exc}); age(39) '
              f'edits fall back to an all-ones region mask, which training never used.')
        return
    object.__setattr__(control_encoder, 'region_parser', parser)


def _attach_content_context(args, control_encoder, device):
    """--content_bank_path runs (models/content_cond.py): load content_encoder
    and attach a ContentContext to control_encoder as .content_ctx.
    edit_single_attribute / edit_multi_attribute read it from there, so every
    script that edits through them (render_preview, the probes, ...) gets the
    same content as training with no change of its own.

    Training gave content to Young(39) edits only; nothing changes for other
    attributes. --disable_content_cond evaluates the same checkpoint with the
    reference withheld (the ablation that shows what content adds)."""
    bank_path = getattr(args, 'content_bank_path', None)
    if not bank_path:
        return
    from models.control_encoder import AttributeControlEncoder
    from models.content_cond import ContentBank, ContentContext, ContentEncoder
    if not isinstance(control_encoder, AttributeControlEncoder):
        print('[WARN] content_bank_path set but control_encoder is the legacy '
              'single-resolution one; content ignored.')
        return
    ckpt = _ckpt_path(args.checkpoint_dir, 'content_encoder', args.step)
    if not os.path.exists(ckpt):
        print(f'[WARN] run was trained with --content_bank_path but {ckpt} is missing; '
              f'evaluating WITHOUT content.')
        return
    if not os.path.exists(bank_path):
        raise FileNotFoundError(
            f'content bank {bank_path} (from config.json) not found. Pass --content_bank_path '
            f'<path> to point at it, or --disable_content_cond to evaluate without content.')
    bank = ContentBank(bank_path, device=device)
    encoder = ContentEncoder(in_dim=bank.dim,
                             out_dim=control_encoder.content_bias_dim).to(device).eval()
    _check_load(encoder.load_state_dict(load_network(ckpt), strict=False), 'content_encoder')
    for p in encoder.parameters():
        p.requires_grad_(False)
    male_local = args.attribute_index.index(20) if 20 in args.attribute_index else None
    enabled = not getattr(args, 'disable_content_cond', False)
    control_encoder.content_ctx = ContentContext(bank, encoder, male_local, seed=0,
                                                 enabled=enabled)
    print(f'Loading content ← {ckpt}  (bank {bank_path}: {bank.summary()})'
          + ('' if enabled else '  [--disable_content_cond: reference WITHHELD]'))


def _content_bias(control_encoder, attr_global_idx, attr_cond, is_rm):
    ctx = getattr(control_encoder, 'content_ctx', None)
    if ctx is None:
        return None
    return ctx.bias(attr_global_idx, attr_cond, is_rm)


def load_models(args):
    global _EDIT_TARGET_MODE
    _EDIT_TARGET_MODE = getattr(args, 'edit_target', None) or 'mirror'
    if _EDIT_TARGET_MODE != 'mirror':
        print(f'[EditTarget] {_EDIT_TARGET_MODE}: edits aim at the training targets '
              f'(0.2/0.8, Eyeglasses 0.1/0.9) -- not comparable with mirror-mode numbers.')
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
    _check_load(prior.load_state_dict(load_network(prior_ckpt), strict=False), 'prior')

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
    _check_load(conditioner.load_state_dict(load_network(cond_ckpt), strict=False),
                'conditioner')

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
            use_attr_lora=getattr(args, 'use_attr_lora', False),
            attr_lora_rank=getattr(args, 'attr_lora_rank', 4),
            signed_magnitude_input=getattr(args, 'signed_magnitude_input', False),
            magnitude_latent_cond=getattr(args, 'magnitude_latent_cond', False),
        ).to(device).eval()
        if os.path.exists(db_ckpt_path):
            # The frozen direction_units are a registered buffer, so they live
            # in this checkpoint too. Snapshot the geometry that was just built
            # from --direction_bank_path BEFORE the checkpoint load, so we can
            # optionally restore it afterwards -- otherwise load_state_dict
            # silently overwrites a spliced/new bank's directions with the ones
            # frozen at training time, and pointing --direction_bank_path at a
            # new file would have no effect at all.
            _bank_dirs = None
            if args.force_bank_directions:
                _bank_dirs = direction_bank.direction_units.detach().clone()
            result = direction_bank.load_state_dict(load_network(db_ckpt_path), strict=False)
            if result.missing_keys:
                print(f'[WARN] Direction bank missing keys: {result.missing_keys[:8]}')
            if result.unexpected_keys:
                print(f'[WARN] Direction bank unexpected keys: {result.unexpected_keys[:8]}')
            print(f'Loading bank   ← {db_ckpt_path}')
            if _bank_dirs is not None:
                if _bank_dirs.shape != direction_bank.direction_units.shape:
                    raise ValueError(
                        f'--force_bank_directions: bank file directions '
                        f'{tuple(_bank_dirs.shape)} do not match the model '
                        f'{tuple(direction_bank.direction_units.shape)}.'
                    )
                with torch.no_grad():
                    direction_bank.direction_units.copy_(_bank_dirs)
                print(f'[ForceBank] re-injected direction_units from '
                      f'{bank_path}, overriding the frozen directions saved in '
                      f'the checkpoint. (Only the direction GEOMETRY is swapped; '
                      f'the trained magnitude_net/gate_net/residual_scale are '
                      f'kept, so use a bank spliced with --keep_original_norms to '
                      f'stay calibrated.)')
        else:
            print(f'Direction bank init (no trained weights at {db_ckpt_path})')
        from models.direction_bank import parse_attr_spec
        for _a in (getattr(args, 'gate_uniform_attrs', None) or []):
            direction_bank.set_uniform_gate(args.attribute_index.index(int(_a)))
            print(f'[Gate] attr {_a}: direction-bank gate forced uniform (averaged slots)')
        for _a, _c in parse_attr_spec(getattr(args, 'residual_scale_cap', None)).items():
            direction_bank.set_residual_cap(args.attribute_index.index(_a), _c)
            print(f'[RunConfig] residual_scale for attr {_a} capped at {_c:g} (as trained)')
        for _a, (_lo, _hi) in parse_attr_spec(getattr(args, 'bank_dir_layers', None),
                                              'range').items():
            direction_bank.set_dir_layers(args.attribute_index.index(_a), _lo, _hi)
            print(f'[RunConfig] direction edit for attr {_a} restricted to W+ layers {_lo}-{_hi}')
        if getattr(args, 'age_gate_by_strata', False):
            _per = direction_bank.enable_strata_routing(
                args.attribute_index.index(39), args.attribute_index.index(20),
                args.attribute_index.index(15))
            print(f'[Routing] age strata routing ON (trained with --age_gate_by_strata): '
                  f'{_per} slot(s) per gender x glasses stratum.')

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

        if 39 in args.attribute_index:
            # StyleGAN's fine W+ layers (index >=4, matching the reg_loss_fine
            # grouping used elsewhere in this codebase) control color/texture,
            # while global/coarse layers (<4) control structure (face shape,
            # wrinkles-as-geometry, hairline). A visual audit confirmed the
            # color-cast artifact on age edits comes 100% from the precomputed
            # direction_units for attribute 39, independent of the flow/
            # residual entirely (residual=0, i.e. pure Direction Bank output,
            # still reproduces it).
            _age_local = args.attribute_index.index(39)
            _start = int(getattr(args, 'age_fine_layer_start', 10))
            _scale = getattr(args, 'age_fine_layer_scale', None)
            if _scale is not None:
                # Explicit override: flat cut, same as before.
                with torch.no_grad():
                    direction_bank.layer_scale[_age_local, _start:] = float(_scale)
                print(f'[Override] age (attr 39) direction layer scale forced to '
                      f'{_scale} for layers [{_start}:18] (layers [0:{_start}] untouched).')
            else:
                # DEFAULT mitigation (was previously off by default, requiring
                # --age_fine_layer_scale to be discovered and set by hand). A
                # blanket cut at layer 4 was measured to also kill real aging
                # signal (500-sample eval: rm-direction AccCLIP 76%->17%),
                # because texture-level aging cues (wrinkles) and the color
                # artifact are not cleanly separable that early. Default here
                # to a GRADUATED ramp starting later (layer 10, matching the
                # "try 10-14" recommendation) down to a floor of 0.6 -- not
                # 0.0 -- at the last layer, so most of the aging signal in
                # layers 10-17 survives while the strongest color-cast
                # contribution (concentrated in the very last, most texture/
                # color-dominated layers) is damped rather than deleted
                # outright. Pass --age_fine_layer_scale explicitly (1.0 to
                # disable this mitigation, 0.0 to reproduce the old hard cut)
                # to override.
                n_fine = direction_bank.layer_scale.shape[1] - _start
                if n_fine > 0:
                    ramp = torch.linspace(1.0, 0.6, steps=n_fine,
                                          device=direction_bank.layer_scale.device)
                    with torch.no_grad():
                        direction_bank.layer_scale[_age_local, _start:] = ramp
                    print(f'[Default] age (attr 39) fine-layer color-cast mitigation: '
                          f'graduated damping 1.0->0.6 over layers [{_start}:18]. '
                          f'Pass --age_fine_layer_scale to override (e.g. 1.0 to disable).')

    # ── ControlNet-style attribute control encoder (optional) ─────────────
    control_encoder = None
    if getattr(args, 'disable_controlnet', False):
        print('[Ablation] --disable_controlnet: control_encoder NOT loaded; the edit runs '
              'through the W+ path (flow + Direction Bank) only, even though this run was '
              'trained with the injection active.')
    elif getattr(args, 'use_controlnet_injection', False):
        from models.control_encoder import (AttributeControlEncoder,
                                            LegacySingleResControlEncoder,
                                            is_legacy_state_dict)
        _cn_res = getattr(args, 'controlnet_res', None) or [args.controlnet_embed_res]
        ce_ckpt_path = _ckpt_path(args.checkpoint_dir, 'control_encoder', args.step)
        # Which architecture to build is decided by the CHECKPOINT, not by the
        # flags: a single-resolution baseline and a multi-resolution challenger
        # are by construction on opposite sides of this change, and comparing
        # them is the whole point. Building the wrong one turns every metric
        # into noise (or, with strict loading, a hard error mid-eval).
        _ce_state = load_network(ce_ckpt_path) if os.path.exists(ce_ckpt_path) else None
        if _ce_state is not None and is_legacy_state_dict(_ce_state):
            if len(_cn_res) > 1:
                print(f'[Compat] control_encoder checkpoint predates multi-resolution '
                      f'injection; building the single-resolution architecture at '
                      f'{args.controlnet_embed_res} and ignoring --controlnet_res {_cn_res}.')
            control_encoder = LegacySingleResControlEncoder(
                num_attrs=num_attrs,
                out_channels=args.controlnet_channels,
                out_res=args.controlnet_embed_res,
                hidden_dim=args.controlnet_hidden_dim,
                init_gain=getattr(args, 'controlnet_init_gain', 1.0),
                per_direction=getattr(args, 'controlnet_per_direction', False),
                latent_cond=getattr(args, 'controlnet_latent_cond', False),
            ).to(device).eval()
        else:
            control_encoder = AttributeControlEncoder(
                num_attrs=num_attrs,
                out_channels=args.controlnet_channels if len(_cn_res) == 1 else None,
                out_res=_cn_res,
                hidden_dim=args.controlnet_hidden_dim,
                init_gain=getattr(args, 'controlnet_init_gain', 1.0),
                per_direction=getattr(args, 'controlnet_per_direction', False),
                latent_cond=getattr(args, 'controlnet_latent_cond', False),
                region_cond=getattr(args, 'controlnet_region_cond', False),
                content_film=bool(getattr(args, 'content_film', False)),
            ).to(device).eval()
        if _ce_state is not None:
            result = control_encoder.load_state_dict(_ce_state, strict=False)
            if result.missing_keys:
                print(f'[WARN] control_encoder missing keys: {result.missing_keys[:8]}')
            if result.unexpected_keys:
                print(f'[WARN] control_encoder unexpected keys: {result.unexpected_keys[:8]}')
            print(f'Loading ctrl   ← {ce_ckpt_path}')
        else:
            print(f'Control encoder init (no trained weights at {ce_ckpt_path})')
        for p in control_encoder.parameters():
            p.requires_grad_(False)
        _attach_region_parser(args, control_encoder, device)
        _attach_content_context(args, control_encoder, device)

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
           attribute_index, direction_bank, control_encoder


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------

# Copy of training/train_sdflow.py SOFT_TARGET_TABLE (not imported: that module
# pulls in the whole training stack). Keep the two in sync.
TRAIN_SOFT_TARGET = {15: (0.10, 0.90), 20: (0.20, 0.80), 39: (0.20, 0.80)}
TRAIN_SOFT_TARGET_DEFAULT = (0.20, 0.80)

# --edit_target, set once by load_models() so every script that edits through
# edit_single_attribute / edit_multi_attribute follows it without passing it.
_EDIT_TARGET_MODE = 'mirror'


def edited_attr_value(src, scale, attr_global_idx, mode=None, direction=None):
    """The attribute value the flow is asked to reach at edit strength `scale`.

    'mirror' (default, what every eval so far used): src*(1-s) + (1-src)*s, so
    s=1 lands on 1-src. 'train': src + s*(hard - src) with hard = 0.20/0.80
    (0.10/0.90 for Eyeglasses), the exact rule training used (soft_flow_target
    in train_sdflow.py).

    Why it matters: the two agree for confident sources and disagree for
    ambiguous ones. A source at 0.68 gets a delta of 0.36 under mirror@1.0 but
    0.48 under train@1.0. Mirror therefore under-edits exactly the sources
    that are hardest to flip, and asks for a smaller change than training
    taught the model to make for that source. Numbers from the two modes are
    not comparable: compare checkpoints under the same mode."""
    if direction is not None:
        # Explicit direction (+1 add, -1 rm), e.g. from the eval judge's own
        # add/rm split: move toward the matching end of [0, 1] by `scale`.
        # Always moves the requested way, even when the conditioner reads the
        # source on the other side of 0.5 (where mirror / train would edit the
        # wrong way, or not at all at src == 0.5).
        end = (direction.to(src.dtype) > 0).to(src.dtype)
        return src + scale * (end - src)
    mode = mode or _EDIT_TARGET_MODE
    if mode == 'train' and attr_global_idx is not None:
        low, high = TRAIN_SOFT_TARGET.get(int(attr_global_idx), TRAIN_SOFT_TARGET_DEFAULT)
        hard = torch.where(src > 0.5, torch.full_like(src, low), torch.full_like(src, high))
        return src + scale * (hard - src)
    return src * (1.0 - scale) + (1.0 - src) * scale


@torch.no_grad()
def composite_faces(face_parser, orig, edited, method='alpha', blur_sigma=15):
    """Blend `edited` face pixels into `orig`'s background/hair, using the
    BiSeNet face mask. orig, edited: (B,3,H,W) tensors in [-1,1].

    method='alpha': mask-weighted linear blend (edited*mask + orig*(1-mask)),
    feathered by `blur_sigma`. Simple, but a seam is visible wherever the
    edited face's overall brightness/color-temperature differs from the
    surrounding background at the boundary -- widening the feather only
    smooths the SHAPE of the seam, it does not fix a color/brightness
    mismatch across it.

    method='poisson': gradient-domain blending (OpenCV seamlessClone). Instead
    of blending pixel VALUES, it solves for a blend whose gradients match the
    inserted content and whose boundary matches the background -- eliminates
    the color-mismatch seam that a feathered alpha blend cannot fix. Runs
    per-sample on CPU (cv2), so it's slower than the alpha path; fine for
    eval/deployment post-processing, not for anything in the training loop.

    MASK SOURCE: the face region is taken as the UNION (elementwise max) of
    the source's and the edited image's BiSeNet face masks, not just the
    source's. Using only `orig` (as an earlier version of this function did)
    silently truncates any structure the edit ADDS that extends past the
    pre-edit face silhouette -- e.g. eyeglasses temple arms reaching past the
    ears, which a glasses-free source has no reason for BiSeNet to have
    labeled "face". Those newly-added pixels would then fall outside the
    orig-only mask and get overwritten by the ORIGINAL (glasses-less)
    background at composite time, i.e. compositing would cut the very
    structure it's supposed to only be protecting the background around --
    directly lowering eyeglasses-add accuracy, not just leaving it flat.
    Taking the union costs a small amount of background/hair protection
    right at that boundary; it does not reintroduce the leakage problem
    compositing exists to fix, since both masks still exclude everything far
    from the face on both sides.
    """
    if method == 'alpha':
        mask = torch.maximum(
            face_parser.get_mask(orig, blur_sigma=int(blur_sigma)),
            face_parser.get_mask(edited, blur_sigma=int(blur_sigma)),
        )
        return edited * mask + orig * (1.0 - mask)

    if method != 'poisson':
        raise ValueError(f'Unknown composite method: {method}')

    import cv2
    import numpy as np
    # hard silhouette (union of both masks -- see docstring); Poisson handles the boundary
    mask = torch.maximum(
        face_parser.get_mask(orig, blur_sigma=0),
        face_parser.get_mask(edited, blur_sigma=0),
    )
    out = []
    for b in range(orig.size(0)):
        o = ((orig[b].clamp(-1, 1) + 1) * 0.5 * 255).byte().permute(1, 2, 0).cpu().numpy()
        e = ((edited[b].clamp(-1, 1) + 1) * 0.5 * 255).byte().permute(1, 2, 0).cpu().numpy()
        m = (mask[b, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
        h, w = m.shape
        if m.sum() < 255 * 50:   # degenerate mask (parser found ~no face) -> keep original
            out.append(orig[b])
            continue
        try:
            blended = cv2.seamlessClone(
                np.ascontiguousarray(e), np.ascontiguousarray(o), m,
                (w // 2, h // 2), cv2.NORMAL_CLONE,
            )
            t = torch.from_numpy(blended).to(orig.device).float().permute(2, 0, 1) / 255.0 * 2 - 1
        except cv2.error:
            t = orig[b]   # mask touched the image border or similar -> fall back safely
        out.append(t)
    return torch.stack(out, dim=0).to(dtype=orig.dtype)


def edit_single_attribute(prior, conditioner, G, id_criterion,
                          img, latent, attr_cond, id_cond,
                          attr_local_idx, edit_scale, direction_bank=None,
                          attr_global_idx=None, bypass_glasses_direction_bank=False,
                          face_parser=None, composite_method='alpha', composite_blur_sigma=15,
                          control_encoder=None, controlnet_max_norm=0.0,
                          controlnet_disable_attrs=None, controlnet_embed_res=64,
                          composite=True, direction=None):
    """
    direction: optional (B,) tensor, +1 = add the attribute, -1 = remove it.
    None (default) = the old behaviour: the direction follows the
    conditioner's own reading of the source (src > 0.5 -> remove). See
    --edit_direction.

    face_parser: if given AND composite=True (the default), composite the
    edited face back onto the SOURCE RECONSTRUCTION's background/hair (see
    composite_faces() above). A long-running complaint in this project was
    that gender/age edits move the background and hair far more than
    intended (the W+ edit is global, not local to the face). Restricting the
    visible change to the BiSeNet face region is a training-free way to cut
    that leakage for EVERY attribute at once, instead of another
    attribute-specific loss tweak. Composites against G(latent) (the source
    RECONSTRUCTION), not the raw source photo, so the blend doesn't inherit
    the encoder's inversion gap as a seam. A first attempt at plain alpha
    blending showed a visible seam (boundary color/brightness mismatch);
    composite_method='poisson' fixes that via gradient-domain blending
    instead of a wider feather.

    composite: set False to pass a face_parser WITHOUT compositing -- needed
    when face_parser is only there to build a --controlnet_region_cond
    region mask (see below) and the caller did not separately ask for
    --composite_face_region. Defaults True so every existing caller (which
    only ever passed face_parser when it wanted compositing) is unaffected.
    """
    B = img.size(0)
    device = img.device
    zero_pad = torch.zeros(B, 18, 1, device=device)

    # Hoisted above the control_encoder call (rather than computed only for
    # compositing at the end, as before): a --controlnet_region_cond encoder
    # needs a region prior built from the UNEDITED source before it runs, for
    # the same inversion-gap reason compositing already used src_recon for.
    # Computed once, reused by both consumers.
    # region_parser: the caller's face_parser if given, else the one
    # load_models() attached for a region_cond encoder (_attach_region_parser)
    # -- used ONLY for the region mask; compositing still needs face_parser.
    region_parser = face_parser if face_parser is not None else \
        getattr(control_encoder, 'region_parser', None)
    need_region = (region_parser is not None and attr_global_idx == 39
                   and getattr(control_encoder, 'region_cond', False))
    src_recon = None
    if face_parser is not None or need_region:
        with torch.no_grad():
            src_recon = G([latent], input_is_latent=True,
                          randomize_noise=False)[0].clamp(-1, 1)

    src_cond = torch.cat([id_cond, attr_cond], dim=1)
    mid_latent, _ = prior(latent, src_cond, zero_pad)

    new_attr_cond = attr_cond.clone()
    src = attr_cond[:, attr_local_idx]
    new_attr_cond[:, attr_local_idx] = edited_attr_value(src, edit_scale, attr_global_idx,
                                                         direction=direction)
    is_rm = (src > 0.5) if direction is None else (direction.to(src.device) < 0)
    new_cond = torch.cat([id_cond, new_attr_cond], dim=1)

    new_latents_raw, _ = prior(mid_latent, new_cond, zero_pad, reverse=True)

    bypass_bank = (
        bypass_glasses_direction_bank
        and attr_global_idx == 15
    )
    control_skips = None
    if direction_bank is not None and not bypass_bank:
        flow_delta = new_latents_raw - latent
        attr_delta = new_attr_cond - attr_cond
        batch_attr_idx = torch.full((B,), attr_local_idx, device=device, dtype=torch.long)
        guided_delta = direction_bank(flow_delta, attr_delta,
                                      attr_idx=batch_attr_idx, latent=latent,
                                      route_scores=attr_cond)
        new_latents = latent + guided_delta
        # controlnet_disable_attrs: some attributes (e.g. eyeglasses) need the
        # ControlNet feature-map injection to synthesize structure the W+
        # direction bank alone can't (AccCeleb add 93%->12% when disabled
        # globally); others (gender/age) got no measurable accuracy benefit
        # from it (~70.7% either way) while it introduced a hairline/collar
        # sparkle artifact -- render_preview visual audit confirmed the
        # artifact tracks with control_encoder being active, not with
        # --scale. Rather than an all-or-nothing --disable_controlnet, this
        # keeps it ONLY for the attributes it actually earns its keep on.
        if control_encoder is not None and (
                controlnet_disable_attrs is None
                or attr_global_idx not in controlnet_disable_attrs):
            region_mask = None
            if getattr(control_encoder, 'region_cond', False):
                # Same convention as training: all-ones (uninformative) for
                # any attribute without a defined region, a real BiSeNet
                # mask for age(39) when face_parser is available.
                region_mask = torch.ones(B, 1, *img.shape[-2:], device=device, dtype=img.dtype)
                if need_region:
                    with torch.no_grad():
                        region_mask = region_parser.get_region_mask(
                            src_recon, AGE_TEXTURE_REGION_CLASS, blur_sigma=5)
            control_skips = control_encoder(attr_delta, batch_attr_idx, is_rm=is_rm,
                                            latent=latent, region_mask=region_mask,
                                            content_bias=_content_bias(
                                                control_encoder, attr_global_idx,
                                                attr_cond, is_rm))
            control_skips = clip_skips(control_skips, controlnet_max_norm)
    else:
        new_latents = new_latents_raw

    edited_face = G([new_latents], skips=control_skips, embed_res=controlnet_embed_res,
                    input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)

    if composite and face_parser is not None:
        edited_face = composite_faces(face_parser, src_recon, edited_face,
                                      method=composite_method,
                                      blur_sigma=composite_blur_sigma)

    return edited_face


def _eval_direction(args, src_probs_clip, local_idx, src_probs_indep=None):
    """--edit_direction clip / indep: tell the model the direction the eval
    scores it on, taken from the source score of that judge (CLIP, or the
    independent classifier; the BiSeNet parser overrides CLIP for Eyeglasses).
    None = conditioner-chosen direction (the old behaviour).

    A direction taken from a judge that mislabels the SOURCE sends the edit
    the wrong way (CLIP calls 40% of bang-less faces "has bangs", so they are
    told to remove bangs and cannot succeed). indep avoids this when the
    independent classifier is the judge that scores the edit."""
    mode = getattr(args, 'edit_direction', 'cond')
    src = {'clip': src_probs_clip, 'indep': src_probs_indep}.get(mode)
    if src is None:
        return None
    return torch.where(src[:, local_idx] > 0.5, -1.0, 1.0)


def edit_multi_attribute(prior, conditioner, G, id_criterion,
                         img, latent, attr_cond, id_cond,
                         attr_local_idxs, edit_scales, direction_bank,
                         attr_global_idxs=None, control_encoder=None, controlnet_max_norm=0.0,
                         controlnet_embed_res=64, controlnet_disable_attrs=None,
                         face_parser=None):
    """Edit several attributes on the same face at once.

    Composes N independently-computed single-attribute guided deltas by
    summation, rather than editing all attr_cond entries in one flow pass
    and calling direction_bank with attr_idx=None. The latter would skip
    direction_bank's per-attribute direction_scale/layer_scale/delta_max_norm
    calibration (those only apply when attr_idx is a single per-sample
    index -- see AttributeDirectionBank.forward), giving untested,
    uncalibrated results. Summing calibrated single-attribute deltas instead
    reuses the exact same code path as edit_single_attribute for each
    attribute, at the cost of one flow reverse-pass per attribute instead
    of one shared pass.

    This does NOT fix the deeper issue that attribute directions aren't
    perfectly orthogonal -- summed edits can still interfere/leak into each
    other. It only avoids ALSO throwing away calibration on top of that.

    attr_local_idxs: list of local indices (into args.attribute_index) to edit.
    edit_scales: float, or list of floats matching attr_local_idxs.
    attr_global_idxs: optional list of absolute CelebA indices, same order
        as attr_local_idxs. Needed for controlnet_disable_attrs (below) to know
        which of these edits are gender/age vs eyeglasses; direction_bank
        itself is always used per attribute regardless.
    controlnet_disable_attrs: absolute attribute indices to skip ControlNet
        injection for, same semantics as edit_single_attribute.
    face_parser: needed only for a --controlnet_region_cond control_encoder
        to get a real region mask for age(39) edits within this composite
        (see edit_single_attribute); every other attribute in the loop gets
        the same all-ones fallback regardless. None (the default) is safe --
        region_cond=True samples then fall back to all-ones for every
        attribute, same as region_cond=False's behavior.
    """
    if isinstance(edit_scales, (int, float)):
        edit_scales = [edit_scales] * len(attr_local_idxs)
    assert len(edit_scales) == len(attr_local_idxs)

    B = img.size(0)
    device = img.device
    zero_pad = torch.zeros(B, 18, 1, device=device)

    region_cond = getattr(control_encoder, 'region_cond', False) if control_encoder is not None else False
    if face_parser is None and control_encoder is not None:
        face_parser = getattr(control_encoder, 'region_parser', None)   # see _attach_region_parser
    src_recon = None
    if region_cond and face_parser is not None:
        with torch.no_grad():
            src_recon = G([latent], input_is_latent=True,
                          randomize_noise=False)[0].clamp(-1, 1)

    src_cond = torch.cat([id_cond, attr_cond], dim=1)
    mid_latent, _ = prior(latent, src_cond, zero_pad)

    combined_delta = torch.zeros_like(latent)
    combined_skips = None
    for i, (local_idx, scale) in enumerate(zip(attr_local_idxs, edit_scales)):
        new_attr_cond = attr_cond.clone()
        src = attr_cond[:, local_idx]
        new_attr_cond[:, local_idx] = edited_attr_value(
            src, scale, attr_global_idxs[i] if attr_global_idxs is not None else None)
        new_cond = torch.cat([id_cond, new_attr_cond], dim=1)

        new_latents_raw, _ = prior(mid_latent, new_cond, zero_pad, reverse=True)
        flow_delta = new_latents_raw - latent
        attr_delta = new_attr_cond - attr_cond
        batch_attr_idx = torch.full((B,), local_idx, device=device, dtype=torch.long)
        guided_delta = direction_bank(flow_delta, attr_delta,
                                      attr_idx=batch_attr_idx, latent=latent,
                                      route_scores=attr_cond)
        combined_delta = combined_delta + guided_delta
        this_global_idx = attr_global_idxs[i] if attr_global_idxs is not None else None
        use_controlnet_here = (control_encoder is not None and not (
            controlnet_disable_attrs is not None and this_global_idx in controlnet_disable_attrs))
        if use_controlnet_here:
            region_mask = None
            if region_cond:
                region_mask = torch.ones(B, 1, *img.shape[-2:], device=device, dtype=img.dtype)
                if face_parser is not None and this_global_idx == 39:
                    with torch.no_grad():
                        region_mask = face_parser.get_region_mask(
                            src_recon, AGE_TEXTURE_REGION_CLASS, blur_sigma=5)
            skip = control_encoder(attr_delta, batch_attr_idx, is_rm=(src > 0.5),
                                   latent=latent, region_mask=region_mask,
                                   content_bias=_content_bias(
                                       control_encoder, this_global_idx, attr_cond, src > 0.5))
            skip = clip_skips(skip, controlnet_max_norm)
            combined_skips = add_skips(combined_skips, skip)

    new_latents = latent + combined_delta
    edited_face = G([new_latents], skips=combined_skips, embed_res=controlnet_embed_res,
                    input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
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


# Failure-progress buckets (change in judge probability toward the target).
NEAR_DELTA = 0.20
STILL_DELTA = 0.05


def is_clear(score, low=0.35, high=0.65):
    return score > high or score < low


def _interp_at(points, x):
    """points: [(x, y)]; linear interpolation of y at x, None outside the range."""
    pts = sorted(points)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return None


def _accuracy_at_matched_id(all_results, args):
    targets = getattr(args, 'report_id_at', None) or []
    scales = [k for k in all_results if isinstance(all_results[k], dict)
              and 'overall' in all_results[k]]
    if not targets or len(scales) < 2:
        return {}
    attr_names = [ATTR_NAMES.get(i, f'attr{i}') for i in args.attribute_index]
    judges = (('acc_indep', 'AccInd'), ('acc_clip', 'AccCLIP'), ('acc_celeb', 'AccCeleb'))
    out = {}
    print(f'\n  Accuracy at matched ID_ind (interpolated over scales {", ".join(scales)}; '
          f'-- = outside the evaluated range)')
    for attr_name in attr_names:
        for t in targets:
            cells = []
            for key, label in judges:
                pts = []
                for sc in scales:
                    row = all_results[sc].get(attr_name) or {}
                    if row.get('id_indep') and row.get(key):
                        pts.append((row['id_indep']['mean'], row[key]['mean']))
                if len(pts) < 2:
                    continue
                v = _interp_at(pts, t)
                out.setdefault(attr_name, {}).setdefault(f'{t:.2f}', {})[key] = v
                cells.append(f'{label} {v * 100:5.1f}%' if v is not None else f'{label}    --')
            if cells:
                print(f'    {attr_name:<12} @ID {t:.2f}:  ' + '   '.join(cells))
    return out


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
        attribute_index, direction_bank, control_encoder = load_models(args)

    clip_judge, indep_id, lpips_fn, indep_teacher, glasses_parser, celeb_judge = \
        build_optional_judges(args, args.attribute_index, id_criterion)
    _glasses_local = args.attribute_index.index(15) if 15 in args.attribute_index else None

    composite_face_parser = None
    if args.composite_face_region or getattr(args, 'controlnet_region_cond', False):
        from common.face_parser import FaceParser
        try:
            composite_face_parser = FaceParser(weights_path=args.face_parser_weights).cuda().eval()
            if args.composite_face_region:
                print('[Composite] Compositing edited face back onto source-reconstruction '
                      'background/hair (common/face_parser.py FaceParser.composite) for ALL '
                      'attributes -- reduces background/hair leakage from global W+ edits, '
                      'training-free.')
            if getattr(args, 'controlnet_region_cond', False):
                print('[ControlNet] region_cond checkpoint: feeding a real BiSeNet skin mask '
                      'for age(39) edits (all-ones fallback for every other attribute).')
        except (FileNotFoundError, RuntimeError) as exc:
            composite_face_parser = None
            print(f'[WARN] face parser unavailable ({exc}); compositing disabled, and any '
                  f'--controlnet_region_cond checkpoint will fall back to an all-ones region '
                  f'mask for age(39) too -- numbers from it will not match what training saw.')

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
            'edit_direction': getattr(args, 'edit_direction', 'cond'),
            'adaptive_ladder': getattr(args, 'adaptive_ladder', None),
            'adaptive_margins': (getattr(args, 'adaptive_margins', None)
                                 if getattr(args, 'adaptive_ladder', None) else None),
        },
    }

    # ── Inversion-gap reference (scale independent, computed once) ─────────
    # Real image vs e4e/StyleGAN reconstruction: this is the quality ceiling
    # every edit inherits before the flow touches anything.
    inv_metrics = defaultdict(list)

    # --adaptive_ladder: each face gets its own edit_scale (see the edit loop);
    # the outer loop then runs over teacher margins instead of fixed scales,
    # each margin giving one point on the accuracy-vs-identity curve.
    adaptive = bool(getattr(args, 'adaptive_ladder', None))
    if adaptive:
        args.adaptive_ladder = sorted(args.adaptive_ladder)
    outer = args.adaptive_margins if adaptive else args.eval_scales
    for edit_scale in outer:
        print(f'\n{"="*60}')
        print(f'adaptive edit_scale, teacher margin = {edit_scale} '
              f'(ladder {args.adaptive_ladder})' if adaptive else f'edit_scale = {edit_scale}')
        print(f'{"="*60}')

        metrics = defaultdict(list)
        # --leak40: per (edited attribute, direction) list of (delta (40,), flipped (40,),
        # clear (40,)) from the independent classifier's full 40-attribute output.
        leak40 = defaultdict(list)
        sample_count = 0
        fid = build_fid(args)
        first_scale = str(edit_scale) == str(outer[0])

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
            if glasses_parser is not None and src_probs_clip is not None and _glasses_local is not None:
                src_probs_clip[:, _glasses_local] = glasses_parser.glasses_prob(src_face_256)
            src_probs_indep_all = torch.sigmoid(indep_teacher(src_face_256)[0]) \
                if indep_teacher is not None else None
            src_probs_indep = src_probs_indep_all[:, attribute_index] \
                if src_probs_indep_all is not None else None
            src_probs_celeb = celeb_judge.scores(src_face_256)[:, attribute_index] \
                if celeb_judge is not None else None
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

                _direction = _eval_direction(args, src_probs_clip, local_idx, src_probs_indep)

                def _edit(scale):
                    return edit_single_attribute(
                        prior, conditioner, G, id_criterion,
                        img, latent, attr_cond, id_cond,
                        local_idx, scale, direction_bank,
                        attr_global_idx=args.attribute_index[local_idx],
                        bypass_glasses_direction_bank=args.bypass_glasses_direction_bank,
                        face_parser=composite_face_parser,
                        composite_method=args.composite_method,
                        composite_blur_sigma=args.composite_blur_sigma,
                        control_encoder=control_encoder,
                        controlnet_max_norm=getattr(args, 'controlnet_max_norm', 0.0),
                        controlnet_disable_attrs=getattr(args, 'controlnet_disable_attrs', None),
                        controlnet_embed_res=getattr(args, 'controlnet_embed_res', 64),
                        # composite_face_parser may now exist ONLY because
                        # --controlnet_region_cond needs it for age's region mask
                        # -- decoupled from whether compositing was actually
                        # requested, so it doesn't silently turn on here too.
                        composite=args.composite_face_region,
                        direction=_direction,
                    )

                if adaptive:
                    # Per-face edit strength: walk up the ladder and keep, for
                    # each face, the FIRST scale at which the training teacher
                    # (r34, not the R50 judge that scores the result) sees the
                    # edit done by `edit_scale` (= the teacher margin here).
                    # Easy faces stop early and keep their identity; faces a
                    # fixed scale leaves unedited ("still" failures) get more.
                    # Faces that never get there keep the largest scale.
                    _gidx = args.attribute_index[local_idx]
                    _add = (_direction > 0) if _direction is not None \
                        else (src_probs_teacher[:, local_idx] < 0.5)
                    _add = _add.to(img.device)
                    edited_face, _done = None, torch.zeros(B, dtype=torch.bool, device=img.device)
                    _chosen = torch.full((B,), float(args.adaptive_ladder[-1]), device=img.device)
                    for _k, _sc in enumerate(args.adaptive_ladder):
                        _ef = _edit(_sc)
                        _pt = torch.sigmoid(attr_teacher(F.interpolate(_ef, (256, 256)))[0])[:, _gidx]
                        _ok = torch.where(_add, _pt > 0.5 + edit_scale, _pt < 0.5 - edit_scale)
                        _take = (~_done) & (_ok | (_k == len(args.adaptive_ladder) - 1))
                        if edited_face is None:
                            edited_face = _ef.clone()
                        edited_face[_take] = _ef[_take]
                        _chosen[_take] = float(_sc)
                        _done |= _take
                        if bool(_done.all()):
                            break
                    metrics[f'chosen_scale_{attr_name}'].extend(_chosen.cpu().tolist())
                else:
                    edited_face = _edit(edit_scale)
                edited_256 = F.interpolate(edited_face, (256, 256))

                edit_id_arc = F.normalize(id_criterion.extract_features(edited_256), dim=1)
                edit_probs_teacher = torch.sigmoid(attr_teacher(edited_256)[0])[:, attribute_index]
                edit_probs_clip = clip_judge.scores(edited_256) if clip_judge is not None else None
                if glasses_parser is not None and edit_probs_clip is not None and _glasses_local is not None:
                    edit_probs_clip[:, _glasses_local] = glasses_parser.glasses_prob(edited_256)
                edit_probs_indep_all = torch.sigmoid(indep_teacher(edited_256)[0]) \
                    if indep_teacher is not None else None
                edit_probs_indep = edit_probs_indep_all[:, attribute_index] \
                    if edit_probs_indep_all is not None else None
                edit_probs_celeb = celeb_judge.scores(edited_256)[:, attribute_index] \
                    if celeb_judge is not None else None
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
                    if src_probs_celeb is not None:
                        judge_sets.append(('celeb', src_probs_celeb, edit_probs_celeb))

                    any_clear = False
                    for jname, sp, ep in judge_sets:
                        s = sp[b, local_idx].item()
                        e = ep[b, local_idx].item()
                        if not is_clear(s):
                            continue
                        any_clear = True
                        _succ = float(strict_success(s, e, args.success_margin))
                        metrics[f'acc_{jname}_{attr_name}'].append(_succ)
                        # Direction split: source lacks the attribute -> this
                        # edit ADDS it; source has it -> this edit REMOVES it.
                        # Aggregate accuracy can hide a large asymmetry (e.g.
                        # glasses removal easy, glasses addition hard).
                        _dir = 'add' if s < 0.5 else 'rm'
                        metrics[f'acc_{jname}_{attr_name}_{_dir}'].append(_succ)
                        if jname == 'teacher':
                            metrics[f'acc_lenient_{attr_name}'].append(
                                float(lenient_success(s, e)))
                        _delta = (e - s) if s < 0.5 else (s - e)   # signed toward target
                        metrics[f'delta_{jname}_{attr_name}'].append(_delta)
                        # How far the edit got on samples the strict test calls
                        # failures: "moved but did not cross 0.5" (judge/label
                        # strictness) vs "did not move" (model/direction).
                        if not _succ:
                            metrics[f'fail_delta_{jname}_{attr_name}_{_dir}'].append(_delta)
                        # Leakage on non-target attributes, same judge
                        for other_idx in range(len(args.attribute_index)):
                            if other_idx == local_idx:
                                continue
                            metrics[f'leak_{jname}_{attr_name}'].append(
                                abs(ep[b, other_idx].item() - sp[b, other_idx].item()))

                    # ── Side effects on all 40 attributes (--leak40) ─────────
                    # What ELSE the edit changed, read by the independent
                    # classifier: signed probability change per attribute, and
                    # whether an attribute whose source score was clear crossed
                    # 0.5. Split by direction, since e.g. lipstick rises in
                    # male->female and falls in female->male and would cancel.
                    if getattr(args, 'leak40', False) and src_probs_indep_all is not None:
                        _st = src_probs_indep[b, local_idx].item()
                        if is_clear(_st):
                            # The classifier emits more than the 40 CelebA
                            # attributes (age outputs follow them); keep the 40.
                            _n40 = len(CELEBA_ALL_ATTRS)
                            _sa = src_probs_indep_all[b, :_n40].float().cpu()
                            _ea = edit_probs_indep_all[b, :_n40].float().cpu()
                            _clr = (_sa > 0.65) | (_sa < 0.35)
                            _ok = strict_success(_st, edit_probs_indep[b, local_idx].item(),
                                                 args.success_margin)
                            leak40[f'{attr_name}_{"add" if _st < 0.5 else "rm"}'].append(
                                (_ea - _sa, ((_sa > 0.5) != (_ea > 0.5)) & _clr, _clr, bool(_ok)))

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
                  f'{"AccT*":>8} {"AccCLIP↑":>9} {"AccInd↑":>8} {"AccCeleb↑":>9} '
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
                (f'acc_celeb_{attr_name}', True),
                (f'delta_teacher_{attr_name}', False),
                (f'delta_clip_{attr_name}', False),
                (f'delta_indep_{attr_name}', False),
                (f'delta_celeb_{attr_name}', False),
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
                  f'{_fmt(row["acc_celeb"], pct=True)} '
                  f'{_fmt(row["lpips"])} '
                  f'{_fmt(row["leak_clip"])}')

        print(f'  (* = same model as the training loss; reference only, '
              f'inflated by construction)')

        # ── Add/remove direction breakdown (CLIP judge) ────────────────────
        # The aggregate accuracy can hide a big asymmetry between adding an
        # attribute and removing it (especially for object-like attributes
        # such as eyeglasses).
        print(f'  Direction split (AccCLIP): add = source lacks attr, rm = source has attr')
        for attr_name in attr_names:
            _a = _summ(metrics[f'acc_clip_{attr_name}_add'])
            _r = _summ(metrics[f'acc_clip_{attr_name}_rm'])
            _a_txt = f'{_a["mean"]*100:5.1f}% (n={_a["n"]})' if _a else '   --'
            _r_txt = f'{_r["mean"]*100:5.1f}% (n={_r["n"]})' if _r else '   --'
            print(f'    {attr_name:<12} add: {_a_txt}   rm: {_r_txt}')
            scale_summary[attr_name]['acc_clip_add'] = _a
            scale_summary[attr_name]['acc_clip_rm'] = _r

        for jname, label, present in (('indep', 'AccInd', indep_teacher is not None),
                                      ('celeb', 'AccCeleb', celeb_judge is not None)):
            if not present:
                continue
            print(f'  Direction split ({label}): add = source lacks attr, rm = source has attr')
            for attr_name in attr_names:
                _a = _summ(metrics[f'acc_{jname}_{attr_name}_add'])
                _r = _summ(metrics[f'acc_{jname}_{attr_name}_rm'])
                _a_txt = f'{_a["mean"]*100:5.1f}% (n={_a["n"]})' if _a else '   --'
                _r_txt = f'{_r["mean"]*100:5.1f}% (n={_r["n"]})' if _r else '   --'
                print(f'    {attr_name:<12} add: {_a_txt}   rm: {_r_txt}')
                scale_summary[attr_name][f'acc_{jname}_add'] = _a
                scale_summary[attr_name][f'acc_{jname}_rm'] = _r

        # ── Progress among failures ───────────────────────────────────────
        # Strict success needs the judge's probability to CROSS 0.5. A sample
        # that went 0.05 -> 0.45 is a failure here although the edit visibly
        # worked, so split the failures by how far they got.
        for jname, label, present in (('indep', 'AccInd', indep_teacher is not None),
                                      ('clip', 'AccCLIP', True)):
            if not present:
                continue
            print(f'  Failures by progress ({label}): "near" = moved >= {NEAR_DELTA:.2f} toward '
                  f'the target without crossing, "still" = moved < {STILL_DELTA:.2f}')
            for attr_name in attr_names:
                cells = []
                for _dir in ('add', 'rm'):
                    d = metrics[f'fail_delta_{jname}_{attr_name}_{_dir}']
                    if not d:
                        cells.append(f'{_dir}: no failures')
                        continue
                    arr = np.asarray(d)
                    prog = {'n_fail': int(arr.size), 'mean_delta': float(arr.mean()),
                            'near': float((arr >= NEAR_DELTA).mean()),
                            'still': float((arr < STILL_DELTA).mean())}
                    scale_summary[attr_name][f'fail_progress_{jname}_{_dir}'] = prog
                    cells.append(f'{_dir}: n={prog["n_fail"]:<3} dP={prog["mean_delta"]:+.2f} '
                                 f'near {prog["near"] * 100:4.0f}% still {prog["still"] * 100:4.0f}%')
                print(f'    {attr_name:<12} ' + '   '.join(cells))

        if leak40:
            top_n = int(getattr(args, 'leak40_top', 6))
            print(f'  Side effects on all 40 attributes (AccInd judge): top {top_n} by mean |dP|, '
                  f'shown as signed mean dP (flip% = clear sources that crossed 0.5)')
            for attr_name in attr_names:
                gidx = args.attribute_index[attr_names.index(attr_name)]
                for _dir in ('add', 'rm'):
                    rows = leak40.get(f'{attr_name}_{_dir}')
                    if not rows:
                        continue
                    others = [j for j in range(rows[0][0].shape[0]) if j != gidx]

                    def _stats(sel):
                        D = torch.stack([r[0] for r in sel])
                        FL = torch.stack([r[1] for r in sel]).float()
                        CL = torch.stack([r[2] for r in sel]).float()
                        return D.mean(0), D.abs().mean(0), FL.sum(0) / CL.sum(0).clamp(min=1)

                    mean_d, abs_d, flip = _stats(rows)
                    ok_rows = [r for r in rows if r[3]]
                    # Side effects of SUCCESSFUL edits only: a setting that barely
                    # edits (low success) also barely changes anything else, so
                    # comparing all-sample averages across settings with different
                    # success rates rewards doing nothing. Ranked on these when
                    # there are enough successes.
                    if len(ok_rows) >= 5:
                        s_mean, s_abs, s_flip = _stats(ok_rows)
                    else:
                        s_mean = s_abs = s_flip = None
                    rank_mean, rank_abs, rank_flip = (s_mean, s_abs, s_flip) if s_abs is not None \
                        else (mean_d, abs_d, flip)
                    order = sorted(others, key=lambda j: -rank_abs[j].item())
                    cells = ', '.join(f'{CELEBA_ALL_ATTRS[j]} {rank_mean[j]:+.2f} ({rank_flip[j] * 100:.0f}%)'
                                      for j in order[:top_n])
                    succ_txt = (f', successful n={len(ok_rows)} mean|dP| others {s_abs[others].mean():.3f}'
                                if s_abs is not None else f', successful n={len(ok_rows)} (too few)')
                    print(f'    {attr_name:<12} {_dir} (all n={len(rows)} mean|dP| others '
                          f'{abs_d[others].mean():.3f}{succ_txt}): '
                          f'{"[successful] " if s_abs is not None else "[all] "}{cells}')

                    def _per(m, a, f):
                        return {CELEBA_ALL_ATTRS[j]: {'mean': float(m[j]), 'abs': float(a[j]),
                                                      'flip': float(f[j])} for j in range(len(m))}
                    scale_summary.setdefault(attr_name, {})[f'leak40_{_dir}'] = {
                        'n': len(rows), 'n_success': len(ok_rows),
                        'mean_abs_others': float(abs_d[others].mean()),
                        'mean_abs_others_success': (float(s_abs[others].mean())
                                                    if s_abs is not None else None),
                        'per_attr': _per(mean_d, abs_d, flip),
                        'per_attr_success': _per(s_mean, s_abs, s_flip) if s_abs is not None else None,
                    }

        if adaptive:
            print('  Chosen edit_scale per face (training-teacher stop rule; '
                  '"at max" = never reached the margin, kept the largest scale):')
            for attr_name in attr_names:
                cs = metrics.get(f'chosen_scale_{attr_name}')
                if not cs:
                    continue
                arr = np.asarray(cs)
                hist = '  '.join(f'{sc:g}:{(arr == sc).mean() * 100:.0f}%' for sc in args.adaptive_ladder)
                print(f'    {attr_name:<12} mean {arr.mean():.2f}  at max {(arr == args.adaptive_ladder[-1]).mean() * 100:4.1f}%   {hist}')
                scale_summary.setdefault(attr_name, {})['chosen_scale'] = _summ(cs)

        # Overall (independent judges only, so the headline number is honest)
        ovr = {}
        for prefix, label in [('id_indep_', 'id_indep'),
                              ('acc_clip_', 'acc_clip'),
                              ('acc_indep_', 'acc_indep'),
                              ('acc_celeb_', 'acc_celeb'),
                              ('lpips_', 'lpips'),
                              ('id_arc_', 'id_arc'),
                              ('acc_teacher_', 'acc_teacher')]:
            vals = [v for k, vl in metrics.items() if k.startswith(prefix) for v in vl]
            ovr[label] = _summ(vals)
        scale_summary['overall'] = ovr
        id_show = ovr['id_indep'] if ovr['id_indep'] else ovr['id_arc']
        # Headline accuracy: the independent evaluation classifier
        # (--independent_attr_weights, e.g. the ResNet-50 CelebA-HQ judge the
        # SDFlow protocol uses) when given, else the CelebA ResNet-18, else
        # CLIP, else the training teacher.
        acc_label, acc_show = next(
            ((lbl, ovr[k]) for k, lbl in (('acc_indep', 'Ind'), ('acc_celeb', 'Celeb'),
                                          ('acc_clip', 'CLIP'), ('acc_teacher', 'teacher'))
             if ovr[k]), ('--', None))
        print(f'  {"-" * 55}')
        print(f'  {"Overall":<12} ID(ind or arc): {_fmt(id_show)}  '
              f'Acc({acc_label}): {_fmt(acc_show, pct=True)}')

        if fid is not None:
            fid_val = float(fid.compute().item())
            scale_summary['fid_edit_vs_recon'] = fid_val
            print(f'  FID (edited vs source recon): {fid_val:.2f}')

        all_results[str(edit_scale)] = scale_summary

    # ── Accuracy at matched identity ──────────────────────────────────────
    # Models trained with different edit magnitudes land at different ID for
    # the same scale, so compare accuracy at the SAME ID_ind, linearly
    # interpolated between the evaluated scales (never extrapolated).
    matched = _accuracy_at_matched_id(all_results, args)
    if matched:
        all_results['acc_at_id'] = matched

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
    out_path = args.out_json or os.path.join(
        args.checkpoint_dir,
        f'eval_v2_step{args.step}_n{args.num_samples}.json',
    )
    # The default name only encodes step and sample count, so a second eval of
    # the same checkpoint (other scales, other --edit_direction) used to
    # overwrite the first. Keep the old file instead of destroying it.
    if os.path.exists(out_path):
        stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(os.path.getmtime(out_path)))
        kept = f'{os.path.splitext(out_path)[0]}_prev{stamp}.json'
        os.replace(out_path, kept)
        print(f'(existing {os.path.basename(out_path)} kept as {os.path.basename(kept)})')
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
    parser.add_argument('--force_bank_directions', action='store_true',
                        help='Re-inject the direction geometry from --direction_bank_path '
                             'AFTER the checkpoint load, overriding the frozen direction_units '
                             'saved in the checkpoint. Without this, swapping --direction_bank_path '
                             'to a new/spliced bank has NO effect at eval time, because '
                             'direction_units is a registered buffer restored from the checkpoint. '
                             'Use with a bank spliced via --keep_original_norms so the trained '
                             'magnitude_net stays calibrated. Diagnostic only.')

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
    parser.add_argument('--use_attr_lora', action='store_true',
                        help='Must match training: whether the checkpoint has a per-attribute '
                             'LoRA adapter (see train_sdflow.py --use_attr_lora). Auto-restored '
                             'from config.json if present.')
    parser.add_argument('--attr_lora_rank', type=int, default=4,
                        help='Must match training --attr_lora_rank if --use_attr_lora is set.')
    parser.add_argument('--signed_magnitude_input', action='store_true',
                        help='Must match training --signed_magnitude_input. Auto-restored from '
                             'config.json.')
    parser.add_argument('--magnitude_latent_cond', action='store_true',
                        help='Must match training --magnitude_latent_cond. Auto-restored from '
                             'config.json.')
    parser.add_argument('--use_controlnet_injection', action='store_true',
                        help='Load the ControlNet-style AttributeControlEncoder and inject its '
                             'predicted skips into StyleGAN2 at embed_res (see '
                             'models/control_encoder.py, train_sdflow.py --use_controlnet_injection). '
                             'Auto-restored from config.json if the checkpoint was trained with it.')
    parser.add_argument('--controlnet_embed_res', type=int, default=64,
                        help='Must match training --controlnet_embed_res if enabled.')
    parser.add_argument('--controlnet_res', nargs='*', type=int, default=None,
                        help='Must match training --controlnet_res if enabled (multi-resolution '
                             'feature injection). Auto-restored from config.json.')
    parser.add_argument('--controlnet_channels', type=int, default=512,
                        help='Must match training --controlnet_channels if enabled.')
    parser.add_argument('--controlnet_hidden_dim', type=int, default=256,
                        help='Must match training --controlnet_hidden_dim if enabled.')
    parser.add_argument('--controlnet_per_direction', action='store_true',
                        help='Must match training --controlnet_per_direction if enabled. '
                             'Auto-restored from config.json.')
    parser.add_argument('--controlnet_latent_cond', action='store_true',
                        help='Must match training --controlnet_latent_cond. Changes the control '
                             'encoder\'s parameter shapes, so a mismatch is a hard checkpoint '
                             'load error rather than silently wrong numbers. Auto-restored from '
                             'config.json.')
    parser.add_argument('--controlnet_region_cond', action='store_true',
                        help='Must match training --controlnet_region_cond. Changes every stage\'s '
                             'input channel count (same class of hard load error as '
                             '--controlnet_latent_cond above), and edit_single_attribute/'
                             'edit_multi_attribute need --face_parser_weights to resolve for age(39) '
                             'edits to get a real region mask instead of the all-ones fallback. '
                             'Auto-restored from config.json.')
    parser.add_argument('--edit_direction', default='cond', choices=['cond', 'clip', 'indep'],
                        help="Who decides add vs remove. cond (default, all previous evals): the "
                             "conditioner's reading of the source. clip: the CLIP judge's add/rm "
                             "split that AccCLIP scores the edit against. The conditioner reads "
                             "some CLIP-young faces as old (middle-aged faces, children; 8%% of men "
                             "vs 2%% of women in v34), so under cond they are edited the wrong way "
                             "or barely at all, and fail at any scale. clip measures editing "
                             "ability with the intended direction given, as a user would; report "
                             "it alongside cond, not instead of it. indep: the same, but the "
                             "direction comes from the independent classifier "
                             "(--independent_attr_weights), the judge that scores AccInd. Prefer "
                             "it to clip whenever CLIP misreads the source (scripts/judge_report.py: "
                             "CLIP TNR 60%% on Bangs, 44%% on Young), because a wrong direction "
                             "means no edit and a guaranteed failure.")
    parser.add_argument('--gate_uniform_attrs', nargs='+', type=int, default=None,
                        help="Force these attributes' direction-bank gate uniform at eval (average "
                             "of the slots instead of the trained gate's pick). Diagnostic on a "
                             "checkpoint trained without it; auto-restored for one trained with it.")
    parser.add_argument('--edit_target', default='mirror', choices=['mirror', 'train'],
                        help="What value an edit of strength s asks the flow for. mirror "
                             "(default, all previous evals): src*(1-s)+(1-src)*s. train: "
                             "src+s*(target-src) with training's 0.2/0.8 targets (0.1/0.9 "
                             "Eyeglasses), which stops ambiguous sources being under-edited "
                             "relative to what training taught. See edited_attr_value(). Compare "
                             "checkpoints under the SAME mode.")
    parser.add_argument('--content_bank_path', type=str, default=None,
                        help='Reference bank of a --content_bank_path training run '
                             '(models/content_cond.py). Auto-restored from config.json; pass it only '
                             'if the bank has moved.')
    parser.add_argument('--disable_content_cond', action='store_true',
                        help='Evaluate a content-trained checkpoint with the reference withheld '
                             '(content bias = 0). Run once with and once without to see what the '
                             'content condition itself contributes.')
    parser.add_argument('--controlnet_init_gain', type=float, default=1.0,
                        help='Must match training --controlnet_init_gain. Only sets the log_gain '
                             'init; the trained value comes from the checkpoint. Auto-restored '
                             'from config.json.')
    parser.add_argument('--disable_controlnet', action='store_true',
                        help='ABLATION: skip loading control_encoder even for a run trained with '
                             '--use_controlnet_injection, so the edit goes through the W+ path '
                             'alone. Run against the same checkpoint as a normal eval to isolate '
                             'what the feature-map injection contributes. Deliberately NOT in '
                             'RUN_CONFIG_KEYS -- an eval-time override, never restored from config.')
    parser.add_argument('--controlnet_disable_attrs', nargs='*', type=int, default=None,
                        help='Keep control_encoder loaded and active, but skip its injection for '
                             'these ABSOLUTE attribute indices only (e.g. 20 39 for gender/age), '
                             'letting others (e.g. eyeglasses) keep it. Found via '
                             '--disable_controlnet + visual audit: eyeglasses ADD accuracy '
                             'collapses without ControlNet (93%%->12%%, it synthesizes frame '
                             'structure the W+ direction bank cannot), but gender/age saw no '
                             'measurable accuracy benefit from it (~70.7%% either way) while it '
                             'introduced a hairline/collar sparkle artifact that --scale sweeps '
                             "(1.0/0.8/0.6) did not change. DEFAULT (when this flag is omitted "
                             'entirely): auto-resolved by resolve_controlnet_disable_attrs() to '
                             '[20, 39] (whichever are in --attribute_index) so gender/age skip '
                             'ControlNet by default and eyeglasses keeps it. Pass this flag '
                             'explicitly (with any list, including none of your attributes) to '
                             'take control back. Deliberately NOT in RUN_CONFIG_KEYS -- an '
                             'eval-time override.')
    parser.add_argument('--controlnet_max_norm', type=float, default=0.0,
                        help='Must match training --controlnet_max_norm if it was set (0 = no cap '
                             'was applied at training time either). Auto-restored from config.json.')

    # Independent judges
    parser.add_argument('--clip_judge_model', default='ViT-L/14',
                        help='CLIP model for the zero-shot attribute judge. Keep it different '
                             'from --clip_prompt_model used in training (default ViT-B/32) so '
                             'the judge is not the teacher.')
    parser.add_argument('--clip_calibration', default=None,
                        help="Recalibrate the CLIP judge's raw pos/neg softmax score per "
                             "attribute so 0.5 lines up with the real visual boundary instead "
                             "of CLIP's raw tie point, analogous to --glasses_area_thresh for "
                             "the parser judge. Format: 'attr_idx:thresh:sharpness,...', e.g. "
                             "'24:0.42:0.15,33:0.60:0.20'. Fit thresh/sharpness with "
                             "scripts/calibrate_clip_thresh.py on a manually-labeled set.")
    parser.add_argument('--independent_attr_weights', default=None,
                        help='Optional second attribute-classifier checkpoint NOT used during '
                             'training. Strongest form of independent attribute judging.')
    parser.add_argument('--independent_attr_backbone', default='r34',
                        help='Backbone for --independent_attr_weights.')
    parser.add_argument('--celeba_attr_judge_weights',
                        default='./data/celeba_attr_resnet18.pth',
                        help='Path to a CelebAAttrClassifierJudge checkpoint (ResNet18, '
                             'https://github.com/Hawaii0821/FaceAttr-Analysis format). '
                             'A supervised 40-attribute classifier that generalizes to any '
                             'CelebA attribute without per-attribute CLIP prompt/threshold '
                             'tuning. Adds an "AccCeleb" column and becomes the preferred '
                             'headline accuracy number when set. Defaults to a conventional '
                             'path under ./data/ so every eval run picks it up automatically '
                             'once the checkpoint is placed there -- no flag needed on the '
                             'command line, and nothing changes if that file does not exist '
                             '(build_judges() catches the missing file and continues without '
                             'AccCeleb, same as before this default existed). Pass a different '
                             'path, or an empty string to force it off, to override.')
    parser.add_argument('--glasses_judge', default='parser', choices=['clip', 'parser'],
                        help="How to score EYEGLASSES (attr 15). 'clip' = CLIP zero-shot "
                             "(under-detects thin frames; a visual audit showed ~44%% of "
                             "glasses-add edits that visibly had glasses were scored as "
                             "failures). 'parser' = BiSeNet face-parser glasses class (label "
                             "6), a purpose-built pixel-level detector that matches the eye "
                             "far better -- REPLACES CLIP for the glasses cell only; gender/age "
                             "still use CLIP. DEFAULT as of this change: parser, with connected-"
                             "component noise filtering (see GlassesParserJudge/"
                             "--glasses_min_component_frac) so stray segmentation pixels can no "
                             "longer saturate the score. Re-run the dumper to visually confirm "
                             "calibration; pass --glasses_judge clip to restore the old default.")
    parser.add_argument('--face_parser_weights', default='./data/parsing_bisenet.pth',
                        help='BiSeNet weights for --glasses_judge parser and '
                             '--composite_face_region.')
    parser.add_argument('--composite_face_region',
                        action=argparse.BooleanOptionalAction, default=False,
                        help='Training-free post-process: composite the edited face back onto '
                             'the source-RECONSTRUCTION background/hair using the BiSeNet face '
                             'mask (common/face_parser.py FaceParser.composite), for every '
                             'attribute. Targets the long-standing complaint that gender/age '
                             'edits move far more of the image than intended (global W+ edits '
                             'leak into background/hair) and directly helps both ID score '
                             '(background/hair no longer contaminate the crop) and perceived '
                             'ghosting/artifacts at the face boundary. Zero training risk -- pure '
                             'inference-time compositing. DEFAULT: off -- the Poisson blend leaves '
                             'a visible tonal seam at the face boundary that inflates LPIPS/depresses '
                             'ID scores relative to the raw uncomposited output (confirmed by '
                             're-evaluating the same checkpoint with and without this flag); pass '
                             '--composite_face_region to enable (falls back automatically with a '
                             'warning if --face_parser_weights is unavailable).')
    parser.add_argument('--composite_method', default='poisson', choices=['alpha', 'poisson'],
                        help="'alpha' (default) feather-blends by mask weight -- fast, but a "
                             "visible seam shows wherever the edited face's brightness/color "
                             "differs from the background at the boundary (a wider "
                             "--composite_blur_sigma only smooths the seam's SHAPE, not the "
                             "color mismatch that causes it). 'poisson' uses gradient-domain "
                             "blending (cv2.seamlessClone) instead, which matches boundary "
                             "illumination directly and removes that seam; slower (runs "
                             "per-sample on CPU) but recommended if 'alpha' shows a visible ring.")
    parser.add_argument('--composite_blur_sigma', type=float, default=15,
                        help='Feather width for --composite_method alpha. Ignored for poisson.')
    parser.add_argument('--glasses_area_thresh', type=float, default=0.0010,
                        help='Glasses-pixel area fraction at which the parser judge outputs '
                             '0.5 (present/absent boundary). Lower = more sensitive to thin '
                             'frames. Calibrate against the dumper.')
    parser.add_argument('--glasses_area_sharpness', type=float, default=0.5,
                        help='Relative width of the parser presence sigmoid; smaller = sharper '
                             '(more binary) present/absent decision.')
    parser.add_argument('--glasses_min_component_frac', type=float, default=0.00015,
                        help='Connected components of the BiSeNet glasses mask smaller than '
                             'this fraction of the 512x512 image are dropped before computing '
                             'the area fraction, so a handful of stray mislabeled pixels cannot '
                             'saturate the score on their own. Default ~39px; raise if the dumper '
                             'still shows noise-driven false positives, lower if it is dropping '
                             'real thin-frame detections.')
    parser.add_argument('--id_indep_pretrained', default='casia-webface',
                        choices=['casia-webface', 'vggface2'],
                        help='Pretrained weights for the facenet-pytorch id_indep judge. '
                             'Default casia-webface differs from the vggface2 weights that '
                             'common/id_loss.py falls back to when insightface is unavailable '
                             '(the common case in this repo), so id_indep is not a literal '
                             'duplicate of id_arc. Still the same architecture either way -- '
                             'see IndependentIDJudge docstring.')
    parser.add_argument('--ignore_run_config', action='store_true',
                        help='Do not auto-load model-structure flags from the run\'s '
                             'config.json (written by train_sdflow.py). By default the '
                             'saved config wins over CLI defaults; explicit CLI flags '
                             'always win over both.')
    parser.add_argument('--override_residual_scale', type=float, default=None,
                        help='Force the direction-bank residual_scale to this value for all '
                             'attributes at eval time (e.g. 0.15 or 0.3), overriding the trained '
                             'value (which tends to be frozen near its 0.05 init). Diagnostic only.')
    parser.add_argument('--age_fine_layer_scale', type=float, default=None,
                        help='Scale the age (attr 39) direction layers [age_fine_layer_start:18] '
                             'by this factor at eval time (e.g. 0.0 to zero them out). '
                             'Diagnostic for the confirmed color-cast artifact living in the '
                             'age direction fine layers -- but a blanket cut at layer 4 also '
                             'kills real aging signal (500-sample eval: rm-direction AccCLIP '
                             '76%->17%), so narrow the range with --age_fine_layer_start.')
    parser.add_argument('--age_fine_layer_start', type=int, default=10,
                        help='First W+ layer index (0-17) affected by the age fine-layer '
                             'color-cast mitigation (see load_models). The reg_loss_fine '
                             'grouping used elsewhere in this codebase starts at layer 4, but '
                             'that boundary was shown to also cut into real aging signal '
                             '(rm-direction AccCLIP 76%%->17%%); default 10 targets only the '
                             'very last, most texture/color-dominated layers.')
    parser.add_argument('--report_id_at', nargs='*', type=float, default=[0.80],
                        help='After all --eval_scales, report each judge\'s accuracy at these '
                             'ID_ind values, interpolated between scales (needs >=2 scales).')
    parser.add_argument('--success_margin', type=float, default=0.0,
                        help='Strict success requires the edited score to cross 0.5 by this '
                             'margin. 0.0 = just cross the decision boundary.')
    parser.add_argument('--compute_fid', action=argparse.BooleanOptionalAction, default=True,
                        help='Compute FID (edited vs source reconstructions). On by default so '
                             'every eval run reports it without remembering the flag; pass '
                             '--no-compute_fid to skip (e.g. a quick smoke-test eval). Needs '
                             'torchmetrics + torch-fidelity -- build_fid() catches ImportError '
                             'and continues without FID, printing a WARN, so a missing '
                             'dependency does not crash the rest of eval.')

    # Eval config
    parser.add_argument('--batch',        type=int,   default=4)
    parser.add_argument('--num_samples',  type=int,   default=500)
    parser.add_argument('--leak40', action='store_true',
                        help='Report what ELSE each edit changed: the independent classifier\'s '
                             'signed probability change on all 40 CelebA attributes, per edited '
                             'attribute and direction, with the rate at which clear source '
                             'attributes crossed 0.5. Needs --independent_attr_weights. Saved '
                             'as leak40_add / leak40_rm in the JSON.')
    parser.add_argument('--leak40_top', type=int, default=6,
                        help='How many of the most-changed other attributes to print per edit.')
    parser.add_argument('--out_json', default=None,
                        help='Where to write the results JSON. Default: '
                             '<checkpoint_dir>/eval_v2_step<N>_n<samples>.json; an existing file '
                             'there is renamed (..._prev<timestamp>.json), never overwritten. '
                             'Several runs of the same checkpoint can be combined with '
                             'scripts/merge_eval_scales.py.')
    parser.add_argument('--eval_scales',  nargs='*',  type=float,
                        default=[0.80, 0.85, 0.90, 0.95])
    parser.add_argument('--adaptive_ladder', nargs='*', type=float, default=None,
                        help='Per-face edit strength instead of one fixed scale: for every face '
                             'and attribute, try these edit_scales from small to large and keep '
                             'the first edit the TRAINING teacher (--attribute_weights, r34) '
                             'scores as done by --adaptive_margins; faces that never get there '
                             'keep the largest. The R50 judge, ID and LPIPS then score the kept '
                             'edit as usual. Replaces --eval_scales: the results are keyed by '
                             'margin, one accuracy-vs-identity point per margin, so keep them in '
                             'their own --out_json. Costs up to len(ladder) generations per edit '
                             '(stops early once every face in the batch is done). '
                             'e.g. 0.5 0.7 0.85 1.0 1.25 1.5')
    parser.add_argument('--adaptive_margins', nargs='*', type=float, default=[0.0, 0.15, 0.3],
                        help='With --adaptive_ladder: stop when the teacher probability is past '
                             '0.5 by this margin in the edit direction. Larger = stronger edits, '
                             'lower ID; each margin is one point of the curve.')

    args = parser.parse_args()
    args = apply_run_config(args)
    args = resolve_controlnet_disable_attrs(args)

    # Auto-detect latest step if not specified
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
        if args.step is None:
            raise ValueError(f'No checkpoints found in {args.checkpoint_dir}/save_models/')
        print(f'Auto-detected latest step: {args.step}')

    evaluate(args)
