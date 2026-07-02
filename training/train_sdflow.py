import argparse
import os,sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'stylegan2'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import optim
from torch.utils import data
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from common.loggerx import WANDBLoggerX
from common.id_loss import IDLoss
from common.ops import load_network
from models.dataset import SDFlowDataset
from models.flows.flow import cnf
from models.flows.utils import modify_one_attribute, standard_normal_logprob
from models.attribute_estimator import AttributeClassifier
from models.conditioner import IdentityAttributeConditioner
from models.direction_bank import AttributeDirectionBank
from models.layer_mask import AttributeLayerMask
from models.stylegan2.model import Generator
    

class LearnableAttributeScales(nn.Module):
    """Per-attribute learnable training edit scales.

    The scale center is exp(attr_log_scales[i]) and is clamped to a conservative
    range. Random noise is detached so only the scale center learns.
    """

    def __init__(self, n_edit_attrs, min_scale=0.3, max_scale=1.5):
        super().__init__()
        self.n_edit_attrs = int(n_edit_attrs)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.attr_log_scales = nn.Parameter(torch.zeros(self.n_edit_attrs))

    def get_attr_train_scale(self, attr_local_idx, base_noise=0.15):
        if not torch.is_tensor(attr_local_idx):
            attr_local_idx = torch.tensor(
                attr_local_idx,
                device=self.attr_log_scales.device,
                dtype=torch.long,
            )
        else:
            attr_local_idx = attr_local_idx.to(
                device=self.attr_log_scales.device,
                dtype=torch.long,
            )

        center = torch.exp(self.attr_log_scales[attr_local_idx]).clamp(
            self.min_scale,
            self.max_scale,
        )
        if base_noise > 0:
            noise = torch.empty_like(center).uniform_(-base_noise, base_noise).detach()
        else:
            noise = torch.zeros_like(center)
        return (center + noise).clamp(self.min_scale, self.max_scale)

    def current_scales(self):
        with torch.no_grad():
            return torch.exp(self.attr_log_scales).clamp(self.min_scale, self.max_scale)


def compute_soft_target(src_vals, attr_idx):
    if attr_idx == 0:  # eyeglasses needs a stronger local-edit signal.
        return torch.where(src_vals > 0.5,
                           torch.full_like(src_vals, 0.10),
                           torch.full_like(src_vals, 0.90))
    elif attr_idx == 1:  # gender should move, but not force a full identity-changing flip.
        return torch.where(src_vals > 0.5,
                           torch.full_like(src_vals, 0.20),
                           torch.full_like(src_vals, 0.80))
    else:  # age is the most identity-sensitive edit; keep the target conservative.
        return torch.where(src_vals > 0.5,
                           torch.full_like(src_vals, 0.20),
                           torch.full_like(src_vals, 0.80))


def compute_soft_targets(src_vals, attr_indices):
    targets = torch.empty_like(src_vals)
    for attr_idx in torch.unique(attr_indices):
        mask = attr_indices == attr_idx
        targets[mask] = compute_soft_target(src_vals[mask], int(attr_idx.item()))
    return targets


def get_eyeglasses_local_idx(attribute_index):
    return attribute_index.index(15) if 15 in attribute_index else None


def build_direction_bank_safety_controls(attribute_index, args, num_layers=18):
    """Build per-attribute Direction Bank controls.

    Age/Young uses conservative layer-aware damping because broad age directions
    often carry texture, hair, lighting, and identity components. Other
    attributes keep the original behavior by default.
    """
    per_attr_direction_scale = []
    per_attr_delta_max_norm = []
    per_attr_layer_scale = []
    for attr_abs_idx in attribute_index:
        if int(attr_abs_idx) == 39:
            per_attr_direction_scale.append(args.age_direction_scale)
            per_attr_delta_max_norm.append(args.age_delta_max_norm)
            layer_scale = torch.ones(num_layers, dtype=torch.float)
            layer_scale[:4] *= args.age_coarse_layer_scale
            layer_scale[4:9] *= args.age_middle_layer_scale
            layer_scale[9:] *= args.age_fine_layer_scale
            per_attr_layer_scale.append(layer_scale.tolist())
        else:
            per_attr_direction_scale.append(1.0)
            per_attr_delta_max_norm.append(0.0)
            per_attr_layer_scale.append([1.0] * num_layers)
    return per_attr_direction_scale, per_attr_layer_scale, per_attr_delta_max_norm


def resolve_resume_save_dir(resume_dir):
    if resume_dir is None:
        return None
    if os.path.basename(os.path.normpath(resume_dir)) == 'save_models':
        return resume_dir
    return os.path.join(resume_dir, 'save_models')


def load_module_checkpoint(module, save_dir, prefix, step, strict=False):
    ckpt_path = os.path.join(save_dir, '{}-{}'.format(prefix, str(step).zfill(7)))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'[Resume] missing {prefix} checkpoint: {ckpt_path}')
    state = torch.load(ckpt_path, map_location='cpu')
    result = module.load_state_dict(state, strict=strict)
    print(f'[Resume] loaded {prefix} from {ckpt_path}')
    if not strict and (result.missing_keys or result.unexpected_keys):
        print(
            f'[Resume] {prefix} non-strict load: '
            f'missing={result.missing_keys}, unexpected={result.unexpected_keys}'
        )


def cap_delta_norm(delta, max_norm):
    if max_norm is None or max_norm <= 0:
        clip = torch.ones(delta.shape[0], device=delta.device, dtype=delta.dtype)
        return delta, clip
    delta_norm = delta.reshape(delta.shape[0], -1).norm(dim=1)
    clip = (float(max_norm) / delta_norm.clamp(min=1e-8)).clamp(max=1.0)
    return delta * clip.view(-1, 1, 1), clip


def apply_direction_bank_with_glasses_bypass(
    direction_bank,
    flow_delta,
    attr_delta,
    attr_idx,
    latent,
    new_latents_raw,
    eyeglasses_local_idx,
    bypass_glasses=True,
):
    if direction_bank is None:
        return new_latents_raw, False, latent.new_tensor(0.0)

    is_glasses = torch.zeros_like(attr_idx, dtype=torch.bool)
    if bypass_glasses and eyeglasses_local_idx is not None:
        is_glasses = attr_idx == int(eyeglasses_local_idx)

    if is_glasses.all():
        return new_latents_raw, False, is_glasses.float().mean()

    guided_delta = direction_bank(flow_delta, attr_delta, attr_idx=attr_idx, latent=latent)
    guided_latents = latent + guided_delta

    if is_glasses.any():
        mask = is_glasses.view(-1, 1, 1)
        guided_latents = torch.where(mask, new_latents_raw, guided_latents)

    return guided_latents, True, is_glasses.float().mean()


def generate_test_image(flow_model:torch.nn.Module,
                        stylegan2_model:torch.nn.Module,
                        test_id_cond:torch.Tensor,
                        test_attr_cond:torch.Tensor,
                        ori_img:torch.Tensor,
                        origin_latent:torch.Tensor,
                        attributes:torch.Tensor,
                        mid_latent:torch.Tensor,
                        img_size=256,
                        layer_mask=None,
                        direction_bank=None,
                        preview_scale=0.6,
                        args=None):
    batchsize = ori_img.shape[0]
    
    #ori_img = F.interpolate(ori_img,(1024,1024))
    #img_ori = torchvision.utils.make_grid(ori_img,nrow=1,normalize=True,value_range=(-1,1))
    #img_recon = stylegan2_model().clamp(-1,1)
    img_recon_batch = stylegan2_model([origin_latent.squeeze(1)],input_is_latent=True,randomize_noise=False)[0].clamp(-1, 1)
    img_recon_batch = F.interpolate(img_recon_batch, ori_img.shape[2:])
    img_recon = torchvision.utils.make_grid(img_recon_batch,nrow=1,normalize=True,value_range=(-1,1))

    # images = [img,ori, img_recon]
    images = [img_recon]

    # generate attributes-change face one by one
    groups = attributes.shape[-1]
    zero_padding = torch.zeros((batchsize,18,1)).to(origin_latent)
    for i in tqdm(range(groups)):
        new_attr_cond = test_attr_cond.detach().clone()
        src = test_attr_cond[:, i]
        new_attr_cond[:, i] = src * (1.0 - preview_scale) + (1.0 - src) * preview_scale
        new_cond = torch.cat([test_id_cond.detach(), new_attr_cond], dim=1)

        new_latents_raw, _ = flow_model(mid_latent, new_cond, zero_padding, reverse=True)
        source_latent = origin_latent.squeeze(1)
        if layer_mask is not None and (args is None or args.velocity_field == 'original'):
            attr_idx = torch.full((batchsize,), i, device=origin_latent.device, dtype=torch.long)
            lm = layer_mask(attr_idx, src, new_attr_cond[:, i]).unsqueeze(-1)
            flow_delta = new_latents_raw - source_latent
            new_latents = source_latent + lm * flow_delta
        elif direction_bank is not None:
            attr_idx = torch.full((batchsize,), i, device=origin_latent.device, dtype=torch.long)
            flow_delta = new_latents_raw - source_latent
            attr_delta = new_attr_cond - test_attr_cond
            eyeglasses_local_idx = None
            if args is not None and getattr(args, 'bypass_glasses_direction_bank', True):
                eyeglasses_local_idx = get_eyeglasses_local_idx(list(args.attribute_index))
            new_latents, _, _ = apply_direction_bank_with_glasses_bypass(
                direction_bank,
                flow_delta,
                attr_delta,
                attr_idx,
                source_latent,
                new_latents_raw,
                eyeglasses_local_idx,
                bypass_glasses=args is None or getattr(args, 'bypass_glasses_direction_bank', True),
            )
        else:
            new_latents = new_latents_raw

        #tmp = stylegan2_model(new_latents).clamp(-1,1)
        tmp = stylegan2_model([new_latents],input_is_latent=True,randomize_noise=False)[0].clamp(-1, 1)
        tmp = F.interpolate(tmp, ori_img.shape[2:])
        tmp = torchvision.utils.make_grid(tmp,nrow=1,normalize=True,value_range=(-1,1))
        images.append(tmp)

    merge = torch.cat(images,dim=2)
    merge = to_pil_image(merge).resize((img_size*(groups+1),img_size*batchsize))
    return merge


def _lookup_dataset_pred(dataset, index):
    file = dataset.image_list[index]
    return dataset._lookup_precomputed(dataset.preds, file)


def _build_fixed_preview_batch(dataset, attribute_index, batch_size):
    attr_ids = [int(i) for i in attribute_index]
    selected = []

    def add_index(idx):
        if idx not in selected and len(selected) < batch_size:
            selected.append(idx)

    preds = []
    for idx in range(len(dataset)):
        pred = _lookup_dataset_pred(dataset, idx)
        preds.append(pred[attr_ids].float())
    preds = torch.stack(preds, dim=0)

    # For each edited attribute, keep one low-score and one high-score source.
    # This makes every preview grid expose add/remove behavior at every checkpoint.
    for local_idx in range(len(attr_ids)):
        values = preds[:, local_idx]
        low_order = torch.argsort(values, descending=False)
        high_order = torch.argsort(values, descending=True)
        for idx in low_order.tolist():
            add_index(idx)
            break
        for idx in high_order.tolist():
            add_index(idx)
            break

    for idx in range(len(dataset)):
        add_index(idx)
        if len(selected) >= batch_size:
            break

    imgs, latents, all_preds = [], [], []
    for idx in selected:
        img, latent, pred = dataset[idx]
        imgs.append(img)
        latents.append(latent)
        all_preds.append(pred)

    return torch.stack(imgs, dim=0), torch.stack(latents, dim=0), torch.stack(all_preds, dim=0), selected


def _collect_dataset_attr_scores(dataset, attribute_index):
    attr_ids = [int(i) for i in attribute_index]
    scores = []
    for idx in range(len(dataset)):
        pred = _lookup_dataset_pred(dataset, idx)
        scores.append(pred[attr_ids].float())
    return torch.stack(scores, dim=0)


class ScoreBalancedBatchSampler(data.Sampler):
    def __init__(self, attr_scores, batch_size, steps_per_epoch,
                 low_threshold=0.35, high_threshold=0.65, seed=0):
        self.attr_scores = attr_scores.float().cpu()
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)
        self.seed = int(seed)
        self.num_attrs = int(attr_scores.shape[1])
        self.epoch = 0
        self._build_pools()

    def _build_pools(self):
        all_indices = torch.arange(self.attr_scores.shape[0])
        self.low_pools = []
        self.high_pools = []
        for attr_idx in range(self.num_attrs):
            scores = self.attr_scores[:, attr_idx]
            low = all_indices[scores <= self.low_threshold]
            high = all_indices[scores >= self.high_threshold]

            if low.numel() == 0:
                low = torch.argsort(scores, descending=False)[:max(self.batch_size, 1)]
            if high.numel() == 0:
                high = torch.argsort(scores, descending=True)[:max(self.batch_size, 1)]

            self.low_pools.append(low.tolist())
            self.high_pools.append(high.tolist())

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        cursors = {}
        shuffled = {}
        for attr_idx in range(self.num_attrs):
            for side, pool in [('low', self.low_pools[attr_idx]), ('high', self.high_pools[attr_idx])]:
                order = torch.randperm(len(pool), generator=generator).tolist()
                shuffled[(attr_idx, side)] = [pool[i] for i in order]
                cursors[(attr_idx, side)] = 0

        def take(attr_idx, side, count):
            key = (attr_idx, side)
            pool = shuffled[key]
            result = []
            while len(result) < count:
                cursor = cursors[key]
                remain = len(pool) - cursor
                if remain <= 0:
                    order = torch.randperm(len(pool), generator=generator).tolist()
                    pool = [pool[i] for i in order]
                    shuffled[key] = pool
                    cursors[key] = 0
                    cursor = 0
                    remain = len(pool)
                n = min(count - len(result), remain)
                result.extend(pool[cursor:cursor + n])
                cursors[key] = cursor + n
            return result

        low_count = self.batch_size // 2
        high_count = self.batch_size - low_count
        for step in range(self.steps_per_epoch):
            attr_idx = step % self.num_attrs
            batch = take(attr_idx, 'low', low_count) + take(attr_idx, 'high', high_count)
            perm = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[i] for i in perm]

    def __len__(self):
        return self.steps_per_epoch


def apply_id_condition_dropout(id_cond, drop_prob):
    drop_prob = max(0.0, min(float(drop_prob), 0.95))
    if drop_prob <= 0:
        return id_cond
    keep_prob = 1.0 - drop_prob
    mask = torch.empty_like(id_cond).bernoulli_(keep_prob)
    return id_cond * mask


def masked_mean(values, mask):
    if mask.any():
        return values[mask].mean()
    return values.new_tensor(0.0)


def collect_lag_dof_losses(flow_model):
    losses = []
    for module in flow_model.modules():
        if hasattr(module, 'odefunc') and hasattr(module.odefunc.diffeq, 'regularization_losses'):
            losses.append(module.odefunc.diffeq.regularization_losses())
    if not losses:
        return None
    out = {}
    for key in losses[0]:
        out[key] = torch.stack([item[key] for item in losses]).mean()
    return out


'''
CUDA_VISIBLE_DEVICES=2 python train_sdflow.py --attribute_index 15 20 39 
'''

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="StyleFlow trainer")
    
    parser.add_argument("--latent_file",default='./data/ffhq_e4e_latents.pth', type=str, help="path to the latents")
    parser.add_argument("--preds_file",default='./data/ffhq_e4e_preds.pth', type=str, help="path to the preds")
    parser.add_argument('--index_file',default='./data/ffhq.txt',type=str)
    parser.add_argument('--image_root',default='data/FFHQ',type=str)
    
    # parameters for model structure
    parser.add_argument("--flow_modules", type=str, default='512-512-512-512-512')
    parser.add_argument("--num_blocks", type=int, default=1)
    parser.add_argument('--velocity_field', default='lag_dof', choices=['original', 'lag', 'dof', 'lag_dof'],
                        help='Use the original shared CNF velocity or LAG-DOF decomposed velocity.')
    parser.add_argument('--lag_gate_hidden_dim', type=int, default=64)
    parser.add_argument('--lag_gate_init_bias', type=float, default=-0.5)
    parser.add_argument("--attribute_index",nargs='*',default=[15,20,39], type=int, help="list of the face attributes index of CelebA")
    parser.add_argument("--stygan2_weights",default='./data/stylegan2-ffhq-config-f.pt',type=str,help='stylegan2 weights path')
    
    # parameters for save and name
    parser.add_argument("--model_name",default='SDFlow',type=str,help="model name")
    parser.add_argument('--run_name',default='default',type=str,help='this run name')
    parser.add_argument('--wandb_project', default='SDFlow', type=str, help='wandb project name')
    parser.add_argument('--wandb_entity', default=None, type=str, help='wandb user or team name')
    parser.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'], help='wandb logging mode')
    parser.add_argument('--print_freq', type=int, default=10,help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1000,help='save frequency')
    
    
    # parameters for training 
    parser.add_argument("--img_size",type=int,default=512,help="image size for model")
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="number of workers")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--id_cond_dim', type=int, default=32)
    parser.add_argument('--id_cond_scale', type=float, default=0.25)
    parser.add_argument('--attr_backbone', default='resnet50',
                        help='ResNet backbone for original/fused conditioner attribute features.')
    parser.add_argument('--conditioner_backbone', default='resnet',
                        choices=['resnet', 'clip', 'resnet_clip'],
                        help='Feature backbone for attr condition: original ResNet estimator, CLIP-only, or ResNet+CLIP fused.')
    parser.add_argument('--clip_model', default='ViT-B/32',
                        help='OpenAI CLIP model name for clip/resnet_clip conditioner.')
    parser.add_argument('--fused_hidden_dim', type=int, default=256,
                        help='Hidden dim of the trainable CLIP/fused projection head.')
    parser.add_argument('--id_cond_dropout', type=float, default=0.2,
                        help='Training-only dropout on identity condition to stop identity from suppressing edits.')
    parser.add_argument('--train_scale_min', type=float, default=0.35)
    parser.add_argument('--train_scale_max', type=float, default=0.55)
    parser.add_argument('--attribute_sampling', default='cycle', choices=['cycle', 'random'],
                        help='cycle trains attributes in a balanced round-robin order.')
    parser.add_argument('--score_balanced_sampling', dest='score_balanced_sampling', action='store_true', default=True,
                        help='Use low/high source-score balanced batches for each cycled attribute.')
    parser.add_argument('--disable_score_balanced_sampling', dest='score_balanced_sampling', action='store_false')
    parser.add_argument('--score_balance_low', type=float, default=0.35)
    parser.add_argument('--score_balance_high', type=float, default=0.65)
    parser.add_argument('--preview_scale', type=float, default=0.50)
    parser.add_argument('--preview_mode', default='fixed_balanced',
                        choices=['fixed_balanced', 'rolling'],
                        help='fixed_balanced reuses the same attribute-balanced preview batch every checkpoint.')
    # parameters for loss weight
    parser.add_argument("--nll_loss_weight", type=float, default=1)
    parser.add_argument("--reg_loss_weight", type=float, default=0.1)
    parser.add_argument('--kd_loss_weight',type=float,default=1)
    parser.add_argument('--id_loss_weight',type=float,default=0.35)
    parser.add_argument('--attribute_weights', default='./data/r34_a40_age_256_classifier.pth', type=str)
    parser.add_argument('--counter_attr_weight', type=float, default=0.6)
    parser.add_argument('--preserve_attr_weight', type=float, default=0.6)
    parser.add_argument('--orth_loss_weight', type=float, default=0.005)
    parser.add_argument('--gate_smooth_weight', type=float, default=0.003)
    parser.add_argument('--reg_fine_weight', type=float, default=0.5)
    parser.add_argument('--direction_bank_path', default=None, type=str,
                        help='Path to precomputed Attribute Direction Bank (.pth).')
    parser.add_argument('--direction_residual_scale', type=float, default=0.05)
    parser.add_argument('--direction_freeze', '--direction-freeze',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--direction_orth_weight', type=float, default=0.0)
    parser.add_argument('--direction_k', type=int, default=1,
                        help='Number of mixture directions per attribute in the Direction Bank.')
    parser.add_argument('--glasses_residual_scale', type=float, default=0.05,
                        help='Residual scale for eyeglasses attribute (local edit; try 0.5-0.8 to give flow more freedom).')
    parser.add_argument('--bypass_glasses_direction_bank',
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Bypass Direction Bank for eyeglasses so local accessory edits keep full flow gradients.')
    parser.add_argument('--direction_guided_delta_max_norm', type=float, default=0.0,
                        help='Optional global max norm inside Direction Bank after all controls. '
                             'Use 10-12 for safe stage fine-tuning; 0 disables.')
    parser.add_argument('--final_delta_max_norm', type=float, default=0.0,
                        help='Optional final max norm after glasses bypass / bank mixing. '
                             'This also caps raw glasses flow deltas; 0 disables.')
    parser.add_argument('--freeze_direction_bank_nets', action='store_true',
                        help='Freeze all Direction Bank trainable nets during fine-tuning. '
                             'Useful when continuing from a good checkpoint and only training the flow/conditioner.')

    # ── Safe resume / stage fine-tuning ─────────────────────────────────────
    parser.add_argument('--resume_dir', default=None, type=str,
                        help='Run directory or save_models directory to resume from.')
    parser.add_argument('--resume_step', default=None, type=int,
                        help='Checkpoint step to resume from, e.g. 55000.')
    parser.add_argument('--resume_optimizer', action=argparse.BooleanOptionalAction, default=False,
                        help='Load optimizer state. Usually false for controlled stage fine-tuning.')
    parser.add_argument('--resume_direction_bank', action=argparse.BooleanOptionalAction, default=False,
                        help='Load direction_bank checkpoint state. Keep false when changing bank path/safety settings.')

    # ── Adaptive residual scale for saturated attributes ───────────────────
    parser.add_argument('--adaptive_residual_scale', action='store_true', default=False,
                        help='Adaptively increase one attribute residual_scale when its target loss EMA is low.')
    parser.add_argument('--adaptive_rs_attr_idx', type=int, default=2,
                        help='Local attribute index to adapt, e.g. 2 for Age when attribute_index is 15 20 39.')
    parser.add_argument('--adaptive_rs_ema_decay', type=float, default=0.95,
                        help='EMA decay for target loss tracking.')
    parser.add_argument('--adaptive_rs_init_ema', type=float, default=0.30,
                        help='Initial target-loss EMA for adaptive residual scale.')
    parser.add_argument('--adaptive_rs_threshold', type=float, default=0.25,
                        help='Increase residual_scale when target loss EMA is below this value.')
    parser.add_argument('--adaptive_rs_step_size', type=float, default=0.02,
                        help='Residual scale increment each time the adaptive rule fires.')
    parser.add_argument('--adaptive_rs_interval', type=int, default=1000,
                        help='Check adaptive residual scale every N local training steps.')
    parser.add_argument('--adaptive_rs_min', type=float, default=0.05,
                        help='Lower bound for adaptive residual scale.')
    parser.add_argument('--adaptive_rs_max', type=float, default=0.25,
                        help='Upper bound for adaptive residual scale.')

    # ── Frozen pretrained diffusion guidance ───────────────────────────────
    parser.add_argument('--use_diffusion_guidance', action='store_true',
                        help='Use a frozen Stable Diffusion model as auxiliary DDS semantic guidance.')
    parser.add_argument('--diffusion_model_id', default='SG161222/Realistic_Vision_V5.1_noVAE', type=str,
                        help='HuggingFace model id or local path for the frozen diffusion model.')
    parser.add_argument('--diffusion_vae_model_id', default='stabilityai/sd-vae-ft-mse', type=str,
                        help='VAE model id for noVAE diffusion checkpoints. Use empty string to disable override.')
    parser.add_argument('--diffusion_guidance_weight', type=float, default=0.01,
                        help='Weight for frozen diffusion DDS loss. Start small: 0.001-0.02.')
    parser.add_argument('--diffusion_guidance_interval', type=int, default=8,
                        help='Run diffusion guidance every N steps to reduce memory/time cost.')
    parser.add_argument('--diffusion_image_size', type=int, default=256,
                        help='Resize generated/source images before VAE encoding.')
    parser.add_argument('--diffusion_timestep_min', type=int, default=50)
    parser.add_argument('--diffusion_timestep_max', type=int, default=700)
    parser.add_argument('--diffusion_guidance_scale', type=float, default=1.0,
                        help='Classifier-free guidance scale inside diffusion noise prediction.')
    parser.add_argument('--diffusion_fp16', action=argparse.BooleanOptionalAction, default=True,
                        help='Load frozen diffusion in fp16 when CUDA is available.')

    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps. Effective batch = batch * grad_accum_steps.')
    parser.add_argument('--residual_max_norm', type=float, default=None,
                        help='Hard clip per-sample residual norm in Direction Bank forward(). '
                             'Prevents residual explosion from large DDS gradients. Suggested: 10.0.')
    parser.add_argument('--age_direction_scale', type=float, default=0.55,
                        help='Age/Young-only multiplier for Direction Bank guided delta. '
                             'Use this to prevent strong age banks from blurring identity.')
    parser.add_argument('--age_coarse_layer_scale', type=float, default=0.75,
                        help='Age/Young-only scale for W+ coarse layers 0:4.')
    parser.add_argument('--age_middle_layer_scale', type=float, default=1.0,
                        help='Age/Young-only scale for W+ middle layers 4:9.')
    parser.add_argument('--age_fine_layer_scale', type=float, default=0.45,
                        help='Age/Young-only scale for W+ fine layers 9:18.')
    parser.add_argument('--age_delta_max_norm', type=float, default=10.0,
                        help='Age/Young-only max norm for final guided W+ delta. '
                             'Set <=0 to disable.')
    parser.add_argument('--dds_fine_layer_start', type=int, default=7,
                        help='W+ layer index from which DDS gradients are blocked (fine layers). '
                             'Set 0 to disable masking.')
    parser.add_argument('--age_diffusion_timestep_min', type=int, default=400,
                        help='Min timestep for age-specific DDS pass (coarse structure).')
    parser.add_argument('--age_diffusion_timestep_max', type=int, default=900,
                        help='Max timestep for age-specific DDS pass.')
    parser.add_argument('--age_diffusion_interval', type=int, default=16,
                        help='Run age DDS guidance every N steps (independent of --diffusion_guidance_interval).')

    # ── Frozen CLIP semantic target loss ───────────────────────────────
    parser.add_argument('--use_clip_prompt_loss', action='store_true',
                        help='Enable frozen CLIP prompt loss for semantic direction supervision.')
    parser.add_argument('--clip_prompt_model', type=str, default='ViT-B/32',
                        help='OpenAI CLIP model name.')
    parser.add_argument('--clip_prompt_weight', type=float, default=0.03,
                        help='Weight for CLIP prompt loss. Suggested range: 0.02–0.05.')
    parser.add_argument('--clip_prompt_temperature', type=float, default=1.0,
                        help='Temperature for softplus sharpness in CLIP loss.')
    parser.add_argument('--clip_prompt_interval', type=int, default=1,
                        help='Compute CLIP loss every N steps (1 = every step).')
    parser.add_argument('--clip_prompt_age_weight', type=float, default=3.0,
                        help='Per-sample weight multiplier for age (attr 39) in CLIP loss.')
    parser.add_argument('--clip_prompt_gender_weight', type=float, default=1.0,
                        help='Per-sample weight multiplier for gender (attr 20) in CLIP loss.')
    args = parser.parse_args()
    torch.manual_seed(0)
    eyeglasses_local_idx = get_eyeglasses_local_idx(list(args.attribute_index))

    os.environ['WANDB_MODE'] = args.wandb_mode

    wandb_kwargs = dict(
        project=args.wandb_project,
        name='{}_{}'.format(args.model_name,args.run_name),
    )
    if args.wandb_entity:
        wandb_kwargs['entity'] = args.wandb_entity
    
    logger = WANDBLoggerX(save_root=os.path.join('./output',args.model_name,args.run_name),
                          print_freq=args.print_freq,
                          config=args,
                          **wandb_kwargs)
    attribute_index = torch.tensor(args.attribute_index,dtype=int)
    base_condition_dim = args.id_cond_dim + len(args.attribute_index)
    condition_dim = base_condition_dim
    prior = cnf(
        512,
        args.flow_modules,
        condition_dim,
        args.num_blocks,
        velocity_field=args.velocity_field,
        num_layers=18,
        gate_hidden_dim=args.lag_gate_hidden_dim,
        gate_init_bias=args.lag_gate_init_bias,
        attr_context_dim=len(args.attribute_index),
        train_T=False,
    ).cuda()
    # prior.load_state_dict(torch.load('./pretrained_models/ffhq_prior.pth',map_location='cpu'),strict=True)
    
    img_transform = T.Compose([
        T.ToTensor(),
        T.Resize((args.img_size,args.img_size)),
        T.Normalize(mean=0.5,std=0.5)
    ])
    train_dataset = SDFlowDataset(index_file=args.index_file,
                                         image_root=args.image_root,
                                         latents_file=args.latent_file,
                                         preds_file=args.preds_file,
                                         train=True,
                                         transform=img_transform)
    
    test_dataset = SDFlowDataset(index_file=args.index_file,
                                        image_root=args.image_root,
                                        latents_file=args.latent_file,
                                        preds_file=args.preds_file,
                                        train=False,
                                        transform=img_transform)
    
    if args.score_balanced_sampling and args.attribute_sampling == 'cycle':
        train_attr_scores = _collect_dataset_attr_scores(train_dataset, args.attribute_index)
        train_sampler = ScoreBalancedBatchSampler(
            train_attr_scores,
            batch_size=args.batch,
            steps_per_epoch=len(train_dataset) // args.batch,
            low_threshold=args.score_balance_low,
            high_threshold=args.score_balance_high,
            seed=0,
        )
        train_loader = data.DataLoader(train_dataset,
                                       batch_sampler=train_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True)
        low_counts = [len(p) for p in train_sampler.low_pools]
        high_counts = [len(p) for p in train_sampler.high_pools]
        print(f'** score-balanced sampler enabled. low pools: {low_counts}, high pools: {high_counts}')
    else:
        train_sampler = None
        train_loader = data.DataLoader(train_dataset,
                                       shuffle=True,
                                       batch_size=args.batch,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       drop_last=True)
    test_batch = args.batch if args.batch<=8 else 8
    test_loader = data.DataLoader(test_dataset, 
                                   shuffle=False, 
                                   batch_size=test_batch,
                                   pin_memory=False,
                                   drop_last=True)
    print('** dataloader initialized successfully !')
    
    
    conditioner = IdentityAttributeConditioner(
        attr_dim=len(args.attribute_index),
        id_dim=args.id_cond_dim,
        id_scale=args.id_cond_scale,
        attr_backbone=args.attr_backbone,
        conditioner_backbone=args.conditioner_backbone,
        clip_model=args.clip_model,
        fused_hidden_dim=args.fused_hidden_dim,
    ).cuda()
    if args.velocity_field == 'original':
        layer_mask = AttributeLayerMask(num_attrs=len(args.attribute_index)).cuda()
    else:
        layer_mask = None
    attr_scales = LearnableAttributeScales(len(args.attribute_index)).cuda()
    trainable_params = list(prior.parameters()) + list(conditioner.parameters())
    if args.velocity_field == 'original':
        trainable_params += list(layer_mask.parameters())
    if args.direction_bank_path:
        # Build per-attribute residual scale: glasses (local) may use a higher scale
        _per_attr_rs = [
            args.glasses_residual_scale if idx == 15 else args.direction_residual_scale
            for idx in args.attribute_index
        ]
        (
            _per_attr_direction_scale,
            _per_attr_layer_scale,
            _per_attr_delta_max_norm,
        ) = build_direction_bank_safety_controls(args.attribute_index, args, num_layers=18)
        direction_bank = AttributeDirectionBank(
            num_attrs=len(args.attribute_index),
            num_layers=18,
            latent_dim=512,
            num_k=args.direction_k,
            bank_path=args.direction_bank_path,
            attribute_index=args.attribute_index,
            residual_scale=args.direction_residual_scale,
            per_attr_residual_scale=_per_attr_rs,
            freeze_directions=args.direction_freeze,
            residual_max_norm=args.residual_max_norm,
            per_attr_direction_scale=_per_attr_direction_scale,
            per_attr_layer_scale=_per_attr_layer_scale,
            per_attr_delta_max_norm=_per_attr_delta_max_norm,
            guided_delta_max_norm=(
                args.direction_guided_delta_max_norm
                if args.direction_guided_delta_max_norm > 0 else None
            ),
        ).cuda()
        if args.freeze_direction_bank_nets:
            for p in direction_bank.parameters():
                p.requires_grad_(False)
            print('** Direction Bank trainable nets frozen for safe fine-tuning')
        trainable_params += [p for p in direction_bank.parameters() if p.requires_grad]
        print(f'** Direction Bank enabled: {args.direction_bank_path}')
    else:
        direction_bank = None
    trainable_params += list(attr_scales.parameters())
    optimizer = optim.Adam(
        [
            {'params': [p for p in trainable_params if p is not attr_scales.attr_log_scales], 'lr': args.lr},
            {'params': [attr_scales.attr_log_scales], 'lr': args.lr * 0.1, 'weight_decay': 0.0},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader) // args.grad_accum_steps, eta_min=1e-6
    )

    start_step = 0
    if args.resume_dir is not None:
        if args.resume_step is None:
            raise ValueError('--resume_step is required when --resume_dir is set')
        resume_save_dir = resolve_resume_save_dir(args.resume_dir)
        start_step = int(args.resume_step)
        print(f'[Resume] fine-tuning from {resume_save_dir} at step {start_step}')
        load_module_checkpoint(prior, resume_save_dir, 'prior', start_step, strict=False)
        load_module_checkpoint(conditioner, resume_save_dir, 'conditioner', start_step, strict=False)
        try:
            load_module_checkpoint(attr_scales, resume_save_dir, 'attr_scales', start_step, strict=False)
        except FileNotFoundError:
            print('[Resume] attr_scales checkpoint not found; using default adaptive scales.')
        if direction_bank is not None and args.resume_direction_bank:
            load_module_checkpoint(direction_bank, resume_save_dir, 'direction_bank', start_step, strict=False)
        elif direction_bank is not None:
            print('[Resume] direction_bank checkpoint not loaded; using current bank path and safety controls.')
        if args.resume_optimizer:
            opt_path = os.path.join(resume_save_dir, 'optimizer-{}'.format(str(start_step).zfill(7)))
            if not os.path.exists(opt_path):
                raise FileNotFoundError(f'[Resume] missing optimizer checkpoint: {opt_path}')
            optimizer.load_state_dict(torch.load(opt_path, map_location='cpu'))
            print(f'[Resume] loaded optimizer from {opt_path}')
        else:
            print('[Resume] optimizer state not loaded; using fresh optimizer for controlled fine-tuning.')

    log_modules = [prior, conditioner, attr_scales, optimizer]
    if args.velocity_field == 'original':
        log_modules.insert(2, layer_mask)
    if direction_bank is not None:
        log_modules.insert(-1, direction_bank)
    logger.modules = log_modules
    
    # Initialization for stylegan2 model
    ckpt = torch.load(args.stygan2_weights,map_location='cpu')
    G = Generator(size=1024,style_dim=512,n_mlp=8)
    G.load_state_dict(ckpt['g_ema'])
    G.cuda().eval()
    for p in G.parameters():
        p.requires_grad_(False)
    print('** StyleGAN2 model initialization success !')

    id_criterion = IDLoss(crop=True).cuda()
    id_criterion.eval()
    for p in id_criterion.parameters():
        p.requires_grad_(False)
    print('** IDLoss (ArcFace) initialization success !')

    attr_teacher = AttributeClassifier(backbone='r34')
    attr_teacher.load_state_dict(load_network(args.attribute_weights))
    attr_teacher.cuda().eval()
    for p in attr_teacher.parameters():
        p.requires_grad_(False)
    print('** Frozen attribute teacher initialization success !')

    diffusion_guidance = None
    if args.use_diffusion_guidance:
        from models.diffusion_guidance import FrozenDiffusionDDSGuidance
        diffusion_guidance = FrozenDiffusionDDSGuidance(
            model_id=args.diffusion_model_id,
            vae_model_id=args.diffusion_vae_model_id or None,
            image_size=args.diffusion_image_size,
            timestep_min=args.diffusion_timestep_min,
            timestep_max=args.diffusion_timestep_max,
            guidance_scale=args.diffusion_guidance_scale,
            fp16=args.diffusion_fp16,
        ).cuda()
        print(f'** Frozen diffusion DDS guidance enabled: {args.diffusion_model_id}  '
              f'weight={args.diffusion_guidance_weight}  interval={args.diffusion_guidance_interval}')

    clip_prompt_loss_fn = None
    if args.use_clip_prompt_loss:
        from models.clip_prompt_loss import FrozenCLIPPromptLoss
        clip_prompt_loss_fn = FrozenCLIPPromptLoss(
            clip_model=args.clip_prompt_model,
            temperature=args.clip_prompt_temperature,
        ).cuda().eval()
        for p in clip_prompt_loss_fn.parameters():
            p.requires_grad_(False)
        print(f'** CLIP prompt loss enabled: {args.clip_prompt_model}  '
              f'weight={args.clip_prompt_weight}  interval={args.clip_prompt_interval}')

    test_loader_iter = iter(test_loader)
    fixed_preview_batch = None
    if args.preview_mode == 'fixed_balanced':
        fixed_preview_batch = _build_fixed_preview_batch(
            test_dataset,
            args.attribute_index,
            test_batch,
        )
        print(f'** fixed preview indices: {fixed_preview_batch[-1]}')

    adaptive_rs_enabled = bool(args.adaptive_residual_scale)
    if adaptive_rs_enabled:
        if direction_bank is None:
            raise ValueError('--adaptive_residual_scale requires --direction_bank_path')
        if not (0 <= args.adaptive_rs_attr_idx < len(args.attribute_index)):
            raise ValueError(
                f'--adaptive_rs_attr_idx must be in [0, {len(args.attribute_index) - 1}], '
                f'got {args.adaptive_rs_attr_idx}'
            )
        if args.adaptive_rs_interval <= 0:
            raise ValueError('--adaptive_rs_interval must be positive')
        if not (0.0 <= args.adaptive_rs_ema_decay < 1.0):
            raise ValueError('--adaptive_rs_ema_decay must be in [0, 1)')

        _ars_target_idx = int(args.adaptive_rs_attr_idx)
        _ars_target_loss_ema = [
            float(args.adaptive_rs_init_ema) for _ in range(len(args.attribute_index))
        ]
        with torch.no_grad():
            initial_rs = float(direction_bank.residual_scale[_ars_target_idx].detach().cpu().item())
            initial_rs = min(max(initial_rs, args.adaptive_rs_min), args.adaptive_rs_max)
            direction_bank.residual_scale[_ars_target_idx] = initial_rs
        _ars_current = initial_rs
        print(
            f'[AdaptiveRS] enabled for attr local_idx={_ars_target_idx} '
            f'(attr={args.attribute_index[_ars_target_idx]}), initial rs={_ars_current:.4f}, '
            f'init_ema={args.adaptive_rs_init_ema}, decay={args.adaptive_rs_ema_decay}, '
            f'threshold={args.adaptive_rs_threshold}, step_size={args.adaptive_rs_step_size}, '
            f'interval={args.adaptive_rs_interval}, '
            f'range=[{args.adaptive_rs_min}, {args.adaptive_rs_max}]'
        )
    else:
        _ars_target_idx = -1
        _ars_target_loss_ema = [1.0 for _ in range(len(args.attribute_index))]
        _ars_current = 0.0
        
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for i, datas in tqdm(enumerate(train_loader),total=len(train_loader)):
            local_step = epoch*len(train_loader)+i
            n_iter = start_step + local_step
            
            img,latent,pred = datas
            img = img.cuda()
            latent = latent.cuda()
            pred = pred.cuda()
            attributes = pred[:,attribute_index]
            zero_pad = torch.zeros(args.batch, 18, 1, device=latent.device, dtype=latent.dtype)
            
            _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
            id_cond_train = apply_id_condition_dropout(id_cond, args.id_cond_dropout)
            src_cond = torch.cat([id_cond_train, attr_cond], dim=1)
            kd_loss = F.mse_loss(attr_cond, attributes)

            if args.attribute_sampling == 'cycle':
                modify_idx = torch.full(
                    (args.batch,),
                    n_iter % len(args.attribute_index),
                    dtype=torch.long,
                    device=attributes.device,
                )
            else:
                modify_idx, _ = modify_one_attribute(attributes, mode='negative')
            batch_indices = torch.arange(args.batch, device=latent.device)
            mid_idx = modify_idx.to(latent.device).view(-1)
            if mid_idx.numel() == 1:
                mid_idx = mid_idx.expand(args.batch)
            src_attr_flow = attr_cond[batch_indices, mid_idx].detach()
            approx21, delta_log_p2 = prior(latent, src_cond, zero_pad)

            # make base distribution standard normal distibution
            approx2 = standard_normal_logprob(approx21).view(args.batch, -1).sum(1, keepdim=True)
            delta_log_p2 = delta_log_p2.view(args.batch, -1).sum(1, keepdim=True)
            log_p2 = -(approx2 - delta_log_p2).mean() / (18*512)

            scale_noise = (args.train_scale_max - args.train_scale_min) / 2.0
            train_scale = attr_scales.get_attr_train_scale(mid_idx, base_noise=scale_noise).to(
                device=latent.device,
                dtype=latent.dtype,
            )
            hard_flow_target = compute_soft_targets(src_attr_flow, mid_idx)
            soft_flow_target = src_attr_flow + train_scale * (hard_flow_target - src_attr_flow)
            new_attr_cond = attr_cond.detach().clone()
            new_attr_cond = new_attr_cond.scatter(1, mid_idx.view(-1, 1), soft_flow_target.view(-1, 1))
            new_cond = torch.cat([id_cond_train.detach(), new_attr_cond], dim=1)
            new_latents_raw, _ = prior(approx21, new_cond, zero_pad, reverse=True)
            lag_dof_losses = collect_lag_dof_losses(prior)
            flow_delta = new_latents_raw - latent
            direction_bank_applied = False
            glasses_bank_bypass_fraction = latent.new_tensor(0.0)
            # is_glasses_bypass: [B] bool — samples whose delta comes from raw flow, not direction bank
            is_glasses_bypass = torch.zeros(latent.shape[0], dtype=torch.bool, device=latent.device)

            if args.velocity_field == 'original':
                lm = layer_mask(mid_idx, src_attr_flow, soft_flow_target).unsqueeze(-1)
                guided_delta = lm * flow_delta
            elif direction_bank is not None:
                attr_delta = new_attr_cond - attr_cond.detach()
                if args.bypass_glasses_direction_bank and eyeglasses_local_idx is not None:
                    is_glasses_bypass = mid_idx == int(eyeglasses_local_idx)
                glasses_bank_bypass_fraction = is_glasses_bypass.float().mean()

                if is_glasses_bypass.all():
                    # All samples are eyeglasses — skip direction bank entirely
                    guided_delta = flow_delta
                else:
                    direction_bank_applied = True
                    _db_delta = direction_bank(flow_delta, attr_delta, attr_idx=mid_idx, latent=latent)
                    if is_glasses_bypass.any():
                        # Mixed batch: glasses samples use raw flow, others use direction bank
                        guided_delta = torch.where(
                            is_glasses_bypass.view(-1, 1, 1), flow_delta, _db_delta
                        )
                    else:
                        guided_delta = _db_delta
            else:
                guided_delta = flow_delta

            final_delta_norm_pre_clip = guided_delta.reshape(guided_delta.shape[0], -1).norm(dim=1).mean().detach()
            guided_delta, final_delta_clip = cap_delta_norm(guided_delta, args.final_delta_max_norm)
            final_delta_norm = guided_delta.reshape(guided_delta.shape[0], -1).norm(dim=1).mean().detach()
            final_delta_clip_factor = final_delta_clip.mean().detach()
            safe_delta = guided_delta
            new_latents = latent + safe_delta
            
            new_face_tensors = G([new_latents],input_is_latent=True,randomize_noise=False)[0].clamp(-1, 1)
            new_face_tensors = F.interpolate(new_face_tensors, (args.img_size, args.img_size))
            reg_loss_global = (new_latents[:, :2, :] - latent[:, :2, :]).pow(2).mean(dim=(1, 2))
            reg_loss_coarse = (new_latents[:, 2:4, :] - latent[:, 2:4, :]).pow(2).mean(dim=(1, 2))
            reg_loss_fine = (new_latents[:, 4:, :] - latent[:, 4:, :]).pow(2).mean(dim=(1, 2))
            reg_loss = (2.0 * reg_loss_global + 1.0 * reg_loss_coarse + args.reg_fine_weight * reg_loss_fine).mean()

            # Identity loss: edited face should preserve identity of original
            id_src_feat = F.normalize(id_criterion.extract_features(img), dim=1).detach()
            id_edit_feat = F.normalize(id_criterion.extract_features(new_face_tensors), dim=1)
            id_loss = 1.0 - F.cosine_similarity(id_edit_feat, id_src_feat, dim=1).mean()

            # Frozen counterfactual teacher: external supervision on visual attribute change
            new_face_256 = F.interpolate(new_face_tensors, (256, 256))
            src_face_256 = F.interpolate(img, (256, 256))
            src_logits, _ = attr_teacher(src_face_256)
            gen_logits, _ = attr_teacher(new_face_256)
            src_probs = torch.sigmoid(src_logits)[:, attribute_index].detach()
            gen_probs = torch.sigmoid(gen_logits)[:, attribute_index]
            target_probs = src_probs.clone()
            src_attr = src_probs[batch_indices, mid_idx]
            hard_teacher_target = compute_soft_targets(src_attr, mid_idx)
            soft_target = src_attr + train_scale * (hard_teacher_target - src_attr)
            soft_target_for_loss = soft_target.detach()
            target_probs[batch_indices, mid_idx] = soft_target_for_loss
            edited_probs = gen_probs[batch_indices, mid_idx]
            changed_mse_per_sample = (edited_probs - soft_target_for_loss).pow(2)
            changed_loss = changed_mse_per_sample.mean()

            _zero = torch.zeros([], device=latent.device, dtype=latent.dtype)
            adaptive_rs_observed = _zero.detach().clone()
            adaptive_rs_fired = _zero.detach().clone()
            if adaptive_rs_enabled and direction_bank is not None:
                ars_mask = mid_idx == _ars_target_idx
                if ars_mask.any():
                    observed = changed_mse_per_sample[ars_mask].detach().mean().item()
                    _ars_target_loss_ema[_ars_target_idx] = (
                        args.adaptive_rs_ema_decay * _ars_target_loss_ema[_ars_target_idx]
                        + (1.0 - args.adaptive_rs_ema_decay) * observed
                    )
                    adaptive_rs_observed = latent.new_tensor(observed)

                if (
                    local_step > 0
                    and local_step % args.adaptive_rs_interval == 0
                    and _ars_target_loss_ema[_ars_target_idx] < args.adaptive_rs_threshold
                ):
                    old_rs = _ars_current
                    _ars_current = min(
                        _ars_current + args.adaptive_rs_step_size,
                        args.adaptive_rs_max,
                    )
                    _ars_current = max(_ars_current, args.adaptive_rs_min)
                    with torch.no_grad():
                        direction_bank.residual_scale[_ars_target_idx] = _ars_current
                    if _ars_current != old_rs:
                        adaptive_rs_fired = torch.ones([], device=latent.device, dtype=latent.dtype).detach()
                        print(
                            f'[AdaptiveRS] step {n_iter}: '
                            f'attr={args.attribute_index[_ars_target_idx]} '
                            f'target_ema={_ars_target_loss_ema[_ars_target_idx]:.4f} '
                            f'< {args.adaptive_rs_threshold}, '
                            f'residual_scale {old_rs:.4f} -> {_ars_current:.4f}'
                        )

            # Vectorized: build a boolean mask [B, num_attrs] where True = non-target attr.
            # In cycle mode all samples share the same mid_idx, but this handles random mode too.
            attr_range = torch.arange(len(args.attribute_index), device=latent.device)
            preserve_mask = attr_range.unsqueeze(0) != mid_idx.unsqueeze(1)  # [B, num_attrs]
            if preserve_mask.any():
                diff_sq = (gen_probs - src_probs).pow(2)  # [B, num_attrs]
                preserve_loss = (diff_sq * preserve_mask.float()).sum() / preserve_mask.float().sum()
            else:
                preserve_loss = _zero.clone()
            counter_attr_loss = changed_loss + args.preserve_attr_weight * preserve_loss

            # ── Frozen CLIP semantic target loss ──────────────────────
            clip_semantic_loss = _zero.clone()
            clip_logs = {}
            if (
                clip_prompt_loss_fn is not None
                and args.clip_prompt_weight > 0
                and (args.clip_prompt_interval <= 1 or n_iter % args.clip_prompt_interval == 0)
            ):
                _clip_abs_idx = torch.tensor(
                    [args.attribute_index[int(j)] for j in mid_idx.detach().cpu().tolist()],
                    device=latent.device, dtype=torch.long,
                )
                clip_loss_each, clip_logs = clip_prompt_loss_fn(
                    images=new_face_tensors,
                    attr_abs_idx=_clip_abs_idx,
                    target_values=soft_target_for_loss,
                    reduction='none',
                )   # clip_loss_each: (B,)
                clip_sample_weight = torch.ones_like(clip_loss_each)
                clip_sample_weight[_clip_abs_idx == 39] = args.clip_prompt_age_weight
                clip_sample_weight[_clip_abs_idx == 20] = args.clip_prompt_gender_weight
                clip_semantic_loss = (clip_sample_weight * clip_loss_each).mean()
                clip_logs['clip_prompt_age_fraction'] = (_clip_abs_idx == 39).float().mean().detach()
                clip_logs['clip_prompt_gender_fraction'] = (_clip_abs_idx == 20).float().mean().detach()

            if lag_dof_losses is None:
                lag_orth = _zero.clone()
                lag_gate_smooth = _zero.clone()
            else:
                lag_orth = lag_dof_losses['orth']
                lag_gate_smooth = lag_dof_losses['gate_smooth']

            id_warmup_steps = 1500
            id_weight = args.id_loss_weight * min(1.0, n_iter / max(1, id_warmup_steps))
            if direction_bank is not None:
                dir_orth_loss = direction_bank.orthogonality_loss()
                dir_logs = direction_bank.last_logs if direction_bank_applied else {}
            else:
                dir_orth_loss = _zero.clone()
                dir_logs = {}

            diffusion_loss = _zero.clone()
            diffusion_logs = {}
            if diffusion_guidance is not None and args.diffusion_guidance_weight > 0:
                mid_abs_idx = torch.tensor(
                    [args.attribute_index[int(j)] for j in mid_idx.detach().cpu().tolist()],
                    device=latent.device,
                    dtype=torch.long,
                )
                is_age = mid_abs_idx == 39

                non_age_fires = (
                    (~is_age).any()
                    and args.diffusion_guidance_interval > 0
                    and n_iter % args.diffusion_guidance_interval == 0
                )
                age_fires = (
                    is_age.any()
                    and args.age_diffusion_interval > 0
                    and n_iter % args.age_diffusion_interval == 0
                )

                # Fine-layer masking: DDS gradients only flow through coarse/mid W+ layers.
                # A separate G forward with detached fine layers avoids contaminating
                # id/reg gradients that legitimately need fine layer information.
                if (non_age_fires or age_fires) and args.dds_fine_layer_start > 0:
                    _nl_dds = torch.cat([
                        new_latents[:, :args.dds_fine_layer_start, :],
                        new_latents[:, args.dds_fine_layer_start:, :].detach(),
                    ], dim=1)
                    _ft_dds = G([_nl_dds], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
                    new_face_for_dds = F.interpolate(_ft_dds, (args.img_size, args.img_size))
                else:
                    new_face_for_dds = new_face_tensors

                # Non-age samples (glasses, gender): standard interval and timestep range
                non_age_mask = ~is_age
                if non_age_fires:
                    _loss, _logs = diffusion_guidance(
                        src_images=img[non_age_mask],
                        edit_images=new_face_for_dds[non_age_mask],
                        attr_abs_idx=mid_abs_idx[non_age_mask],
                        target_values=soft_target[non_age_mask].detach(),
                    )
                    diffusion_loss = diffusion_loss + _loss
                    diffusion_logs.update(_logs)

                # Age samples: coarse timestep range [400, 900], longer interval
                if age_fires:
                    _loss, _logs = diffusion_guidance(
                        src_images=img[is_age],
                        edit_images=new_face_for_dds[is_age],
                        attr_abs_idx=mid_abs_idx[is_age],
                        target_values=soft_target[is_age].detach(),
                        timestep_min=args.age_diffusion_timestep_min,
                        timestep_max=args.age_diffusion_timestep_max,
                    )
                    diffusion_loss = diffusion_loss + _loss
                    diffusion_logs.update({f'age_{k}': v for k, v in _logs.items()})

            loss = args.kd_loss_weight * kd_loss +\
                args.nll_loss_weight * log_p2 +\
                args.reg_loss_weight * reg_loss +\
                id_weight * id_loss +\
                args.counter_attr_weight * counter_attr_loss +\
                args.orth_loss_weight * lag_orth +\
                args.gate_smooth_weight * lag_gate_smooth +\
                args.direction_orth_weight * dir_orth_loss +\
                args.diffusion_guidance_weight * diffusion_loss +\
                args.clip_prompt_weight * clip_semantic_loss

            attr_scale_grad_norm = _zero.detach().clone()
            (loss / args.grad_accum_steps).backward()
            if (i + 1) % args.grad_accum_steps == 0:
                if attr_scales.attr_log_scales.grad is None:
                    attr_scale_grad_norm = _zero.detach().clone()
                else:
                    attr_scale_grad_norm = attr_scales.attr_log_scales.grad.detach().norm()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            
            if n_iter % args.save_freq==0:
                if fixed_preview_batch is not None:
                    test_img, test_latent, test_pred, _ = fixed_preview_batch
                else:
                    try:
                        _tb = next(test_loader_iter)
                    except StopIteration:
                        test_loader_iter = iter(test_loader)
                        _tb = next(test_loader_iter)
                    test_img, test_latent, test_pred = _tb
                test_img = test_img.cuda()
                test_latent = test_latent.cuda()
                test_attributes = test_pred[:, attribute_index].cuda()
                with torch.no_grad():
                    test_cond, test_id_cond, test_attr_cond = conditioner.make_condition(
                        test_img, test_latent, id_criterion
                    )
                    test_mid_latent, _ = prior(test_latent, test_cond, torch.zeros(test_batch, 18, 1).to(test_latent))
                    grid_img = generate_test_image(
                        prior, G, test_id_cond, test_attr_cond, test_img, test_latent, test_attributes,
                        test_mid_latent, layer_mask=layer_mask, direction_bank=direction_bank,
                        preview_scale=args.preview_scale,
                        args=args,
                    )
                    logger.save_image(grid_img,n_iter,'test')
                logger.checkpoints(n_iter)
                    
            _log_dict = {
                'loss_total': loss,
                'loss_nll': log_p2,
                'loss_target': changed_loss,
                'loss_leakage': preserve_loss,
                'loss_reg': reg_loss,
                'loss_id': id_loss,
                'id_weight': torch.tensor(id_weight),
                'reg_loss_global': reg_loss_global.mean(),
                'reg_loss_coarse': reg_loss_coarse.mean(),
                'reg_loss_fine': reg_loss_fine.mean(),
                'lag_orth': lag_orth,
                'lag_gate_smooth': lag_gate_smooth,
                'edit_scale': train_scale.detach().mean(),
                'attr_scale_grad_norm': attr_scale_grad_norm,
                'final_delta_norm_pre_clip': final_delta_norm_pre_clip,
                'final_delta_norm': final_delta_norm,
                'final_delta_clip_factor': final_delta_clip_factor,
                'final_delta_max_norm': torch.tensor(args.final_delta_max_norm),
                'adaptive_rs_ema': torch.tensor(_ars_target_loss_ema[_ars_target_idx])
                if adaptive_rs_enabled else _zero.detach().clone(),
                'adaptive_rs_current': torch.tensor(_ars_current)
                if adaptive_rs_enabled else _zero.detach().clone(),
                'adaptive_rs_observed': adaptive_rs_observed,
                'adaptive_rs_fired': adaptive_rs_fired,
                'dir_orth': dir_orth_loss,
                'dir_bank_flow_delta_norm': dir_logs.get('dir_bank_flow_delta_norm', _zero.detach().clone()),
                'dir_bank_dir_delta_norm': dir_logs.get('dir_bank_dir_delta_norm', _zero.detach().clone()),
                'dir_bank_residual_norm': dir_logs.get('dir_bank_residual_norm', _zero.detach().clone()),
                'dir_bank_guided_delta_norm_pre_clip': dir_logs.get('dir_bank_guided_delta_norm_pre_clip', _zero.detach().clone()),
                'dir_bank_guided_delta_norm': dir_logs.get('dir_bank_guided_delta_norm', _zero.detach().clone()),
                'dir_bank_residual_scale': dir_logs.get('dir_bank_residual_scale', _zero.detach().clone()),
                'dir_bank_active_direction_scale': dir_logs.get('dir_bank_active_direction_scale', _zero.detach().clone()),
                'dir_bank_active_delta_max_norm': dir_logs.get('dir_bank_active_delta_max_norm', _zero.detach().clone()),
                'dir_bank_global_delta_max_norm': dir_logs.get('dir_bank_global_delta_max_norm', _zero.detach().clone()),
                'dir_gate_entropy': dir_logs.get('dir_gate_entropy', _zero.detach().clone()),
                'glasses_bank_bypass_fraction': glasses_bank_bypass_fraction,
                'loss_diffusion_dds': diffusion_loss,
                'loss_clip_prompt':   clip_semantic_loss,
                'clip_score_mean':     clip_logs.get('clip_score_mean',     latent.new_tensor(0.0)),
                'clip_score_pos_mean': clip_logs.get('clip_score_pos_mean', latent.new_tensor(0.0)),
                'clip_score_neg_mean': clip_logs.get('clip_score_neg_mean', latent.new_tensor(0.0)),
                'clip_direction_mean': clip_logs.get('clip_direction_mean', latent.new_tensor(0.0)),
                'clip_prompt_weight':  torch.tensor(args.clip_prompt_weight),
                'clip_prompt_age_fraction':    clip_logs.get('clip_prompt_age_fraction',    latent.new_tensor(0.0)),
                'clip_prompt_gender_fraction': clip_logs.get('clip_prompt_gender_fraction', latent.new_tensor(0.0)),
            }
            current_attr_scales = attr_scales.current_scales()
            for _i, _attr_abs_idx in enumerate(args.attribute_index):
                _log_dict[f'attr_scale/attr_{_attr_abs_idx}'] = current_attr_scales[_i]
            for _k, _v in diffusion_logs.items():
                _log_dict[_k] = _v
            logger.msg(_log_dict, n_iter)
            
