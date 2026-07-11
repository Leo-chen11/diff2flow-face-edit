import argparse
import copy
import json
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


def _inverse_softplus(x):
    x = x.clamp(min=1e-6)
    return torch.log(torch.expm1(x))


class LearnableRegLossWeights(nn.Module):
    """Per-attribute learnable weights for the global/coarse/fine W+ regularization
    loss groups, replacing the fixed 2.0/1.0/reg_fine_weight constants that used to be
    shared identically by every attribute. Softplus reparam (same style as
    AttributeDirectionBank.residual_scale_raw) keeps weights positive with no upper
    bound; each attribute finds its own global/coarse/fine balance via gradient
    descent on the same losses already driving training (id_loss pulls a group's
    weight up when moving there is hurting identity, counter_attr_loss pulls it down
    when more freedom in that group is needed to hit the target).
    """

    def __init__(self, n_edit_attrs, init_global=2.0, init_coarse=1.0, init_fine=0.5):
        super().__init__()
        self.n_edit_attrs = int(n_edit_attrs)
        init = torch.tensor([float(init_global), float(init_coarse), float(init_fine)])
        init = init.unsqueeze(0).repeat(self.n_edit_attrs, 1)   # (n_edit_attrs, 3)
        self.log_weights_raw = nn.Parameter(_inverse_softplus(init))

    def weights_for(self, attr_local_idx):
        """attr_local_idx: LongTensor [B] -> (B, 3) tensor of [global, coarse, fine] weights."""
        weights = F.softplus(self.log_weights_raw)   # (n_edit_attrs, 3)
        return weights[attr_local_idx]

    def current_weights(self):
        with torch.no_grad():
            return F.softplus(self.log_weights_raw)   # (n_edit_attrs, 3)


class CrossAttributeLossBalancer:
    """Keeps changed_loss progress comparable across attributes sharing one
    training loop, instead of letting a fixed --counter_attr_weight apply
    equally regardless of how hard id_loss/reg_loss naturally fight each
    attribute's edit (e.g. aging a face moves the ArcFace embedding far more
    than adding glasses does, for the same semantic progress).

    Tracks an EMA of changed_loss per attribute relative to where that
    attribute's loss started, and raises the changed_loss weight for
    whichever attribute is lagging behind the others' relative progress.
    The same update rule runs identically for every attribute index; nothing
    here is keyed to a specific attribute, so it applies unchanged if
    attribute_index gains or loses entries.
    """

    def __init__(self, num_attrs, ema_decay=0.98, adapt_rate=0.05,
                 min_weight=0.25, max_weight=4.0, device='cuda'):
        self.num_attrs = int(num_attrs)
        self.ema_decay = float(ema_decay)
        self.adapt_rate = float(adapt_rate)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.initial_loss = torch.zeros(self.num_attrs, device=device)
        self.ema_loss = torch.zeros(self.num_attrs, device=device)
        self.initialized = torch.zeros(self.num_attrs, dtype=torch.bool, device=device)
        self.weights = torch.ones(self.num_attrs, device=device)

    @torch.no_grad()
    def update(self, attr_local_idx, changed_loss_per_sample):
        """attr_local_idx: LongTensor [B]. changed_loss_per_sample: FloatTensor [B]."""
        for a in range(self.num_attrs):
            mask = attr_local_idx == a
            if not mask.any():
                continue
            v = changed_loss_per_sample[mask].mean()
            if not self.initialized[a]:
                self.initial_loss[a] = v
                self.ema_loss[a] = v
                self.initialized[a] = True
            else:
                self.ema_loss[a] = self.ema_decay * self.ema_loss[a] + (1.0 - self.ema_decay) * v

        if bool(self.initialized.all()):
            ratio = self.ema_loss / self.initial_loss.clamp(min=1e-8)
            mean_ratio = ratio.mean().clamp(min=1e-8)
            relative_lag = (ratio / mean_ratio).clamp(min=1e-3)
            self.weights = (self.weights * relative_lag.pow(self.adapt_rate)).clamp(
                self.min_weight, self.max_weight
            )
            # Renormalize to mean 1 so the overall loss scale (and therefore
            # --counter_attr_weight and every other fixed weight) stays
            # meaningful without retuning.
            self.weights = self.weights / self.weights.mean().clamp(min=1e-8)

    def weights_for(self, attr_local_idx):
        """attr_local_idx: LongTensor [B] -> FloatTensor [B] of per-sample weights."""
        return self.weights[attr_local_idx]


# Soft-target policy keyed by ABSOLUTE CelebA attribute index. The old version
# keyed on local position (0/1/2 assumed to be glasses/gender/age), so any
# other --attribute_index ordering or attribute set silently mis-targeted
# every edit with no error.
SOFT_TARGET_TABLE = {
    15: (0.10, 0.90),   # eyeglasses needs a stronger local-edit signal
    20: (0.20, 0.80),   # gender should move without forcing a full identity flip
    39: (0.20, 0.80),   # age is the most identity-sensitive edit; conservative
}
DEFAULT_SOFT_TARGET = (0.20, 0.80)


def compute_soft_targets(src_vals, attr_local_idx, attribute_index):
    """attribute_index: the --attribute_index list mapping local -> absolute idx."""
    targets = torch.empty_like(src_vals)
    for local in torch.unique(attr_local_idx):
        abs_idx = int(attribute_index[int(local.item())])
        low, high = SOFT_TARGET_TABLE.get(abs_idx, DEFAULT_SOFT_TARGET)
        mask = attr_local_idx == local
        targets[mask] = torch.where(src_vals[mask] > 0.5,
                                    torch.full_like(src_vals[mask], low),
                                    torch.full_like(src_vals[mask], high))
    return targets


def teacher_augment(src_images, edit_images, enabled=True, noise_std=0.02, out_size=256):
    """Shared-parameter, differentiable augmentation applied to BOTH the source
    and the edited image before the frozen attribute teacher.

    Purpose: break adversarial teacher-fooling. The independent-judge eval shows
    a ~30pp gap between teacher accuracy and CLIP accuracy (e.g. Eyeglasses 91%
    vs 58%), i.e. the generator learns high-frequency patterns that move the
    teacher's logits without a real semantic change. Random crop/flip/resize and
    shared noise destroy such patterns while leaving true semantics intact.
    Gradients still flow to the generator (crop/flip/interpolate/add are all
    differentiable); parameters are shared so src/edit scores stay comparable
    for preserve_loss.
    """
    if not enabled:
        return (F.interpolate(src_images, (out_size, out_size)),
                F.interpolate(edit_images, (out_size, out_size)))
    H, W = src_images.shape[-2:]
    s = float(torch.empty(1).uniform_(0.85, 1.0))
    ch, cw = int(H * s), int(W * s)
    top = int(torch.randint(0, H - ch + 1, (1,)))
    left = int(torch.randint(0, W - cw + 1, (1,)))
    src = src_images[:, :, top:top + ch, left:left + cw]
    edit = edit_images[:, :, top:top + ch, left:left + cw]
    if float(torch.rand(1)) < 0.5:
        src = torch.flip(src, [-1])
        edit = torch.flip(edit, [-1])
    src = F.interpolate(src, (out_size, out_size), mode='bilinear', align_corners=False)
    edit = F.interpolate(edit, (out_size, out_size), mode='bilinear', align_corners=False)
    if noise_std > 0:
        noise = torch.randn_like(edit) * noise_std
        src = (src + noise).clamp(-1, 1)
        edit = (edit + noise).clamp(-1, 1)
    return src, edit


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


@torch.no_grad()
def update_ema(ema_module, module, decay):
    """In-place EMA update: ema_module <- decay * ema_module + (1 - decay) * module.
    Buffers (e.g. frozen direction_units) are copied directly, not averaged."""
    ema_params = dict(ema_module.named_parameters())
    for name, p in module.named_parameters():
        ema_params[name].mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    ema_buffers = dict(ema_module.named_buffers())
    for name, b in module.named_buffers():
        ema_buffers[name].copy_(b)


def make_ema_copy(module):
    """Deep-copy a module into a frozen (no-grad, eval-mode) EMA shadow starting
    identical to the live weights."""
    ema = copy.deepcopy(module)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


def save_ema_checkpoints(save_root, step, ema_modules):
    """Save EMA shadow modules under save_models_ema/ using the same prefix
    naming convention as logger.checkpoints() uses under save_models/, so
    eval scripts can load them unmodified by pointing --ckpt_dir at
    save_models_ema/ instead of save_models/."""
    ema_dir = os.path.join(save_root, 'save_models_ema')
    os.makedirs(ema_dir, exist_ok=True)
    for name, module in ema_modules.items():
        torch.save(
            module.state_dict(),
            os.path.join(ema_dir, '{}-{}'.format(name, str(step).zfill(7))),
        )


def cap_delta_norm(delta, max_norm):
    if max_norm is None or max_norm <= 0:
        clip = torch.ones(delta.shape[0], device=delta.device, dtype=delta.dtype)
        return delta, clip
    delta_norm = delta.reshape(delta.shape[0], -1).norm(dim=1)
    clip = (float(max_norm) / delta_norm.clamp(min=1e-8)).clamp(max=1.0)
    return delta * clip.view(-1, 1, 1), clip


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
            guided_delta = direction_bank(flow_delta, attr_delta, attr_idx=attr_idx, latent=source_latent)
            new_latents = source_latent + guided_delta
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
    parser.add_argument('--meta_lr_mult', type=float, default=1.0,
                        help='LR multiplier for magnitude meta-params (attr_scales, '
                             'reg_loss_weights, direction-bank residual_scale). The old '
                             'hardcoded 0.1 froze them at init for the whole run.')
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
    # Independent-judge eval evidence: the model is deployed at edit strengths
    # 0.9-1.25, but the old 0.35-0.55 range meant training never saw anything
    # stronger than 0.55 and inference had to extrapolate. Cover the deployed
    # range instead.
    parser.add_argument('--train_scale_min', type=float, default=0.5)
    parser.add_argument('--train_scale_max', type=float, default=0.9)
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
    parser.add_argument('--teacher_aug', action=argparse.BooleanOptionalAction, default=True,
                        help='Shared-parameter random crop/flip/noise on src+edited images '
                             'before the frozen attribute teacher, to break adversarial '
                             'teacher-fooling (independent-judge eval showed ~30pp gap '
                             'between teacher and CLIP accuracy without it).')
    parser.add_argument('--teacher_aug_noise', type=float, default=0.02,
                        help='Std of the shared gaussian noise in --teacher_aug.')

    # ── Cross-attribute loss balancing ──────────────────────────────────────
    parser.add_argument('--balance_attr_losses', action=argparse.BooleanOptionalAction, default=False,
                        help='Reweight changed_loss per attribute based on measured relative '
                             'training progress (see CrossAttributeLossBalancer), instead of a '
                             'single --counter_attr_weight applied identically to every attribute.')
    parser.add_argument('--balance_ema_decay', type=float, default=0.98,
                        help='EMA decay for each attribute changed_loss estimate used by the balancer.')
    parser.add_argument('--balance_adapt_rate', type=float, default=0.05,
                        help='How aggressively weights move toward equalizing relative progress each update.')
    parser.add_argument('--balance_min_weight', type=float, default=0.25)
    parser.add_argument('--balance_max_weight', type=float, default=4.0)
    parser.add_argument('--orth_loss_weight', type=float, default=0.005)
    parser.add_argument('--gate_smooth_weight', type=float, default=0.003)
    parser.add_argument('--reg_global_weight_init', type=float, default=2.0,
                        help='Initial global-layer reg_loss weight, per attribute; then learned '
                             '(see LearnableRegLossWeights). Has no effect after the first step.')
    parser.add_argument('--reg_coarse_weight_init', type=float, default=1.0,
                        help='Initial coarse-layer reg_loss weight, per attribute; then learned.')
    parser.add_argument('--reg_fine_weight', type=float, default=0.5,
                        help='Initial fine-layer reg_loss weight, per attribute; then learned.')
    parser.add_argument('--direction_bank_path', default=None, type=str,
                        help='Path to precomputed Attribute Direction Bank (.pth).')
    # Independent-judge eval evidence: with the old 0.05 init the residual (the
    # flow's per-sample contribution) stayed ~5% of the final delta for the whole
    # run, so Eyeglasses/Young hit a dataset-mean-direction ceiling (~60% real
    # accuracy). Start higher so the flow's personalization is actually in play.
    parser.add_argument('--direction_residual_scale', type=float, default=0.15)
    parser.add_argument('--direction_freeze', '--direction-freeze',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--direction_orth_weight', type=float, default=0.0)
    parser.add_argument('--direction_k', type=int, default=1,
                        help='Number of mixture directions per attribute in the Direction Bank.')
    parser.add_argument('--direction_guided_delta_max_norm', type=float, default=0.0,
                        help='Optional global max norm inside Direction Bank after all controls. '
                             'Use 10-12 for safe stage fine-tuning; 0 disables.')
    parser.add_argument('--final_delta_max_norm', type=float, default=0.0,
                        help='Optional final max norm after direction-bank mixing. 0 disables.')
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

    # ── EMA (exponential moving average) of trainable weights ──────────────
    parser.add_argument('--use_ema', action=argparse.BooleanOptionalAction, default=True,
                        help='Track an EMA shadow copy of prior/conditioner/attr_scales/'
                             'reg_loss_weights/direction_bank, saved alongside the live '
                             'checkpoints under save_models_ema/. Point --ckpt_dir at that '
                             'directory at eval time to use the EMA weights instead of the '
                             'raw (noisier) live weights; no eval code changes needed since '
                             'the file naming matches save_models/ exactly.')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay per optimizer step. 0.999 ~= averaging over the last '
                             '~1000 steps. Higher = smoother but slower to reflect recent training.')

    # ── Hinge identity loss ─────────────────────────────────────────────────
    parser.add_argument('--id_loss_hinge', action=argparse.BooleanOptionalAction, default=False,
                        help='Replace the continuous id_loss (1 - cosine_sim, always pulling) with '
                             'a hinge loss that is exactly zero once id cosine similarity is at or '
                             'above --id_hinge_threshold, and only pulls below that floor. Lets the '
                             'model spend its full editing budget above the safety line instead of '
                             'constantly fighting a continuous pull, without weakening the floor '
                             'itself. This threshold is a fixed safety policy, not a per-attribute '
                             'learnable target -- a learnable floor could degenerate toward 0 and '
                             'remove the safety net it exists to provide.')
    parser.add_argument('--id_hinge_threshold', type=float, default=0.8,
                        help='Identity cosine-similarity floor for --id_loss_hinge.')
    args = parser.parse_args()
    torch.manual_seed(0)

    os.environ['WANDB_MODE'] = args.wandb_mode

    wandb_kwargs = dict(
        project=args.wandb_project,
        name='{}_{}'.format(args.model_name,args.run_name),
    )
    if args.wandb_entity:
        wandb_kwargs['entity'] = args.wandb_entity
    
    save_root = os.path.join('./output', args.model_name, args.run_name)
    logger = WANDBLoggerX(save_root=save_root,
                          print_freq=args.print_freq,
                          config=args,
                          **wandb_kwargs)
    # Persist the full run config next to the checkpoints. evaluate_sdflow.py
    # auto-loads this file so eval model-structure flags can never silently
    # drift from what was trained (the strict=False loads would otherwise hide
    # such a mismatch completely).
    os.makedirs(save_root, exist_ok=True)
    with open(os.path.join(save_root, 'config.json'), 'w') as _f:
        json.dump(vars(args), _f, indent=2, default=str)
    print(f'** run config saved to {os.path.join(save_root, "config.json")}')
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
    reg_loss_weights = LearnableRegLossWeights(
        len(args.attribute_index),
        init_global=args.reg_global_weight_init,
        init_coarse=args.reg_coarse_weight_init,
        init_fine=args.reg_fine_weight,
    ).cuda()
    loss_balancer = None
    if args.balance_attr_losses:
        loss_balancer = CrossAttributeLossBalancer(
            len(args.attribute_index),
            ema_decay=args.balance_ema_decay,
            adapt_rate=args.balance_adapt_rate,
            min_weight=args.balance_min_weight,
            max_weight=args.balance_max_weight,
            device='cuda',
        )
        print(f'** Cross-attribute loss balancing enabled for attrs {args.attribute_index}')
    trainable_params = list(prior.parameters()) + list(conditioner.parameters())
    if args.velocity_field == 'original':
        trainable_params += list(layer_mask.parameters())
    if args.direction_bank_path:
        # Read K from the bank file itself rather than trusting --direction_k:
        # editor.py and evaluate_sdflow.py both do the same at load time, so
        # --direction_k silently disagreeing with the bank's real K would build
        # a checkpoint here (e.g. K truncated 4->1, discarding 3 of 4 stratified
        # directions per attribute) that those scripts then fail to load with a
        # direction_units shape mismatch. There is only one correct K per bank
        # file; don't let two independently-set numbers disagree about it.
        _bank_meta = torch.load(args.direction_bank_path, map_location='cpu')
        _bank_num_k = int(_bank_meta.get('num_k', args.direction_k)) if isinstance(_bank_meta, dict) else args.direction_k
        if _bank_num_k != args.direction_k:
            print(f'** Direction Bank: --direction_k={args.direction_k} ignored, '
                  f'using num_k={_bank_num_k} from {args.direction_bank_path}')
        # No per-attribute direction_scale/layer_scale/delta_max_norm, and no
        # per-attribute residual_scale init either: every attribute starts from
        # the same residual_scale value and learns its own from there via
        # gradient descent (see AttributeDirectionBank.residual_scale_raw). The
        # only magnitude safety net set here in advance is guided_delta_max_norm.
        direction_bank = AttributeDirectionBank(
            num_attrs=len(args.attribute_index),
            num_layers=18,
            latent_dim=512,
            num_k=_bank_num_k,
            bank_path=args.direction_bank_path,
            attribute_index=args.attribute_index,
            residual_scale=args.direction_residual_scale,
            freeze_directions=args.direction_freeze,
            residual_max_norm=args.residual_max_norm,
            guided_delta_max_norm=(
                args.direction_guided_delta_max_norm
                if args.direction_guided_delta_max_norm > 0 else None
            ),
        ).cuda()
        if args.freeze_direction_bank_nets:
            for p in direction_bank.parameters():
                p.requires_grad_(False)
            print('** Direction Bank trainable nets frozen for safe fine-tuning')
        trainable_params += [
            p for p in direction_bank.parameters()
            if p.requires_grad and p is not direction_bank.residual_scale_raw
        ]
        print(f'** Direction Bank enabled: {args.direction_bank_path}')
    else:
        direction_bank = None
    trainable_params += list(attr_scales.parameters())
    trainable_params += list(reg_loss_weights.parameters())

    # Magnitude-controlling meta-parameters (edit-strength center, direction-bank
    # residual trust, reg_loss layer-group weights). The old hardcoded 0.1x lr,
    # combined with the softplus/exp reparam shrinking gradients near small
    # values, froze all of these at their init for the entire run (wandb:
    # residual_scale 0.05->0.0525 and attr_scale 1.0->1.02 over 65k steps) --
    # "learnable" in name only. Default multiplier is now 1.0 so they actually
    # learn; pass --meta_lr_mult 0.1 to reproduce the old frozen behavior.
    low_lr_params = [attr_scales.attr_log_scales, reg_loss_weights.log_weights_raw]
    if direction_bank is not None and direction_bank.residual_scale_raw.requires_grad:
        low_lr_params.append(direction_bank.residual_scale_raw)
    low_lr_ids = {id(p) for p in low_lr_params}

    optimizer = optim.Adam(
        [
            {'params': [p for p in trainable_params if id(p) not in low_lr_ids], 'lr': args.lr},
            {'params': low_lr_params, 'lr': args.lr * args.meta_lr_mult, 'weight_decay': 0.0},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader) // args.grad_accum_steps, eta_min=1e-6
    )

    ema_pairs = []       # [(live_module, ema_module, save_name), ...]
    ema_modules_for_save = {}
    if args.use_ema:
        prior_ema = make_ema_copy(prior)
        conditioner_ema = make_ema_copy(conditioner)
        attr_scales_ema = make_ema_copy(attr_scales)
        reg_loss_weights_ema = make_ema_copy(reg_loss_weights)
        ema_pairs = [
            (prior, prior_ema, 'prior'),
            (conditioner, conditioner_ema, 'conditioner'),
            (attr_scales, attr_scales_ema, 'attr_scales'),
            (reg_loss_weights, reg_loss_weights_ema, 'reg_loss_weights'),
        ]
        ema_modules_for_save = {name: ema for _, ema, name in ema_pairs}
        if direction_bank is not None:
            direction_bank_ema = make_ema_copy(direction_bank)
            ema_pairs.append((direction_bank, direction_bank_ema, 'direction_bank'))
            ema_modules_for_save['direction_bank'] = direction_bank_ema
        print(f'** EMA enabled (decay={args.ema_decay}); shadow weights saved to '
              f'save_models_ema/, point --ckpt_dir there at eval time to use them.')

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
        try:
            load_module_checkpoint(reg_loss_weights, resume_save_dir, 'reg_loss_weights', start_step, strict=False)
        except FileNotFoundError:
            print('[Resume] reg_loss_weights checkpoint not found; using default init weights.')
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

        if args.use_ema:
            # EMA shadow copies were made from the pre-resume (freshly constructed)
            # weights above; now that the live weights have been resumed, either
            # resume the EMA shadow too (if it was saved) or re-sync it to match
            # the resumed live weights so it isn't stuck at the old init.
            ema_resume_dir = os.path.join(os.path.dirname(resume_save_dir), 'save_models_ema')
            for live_module, ema_module, name in ema_pairs:
                try:
                    load_module_checkpoint(ema_module, ema_resume_dir, name, start_step, strict=False)
                except FileNotFoundError:
                    ema_module.load_state_dict(live_module.state_dict())
                    print(f'[Resume] {name}_ema checkpoint not found; re-synced EMA to resumed live weights.')

    log_modules = [prior, conditioner, attr_scales, reg_loss_weights, optimizer]
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
            hard_flow_target = compute_soft_targets(src_attr_flow, mid_idx, args.attribute_index)
            soft_flow_target = src_attr_flow + train_scale * (hard_flow_target - src_attr_flow)
            new_attr_cond = attr_cond.detach().clone()
            new_attr_cond = new_attr_cond.scatter(1, mid_idx.view(-1, 1), soft_flow_target.view(-1, 1))
            new_cond = torch.cat([id_cond_train.detach(), new_attr_cond], dim=1)
            new_latents_raw, _ = prior(approx21, new_cond, zero_pad, reverse=True)
            lag_dof_losses = collect_lag_dof_losses(prior)
            flow_delta = new_latents_raw - latent
            direction_bank_applied = False

            if args.velocity_field == 'original':
                lm = layer_mask(mid_idx, src_attr_flow, soft_flow_target).unsqueeze(-1)
                guided_delta = lm * flow_delta
            elif direction_bank is not None:
                attr_delta = new_attr_cond - attr_cond.detach()
                direction_bank_applied = True
                guided_delta = direction_bank(flow_delta, attr_delta, attr_idx=mid_idx, latent=latent)
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
            reg_weights = reg_loss_weights.weights_for(mid_idx)   # (B, 3): [global, coarse, fine]
            reg_loss = (
                reg_weights[:, 0] * reg_loss_global
                + reg_weights[:, 1] * reg_loss_coarse
                + reg_weights[:, 2] * reg_loss_fine
            ).mean()

            # Identity loss: edited face should preserve identity of original.
            # Hinge mode (--id_loss_hinge) is exactly zero once id cosine similarity
            # is at or above --id_hinge_threshold, so the model can spend its full
            # editing budget above that floor instead of a continuous pull fighting
            # counter_attr_loss even when identity is already well preserved.
            id_src_feat = F.normalize(id_criterion.extract_features(img), dim=1).detach()
            id_edit_feat = F.normalize(id_criterion.extract_features(new_face_tensors), dim=1)
            id_cos_sim = F.cosine_similarity(id_edit_feat, id_src_feat, dim=1)
            if args.id_loss_hinge:
                id_loss = F.relu(args.id_hinge_threshold - id_cos_sim).mean()
            else:
                id_loss = 1.0 - id_cos_sim.mean()

            # Frozen counterfactual teacher: external supervision on visual attribute change.
            # Shared-parameter augmentation before the teacher breaks adversarial
            # teacher-fooling (see teacher_augment docstring).
            src_face_256, new_face_256 = teacher_augment(
                img, new_face_tensors,
                enabled=args.teacher_aug,
                noise_std=args.teacher_aug_noise,
            )
            src_logits, _ = attr_teacher(src_face_256)
            gen_logits, _ = attr_teacher(new_face_256)
            src_probs = torch.sigmoid(src_logits)[:, attribute_index].detach()
            gen_probs = torch.sigmoid(gen_logits)[:, attribute_index]
            target_probs = src_probs.clone()
            src_attr = src_probs[batch_indices, mid_idx]
            hard_teacher_target = compute_soft_targets(src_attr, mid_idx, args.attribute_index)
            soft_target = src_attr + train_scale * (hard_teacher_target - src_attr)
            soft_target_for_loss = soft_target.detach()
            target_probs[batch_indices, mid_idx] = soft_target_for_loss
            edited_probs = gen_probs[batch_indices, mid_idx]
            changed_mse_per_sample = (edited_probs - soft_target_for_loss).pow(2)
            if loss_balancer is not None:
                loss_balancer.update(mid_idx.detach(), changed_mse_per_sample.detach())
                balance_weights = loss_balancer.weights_for(mid_idx).detach()
                changed_loss = (balance_weights * changed_mse_per_sample).mean()
            else:
                changed_loss = changed_mse_per_sample.mean()

            _zero = torch.zeros([], device=latent.device, dtype=latent.dtype)
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
                if args.use_ema:
                    for live_module, ema_module, _name in ema_pairs:
                        update_ema(ema_module, live_module, args.ema_decay)
            
            
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
                if args.use_ema:
                    save_ema_checkpoints(logger.save_root, n_iter, ema_modules_for_save)
                    
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
            if direction_bank is not None:
                current_residual_scales = direction_bank.current_residual_scale().detach()
                for _i, _attr_abs_idx in enumerate(args.attribute_index):
                    _log_dict[f'residual_scale/attr_{_attr_abs_idx}'] = current_residual_scales[_i]
            current_reg_weights = reg_loss_weights.current_weights()
            for _i, _attr_abs_idx in enumerate(args.attribute_index):
                _log_dict[f'reg_weight_global/attr_{_attr_abs_idx}'] = current_reg_weights[_i, 0]
                _log_dict[f'reg_weight_coarse/attr_{_attr_abs_idx}'] = current_reg_weights[_i, 1]
                _log_dict[f'reg_weight_fine/attr_{_attr_abs_idx}'] = current_reg_weights[_i, 2]
            if loss_balancer is not None:
                for _i, _attr_abs_idx in enumerate(args.attribute_index):
                    _log_dict[f'balance_weight/attr_{_attr_abs_idx}'] = loss_balancer.weights[_i]
                    _log_dict[f'balance_ema_loss/attr_{_attr_abs_idx}'] = loss_balancer.ema_loss[_i]
            for _k, _v in diffusion_logs.items():
                _log_dict[_k] = _v
            logger.msg(_log_dict, n_iter)
            
