import argparse
import copy
import json
import os,shutil,sys

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

    def __init__(self, n_edit_attrs, init_global=2.0, init_coarse=1.0, init_fine=0.5,
                 min_weight=0.1):
        super().__init__()
        self.n_edit_attrs = int(n_edit_attrs)
        # Floor on every learned weight. Every main training loss pushes these
        # weights DOWN (regularization only ever hurts the other objectives),
        # and nothing pushes them up, so at full meta lr they degenerate to ~0
        # (observed: fine/attr_39 collapsed 0.5 -> 5e-6 by 46k steps, freeing
        # the fine W+ layers and producing texture/color artifacts). The floor
        # keeps per-attribute REBALANCING learnable while making total collapse
        # impossible.
        self.min_weight = float(min_weight)
        init = torch.tensor([float(init_global), float(init_coarse), float(init_fine)])
        init = init.unsqueeze(0).repeat(self.n_edit_attrs, 1)   # (n_edit_attrs, 3)
        self.log_weights_raw = nn.Parameter(_inverse_softplus(init))

    def weights_for(self, attr_local_idx):
        """attr_local_idx: LongTensor [B] -> (B, 3) tensor of [global, coarse, fine] weights."""
        weights = F.softplus(self.log_weights_raw).clamp(min=self.min_weight)
        return weights[attr_local_idx]

    def current_weights(self):
        with torch.no_grad():
            return F.softplus(self.log_weights_raw).clamp(min=self.min_weight)


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


class JudgePeakDeclineBalancer:
    """Weights CLIP-prompt loss by how far each attribute's independent-judge
    accuracy has fallen from its own best-seen (EMA-smoothed) value, instead
    of CrossAttributeLossBalancer's relative-progress-since-init signal.

    Built to fix a specific failure: a run's judge_celeb_acc rose to a
    plateau by ~10k steps, then Male declined steadily for the rest of
    training (0.68 -> 0.58 by 60k) while Eyeglasses kept climbing the whole
    time (0.3 -> 0.9). CrossAttributeLossBalancer never caught this, because
    it tracks the CLIP LOSS's own relative progress since its initial value
    -- Male's CLIP loss had dropped early and stayed low, so the balancer
    read it as "ahead" and kept its weight near the floor right through the
    accuracy decline the loss was blind to.

    The judge itself never receives gradient (scored under torch.no_grad() in
    the training loop's monitor block), so raising an attribute's weight here
    gives the model no way to earn it by fooling the judge -- only by
    genuinely recovering accuracy does its own EMA rise back toward its peak
    and its weight relax. The judge only routes more gradient through the
    REAL supervised loss (loss_clip_prompt); it never becomes the target.

    per_direction (default on) tracks add and rm as SEPARATE slots. Keying on
    the attribute alone measures the mean over both edit directions, and that
    mean can be restored by improving whichever direction is cheaper rather
    than the one that actually declined. That is exactly what the first run of
    this balancer did: Male's weight climbed from 20k on, its judge accuracy
    peaked HIGHER than the unbalanced baseline (0.75-0.8 vs 0.65-0.7) -- and
    the eval showed the gain had been bought entirely on the easy side, Male
    add 83.6% -> 92.8% while Male rm collapsed 68.1% -> 34.1%, a 58.7pp split.
    The judge was never fooled; the SUMMARY STATISTIC was satisfiable one-sided.
    Splitting the slots closes that route: a saturated add direction sits at
    its own peak and earns no uplift, so only the direction genuinely falling
    away from its own best is funded.
    """

    def __init__(self, num_attrs, ema_decay=0.98, gain=4.0,
                 min_weight=0.25, max_weight=4.0, warmup_updates=5,
                 per_direction=True, device='cuda'):
        self.num_attrs = int(num_attrs)
        self.per_direction = bool(per_direction)
        self.num_slots = self.num_attrs * 2 if self.per_direction else self.num_attrs
        self.ema_decay = float(ema_decay)
        self.gain = float(gain)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.warmup_updates = int(warmup_updates)
        self.ema_acc = torch.zeros(self.num_slots, device=device)
        self.peak_acc = torch.zeros(self.num_slots, device=device)
        self.n_updates = torch.zeros(self.num_slots, dtype=torch.long, device=device)
        self.weights = torch.ones(self.num_slots, device=device)

    def slot_index(self, attr_local_idx, is_rm=None):
        """Map (attribute, edit direction) -> slot. Layout is attr-major:
        slot 2a is attribute a's add direction, 2a+1 its rm direction."""
        if not self.per_direction:
            return attr_local_idx
        if is_rm is None:
            raise ValueError('per_direction balancer needs is_rm per sample')
        return attr_local_idx * 2 + is_rm.long()

    def slot_name(self, slot, attribute_index):
        """Human-readable key for logging, e.g. 'attr_20_rm'."""
        if not self.per_direction:
            return f'attr_{attribute_index[slot]}'
        return f'attr_{attribute_index[slot // 2]}_{"rm" if slot % 2 else "add"}'

    @torch.no_grad()
    def update(self, attr_local_idx, correct_per_sample, is_rm=None):
        """attr_local_idx: LongTensor [B] of attributes present at this judge
        tick. correct_per_sample: FloatTensor [B] in {0,1}, whether the judge
        called that sample's edit a success (same rule as
        evaluate_sdflow.strict_success). is_rm: BoolTensor [B], True where the
        source already HAS the attribute so the edit removes it -- the same
        add/rm convention evaluate_sdflow reports its direction split under."""
        slots = self.slot_index(attr_local_idx, is_rm)
        for a in torch.unique(slots).tolist():
            mask = slots == a
            v = correct_per_sample[mask].mean()
            if self.n_updates[a] == 0:
                self.ema_acc[a] = v
            else:
                self.ema_acc[a] = self.ema_decay * self.ema_acc[a] + (1.0 - self.ema_decay) * v
            self.n_updates[a] += 1
            # Only start tracking a peak once the EMA has had a few ticks to
            # settle -- the first handful are the climb off zero, not a peak
            # to decline from.
            if self.n_updates[a] > self.warmup_updates:
                self.peak_acc[a] = torch.maximum(self.peak_acc[a], self.ema_acc[a])

        settled = self.n_updates > self.warmup_updates
        if settled.any():
            decline = (self.peak_acc - self.ema_acc).clamp(min=0.0)
            target = 1.0 + self.gain * decline
            target = torch.where(settled, target, torch.ones_like(target))
            self.weights = target.clamp(self.min_weight, self.max_weight)
            # Renormalize to mean 1 over the settled slots only, so a slot
            # still in warmup doesn't get silently down-weighted by the
            # renormalization before it has a real peak to compare to. With
            # per_direction on, this is what funds a declining rm direction
            # out of a saturated add direction rather than out of thin air.
            denom = self.weights[settled].mean().clamp(min=1e-8) if settled.any() else 1.0
            self.weights = self.weights / denom

    def weights_for(self, attr_local_idx, is_rm=None):
        return self.weights[self.slot_index(attr_local_idx, is_rm)]


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

# Local attributes: edits that should only touch a specific facial region.
# Maps absolute CelebA attribute index -> BiSeNet parsing classes defining the
# region the edit is ALLOWED to change (see common/face_parser.py label map).
# Eyeglasses: brows(2,3) + eyes(4,5) + glasses(6). Global attributes such as
# gender(20)/age(39) are intentionally absent — a whole-face region mask would
# make the loss a no-op for them.
LOCAL_REGION_CLASSES = {
    15: [2, 3, 4, 5, 6],
}

# BiSeNet 'skin' class, used by --color_shift_loss_weight to measure whether
# an edit shifted the overall skin tone rather than changing texture/geometry.
SKIN_CLASS = [1]


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


def save_best_checkpoints(save_root, step, logger, ema_modules, metric, metric_name):
    """Mirror the run layout under <save_root>/best/ so eval tooling works unchanged.

    Writes save_models/, save_models_ema/ and a copy of config.json, which is
    what apply_run_config() and _latest_step() expect -- so evaluating the best
    checkpoint is just --checkpoint_dir <run>/best with no other flags. Previous
    contents are cleared first, so exactly one step remains and auto-detection
    picks it.
    """
    best_root = os.path.join(save_root, 'best')
    models_dir = os.path.join(best_root, 'save_models')
    ema_dir = os.path.join(best_root, 'save_models_ema')
    for d in (models_dir, ema_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    for name, module in zip(logger.module_names, logger.modules):
        torch.save(module.state_dict(),
                   os.path.join(models_dir, '{}-{}'.format(name, str(step).zfill(7))))
    for name, module in (ema_modules or {}).items():
        torch.save(module.state_dict(),
                   os.path.join(ema_dir, '{}-{}'.format(name, str(step).zfill(7))))

    cfg_src = os.path.join(save_root, 'config.json')
    if os.path.exists(cfg_src):
        shutil.copy2(cfg_src, os.path.join(best_root, 'config.json'))

    with open(os.path.join(best_root, 'best_info.json'), 'w') as f:
        json.dump({'step': int(step), 'metric_name': metric_name,
                   'metric': float(metric)}, f, indent=2)
    print(f'** new best {metric_name}={metric:.4f} at step {step} -> {best_root}')


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
    parser.add_argument('--save_freq', type=int, default=5000,help='save frequency')
    
    
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
    parser.add_argument('--local_region_loss_weight', type=float, default=0.5,
                        help='Weight for the face-parser locality loss on LOCAL attributes '
                             '(currently eyeglasses, see LOCAL_REGION_CLASSES): outside the '
                             'allowed region, the edited image must match the source '
                             'reconstruction pixel-wise. Directly attacks the ~55-60%% real '
                             'eyeglasses accuracy ceiling by forcing the edit budget into '
                             'the eye region instead of a diffuse whole-face "glasses-ness". '
                             'DEFAULT changed from 0.0 (off) to 0.5 (the previously-suggested '
                             'value) -- eyeglasses structure was found incomplete/imperfect '
                             'without it. Pass 0 to fully disable.')
    parser.add_argument('--face_parser_weights', default='./data/parsing_bisenet.pth',
                        help='BiSeNet weights for --local_region_loss_weight and '
                             '--dds_face_mask.')
    parser.add_argument('--dds_face_mask', action=argparse.BooleanOptionalAction, default=True,
                        help='Restrict the DDS diffusion-guidance gradient (models/'
                             'diffusion_guidance.py) to the region the edit is allowed to touch, '
                             'instead of the whole latent. Unmasked, DDS asks the frozen '
                             'diffusion model to denoise the ENTIRE image toward edit_prompt, so '
                             'background/hair/clothing all get nonzero gradient even though the '
                             'prompt has no real opinion about them -- wasted signal that '
                             'directly funds LeakCLIP and ID drift. Eyeglasses uses '
                             'LOCAL_REGION_CLASSES (brows/eyes/glasses only); Male/Young use '
                             'FaceParser.FACE_CLASSES, which already INCLUDES hair -- age\'s '
                             '"receding hairline" prompt cue is not cut by this. Loads a '
                             'FaceParser the same way --local_region_loss_weight does (shared '
                             'instance if both are set). 0 risk to identity/leakage, upside '
                             'only if DDS was actually spending gradient off-face. DEFAULT '
                             'changed from off to on -- pass --no-dds_face_mask to resume an '
                             'existing --resume_dir run unmasked (matching the DDS numerics it '
                             'was trained with) instead of introducing the mask mid-run.')
    parser.add_argument('--local_region_add_blur', type=float, default=15,
                        help='Mask dilation (gaussian blur sigma) for ADDITION-direction '
                             'local edits. The source face has no glasses pixels for '
                             'BiSeNet to label, so the precise mask cannot contain a '
                             'future frame; a heavily dilated eye/brow region acts as a '
                             'geometric prior for the frame footprint while still '
                             'forbidding hair/mouth/background changes. Removal edits '
                             'keep the precise mask (sigma 5).')
    parser.add_argument('--losses_vs_recon', action=argparse.BooleanOptionalAction, default=True,
                        help='Compare the edited image against the source RECONSTRUCTION '
                             'G(latent) instead of the real photo in id_loss, the directional '
                             'CLIP loss, and the DDS diffusion guidance. The real photo differs '
                             'from any generated image by (inversion gap) + (edit); referencing '
                             'it charges the fixed e4e/StyleGAN reconstruction error to the edit, '
                             'which the flow cannot remove without spending W+ budget on '
                             'reconstruction instead of the attribute. Concretely: id_loss then '
                             'optimizes a different quantity than evaluate_sdflow.py reports '
                             '(it measures edited-vs-reconstruction), the directional CLIP delta '
                             'carries a constant inversion offset that pulls it off the pos/neg '
                             'text axis, and the DDS source-branch subtraction no longer cancels '
                             'cleanly. The locality and color-shift losses already compare '
                             'against G(latent) for exactly this reason -- this makes the other '
                             'three consistent with them. Costs one extra frozen G forward per '
                             'step (no_grad) when no face-parser loss already needed it. Pass '
                             '--no-losses_vs_recon to restore the old real-photo references.')
    parser.add_argument('--color_shift_loss_weight', type=float, default=1.0,
                        help='Penalize the mean-RGB shift of the BiSeNet skin region between '
                             'source and edited face, for --color_shift_attrs samples. '
                             'A visual audit (scripts/dump_attr_failures.py, attr 39 direction '
                             'rm) found the model satisfies the aging edit largely via a '
                             'uniform red/orange skin-tone shift rather than genuine structural '
                             'aging (wrinkles, gray hair) -- confirmed across every fix tried so '
                             'far (direction changes, scale changes, directional CLIP loss all '
                             'left the color-cast unchanged). Unlike those, this acts directly '
                             'on the generated image pixels: it does not touch the direction '
                             'vector or edit magnitude, it makes the color-shift shortcut itself '
                             'costly, so satisfying the attribute loss has to come from '
                             'elsewhere. Genuine structural aging is a texture/geometry change, '
                             'not a uniform tone shift, so it is barely affected. DEFAULT raised '
                             'from 0.0 (off) to 1.0, the middle of the previously-suggested '
                             '0.5-2.0 range: this is the only mechanism in this file that acts '
                             'directly on the color-cast shortcut, and the reported symptom '
                             '("aging looks like a color wash, not real aging") is exactly what '
                             'it targets. Requires a loadable --face_parser_weights. 0 disables.')
    parser.add_argument('--color_shift_attrs', nargs='*', type=int, default=[39],
                        help='Attribute indices the color-shift regularizer applies to. '
                             'Default is age (39) only, since that is the attribute the visual '
                             'audit found relying on the color-shift shortcut.')

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
    parser.add_argument('--gate_sparse_weight', type=float, default=0.01,
                        help='L_sparse = mean|g_a|, from the method doc but never wired '
                             'into the loss until now. gate_smooth only pulls adjacent '
                             'layers toward each other -- it has no gradient pushing any '
                             'layer toward 0, so the gate has no incentive to ever close a '
                             'layer. scripts/inspect_gate.py confirmed this on a trained '
                             'checkpoint: every layer stayed in 0.79-0.99 regardless of '
                             'attribute or ODE time (degenerated to an always-open, '
                             'attribute-agnostic constant). Method doc suggested 0.01, '
                             'try 0.003 if edits become too weak.')
    parser.add_argument('--reg_global_weight_init', type=float, default=2.0,
                        help='Initial global-layer reg_loss weight, per attribute; then learned '
                             '(see LearnableRegLossWeights). Has no effect after the first step.')
    parser.add_argument('--reg_coarse_weight_init', type=float, default=1.0,
                        help='Initial coarse-layer reg_loss weight, per attribute; then learned.')
    parser.add_argument('--reg_fine_weight', type=float, default=0.5,
                        help='Initial fine-layer reg_loss weight, per attribute; then learned.')
    parser.add_argument('--reg_weight_min', type=float, default=0.1,
                        help='Hard floor for every learned reg-loss weight. All main losses '
                             'push these weights down and nothing pushes them up, so without '
                             'a floor they collapse to ~0 at full meta lr (v7: fine/attr_39 '
                             'hit 5e-6, freeing fine layers -> texture/color artifacts).')
    parser.add_argument('--direction_bank_path', default=None, type=str,
                        help='Path to precomputed Attribute Direction Bank (.pth).')
    # Independent-judge eval evidence: with the old 0.05 init the residual (the
    # flow's per-sample contribution) stayed ~5% of the final delta for the whole
    # run, so Eyeglasses/Young hit a dataset-mean-direction ceiling (~60% real
    # accuracy). Start higher so the flow's personalization is actually in play.
    parser.add_argument('--direction_residual_scale', type=float, default=0.15)
    parser.add_argument('--glasses_residual_scale', type=float, default=0.35,
                        help='Separate INITIAL residual_scale for eyeglasses (attr 15) at TRAINING '
                             'time, same mechanism as --age_residual_scale. scripts/'
                             'validate_direction_bank.py measures the eyeglasses direction ALONE '
                             '(no residual, no ControlNet, no local_region_loss) reaching only '
                             '~12%% AccCLIP even at 1.5x its natural magnitude -- markedly weaker '
                             'than gender or age\'s response curve at the same alphas. This is '
                             'architectural, not a calibration bug: eyeglasses is a discrete, '
                             'multi-modal, spatially-precise structure (thin/thick frames, '
                             'rimless, sunglasses, all at slightly different positions), and a '
                             'single per-stratum mean-difference direction averages those styles '
                             'together, which for a high-frequency local structure washes out '
                             'detail rather than reinforcing it (unlike age/gender, which are '
                             'smoother, closer to single-axis semantic shifts that a linear '
                             'direction represents well). DEFAULT 0.35 (more than double the '
                             'shared 0.15) hands the trainable flow residual more of the budget '
                             'for glasses specifically, instead of leaning on a frozen direction '
                             'that is confirmed too weak to carry the edit alone -- the flow, '
                             'local_region_loss, ControlNet (if enabled) and '
                             '--clip_prompt_glasses_weight do the real work; the direction only '
                             'needs to point roughly the right way. Still just an INIT value, '
                             'learned further via gradient descent from there. <0 falls back to '
                             '--direction_residual_scale (old behavior).')
    parser.add_argument('--age_residual_scale', type=float, default=-1.0,
                        help='Separate INITIAL residual_scale for age (attr 39) at TRAINING time. '
                             '<0 (default) falls back to the shared --direction_residual_scale, i.e. '
                             'age starts with the same residual budget as every other attribute even '
                             "though the age direction is the one confirmed (via visual audit) to carry "
                             'a baked-in color-cast shortcut. AttributeDirectionBank already supports '
                             'per-attribute residual_scale init (per_attr_residual_scale) -- evaluation/'
                             'evaluate_sdflow.py already uses it for --glasses_residual_scale, but '
                             'training never wired it up for any attribute until now. Raising this '
                             '(e.g. 0.15-0.30) gives the flow residual a bigger starting share of the '
                             "final age edit, so whatever real aging texture --age_diffusion_weight/"
                             '--age_dds_fine_layer_start teach the flow actually has weight in the '
                             'output, instead of the frozen (~95% of the edit) direction dominating '
                             'regardless of what the residual learns. residual_scale is still learned '
                             'via gradient descent from this starting point, not fixed.')
    parser.add_argument('--male_residual_scale', type=float, default=-1.0,
                        help='Separate INITIAL residual_scale for gender (attr 20) at TRAINING '
                             'time, same mechanism as --age_residual_scale/--glasses_residual_scale. '
                             '<0 (default) falls back to the shared --direction_residual_scale. '
                             'Independent-judge eval on a trained checkpoint found gender the most '
                             'expensive of the three attributes per unit of accuracy: highest LPIPS '
                             '(~2x eyeglasses/age at every tested scale), fastest-declining ID_ind '
                             'across the scale sweep, and the highest LeakCLIP -- while its AccCLIP '
                             'was already strong. Unlike eyeglasses/age, gender was never given its '
                             'own reduced budget, so it shares the same residual_scale as every other '
                             'attribute despite already having a well-calibrated (LDA + cross-attr-'
                             'decorrelated) frozen direction to lean on instead. LOWERING this (e.g. '
                             '0.05-0.10, below the shared default 0.15) hands gender LESS of the edit '
                             'budget from the freely-learned, less-constrained flow residual and MORE '
                             'from the frozen direction, trading a bit of headroom on AccCLIP (already '
                             'comfortably ahead of eyeglasses/age) for less collateral pixel change and '
                             'identity drift. residual_scale is still learned via gradient descent from '
                             'this starting point, not fixed.')
    parser.add_argument('--direction_freeze', '--direction-freeze',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--direction_orth_weight', type=float, default=0.0)
    parser.add_argument('--dir_gate_diversity_weight', type=float, default=0.0,
                        help='Weight on AttributeDirectionBank.gate_load_balance_loss(), which '
                             'penalizes the K-mixture gate (gate_net, active whenever the bank has '
                             'num_k>1, e.g. from --K/--age_k combined with --substyle_k) for '
                             'collapsing onto a minority of its K slots regardless of the source '
                             'face. WHY THIS EXISTS: nothing in this project previously supervised '
                             'gate routing at all -- it only received gradient indirectly through '
                             'the final guided_delta, with no signal rewarding correct or even '
                             'diverse routing. Measured on a real trained checkpoint (Eyeglasses, '
                             'K=12 from --K 4 x --substyle_k 3): the gate collapsed within the first '
                             'few thousand steps onto ~2 of the 12 slots for 81% of samples '
                             'regardless of the source face\'s actual gender/age, including almost '
                             'NEVER routing female_young samples to the female_young-conditioned '
                             'slots --extreme_min_conf specifically cleaned up in '
                             'precompute_directions_stratified.py -- which is why that direction-'
                             'bank fix alone did not move the eyeglasses-add failure rate. This '
                             'loss does NOT know which slot is demographically correct for a given '
                             'face (no label fed in for that) -- it only discourages collapsing onto '
                             'too few slots, a necessary but not sufficient condition for a clean '
                             'stratum-level direction to actually get used. 0 (default) disables; '
                             'try 0.05-0.2 to start. No effect when num_k<=1 (K=1, no substyle_k).')
    parser.add_argument('--gate_usage_ema_decay', type=float, default=0.98,
                        help='EMA decay for AttributeDirectionBank.gate_usage_ema (per-attribute, '
                             'per-K-slot usage, used by --dir_gate_diversity_weight and the '
                             'dir_gate_entropy_per_attr wandb logs). Higher = smoother/slower to '
                             'react; matches the convention of --balance_ema_decay.')
    parser.add_argument('--direction_k', type=int, default=1,
                        help='Number of mixture directions per attribute in the Direction Bank.')
    parser.add_argument('--direction_guided_delta_max_norm', type=float, default=0.0,
                        help='Optional global max norm inside Direction Bank after all controls. '
                             'Use 10-12 for safe stage fine-tuning; 0 disables.')
    parser.add_argument('--final_delta_max_norm', type=float, default=0.0,
                        help='Optional final max norm after direction-bank mixing. 0 disables.')
    parser.add_argument('--use_attr_lora', action='store_true',
                        help='Give each attribute its own small low-rank (LoRA-style) additive '
                             'correction on top of magnitude_net\'s shared hidden representation, '
                             'instead of every attribute sharing the exact same MLP path with only '
                             'a scalar residual_scale/direction_scale to differ by. '
                             'Zero-initialized (B matrix), so this is a strict no-op at step 0 -- '
                             'safe to enable on a bank that already has a trained magnitude_net.')
    parser.add_argument('--attr_lora_rank', type=int, default=4,
                        help='Rank of the per-attribute LoRA adapter (--use_attr_lora). Higher = '
                             'more per-attribute expressiveness, more parameters to learn.')
    parser.add_argument('--signed_magnitude_input', action='store_true',
                        help="Feed magnitude_net the SIGNED attr_delta instead of its absolute "
                             "value, so an attribute can travel a different distance when added "
                             "than when removed. Under .abs() the two are forced equal, but a "
                             "scale sweep shows the weak side of each attribute keeps improving "
                             "with more displacement while the strong side has saturated -- and "
                             "which side is weak differs per attribute (Male on rm, Young on add). "
                             "Raising edit_scale globally therefore overpays, buying nothing on "
                             "the saturated directions while costing identity everywhere. This "
                             "moves the compensation inside the model so edit_scale stays at 1.0. "
                             "Parameter shapes are unchanged, so an existing checkpoint can be "
                             "fine-tuned with --resume_dir rather than retrained from scratch.")
    parser.add_argument('--use_controlnet_injection', action='store_true',
                        help='ControlNet-style additive injection into an INTERMEDIATE StyleGAN2 '
                             'feature map (models/stylegan2/model.py Generator\'s dormant `skips` '
                             'hook at embed_res resolution), instead of only at the W+ input like '
                             'every other mechanism (Direction Bank, flow residual, LoRA). Each '
                             'attribute gets its own small decoder head (models/control_encoder.py'
                             '), zero-initialized so this is a no-op at step 0. Requires '
                             '--direction_bank_path. See models/control_encoder.py module '
                             'docstring for the motivation.')
    parser.add_argument('--controlnet_embed_res', type=int, default=64,
                        help='Must match Generator.forward()\'s embed_res (default 64) -- the '
                             'feature-map resolution the injection targets.')
    parser.add_argument('--controlnet_channels', type=int, default=512,
                        help='Must match StyleGAN2\'s channel count at --controlnet_embed_res '
                             '(512 for the default channel_multiplier=2 at 64x64).')
    parser.add_argument('--controlnet_hidden_dim', type=int, default=256,
                        help='Hidden width of the shared trunk in AttributeControlEncoder.')
    parser.add_argument('--controlnet_reg_weight', type=float, default=0.01,
                        help='L2 penalty weight on the per-sample norm of control_skips (the '
                             'AttributeControlEncoder output actually added into the StyleGAN2 '
                             'feature map). Unlike the W+ guided_delta, control_skips has NO loss '
                             'term constraining its magnitude directly -- it is only shaped '
                             'indirectly through downstream classifier/CLIP losses, which reward '
                             'moving attribute scores but never penalize HOW that movement looks. '
                             'Over long fine-tunes this let the injected signal grow large enough '
                             'to visibly corrupt images (LPIPS blew up, AccCeleb collapsed, while '
                             'ID stayed misleadingly high) with no warning in any other loss curve. '
                             'DEFAULT raised from 0.001 to 0.01: eval found ControlNet injection '
                             'earns its keep on eyeglasses but gives gender/age no measured '
                             'accuracy benefit while adding a sparkle artifact -- the stronger '
                             'penalty lets training itself shrink the injection toward zero for '
                             'attributes that do not need it, rather than a hard eval-time '
                             '--controlnet_disable_attrs override. Set 0 to disable.')
    parser.add_argument('--controlnet_max_norm', type=float, default=0.0,
                        help='Hard per-sample cap on control_skips norm (like guided_delta_max_norm '
                             'for the W+ path). 0 disables the cap; only the L2 penalty above still '
                             'applies. Use this if the L2 penalty alone does not keep training stable.')
    parser.add_argument('--controlnet_warmup_steps', type=int, default=0,
                        help='Hold the injection off for this many steps so the W+ path (flow + '
                             'Direction Bank) learns to edit on its own first. Training both from '
                             'scratch lets the control branch win the work outright: it reaches '
                             'the losses more easily and the W+ path then stops developing -- a '
                             'measured run had W+ delta norm peak early and DECLINE to ~8 while '
                             'the same recipe without injection climbed to ~15, and disabling the '
                             'branch at eval dropped accuracy from 71.8% to 28.1%. That hurts '
                             'exactly the attributes the branch cannot handle: the injection sits '
                             'at embed_res, downstream of the coarse layers where face geometry is '
                             'decided, so it is strong on spatially-local attributes and cannot do '
                             'the jaw/skull/hairline changes age and gender need.')
    parser.add_argument('--controlnet_init_gain', type=float, default=1.0,
                        help='Starting norm of the injected control_skips signal (the encoder '
                             'output is L2-normalized, then scaled by a learnable per-attribute '
                             'gain initialized here). This IS control_skip_norm at step 0, so set '
                             'it against the measured magnitude of the feature map it is added to '
                             '-- run scripts/measure_feature_norm.py rather than guessing.')
    parser.add_argument('--controlnet_lr_mult', type=float, default=1.0,
                        help='LR multiplier for control_encoder only (own Adam param group).')
    parser.add_argument('--controlnet_latent_cond', action='store_true',
                        help='Condition the injected feature map on the SOURCE LATENT, not just '
                             'on attr_delta. WITHOUT this, AttributeControlEncoder.forward() sees '
                             'only a (B, num_attrs) vector that is effectively identical for every '
                             'sample editing the same attribute in the same direction -- so it '
                             'adds ONE fixed, face-agnostic 512x64x64 pattern to a generator '
                             'feature map that differs for every identity, pose and framing. For '
                             'eyeglasses that means the perturbation lands at fixed spatial '
                             'coordinates instead of on THIS face\'s eyes: enough glasses-like '
                             'texture for a detector to fire, never a correctly placed frame. The '
                             'real ControlNet is conditioned on a spatial input for exactly this '
                             'reason. RECOMMENDED for new runs. Off by default because it changes '
                             'the module\'s parameter shapes -- a checkpoint trained without it '
                             'cannot be loaded into a model built with it, or vice versa.')
    parser.add_argument('--controlnet_per_direction',
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Give add and rm their own decoder head and their own learnable gain "
                             "per attribute, instead of sharing one of each per attribute. Sharing "
                             "forces the SAME weights to learn two very different tasks -- e.g. "
                             "eyeglasses add must synthesize frame structure from nothing while rm "
                             "only erases an existing one. A 128x128 run (--controlnet_embed_res "
                             "128) showed the predicted failure: judge_celeb_acc/attr_15_add "
                             "climbed slowly from ~0 while judge_celeb_acc/attr_15_rm drifted down "
                             "over the same steps -- the harder add task pulling shared capacity "
                             "away from rm, the same entanglement --per_direction already fixed for "
                             "AttributeDirectionBank and never fixed here. Off by default: changes "
                             "control_encoder's parameter shapes, so a checkpoint saved with the "
                             "old (shared) layout cannot --resume_dir into a run with this flag set.")
    parser.add_argument('--target_loss', default='mse', choices=['mse', 'hinge'],
                        help="Shape of the target-attribute loss. 'mse' squares the distance to "
                             "soft_target, which keeps pulling on samples that already crossed the "
                             "decision boundary and leaves the ones far past it holding a surplus "
                             "the optimiser can spend elsewhere. 'hinge' penalises only the "
                             "shortfall relative to crossing 0.5 by --target_hinge_margin, matching "
                             "what AccCeleb actually measures, and gives zero gradient once a "
                             "sample is safely across.")
    parser.add_argument('--target_hinge_margin_frac', type=float, default=0.6,
                        help="Per-attribute --target_loss hinge margin, expressed as a FRACTION "
                             "of that attribute's own |SOFT_TARGET_TABLE target - 0.5| gap, not an "
                             "absolute probability offset. An absolute margin is structurally wrong "
                             "here: SOFT_TARGET_TABLE deliberately gives Eyeglasses a wide target "
                             "gap (0.10/0.90, gap 0.40) and Male/Young a conservative one "
                             "(0.20/0.80, gap 0.30, chosen to avoid forcing a full identity flip). "
                             "A margin of 0.35 sat comfortably inside Eyeglasses' 0.40 gap (no "
                             "effect -- soft_target always won the "
                             "max(soft_target,boundary)/min(...) comparison) but exceeded Male/"
                             "Young's 0.30 gap, pushing their boundary PAST their own intended "
                             "target and making loss behaviour there hypersensitive to exactly "
                             "where scores land relative to a boundary that no longer means what "
                             "it was supposed to. Male rm got weaker (LPIPS 0.1694 -> 0.1359, "
                             "AccCeleb 54.8% -> 44.4%) under that setup, not stronger. Scaling by "
                             "each attribute's own gap keeps the buffer proportionate: 0.6 asks "
                             "for 60% of the way from 0.5 to the intended target, for every "
                             "attribute, regardless of how aggressive that target already is.")
    parser.add_argument('--celeba_attr_judge_weights', default=None,
                        help='Optional independent CelebA attribute classifier (the same weights '
                             'evaluate_sdflow.py scores AccCeleb with). Used for LOGGING ONLY -- it '
                             'never enters the loss, so AccCeleb stays an independent measure. The '
                             'training teacher saturates well before that judge does (AccT* reads '
                             '100% on Eyeglasses where AccCeleb reads 88%), so without this the '
                             'real metric is invisible until a run finishes. Logged as '
                             'judge_celeb_acc/attr_N and judge_celeb_acc_mean.')
    parser.add_argument('--save_best', action=argparse.BooleanOptionalAction, default=True,
                        help='Also keep the best-scoring checkpoint under <run>/best/, judged by a '
                             'smoothed judge_celeb_acc_mean. Requires --celeba_attr_judge_weights, '
                             'since that monitor is the only signal here that tracks the reported '
                             'metric rather than the training teacher (which saturates well before '
                             'it -- AccT* reads 100% on Eyeglasses where AccCeleb reads 88%). The '
                             'best directory mirrors the run layout, so evaluating it is just '
                             '--checkpoint_dir <run>/best.')
    parser.add_argument('--best_metric_ema', type=float, default=0.98,
                        help='Smoothing for the best-checkpoint metric. The raw monitor scores one '
                             'batch of --batch samples, of which typically only one or two are '
                             'editing any given attribute, so it lands on 0/0.5/1 and is far too '
                             'noisy to select on directly.')
    parser.add_argument('--best_min_step', type=int, default=10000,
                        help='Ignore best-checkpoint candidates before this step, while the metric '
                             'is still climbing out of its initial transient.')
    parser.add_argument('--celeba_judge_interval', type=int, default=50,
                        help='Steps between --celeba_attr_judge_weights monitor evaluations.')
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
    parser.add_argument('--use_diffusion_guidance', action=argparse.BooleanOptionalAction, default=True,
                        help='Use a frozen Stable Diffusion model as auxiliary DDS semantic '
                             'guidance. DEFAULT changed from off to on -- this is what lets '
                             '--age_dds_fine_layer_start/--age_diffusion_weight/'
                             '--age_diffusion_interval (below) do anything; without it those are '
                             'silently no-ops. COST: downloads/loads --diffusion_model_id (a few '
                             'GB from HuggingFace on first run) and adds a forward/backward pass '
                             'through it every --diffusion_guidance_interval steps -- meaningfully '
                             'more VRAM and time per step. Pass --no-use_diffusion_guidance to '
                             'restore the old default if you don\'t have network access to '
                             'HuggingFace or want the old resource footprint.')
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
    parser.add_argument('--residual_max_norm', type=float, default=10.0,
                        help='Hard clip per-sample residual norm in Direction Bank forward(). '
                             'Prevents residual explosion from large DDS gradients. DEFAULT '
                             'changed from None (off) to 10.0 (the previously-suggested value). '
                             'There is no CLI value that means "off" any more (the consuming code '
                             'in AttributeDirectionBank checks `is not None`, and a negative norm '
                             'would flip the residual\'s sign rather than disable clipping) -- pass '
                             'a very large value (e.g. 1e6) to make the clip effectively a no-op, '
                             'or edit this default back to None to fully restore the old behavior.')
    parser.add_argument('--dds_fine_layer_start', type=int, default=7,
                        help='W+ layer index from which DDS gradients are blocked (fine layers). '
                             'Set 0 to disable masking.')
    parser.add_argument('--age_diffusion_timestep_min', type=int, default=400,
                        help='Min timestep for age-specific DDS pass (coarse structure).')
    parser.add_argument('--age_diffusion_timestep_max', type=int, default=900,
                        help='Max timestep for age-specific DDS pass.')
    parser.add_argument('--age_diffusion_interval', type=int, default=8,
                        help='Run age DDS guidance every N steps (independent of --diffusion_guidance_interval). '
                             'DEFAULT lowered from 16 to 8 (matching non-age) -- 16 was LESS '
                             'frequent than non-age despite age being the hardest attribute, i.e. '
                             'the hardest attribute was getting the least diffusion supervision.')
    parser.add_argument('--age_diffusion_weight', type=float, default=0.1,
                        help='Separate loss weight for the AGE DDS pass. DEFAULT changed from -1 '
                             '(fall back to the shared --diffusion_guidance_weight, 0.01, same tiny '
                             'weight as glasses/gender) to 0.1 -- gives the diffusion teacher real '
                             'pull on aging without touching the other attributes. Pass a negative '
                             'value to restore the old fall-back-to-shared-weight behavior.')
    parser.add_argument('--age_dds_fine_layer_start', type=int, default=12,
                        help='Fine-layer cutoff for the AGE DDS pass specifically. DEFAULT changed '
                             'from -1 (fall back to the shared --dds_fine_layer_start, 7, which '
                             'blocks DDS gradients from the fine W+ layers 7-17 -- exactly the '
                             'layers that carry wrinkles/skin texture/gray hair, so the diffusion '
                             'teacher could not teach real aging texture and the model fell back on '
                             'the coarse/global color-shift shortcut) to 12: the documented '
                             'middle-ground value, letting the teacher reach most texture layers. '
                             'Set to 18 for fully unblocked (higher risk), or a negative value to '
                             'restore the old fall-back-to-7 behavior.')

    # ── Frozen CLIP semantic target loss ───────────────────────────────
    parser.add_argument('--use_clip_prompt_loss', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable frozen CLIP prompt loss for semantic direction supervision. '
                             'DEFAULT changed from off to on -- required for --clip_prompt_mode/'
                             '--clip_prompt_glasses_weight/--clip_prompt_age_weight (below) to have '
                             'any effect; without it those are silently no-ops. Pass '
                             '--no-use_clip_prompt_loss to restore the old default.')
    parser.add_argument('--clip_prompt_model', type=str, default='ViT-B/32',
                        help='OpenAI CLIP model name.')
    parser.add_argument('--clip_prompt_weight', type=float, default=0.03,
                        help='Weight for CLIP prompt loss. Suggested range: 0.02–0.05.')
    parser.add_argument('--clip_prompt_temperature', type=float, default=1.0,
                        help='Temperature for softplus sharpness in CLIP loss (absolute mode only).')
    parser.add_argument('--clip_prompt_mode', default='directional', choices=['absolute', 'directional'],
                        help="'absolute' pulls the edited image toward the target prompt "
                             "regardless of the source. A visual audit (attr 39, direction rm) "
                             "found this lets the model reach a high CLIP 'looks old' score via a "
                             "red/orange color shift instead of real structural aging -- a "
                             "shortcut, not the intended edit, and the actual source of the "
                             "age color-cast artifact. 'directional' (StyleGAN-NADA style, now "
                             "DEFAULT, was 'absolute') instead rewards moving the image, from ITS "
                             "OWN source, along the same CLIP-space axis that separates the pos/"
                             "neg prompts -- closing off shortcuts that shift every image the same "
                             "way regardless of content. Costs one extra CLIP image encode per "
                             "step (source image). Pass 'absolute' to restore the old default.")
    parser.add_argument('--clip_prompt_interval', type=int, default=1,
                        help='Compute CLIP loss every N steps (1 = every step).')
    parser.add_argument('--clip_prompt_num_augs', type=int, default=4,
                        help='Number of random crop/flip views the CLIP prompt loss averages '
                             'its score over, per image. 1 = the old single fixed full-frame '
                             'view, which the generator can attack with spatially-fixed '
                             'high-frequency patterns that raise the CLIP score with no semantic '
                             'change -- the CLIP-side twin of the r34 teacher-fooling --teacher_aug '
                             'already defends against. Averaging over views forces a pattern to '
                             'survive translation/scale/flip to keep scoring. Costs this many CLIP '
                             'image encodes per step (doubled in directional mode); lower it to 2 '
                             'if step time matters, 1 to restore the old behavior.')
    parser.add_argument('--clip_prompt_aug_min_scale', type=float, default=0.75,
                        help='Smallest random crop side, as a fraction of the image, for '
                             '--clip_prompt_num_augs. Too small and the crop can miss the '
                             'attribute entirely (e.g. crop out the eyes on a glasses edit).')
    parser.add_argument('--clip_prompt_age_weight', type=float, default=3.0,
                        help='Per-sample weight multiplier for age (attr 39) in CLIP loss.')
    parser.add_argument('--clip_prompt_gender_weight', type=float, default=1.0,
                        help='Per-sample weight multiplier for gender (attr 20) in CLIP loss.')
    parser.add_argument('--clip_prompt_glasses_weight', type=float, default=3.0,
                        help='Per-sample weight multiplier for eyeglasses (attr 15) in CLIP '
                             'loss. Eyeglasses-add is the biggest independent-judge gap (r34 '
                             'teacher ~91%% but CLIP judge ~44%%, i.e. teacher-fooling) AND '
                             'was historically the attribute that got the LEAST semantic help '
                             '(age has a 3x CLIP weight and its own diffusion guidance; '
                             'glasses had neither). DEFAULT raised from 1.0 to 3.0 to force '
                             'real, CLIP-visible glasses instead of a decision-boundary trick '
                             'the frozen r34 classifier alone rewards. Try up to 5.0 if '
                             'eyeglasses structure is still incomplete.')
    parser.add_argument('--balance_clip_prompt_loss', action=argparse.BooleanOptionalAction, default=False,
                        help='Replace the fixed --clip_prompt_{age,gender,glasses}_weight constants '
                             'with a CrossAttributeLossBalancer tracking CLIP loss progress per '
                             'attribute, instead of hand-tuning weights after each eval. IMPORTANT: '
                             'this tracks the CLIP loss itself, not changed_loss/--balance_attr_losses '
                             '(that one watches the r34 teacher, which already reads eyeglasses as '
                             "~91% -- it would never flag eyeglasses as lagging, since teacher-fooling "
                             "is exactly a gap between teacher and CLIP judgment). A separate balancer "
                             'keyed to CLIP loss is required to auto-detect that gap and correct it. '
                             'Uses the same --balance_ema_decay/--balance_adapt_rate/--balance_min_weight/'
                             '--balance_max_weight hyperparameters as --balance_attr_losses.')
    parser.add_argument('--clip_balance_signal', default='loss', choices=['loss', 'judge'],
                        help="What --balance_clip_prompt_loss tracks to set weights. 'loss' uses "
                             "CrossAttributeLossBalancer on the CLIP loss's own relative progress "
                             "(original behaviour). 'judge' uses JudgePeakDeclineBalancer instead: it "
                             "reads --celeba_attr_judge_weights accuracy and raises an attribute's "
                             "weight the further its OWN smoothed accuracy has fallen from its own "
                             "best-seen value. This exists because CLIP-loss progress and judge "
                             "accuracy can disagree for a long stretch of training -- a run's Male "
                             "judge accuracy peaked by ~10k steps then eroded to 60k while its CLIP "
                             "loss stayed low the whole time (having dropped early), so the "
                             "loss-based balancer read Male as ahead and kept deprioritizing it right "
                             "through the decline. Requires --celeba_attr_judge_weights.")
    parser.add_argument('--judge_balance_gain', type=float, default=4.0,
                        help='--clip_balance_signal judge: weight = 1 + gain * (own peak accuracy - '
                             'current smoothed accuracy), before clamping to [--balance_min_weight, '
                             '--balance_max_weight] and renormalizing to mean 1.')
    parser.add_argument('--judge_balance_warmup_updates', type=int, default=5,
                        help='--clip_balance_signal judge: judge ticks (each --celeba_judge_interval '
                             "steps) an attribute needs before its EMA starts counting toward its own "
                             "peak. Skips the initial climb off zero so it is never mistaken for a "
                             "decline.")
    parser.add_argument('--judge_balance_per_direction',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='--clip_balance_signal judge: track add and rm as separate slots, each '
                             'with its own peak and weight. On by default because keying on the '
                             'attribute alone weights a mean over both directions, and that mean is '
                             'satisfiable one-sided: the first run of this balancer lifted Male add '
                             '83.6%% -> 92.8%% while Male rm fell 68.1%% -> 34.1%%, so the attribute '
                             'average looked healthy (it peaked HIGHER than the unbalanced baseline) '
                             'while the edit everyone actually cares about got worse. Pass '
                             '--no-judge_balance_per_direction to reproduce that per-attribute run.')

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
        min_weight=args.reg_weight_min,
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

    clip_loss_balancer = None
    judge_balancer = None   # built after celeb_monitor exists, see below
    if args.balance_clip_prompt_loss:
        if args.clip_balance_signal == 'judge':
            if not args.celeba_attr_judge_weights:
                raise ValueError('--clip_balance_signal judge requires --celeba_attr_judge_weights.')
            print(f'** CLIP-prompt-loss auto-balancing enabled for attrs {args.attribute_index} '
                  f'(overrides --clip_prompt_{{age,gender,glasses}}_weight), signal=judge '
                  f'(JudgePeakDeclineBalancer built once the judge is loaded below)')
        else:
            clip_loss_balancer = CrossAttributeLossBalancer(
                len(args.attribute_index),
                ema_decay=args.balance_ema_decay,
                adapt_rate=args.balance_adapt_rate,
                min_weight=args.balance_min_weight,
                max_weight=args.balance_max_weight,
                device='cuda',
            )
            print(f'** CLIP-prompt-loss auto-balancing enabled for attrs {args.attribute_index} '
                  f'(overrides --clip_prompt_{{age,gender,glasses}}_weight), signal=loss')
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
        # No per-attribute direction_scale/layer_scale/delta_max_norm. Age,
        # glasses, and gender DO get their own residual_scale INIT
        # (--age_residual_scale, --glasses_residual_scale,
        # --male_residual_scale) -- every other attribute still starts from
        # the same shared --direction_residual_scale. All of them keep
        # learning their own value from that starting point via gradient
        # descent (see AttributeDirectionBank.residual_scale_raw); the only
        # magnitude safety net fixed in advance is guided_delta_max_norm.
        _per_attr_residual_scale = [
            args.age_residual_scale if (idx == 39 and args.age_residual_scale >= 0)
            else args.glasses_residual_scale if (idx == 15 and args.glasses_residual_scale >= 0)
            else args.male_residual_scale if (idx == 20 and args.male_residual_scale >= 0)
            else args.direction_residual_scale
            for idx in args.attribute_index
        ]
        direction_bank = AttributeDirectionBank(
            num_attrs=len(args.attribute_index),
            num_layers=18,
            latent_dim=512,
            num_k=_bank_num_k,
            bank_path=args.direction_bank_path,
            attribute_index=args.attribute_index,
            residual_scale=args.direction_residual_scale,
            per_attr_residual_scale=_per_attr_residual_scale,
            freeze_directions=args.direction_freeze,
            residual_max_norm=args.residual_max_norm,
            guided_delta_max_norm=(
                args.direction_guided_delta_max_norm
                if args.direction_guided_delta_max_norm > 0 else None
            ),
            use_attr_lora=args.use_attr_lora,
            attr_lora_rank=args.attr_lora_rank,
            signed_magnitude_input=args.signed_magnitude_input,
            gate_usage_ema_decay=args.gate_usage_ema_decay,
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

    control_encoder = None
    if args.use_controlnet_injection:
        if direction_bank is None:
            raise ValueError('--use_controlnet_injection requires --direction_bank_path '
                              '(and --velocity_field != original) -- attr_delta, needed to '
                              'condition the control encoder, is only computed on that branch.')
        from models.control_encoder import AttributeControlEncoder
        control_encoder = AttributeControlEncoder(
            num_attrs=len(args.attribute_index),
            out_channels=args.controlnet_channels,
            out_res=args.controlnet_embed_res,
            hidden_dim=args.controlnet_hidden_dim,
            init_gain=args.controlnet_init_gain,
            per_direction=args.controlnet_per_direction,
            latent_cond=args.controlnet_latent_cond,
        ).cuda()
        trainable_params += list(control_encoder.parameters())
        _warm = args.controlnet_warmup_steps
        print(f'** ControlNet-style feature injection enabled: embed_res='
              f'{args.controlnet_embed_res}, channels={args.controlnet_channels}, '
              f'init_gain={args.controlnet_init_gain} (= control_skip_norm once active)'
              + (f'; held off until step {_warm} so the W+ path matures first'
                 if _warm > 0 else '; active from step 0'))

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

    # control_encoder gets its own group so --controlnet_lr_mult can move the
    # branch independently of the W+ path it runs alongside.
    control_params = list(control_encoder.parameters()) if control_encoder is not None else []
    control_ids = {id(p) for p in control_params}

    param_groups = [
        {'params': [p for p in trainable_params
                    if id(p) not in low_lr_ids and id(p) not in control_ids],
         'lr': args.lr},
        {'params': low_lr_params, 'lr': args.lr * args.meta_lr_mult, 'weight_decay': 0.0},
    ]
    if control_params:
        param_groups.append({'params': control_params,
                             'lr': args.lr * args.controlnet_lr_mult})
        print(f'** control_encoder lr = {args.lr * args.controlnet_lr_mult:g} '
              f'({args.controlnet_lr_mult}x base lr)')

    optimizer = optim.Adam(param_groups)
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
        if control_encoder is not None:
            control_encoder_ema = make_ema_copy(control_encoder)
            ema_pairs.append((control_encoder, control_encoder_ema, 'control_encoder'))
            ema_modules_for_save['control_encoder'] = control_encoder_ema
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
        if control_encoder is not None:
            try:
                load_module_checkpoint(control_encoder, resume_save_dir, 'control_encoder', start_step, strict=False)
            except FileNotFoundError:
                print('[Resume] control_encoder checkpoint not found; starting from zero-init (no-op).')
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
    if control_encoder is not None:
        log_modules.insert(-1, control_encoder)

    # Best-checkpoint tracking. The metric is the AccCeleb monitor, smoothed --
    # the raw value scores one small batch and lands on 0/0.5/1, so selecting on
    # it unsmoothed would just pick a lucky batch.
    best_metric_ema = None
    best_metric_seen = float('-inf')
    if args.save_best and not args.celeba_attr_judge_weights:
        print('** --save_best ignored: it selects on judge_celeb_acc_mean, which needs '
              '--celeba_attr_judge_weights')
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

    # Independent AccCeleb monitor. Deliberately outside the loss: this is the
    # same classifier evaluate_sdflow.py reports AccCeleb with, so feeding it
    # gradients would turn the headline metric into self-grading. It exists
    # only so the real number is visible during a run instead of after it --
    # the training teacher saturates far earlier (AccT* 100% on Eyeglasses
    # against AccCeleb 88%), which is why loss curves kept looking healthy
    # while the independent judge disagreed.
    celeb_monitor = None
    if args.celeba_attr_judge_weights:
        from evaluation.evaluate_sdflow import CelebAAttrClassifierJudge
        celeb_monitor = CelebAAttrClassifierJudge(args.celeba_attr_judge_weights, 'cuda')
        for p_ in celeb_monitor.parameters():
            p_.requires_grad_(False)
        print(f'** AccCeleb monitor enabled (logging only, every '
              f'{args.celeba_judge_interval} steps): {args.celeba_attr_judge_weights}')

    if args.balance_clip_prompt_loss and args.clip_balance_signal == 'judge':
        judge_balancer = JudgePeakDeclineBalancer(
            len(args.attribute_index),
            ema_decay=args.balance_ema_decay,
            gain=args.judge_balance_gain,
            min_weight=args.balance_min_weight,
            max_weight=args.balance_max_weight,
            warmup_updates=args.judge_balance_warmup_updates,
            per_direction=args.judge_balance_per_direction,
            device='cuda',
        )
        print(f'** JudgePeakDeclineBalancer built (gain={args.judge_balance_gain}, '
              f'warmup={args.judge_balance_warmup_updates} judge ticks, '
              f'{"per add/rm direction" if args.judge_balance_per_direction else "per attribute"}, '
              f'{judge_balancer.num_slots} slots)')

    face_parser = None
    if args.local_region_loss_weight > 0 or args.color_shift_loss_weight > 0 or args.dds_face_mask:
        from common.face_parser import FaceParser
        try:
            face_parser = FaceParser(weights_path=args.face_parser_weights).cuda().eval()
            if args.local_region_loss_weight > 0:
                local_attrs = [a for a in args.attribute_index if a in LOCAL_REGION_CLASSES]
                print(f'** Face-parser locality loss enabled (weight='
                      f'{args.local_region_loss_weight}) for local attrs {local_attrs}')
            if args.color_shift_loss_weight > 0:
                print(f'** Color-shift regularizer enabled (weight='
                      f'{args.color_shift_loss_weight}) for attrs {args.color_shift_attrs}')
            if args.dds_face_mask:
                print('** DDS face mask enabled: diffusion-guidance gradient restricted to '
                      'the region each attribute is allowed to touch.')
        except (FileNotFoundError, RuntimeError) as exc:
            face_parser = None
            print(f'[WARN] Face parser unavailable ({exc}); locality/color-shift/DDS-mask '
                  f'disabled.')

    diffusion_guidance = None
    if args.use_diffusion_guidance:
        try:
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
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                f'--use_diffusion_guidance is on by default but failed to load '
                f'{args.diffusion_model_id} ({exc}). This needs the `diffusers` package plus '
                f'either network access to HuggingFace or a local cache of the model. Pass '
                f'--no-use_diffusion_guidance to train without it (age gets less diffusion '
                f'supervision but everything else is unaffected).'
            ) from exc

    clip_prompt_loss_fn = None
    if args.use_clip_prompt_loss:
        try:
            from models.clip_prompt_loss import FrozenCLIPPromptLoss
        except ImportError as exc:
            raise RuntimeError(
                f'--use_clip_prompt_loss is on by default but failed to import ({exc}). Install '
                f'OpenAI CLIP (pip install git+https://github.com/openai/CLIP.git) or pass '
                f'--no-use_clip_prompt_loss to train without it.'
            ) from exc
        clip_prompt_loss_fn = FrozenCLIPPromptLoss(
            clip_model=args.clip_prompt_model,
            temperature=args.clip_prompt_temperature,
            mode=args.clip_prompt_mode,
            num_augs=args.clip_prompt_num_augs,
            aug_min_scale=args.clip_prompt_aug_min_scale,
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

            control_skips = None
            control_skip_norm = torch.zeros([], device=latent.device, dtype=latent.dtype)
            loss_control_reg = torch.zeros([], device=latent.device, dtype=latent.dtype)
            controlnet_active = (control_encoder is not None
                                 and n_iter >= args.controlnet_warmup_steps)
            if controlnet_active:
                # ControlNet-style injection: an additive correction at an
                # INTERMEDIATE StyleGAN2 feature map (embed_res, default
                # 64x64), not at the W+ input like every other mechanism in
                # this project. attr_delta is only defined on the
                # direction_bank branch above; control_encoder requires
                # direction_bank to be enabled (checked at arg-parse time).
                control_skips = control_encoder(attr_delta, mid_idx,
                                                is_rm=(src_attr_flow > 0.5),
                                                latent=latent)

                # control_skips has no OTHER loss term constraining its
                # magnitude -- it's only shaped indirectly through downstream
                # classifier/CLIP losses, which reward moving attribute
                # scores but never penalize how that movement looks. Long
                # fine-tunes let this grow unbounded (LPIPS spiked, AccCeleb
                # collapsed, while ID stayed misleadingly high because the
                # corruption didn't destroy ArcFace's coarse face embedding).
                # An explicit L2 penalty plus optional hard cap closes that
                # gap, mirroring guided_delta_max_norm on the W+ path.
                skip_norm_per_sample = control_skips.reshape(control_skips.shape[0], -1).norm(dim=1)
                loss_control_reg = skip_norm_per_sample.pow(2).mean()
                if args.controlnet_max_norm > 0:
                    clip = (args.controlnet_max_norm / skip_norm_per_sample.clamp(min=1e-8)).clamp(max=1.0)
                    control_skips = control_skips * clip.view(-1, 1, 1, 1)
                control_skip_norm = skip_norm_per_sample.mean().detach()

            new_face_tensors = G([new_latents], skips=control_skips,
                                 embed_res=args.controlnet_embed_res,
                                 input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
            new_face_tensors = F.interpolate(new_face_tensors, (args.img_size, args.img_size))

            # ── Face-parser locality loss (local attributes only) ────────────
            # Outside the attribute's allowed facial region, the edited image
            # must match the source reconstruction. Compared against G(latent)
            # (not the real photo) so the inversion gap is not charged to the
            # edit. In cycle mode this fires on the local attribute's steps only.
            #
            # REMOVAL-ONLY: the region mask is computed from the SOURCE
            # reconstruction. For a face that does not yet have glasses, BiSeNet
            # only labels the bare eyes/brows (no "glasses" class pixels exist
            # yet), which is much smaller than the actual frame footprint (it
            # extends toward the temples/cheeks). Enforcing "no change outside
            # this tight mask" on an add-glasses edit makes it geometrically
            # impossible to paint a full frame, and the model just gives up
            # drawing one (observed: v11 preview grid stopped adding glasses
            # entirely). For a face that already has glasses, the source mask
            # correctly covers the full frame, so removal edits are safe.
            # Restrict the loss to src_attr_flow > 0.5 (removal direction only)
            # so addition edits fall back to the unconstrained v10 behavior.
            # Direction-aware masking:
            #   removal (source HAS the attribute): BiSeNet sees the full frame
            #     on the source, so a precise mask (default blur) is safe.
            #   addition (source LACKS it): no glasses pixels exist yet, so the
            #     precise mask is far smaller than a real frame's footprint and
            #     would make painting one impossible (v11 stopped adding glasses
            #     entirely). Instead use a heavily dilated eye/brow mask
            #     (--local_region_add_blur) as a geometric prior for where a
            #     frame may appear; hair/mouth/background stay forbidden.
            local_region_loss = torch.zeros([], device=latent.device, dtype=latent.dtype)
            # ── Color-shift regularizer (--color_shift_loss_weight) ──────────
            # A visual audit (scripts/dump_attr_failures.py, attr 39 direction
            # rm) found the model satisfies the aging edit largely by shifting
            # the whole face's skin tone toward red/orange, not by drawing
            # genuine texture/geometry aging (wrinkles, gray hair) -- and this
            # was unchanged by every fix tried so far that operates on the
            # direction vector or edit magnitude (direction swaps, scale
            # sweeps, residual overrides, directional CLIP loss). This
            # regularizer instead acts directly on the generated pixels: it
            # penalizes the change in mean skin-region RGB between source and
            # edited face, making the color-shift shortcut itself costly so
            # the attribute loss has to be satisfied some other way. Genuine
            # aging is a texture/geometry change, not a uniform tone shift,
            # so it should be largely unaffected by this penalty.
            color_shift_loss = torch.zeros([], device=latent.device, dtype=latent.dtype)

            # ── Source RECONSTRUCTION, G(latent) ────────────────────────────
            # Hoisted out of the face-parser branch below so the identity /
            # CLIP / DDS losses can use it as their reference too (see
            # --losses_vs_recon). Computed at most once per step and reused.
            #
            # WHY THIS MATTERS: `img` is the REAL photo; `new_face_tensors` is
            # G(latent + delta). The difference between them is (inversion gap)
            # + (edit). Any loss that compares the edited image against `img`
            # therefore charges the e4e/StyleGAN inversion error to the edit --
            # error the flow did not cause and cannot usefully remove, since
            # fixing it would mean spending W+ budget on reconstruction rather
            # than on the attribute. The locality/color-shift losses below
            # already deliberately compare against src_recon for exactly this
            # reason; --losses_vs_recon extends the same principle to the other
            # three losses that were still referencing `img`.
            src_recon = None
            if (args.losses_vs_recon
                    or face_parser is not None):
                with torch.no_grad():
                    src_recon = G([latent], input_is_latent=True,
                                  randomize_noise=False)[0].clamp(-1, 1)
                    src_recon = F.interpolate(src_recon, (args.img_size, args.img_size))
            # Reference image for id_loss / directional CLIP / DDS.
            loss_ref_img = src_recon if (args.losses_vs_recon and src_recon is not None) else img

            if face_parser is not None:
                _mid_abs = [args.attribute_index[int(j)] for j in mid_idx.detach().cpu().tolist()]
                _is_local = torch.tensor([a in LOCAL_REGION_CLASSES for a in _mid_abs],
                                         device=latent.device)
                _is_color = torch.tensor([a in args.color_shift_attrs for a in _mid_abs],
                                         device=latent.device)
                if _is_local.any() or _is_color.any():
                    if _is_local.any():
                        _is_removal = src_attr_flow > 0.5
                        _terms = []
                        for _abs_idx in set(a for a in _mid_abs if a in LOCAL_REGION_CLASSES):
                            _attr_sel = torch.tensor([a == _abs_idx for a in _mid_abs],
                                                     device=latent.device)
                            for _dir_sel, _sigma in [
                                (_attr_sel & _is_removal, 5),
                                (_attr_sel & ~_is_removal, int(args.local_region_add_blur)),
                            ]:
                                if not _dir_sel.any():
                                    continue
                                with torch.no_grad():
                                    region = face_parser.get_region_mask(
                                        src_recon[_dir_sel],
                                        LOCAL_REGION_CLASSES[_abs_idx],
                                        blur_sigma=_sigma,
                                    )
                                outside = (1.0 - region)
                                diff_sq = (new_face_tensors[_dir_sel] - src_recon[_dir_sel]).pow(2)
                                _terms.append(
                                    (diff_sq * outside).sum()
                                    / (outside.sum() * diff_sq.shape[1]).clamp(min=1e-6)
                                )
                        if _terms:
                            local_region_loss = torch.stack(_terms).mean()

                    if _is_color.any():
                        with torch.no_grad():
                            skin_mask = face_parser.get_region_mask(
                                src_recon[_is_color], SKIN_CLASS, blur_sigma=5,
                            )   # (N, 1, H, W)
                        _src_c = src_recon[_is_color]
                        _edit_c = new_face_tensors[_is_color]
                        _pixel_count = skin_mask.sum(dim=(2, 3)).clamp(min=1e-6)   # (N, 1)
                        mean_src = (_src_c * skin_mask).sum(dim=(2, 3)) / _pixel_count    # (N, 3)
                        mean_edit = (_edit_c * skin_mask).sum(dim=(2, 3)) / _pixel_count  # (N, 3)
                        color_shift_loss = (mean_edit - mean_src).pow(2).sum(dim=1).mean()
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
            # Reference is the source RECONSTRUCTION by default (--losses_vs_recon),
            # not the real photo: otherwise the fixed inversion gap sits inside
            # id_loss as a constant penalty the edit cannot remove, and training
            # optimizes a different quantity than evaluate_sdflow.py reports
            # (which measures edited-vs-reconstruction identity).
            id_src_feat = F.normalize(id_criterion.extract_features(loss_ref_img), dim=1).detach()
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
            # Which way is this edit going? Same convention evaluate_sdflow
            # reports its direction split under: rm = the source already has
            # the attribute. Computed once here so the judge balancer buckets a
            # sample the same way when it reads a weight out (clip-prompt loss,
            # below) and when it writes an accuracy back in (monitor block).
            edit_is_rm = (src_attr > 0.5).detach()
            hard_teacher_target = compute_soft_targets(src_attr, mid_idx, args.attribute_index)
            soft_target = src_attr + train_scale * (hard_teacher_target - src_attr)
            soft_target_for_loss = soft_target.detach()
            target_probs[batch_indices, mid_idx] = soft_target_for_loss
            edited_probs = gen_probs[batch_indices, mid_idx]
            if args.target_loss == 'hinge':
                # AccCeleb only asks whether the edited score CROSSED 0.5; it is
                # blind to how far past the boundary a sample lands. A squared
                # error to soft_target is not: a sample sitting at 0.02 when the
                # target is 0 contributes almost nothing, so from the loss's view
                # it holds a large surplus that can be spent elsewhere -- moving
                # it to 0.45 costs only ~0.2 while freeing capacity for a sample
                # stuck at 0.85 whose squared error is 0.72. Accuracy sees that
                # trade very differently: 0.02 and 0.45 are both successes, but
                # 0.55 is a total loss, and the squared error barely changes
                # across that cliff (0.20 -> 0.30). Enabling --signed_magnitude_input
                # gave the model the freedom to make exactly that trade, and it
                # did: Eyeglasses rm fell 98.4% -> 90.6% funding a Male rm gain.
                #
                # The hinge penalises only the shortfall relative to crossing the
                # boundary by --target_hinge_margin_frac (scaled per attribute --
                # see below), and only ever asks for the
                # easier of (soft_target, boundary) so a partial train_scale is
                # never turned into a demand for a full flip. Past the boundary
                # the gradient is zero, so there is no surplus left to harvest.
                _sign = torch.sign(hard_teacher_target.detach() - src_attr)
                # Margin as a fraction of THIS attribute's own target gap, not
                # a shared absolute offset -- see --target_hinge_margin_frac.
                _gap = torch.empty_like(src_attr)
                for _local in torch.unique(mid_idx):
                    _abs = int(args.attribute_index[int(_local.item())])
                    _low, _high = SOFT_TARGET_TABLE.get(_abs, DEFAULT_SOFT_TARGET)
                    _gap[mid_idx == _local] = (_high - _low) / 2.0
                _boundary = 0.5 + _sign * args.target_hinge_margin_frac * _gap
                _eff_target = torch.where(
                    _sign > 0,
                    torch.minimum(soft_target_for_loss, _boundary),
                    torch.maximum(soft_target_for_loss, _boundary),
                )
                changed_mse_per_sample = torch.relu(
                    _sign * (_eff_target - edited_probs)
                ).pow(2)
            else:
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
                    # Directional CLIP measures cos(clip(edit) - clip(src), text_delta).
                    # The source MUST be the reconstruction, not the real photo:
                    # with the real photo, every img_delta carries the same
                    # inversion-gap offset in CLIP space on top of the actual
                    # edit direction, which drags the cosine toward that offset
                    # and away from the pos/neg text axis the loss is supposed
                    # to align to. That is very likely why this file's own notes
                    # record "directional CLIP loss left the color-cast
                    # unchanged" -- the directional signal was diluted, so it
                    # never got a fair test.
                    src_images=loss_ref_img if args.clip_prompt_mode == 'directional' else None,
                )   # clip_loss_each: (B,)
                if clip_loss_balancer is not None:
                    # Auto-balanced: tracks THIS loss's own per-attribute progress
                    # (not changed_loss/r34, which would miss teacher-fooling gaps
                    # like eyeglasses reading ~91% to the teacher but far lower to
                    # CLIP -- see --balance_clip_prompt_loss help).
                    clip_loss_balancer.update(mid_idx.detach(), clip_loss_each.detach())
                    clip_sample_weight = clip_loss_balancer.weights_for(mid_idx).detach()
                elif judge_balancer is not None:
                    # --clip_balance_signal judge: weights come from
                    # JudgePeakDeclineBalancer, updated in the AccCeleb monitor
                    # block below (every --celeba_judge_interval steps) from the
                    # judge's actual pass/fail calls, not this loss. Reading
                    # .weights_for() here just applies whatever the balancer's
                    # state currently is -- it lags by at most one judge tick,
                    # same as every other EMA-based balancer in this file.
                    clip_sample_weight = judge_balancer.weights_for(
                        mid_idx, edit_is_rm).detach()
                else:
                    clip_sample_weight = torch.ones_like(clip_loss_each)
                    clip_sample_weight[_clip_abs_idx == 39] = args.clip_prompt_age_weight
                    clip_sample_weight[_clip_abs_idx == 20] = args.clip_prompt_gender_weight
                    clip_sample_weight[_clip_abs_idx == 15] = args.clip_prompt_glasses_weight
                clip_semantic_loss = (clip_sample_weight * clip_loss_each).mean()
                clip_logs['clip_prompt_age_fraction'] = (_clip_abs_idx == 39).float().mean().detach()
                clip_logs['clip_prompt_gender_fraction'] = (_clip_abs_idx == 20).float().mean().detach()
                clip_logs['clip_prompt_glasses_fraction'] = (_clip_abs_idx == 15).float().mean().detach()

            if lag_dof_losses is None:
                lag_orth = _zero.clone()
                lag_gate_smooth = _zero.clone()
                lag_gate_sparse = _zero.clone()
            else:
                lag_orth = lag_dof_losses['orth']
                lag_gate_smooth = lag_dof_losses['gate_smooth']
                lag_gate_sparse = lag_dof_losses['gate_sparse']

            id_warmup_steps = 1500
            id_weight = args.id_loss_weight * min(1.0, n_iter / max(1, id_warmup_steps))
            if direction_bank is not None:
                dir_orth_loss = direction_bank.orthogonality_loss()
                dir_logs = direction_bank.last_logs if direction_bank_applied else {}
                # gate_load_balance_loss() returns a value computed INSIDE the
                # most recent direction_bank(...) forward call (see
                # _last_gate_diversity_loss in direction_bank.py) -- only valid
                # to read/backward through when that forward call actually
                # happened THIS step (direction_bank_applied). Reading it on a
                # step where direction_bank wasn't called would reuse a stale
                # tensor from a previous step whose autograd graph has already
                # been freed by that step's own .backward().
                dir_gate_diversity_loss = (
                    direction_bank.gate_load_balance_loss()
                    if (args.dir_gate_diversity_weight > 0 and direction_bank_applied)
                    else _zero.clone()
                )
            else:
                dir_orth_loss = _zero.clone()
                dir_logs = {}
                dir_gate_diversity_loss = _zero.clone()

            diffusion_loss = _zero.clone()       # non-age DDS (glasses/gender)
            age_diffusion_loss = _zero.clone()   # age DDS, separately weighted
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

                # Fine-layer masking: DDS gradients only flow through W+ layers below
                # the cutoff; layers at/above it are detached. A separate G forward
                # avoids contaminating id/reg gradients that need fine layer info.
                # Age can use a DIFFERENT cutoff (--age_dds_fine_layer_start) so the
                # diffusion teacher can reach the fine layers that carry wrinkle/skin
                # texture, which the default cutoff (7) blocks -- see arg help.
                def _dds_face(fine_start):
                    if fine_start > 0 and fine_start < new_latents.size(1):
                        _nl = torch.cat([
                            new_latents[:, :fine_start, :],
                            new_latents[:, fine_start:, :].detach(),
                        ], dim=1)
                        _ft = G([_nl], skips=control_skips, embed_res=args.controlnet_embed_res,
                               input_is_latent=True,
                               randomize_noise=False)[0].clamp(-1, 1)
                        return F.interpolate(_ft, (args.img_size, args.img_size))
                    return new_face_tensors

                # --dds_face_mask: restrict the DDS gradient to the region each
                # attribute is allowed to touch, per-sample (a batch can mix
                # glasses and gender). Eyeglasses gets the narrow
                # LOCAL_REGION_CLASSES mask; everything else gets FaceParser's
                # default FACE_CLASSES mask, which already includes hair.
                def _dds_mask(images, abs_idx_1d):
                    if face_parser is None or not args.dds_face_mask:
                        return None
                    mask = torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3],
                                       device=images.device, dtype=images.dtype)
                    for _a in set(abs_idx_1d.detach().cpu().tolist()):
                        _sel = abs_idx_1d == _a
                        if _a in LOCAL_REGION_CLASSES:
                            mask[_sel] = face_parser.get_region_mask(
                                images[_sel], LOCAL_REGION_CLASSES[_a])
                        else:
                            mask[_sel] = face_parser.get_mask(images[_sel])
                    return mask

                # Non-age samples (glasses, gender): standard cutoff and timestep range
                non_age_mask = ~is_age
                if non_age_fires:
                    _face_non_age = _dds_face(args.dds_fine_layer_start)
                    # DDS subtracts the source branch's noise residual as a bias
                    # correction; that cancellation is only valid when src and
                    # edit differ ONLY by the edit. Passing the real photo makes
                    # the residual difference also contain the inversion gap,
                    # leaving an uncancelled "fix the reconstruction" component
                    # in the gradient. See --losses_vs_recon.
                    _loss, _logs = diffusion_guidance(
                        src_images=loss_ref_img[non_age_mask],
                        edit_images=_face_non_age[non_age_mask],
                        attr_abs_idx=mid_abs_idx[non_age_mask],
                        target_values=soft_target[non_age_mask].detach(),
                        face_mask=_dds_mask(_face_non_age[non_age_mask], mid_abs_idx[non_age_mask]),
                    )
                    diffusion_loss = diffusion_loss + _loss
                    diffusion_logs.update(_logs)

                # Age samples: coarse timestep range, own interval, own fine-layer
                # cutoff (may reach fine layers), accumulated into age_diffusion_loss
                # so it can carry its own --age_diffusion_weight.
                if age_fires:
                    _age_fine_start = (args.age_dds_fine_layer_start
                                       if args.age_dds_fine_layer_start >= 0
                                       else args.dds_fine_layer_start)
                    _face_age = _dds_face(_age_fine_start)
                    _loss, _logs = diffusion_guidance(
                        src_images=loss_ref_img[is_age],
                        edit_images=_face_age[is_age],
                        attr_abs_idx=mid_abs_idx[is_age],
                        target_values=soft_target[is_age].detach(),
                        timestep_min=args.age_diffusion_timestep_min,
                        timestep_max=args.age_diffusion_timestep_max,
                        face_mask=_dds_mask(_face_age[is_age], mid_abs_idx[is_age]),
                    )
                    age_diffusion_loss = age_diffusion_loss + _loss
                    diffusion_logs.update({f'age_{k}': v for k, v in _logs.items()})

            loss = args.kd_loss_weight * kd_loss +\
                args.nll_loss_weight * log_p2 +\
                args.reg_loss_weight * reg_loss +\
                id_weight * id_loss +\
                args.counter_attr_weight * counter_attr_loss +\
                args.orth_loss_weight * lag_orth +\
                args.gate_smooth_weight * lag_gate_smooth +\
                args.gate_sparse_weight * lag_gate_sparse +\
                args.direction_orth_weight * dir_orth_loss +\
                args.dir_gate_diversity_weight * dir_gate_diversity_loss +\
                args.diffusion_guidance_weight * diffusion_loss +\
                (args.age_diffusion_weight if args.age_diffusion_weight >= 0
                 else args.diffusion_guidance_weight) * age_diffusion_loss +\
                args.clip_prompt_weight * clip_semantic_loss +\
                args.local_region_loss_weight * local_region_loss +\
                args.color_shift_loss_weight * color_shift_loss +\
                args.controlnet_reg_weight * loss_control_reg

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

                if (args.save_best and best_metric_ema is not None
                        and n_iter >= args.best_min_step
                        and best_metric_ema > best_metric_seen):
                    best_metric_seen = best_metric_ema
                    save_best_checkpoints(
                        logger.save_root, n_iter, logger,
                        ema_modules_for_save if args.use_ema else None,
                        best_metric_ema, 'judge_celeb_acc_ema')
                    
            # AccCeleb monitor: same success rule as evaluate_sdflow.strict_success
            # (did the edited score cross 0.5), scored by the independent judge.
            # No gradient -- purely a readout.
            celeb_acc_log = {}
            if celeb_monitor is not None and n_iter % args.celeba_judge_interval == 0:
                with torch.no_grad():
                    _src_c = celeb_monitor.scores(
                        F.interpolate(src_face_256, (256, 256)))[:, attribute_index]
                    _edit_c = celeb_monitor.scores(
                        F.interpolate(new_face_256, (256, 256)))[:, attribute_index]
                    _s = _src_c[batch_indices, mid_idx]
                    _e = _edit_c[batch_indices, mid_idx]
                    _ok = torch.where(_s > 0.5, _e < 0.5, _e > 0.5).float()
                    if judge_balancer is not None:
                        judge_balancer.update(mid_idx.detach(), _ok.detach(), edit_is_rm)
                        for _slot in range(judge_balancer.num_slots):
                            _key = judge_balancer.slot_name(_slot, args.attribute_index)
                            celeb_acc_log[f'judge_balance_weight/{_key}'] = \
                                judge_balancer.weights[_slot]
                            celeb_acc_log[f'judge_balance_peak/{_key}'] = \
                                judge_balancer.peak_acc[_slot]
                    for _local in torch.unique(mid_idx):
                        _m = mid_idx == _local
                        _abs = int(args.attribute_index[int(_local.item())])
                        celeb_acc_log[f'judge_celeb_acc/attr_{_abs}'] = _ok[_m].mean()
                        # Also split by direction. The per-attribute mean hid a
                        # 58.7pp add/rm gap for a whole run; logging both halves
                        # makes that visible in wandb as it happens instead of
                        # only in the eval afterwards.
                        for _rm in (False, True):
                            _md = _m & (edit_is_rm == _rm)
                            if _md.any():
                                celeb_acc_log[
                                    f'judge_celeb_acc/attr_{_abs}_{"rm" if _rm else "add"}'
                                ] = _ok[_md].mean()
                    celeb_acc_log['judge_celeb_acc_mean'] = _ok.mean()
                    _m = float(_ok.mean())
                    best_metric_ema = _m if best_metric_ema is None else (
                        args.best_metric_ema * best_metric_ema
                        + (1.0 - args.best_metric_ema) * _m)
                    celeb_acc_log['judge_celeb_acc_ema'] = torch.tensor(best_metric_ema)

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
                'lag_gate_sparse': lag_gate_sparse,
                'edit_scale': train_scale.detach().mean(),
                'attr_scale_grad_norm': attr_scale_grad_norm,
                'final_delta_norm_pre_clip': final_delta_norm_pre_clip,
                'final_delta_norm': final_delta_norm,
                'final_delta_clip_factor': final_delta_clip_factor,
                'final_delta_max_norm': torch.tensor(args.final_delta_max_norm),
                'control_skip_norm': control_skip_norm,
                'loss_control_reg': loss_control_reg.detach(),
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
                'dir_gate_diversity_loss': dir_gate_diversity_loss,
                'loss_diffusion_dds': diffusion_loss,
                'loss_age_diffusion_dds': age_diffusion_loss,
                'loss_clip_prompt':   clip_semantic_loss,
                'loss_local_region':  local_region_loss,
                'loss_color_shift':   color_shift_loss,
                'clip_score_mean':     clip_logs.get('clip_score_mean',     latent.new_tensor(0.0)),
                'clip_score_pos_mean': clip_logs.get('clip_score_pos_mean', latent.new_tensor(0.0)),
                'clip_score_neg_mean': clip_logs.get('clip_score_neg_mean', latent.new_tensor(0.0)),
                'clip_directional_cos': clip_logs.get('clip_directional_cos', latent.new_tensor(0.0)),
                'clip_direction_mean': clip_logs.get('clip_direction_mean', latent.new_tensor(0.0)),
                'clip_prompt_weight':  torch.tensor(args.clip_prompt_weight),
                'clip_prompt_age_fraction':    clip_logs.get('clip_prompt_age_fraction',    latent.new_tensor(0.0)),
                'clip_prompt_gender_fraction': clip_logs.get('clip_prompt_gender_fraction', latent.new_tensor(0.0)),
                'clip_prompt_glasses_fraction': clip_logs.get('clip_prompt_glasses_fraction', latent.new_tensor(0.0)),
            }
            current_attr_scales = attr_scales.current_scales()
            for _i, _attr_abs_idx in enumerate(args.attribute_index):
                _log_dict[f'attr_scale/attr_{_attr_abs_idx}'] = current_attr_scales[_i]
            if direction_bank is not None:
                current_residual_scales = direction_bank.current_residual_scale().detach()
                for _i, _attr_abs_idx in enumerate(args.attribute_index):
                    _log_dict[f'residual_scale/attr_{_attr_abs_idx}'] = current_residual_scales[_i]
                _gate_entropy_per_attr = dir_logs.get('dir_gate_entropy_per_attr')
                if _gate_entropy_per_attr is not None:
                    for _i, _attr_abs_idx in enumerate(args.attribute_index):
                        _log_dict[f'dir_gate_entropy_ema/attr_{_attr_abs_idx}'] = _gate_entropy_per_attr[_i]
            current_reg_weights = reg_loss_weights.current_weights()
            for _i, _attr_abs_idx in enumerate(args.attribute_index):
                _log_dict[f'reg_weight_global/attr_{_attr_abs_idx}'] = current_reg_weights[_i, 0]
                _log_dict[f'reg_weight_coarse/attr_{_attr_abs_idx}'] = current_reg_weights[_i, 1]
                _log_dict[f'reg_weight_fine/attr_{_attr_abs_idx}'] = current_reg_weights[_i, 2]
            if loss_balancer is not None:
                for _i, _attr_abs_idx in enumerate(args.attribute_index):
                    _log_dict[f'balance_weight/attr_{_attr_abs_idx}'] = loss_balancer.weights[_i]
                    _log_dict[f'balance_ema_loss/attr_{_attr_abs_idx}'] = loss_balancer.ema_loss[_i]
            if clip_loss_balancer is not None:
                for _i, _attr_abs_idx in enumerate(args.attribute_index):
                    _log_dict[f'clip_balance_weight/attr_{_attr_abs_idx}'] = clip_loss_balancer.weights[_i]
                    _log_dict[f'clip_balance_ema_loss/attr_{_attr_abs_idx}'] = clip_loss_balancer.ema_loss[_i]
            for _k, _v in diffusion_logs.items():
                _log_dict[_k] = _v
            for _k, _v in celeb_acc_log.items():
                _log_dict[_k] = _v
            logger.msg(_log_dict, n_iter)
            
