import glob
import os

import torch

from common.id_loss import IDLoss
from models.conditioner import IdentityAttributeConditioner
from models.direction_bank import AttributeDirectionBank
from models.flows.flow import cnf
from models.layer_mask import AttributeLayerMask


def _format_ckpt_step(step):
    if step is None:
        return None
    return str(int(step)).zfill(7)


def _find_latest_ckpt(ckpt_dir, prefix, step=None):
    step = _format_ckpt_step(step)
    if step is not None:
        path = os.path.join(ckpt_dir, f'{prefix}-{step}')
        if not os.path.exists(path):
            raise ValueError(f'No {prefix}-{step} checkpoint found in {ckpt_dir}')
        return path
    files = glob.glob(os.path.join(ckpt_dir, f'{prefix}-*'))
    if not files:
        raise ValueError(f'No {prefix} checkpoints found in {ckpt_dir}')
    return sorted(files)[-1]


def _find_latest_ckpt_optional(ckpt_dir, prefix, step=None):
    step = _format_ckpt_step(step)
    if step is not None:
        path = os.path.join(ckpt_dir, f'{prefix}-{step}')
        return path if os.path.exists(path) else None
    files = glob.glob(os.path.join(ckpt_dir, f'{prefix}-*'))
    if not files:
        return None
    return sorted(files)[-1]



def _load_state_allow_old_pos(model, path):
    state = torch.load(path, map_location='cpu')
    result = model.load_state_dict(state, strict=False)
    allowed_missing = [
        key for key in result.missing_keys
        if key.endswith('pos_emb') or key.endswith('pos_alpha')
    ]
    real_missing = sorted(set(result.missing_keys) - set(allowed_missing))
    if real_missing:
        raise RuntimeError(
            f'Checkpoint {path} does not match the requested flow structure. '
            f'Missing keys include: {real_missing[:8]}'
        )
    if result.unexpected_keys:
        print(f'[WARN] Unexpected keys while loading {path}: {result.unexpected_keys[:8]}')
    if allowed_missing:
        print(f'[WARN] Loaded old flow checkpoint without position embedding keys: {path}')


class SDFlow(object):
    def __init__(self, ckpt_dir, attr_num, attr_list=[15, 20, 39], scale=1.0, device='cuda',
                 id_cond_dim=32, id_cond_scale=0.25, attr_backbone='resnet50',
                 conditioner_backbone='resnet', clip_model='ViT-B/32', fused_hidden_dim=256,
                 flow_modules='512-512-512-512-512', num_blocks=1,
                 velocity_field='original', lag_gate_hidden_dim=64,
                 lag_gate_init_bias=-1.5, direction_bank_path=None,
                 direction_residual_scale=0.05, direction_freeze=True,
                 ckpt_step=None, bypass_glasses_direction_bank=True,
                 guided_delta_max_norm=None) -> None:
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.attr_num = attr_num
        self.scale = scale
        self.id_cond_dim = id_cond_dim
        self.id_cond_scale = id_cond_scale
        self.attr_backbone = attr_backbone
        self.conditioner_backbone = conditioner_backbone
        self.clip_model = clip_model
        self.fused_hidden_dim = fused_hidden_dim
        self.flow_modules = flow_modules
        self.num_blocks = num_blocks
        self.velocity_field = velocity_field
        self.lag_gate_hidden_dim = lag_gate_hidden_dim
        self.lag_gate_init_bias = lag_gate_init_bias
        self.direction_bank_path = direction_bank_path
        self.direction_residual_scale = direction_residual_scale
        self.direction_freeze = direction_freeze
        self.ckpt_step = ckpt_step
        self.bypass_glasses_direction_bank = bypass_glasses_direction_bank
        # Single global cap shared by every attribute, replacing the old
        # per-attribute (age-only) direction_scale/layer_scale/delta_max_norm
        # hack. None disables the cap so the raw, LDA-direction-guided delta
        # can be measured honestly before deciding whether any cap is needed.
        self.guided_delta_max_norm = guided_delta_max_norm

        self.class_indices = attr_list

        self.id_extractor = IDLoss(crop=True).to(self.device).eval()
        for p in self.id_extractor.parameters():
            p.requires_grad_(False)

        self.conditioner = IdentityAttributeConditioner(
            attr_dim=len(self.class_indices),
            id_dim=id_cond_dim,
            attr_backbone=attr_backbone,
            id_scale=id_cond_scale,
            conditioner_backbone=conditioner_backbone,
            clip_model=clip_model,
            fused_hidden_dim=fused_hidden_dim,
        ).to(self.device)
        filename = _find_latest_ckpt(ckpt_dir, 'conditioner', ckpt_step)
        self.conditioner.load_state_dict(torch.load(filename, map_location='cpu'), strict=True)
        self.conditioner.eval()
        print(f'Loaded conditioner from {filename}')

        flow_condition_dim = self.conditioner.condition_dim
        self.flow = cnf(
            512,
            flow_modules,
            flow_condition_dim,
            num_blocks,
            velocity_field=self.velocity_field,
            num_layers=18,
            gate_hidden_dim=lag_gate_hidden_dim,
            gate_init_bias=lag_gate_init_bias,
            attr_context_dim=len(self.class_indices),
            train_T=False,
        ).to(self.device)
        filename = _find_latest_ckpt(ckpt_dir, 'prior', ckpt_step)
        _load_state_allow_old_pos(self.flow, filename)
        self.flow.eval()
        print(f'Loaded flow from {filename}')

        self.layer_mask = None
        if self.velocity_field == 'original':
            self.layer_mask = AttributeLayerMask(num_attrs=len(self.class_indices)).to(self.device)
            filename = _find_latest_ckpt_optional(ckpt_dir, 'layer_mask', ckpt_step)
            if filename is not None:
                self.layer_mask.load_state_dict(torch.load(filename, map_location='cpu'), strict=True)
                self.layer_mask.eval()
                print(f'Loaded layer_mask from {filename}')

        self.direction_bank = None
        if direction_bank_path is not None:
            _bank_meta = torch.load(direction_bank_path, map_location="cpu")
            _num_k = int(_bank_meta.get("num_k", 1)) if isinstance(_bank_meta, dict) else 1
            # No per-attribute direction_scale/layer_scale/delta_max_norm here:
            # every attribute gets the same (default) treatment, and the only
            # magnitude safety net is the shared guided_delta_max_norm below.
            self.direction_bank = AttributeDirectionBank(
                num_attrs=len(self.class_indices),
                num_layers=18,
                latent_dim=512,
                num_k=_num_k,
                bank_path=direction_bank_path,
                attribute_index=self.class_indices,
                residual_scale=direction_residual_scale,
                freeze_directions=direction_freeze,
                guided_delta_max_norm=self.guided_delta_max_norm,
            ).to(self.device)
            filename = _find_latest_ckpt_optional(ckpt_dir, 'direction_bank', ckpt_step)
            if filename is not None:
                result = self.direction_bank.load_state_dict(
                    torch.load(filename, map_location='cpu'),
                    strict=False,
                )
                if result.missing_keys:
                    print(f'[WARN] Direction bank missing keys: {result.missing_keys[:8]}')
                if result.unexpected_keys:
                    print(f'[WARN] Direction bank unexpected keys: {result.unexpected_keys[:8]}')
                print(f'Loaded direction_bank from {filename}')
            else:
                print(f'Loaded direction_bank initialization from {direction_bank_path}')
            self.direction_bank.eval()

    def samples(self, targets):
        """Sample W+ latents from either full condition or attribute-only targets.

        targets can be [B, condition_dim] or [B, attr_dim]. For attribute-only
        targets, a zero identity condition is prepended.
        """
        targets = targets.to(self.device)
        batch = targets.shape[0]
        attr_dim = len(self.class_indices)
        if targets.dim() != 2:
            raise ValueError(f'targets must be 2D, got shape {tuple(targets.shape)}')
        if targets.size(1) == attr_dim:
            zero_id = torch.zeros(batch, self.id_cond_dim, device=self.device, dtype=targets.dtype)
            targets = torch.cat([zero_id, targets], dim=1)
        if targets.size(1) != self.conditioner.condition_dim:
            raise ValueError(
                f'targets must have attr_dim={attr_dim} or '
                f'condition_dim={self.conditioner.condition_dim}, got {targets.size(1)}'
            )
        z = torch.randn(batch, 18, 512, device=self.device)
        new_styles, _ = self.flow(
            z,
            targets,
            torch.zeros(batch, 18, 1, device=self.device),
            reverse=True,
        )
        return new_styles

    def transform(self, inputs, sources, images=None):
        if images is None:
            raise ValueError('SDFlow.transform requires aligned source images for identity conditioning.')

        targets = sources.clone()
        strength = max(0.0, min(float(self.scale), 4.0))
        source_attr = sources[:, self.attr_num]
        targets[:, self.attr_num] = source_attr * (1.0 - strength) + (1.0 - source_attr) * strength

        sources = sources[:, self.class_indices]
        targets = targets[:, self.class_indices]

        src_cond, id_cond, attr_cond = self.conditioner.make_condition(images, inputs, self.id_extractor)
        attr_local_idx = self.class_indices.index(self.attr_num)
        attr_idx = torch.full((inputs.size(0),), attr_local_idx, device=self.device, dtype=torch.long)
        z = self.flow(inputs, src_cond, torch.zeros(targets.size(0), inputs.size(1), 1).to(self.device))[0]

        new_attr_cond = attr_cond.clone()
        new_attr_cond[:, attr_local_idx] = targets[:, attr_local_idx]
        new_cond = torch.cat([id_cond, new_attr_cond], dim=1)
        new_styles_raw, _ = self.flow(
            z,
            new_cond,
            torch.zeros(targets.size(0), inputs.size(1), 1).to(self.device),
            reverse=True,
        )
        flow_delta = new_styles_raw - inputs
        if self.velocity_field == 'original' and self.layer_mask is not None:
            lm = self.layer_mask(
                attr_idx,
                attr_cond[:, attr_local_idx],
                new_attr_cond[:, attr_local_idx],
            ).unsqueeze(-1)
            new_styles = inputs + lm * flow_delta
        elif (self.direction_bank is not None
              and not (self.bypass_glasses_direction_bank and self.attr_num == 15)):
            attr_delta = new_attr_cond - attr_cond
            guided_delta = self.direction_bank(flow_delta, attr_delta, attr_idx=attr_idx, latent=inputs)
            new_styles = inputs + guided_delta
        else:
            new_styles = inputs + flow_delta
        return new_styles
