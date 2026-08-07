"""ControlNet-style feature-map injection for StyleGAN2 attribute editing.

Every editing mechanism in this project so far (Direction Bank, flow
residual, LoRA magnitude adapter) injects its correction at exactly ONE
point: the W+ latent code, which then gets broadcast through ALL 18
StyleGAN2 layers via the standard AdaIN-style modulation. That single
global injection point is the likely root cause of a recurring pattern
this session: attributes that should stay local (glasses) or global (age,
gender) end up fighting for control of the same W+ vector, and
post-hoc fixes at that level (cross-attribute direction decorrelation,
DDS-anchored refinement, LoRA on the magnitude network) have not resolved
the entanglement -- see scripts/dump_attr_failures.py --watch_attrs
results for age vs gender.

models/stylegan2/model.py's Generator.forward() already has a dormant
hook for exactly this: a `skips` argument that gets ADDED to the internal
feature map at the resolution matching `embed_res` (default 64x64,
channels[64] = 512 for the default channel_multiplier=2):

    if out.shape[-1] == embed_res and skips is not None:
        out = out + skips

Nothing in this project calls G(..., skips=...) -- it was inherited
unused from whatever base StyleGAN2 fork this was built on (likely a
high-fidelity-inversion codebase's "distortion consultation" mechanism)
and never wired up. AttributeControlEncoder predicts that `skips` tensor
from the attribute being edited, giving each attribute a private decoder
head that injects control DIRECTLY into an intermediate generator feature
map instead of only at the W+ input. Coarse layers dominate global
structure and fine layers dominate local detail/texture in StyleGAN2's
hierarchy; a 64x64 feature map sits in the middle of that hierarchy, so
this is a genuinely different lever than anything tried so far, not
another variant of "adjust the W+ direction."

Each attribute's decoder head ends in a zero-initialized conv, so at
step 0 this contributes nothing (safe to enable on top of an existing
trained run, same convention as the LoRA adapter in direction_bank.py).
"""
import math

import torch
import torch.nn as nn


class AttributeControlEncoder(nn.Module):
    def __init__(self, num_attrs, out_channels=512, out_res=64, seed_res=4, hidden_dim=256):
        super().__init__()
        self.num_attrs = int(num_attrs)
        self.out_channels = int(out_channels)
        self.out_res = int(out_res)
        self.seed_res = int(seed_res)
        assert self.out_res % self.seed_res == 0 and \
            (self.out_res // self.seed_res) & (self.out_res // self.seed_res - 1) == 0, \
            "out_res / seed_res must be a power of 2"
        num_upsamples = int(math.log2(self.out_res // self.seed_res))

        # Shared trunk: attribute delta -> a small spatial "seed" feature map.
        # Shared across attributes (like magnitude_net), since this only sets
        # up a generic starting point; per-attribute specialization happens
        # entirely in each attribute's own decoder head below.
        self.fc = nn.Sequential(
            nn.Linear(self.num_attrs, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.seed_proj = nn.Linear(hidden_dim, self.out_channels * self.seed_res * self.seed_res)

        # Per-attribute decoder head: upsamples the shared seed to the target
        # resolution. Separate weights per attribute (not LoRA-shared) since
        # this module is new and unvalidated -- keep it simple, one variable
        # at a time, matching this project's own experimental discipline.
        self.attr_heads = nn.ModuleList([
            self._build_head(self.out_channels, num_upsamples) for _ in range(self.num_attrs)
        ])

    def _build_head(self, channels, num_upsamples):
        layers = []
        for _ in range(num_upsamples):
            layers += [
                nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        final = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        return nn.Sequential(*layers)

    def forward(self, attr_delta, attr_idx):
        """
        attr_delta: (B, num_attrs) -- same tensor passed to AttributeDirectionBank.
        attr_idx:   (B,) long -- which attribute is being edited, per sample.
        Returns: (B, out_channels, out_res, out_res), to pass as
                 StyleGAN2 Generator's `skips=` argument.
        """
        B = attr_delta.size(0)
        device, dtype = attr_delta.device, attr_delta.dtype
        hidden = self.fc(attr_delta)
        seed = self.seed_proj(hidden).view(B, self.out_channels, self.seed_res, self.seed_res)

        out = torch.zeros(B, self.out_channels, self.out_res, self.out_res,
                           device=device, dtype=dtype)
        attr_idx = attr_idx.view(-1).long()
        for a in range(self.num_attrs):
            mask = attr_idx == a
            if mask.any():
                out[mask] = self.attr_heads[a](seed[mask])
        return out
