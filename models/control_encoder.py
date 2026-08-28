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
import torch.nn.functional as F


class AttributeControlEncoder(nn.Module):
    """Magnitude is decoupled from direction on purpose.

    The head network only decides the SHAPE of the injected signal; its
    output is L2-normalized per sample and then scaled by a single
    learnable gain per attribute. Letting the conv stack set the magnitude
    directly turned out to be unusable in both directions: at the shared
    base LR the zero-initialized stack never grew past ~0.5 against a
    512x64x64 feature map whose own norm measures ~2931 (a 0.017%
    perturbation -- --disable_controlnet reproduced identical accuracy, ID,
    LPIPS and leakage, so the module was inert), and at 20x LR with no
    penalty it diverged to ~6e7 within 70 steps. Four stacked 512-channel
    ConvTranspose2d layers compound weight growth multiplicatively, so the
    output norm responds to LR far too sharply to tune by trial.

    With this split, control_skip_norm in the training log IS the gain:
    directly observable, settable via --controlnet_init_gain, and bounded
    by --controlnet_max_norm. The gain stays learnable so the model can
    still choose how much to lean on this branch.

    per_direction (default off, opt-in via --controlnet_per_direction) gives
    add and rm their own decoder head and their own gain per attribute,
    instead of one shared head/gain per attribute handling both directions.
    Sharing forces the SAME weights to learn two very different tasks at
    once -- e.g. eyeglasses add must synthesize frame structure from nothing
    (source has no glasses pixels at all) while rm only has to erase an
    existing one, an easier task. A 128x128 run (v41) showed exactly the
    predicted failure mode: judge_celeb_acc/attr_15_add climbed slowly from
    ~0 while judge_celeb_acc/attr_15_rm drifted down over the same steps --
    the harder add task pulling shared capacity away from rm, the same
    entanglement --per_direction already fixed for AttributeDirectionBank
    (direction_bank.py) and never fixed here. Splitting removes the shared
    capacity these two are competing for.
    """

    def __init__(self, num_attrs, out_channels=512, out_res=64, seed_res=4, hidden_dim=256,
                 init_gain=1.0, per_direction=False):
        super().__init__()
        self.num_attrs = int(num_attrs)
        self.out_channels = int(out_channels)
        self.out_res = int(out_res)
        self.seed_res = int(seed_res)
        self.per_direction = bool(per_direction)
        self.num_slots = self.num_attrs * 2 if self.per_direction else self.num_attrs
        assert self.out_res % self.seed_res == 0 and \
            (self.out_res // self.seed_res) & (self.out_res // self.seed_res - 1) == 0, \
            "out_res / seed_res must be a power of 2"
        num_upsamples = int(math.log2(self.out_res // self.seed_res))

        # Shared trunk: attribute delta -> a small spatial "seed" feature map.
        # Shared across attributes AND directions (like magnitude_net), since
        # this only sets up a generic starting point; per-slot specialization
        # happens entirely in each slot's own decoder head below.
        self.fc = nn.Sequential(
            nn.Linear(self.num_attrs, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.seed_proj = nn.Linear(hidden_dim, self.out_channels * self.seed_res * self.seed_res)

        # Per-slot decoder head: upsamples the shared seed to the target
        # resolution. Separate weights per slot (not LoRA-shared) since
        # this module is new and unvalidated -- keep it simple, one variable
        # at a time, matching this project's own experimental discipline.
        # Slot layout is attr-major when per_direction: slot 2a is attribute
        # a's add direction, 2a+1 its rm direction (same convention as
        # direction_bank.py / JudgePeakDeclineBalancer / LearnableAttributeScales).
        self.attr_heads = nn.ModuleList([
            self._build_head(self.out_channels, num_upsamples) for _ in range(self.num_slots)
        ])

        # One learnable magnitude per slot, through softplus so it stays
        # positive. Stored inverted from init_gain so that at step 0 the
        # injected signal has exactly norm == init_gain.
        init_gain = max(float(init_gain), 1e-4)
        raw = math.log(math.expm1(init_gain))
        self.log_gain = nn.Parameter(torch.full((self.num_slots,), raw))

    def slot_index(self, attr_idx, is_rm=None):
        if not self.per_direction:
            return attr_idx
        if is_rm is None:
            raise ValueError('per_direction AttributeControlEncoder needs is_rm per sample')
        return attr_idx * 2 + is_rm.long()

    def _build_head(self, channels, num_upsamples):
        layers = []
        for _ in range(num_upsamples):
            layers += [
                nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        # NOT zero-initialized: the output is L2-normalized in forward(), so a
        # zero head would divide by zero. Magnitude safety comes from log_gain
        # (and --controlnet_max_norm) instead.
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        return nn.Sequential(*layers)

    def forward(self, attr_delta, attr_idx, is_rm=None):
        """
        attr_delta: (B, num_attrs) -- same tensor passed to AttributeDirectionBank.
        attr_idx:   (B,) long -- which attribute is being edited, per sample.
        is_rm:      (B,) bool -- True where the source already HAS the
                    attribute (this edit removes it). Required when
                    per_direction=True, same convention as
                    AttributeDirectionBank/JudgePeakDeclineBalancer/
                    LearnableAttributeScales; ignored otherwise.
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
        slot_idx = self.slot_index(attr_idx, is_rm)
        for s in range(self.num_slots):
            mask = slot_idx == s
            if mask.any():
                out[mask] = self.attr_heads[s](seed[mask])

        # Direction from the head, magnitude from the per-slot gain.
        out = F.normalize(out.reshape(B, -1), dim=1).view_as(out)
        gain = F.softplus(self.log_gain)[slot_idx].to(device=device, dtype=dtype)
        return out * gain.view(-1, 1, 1, 1)
