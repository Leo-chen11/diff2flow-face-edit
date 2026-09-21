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
feature map. AttributeControlEncoder predicts those tensors from the
attribute being edited, giving each attribute a private decoder head that
injects control DIRECTLY into intermediate generator feature maps instead
of only at the W+ input.

WHY MULTI-RESOLUTION
--------------------
The first version of this module injected at ONE resolution (64x64), and
the eval story that came out of it was lopsided: load-bearing for
eyeglasses (AccCeleb add 93% -> 12% with it disabled) but no measurable
benefit for age or gender (~70.7% either way), enough that
evaluate_sdflow.resolve_controlnet_disable_attrs now switches it OFF for
20/39 by default.

That asymmetry is what a single 64x64 injection point predicts. StyleGAN2's
hierarchy puts coarse structure in low-resolution layers and fine
detail/texture in high-resolution ones; 64x64 sits where mid-level
STRUCTURE lives. An eyeglass frame is mid-level structure, so 64x64 can
carry it. Wrinkles, skin pores, age spots and stubble are fine TEXTURE,
living at 128-512 -- a 64x64 injection has no path to them at all, so
"no benefit for age" was the expected result, not evidence that
feature injection cannot help age.

The two worst edits in the current eval are exactly the two that must
SYNTHESIZE structure the source image does not contain:

    Eyeglasses add   58.0%   (needs a frame that isn't there)
    Young rm         58.8%   (needs wrinkles/texture that aren't there)

every other direction, all of which only have to REMOVE or displace
existing structure, scores 72-96%. W+ cannot synthesize -- it moves the
face along the generator's manifold, it cannot ask for new detail at a
specific place. Feature injection can, but only at the resolutions it
actually reaches. Hence out_res is now a LIST.

SHARED TRUNK, PER-SLOT OUTPUT CONVS
-----------------------------------
The single-resolution version gave each slot its own full decoder stack
(4 ConvTranspose2d at 512 channels ~ 16.8M parameters each). That does
not survive the move to several resolutions with per_direction on:
6 slots x 3 resolutions of private stacks is >100M parameters on top of a
1024 StyleGAN2, Stable Diffusion, CLIP, ArcFace and BiSeNet already
resident.

So the upsampling trunk is shared and only the final output conv at each
resolution is per-slot. The trunk still sees slot-specific input (the
seed is conditioned on attr_delta and, with latent_cond, on this face's
W+ code), and the per-slot conv is what decides WHAT PATTERN gets written
into the feature map at that band -- which is where the specialization
between "draw a frame" and "erase a frame" actually belongs.

Each resolution carries its own learnable gain per slot, because the
generator's own feature-map norms differ by roughly an order of magnitude
across the hierarchy: one shared gain would force the fine band to live at
whatever scale the coarse band settled on.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def stylegan2_channels(channel_multiplier=2):
    """Channel count per resolution, mirroring Generator.channels in
    models/stylegan2/model.py. Kept here so the injected tensors are built
    with the exact width the generator expects to add them to."""
    return {
        4: 512,
        8: 512,
        16: 512,
        32: 512,
        64: 256 * channel_multiplier,
        128: 128 * channel_multiplier,
        256: 64 * channel_multiplier,
        512: 32 * channel_multiplier,
        1024: 16 * channel_multiplier,
    }


# ── helpers shared by training and evaluation ────────────────────────────
# skips is a dict {resolution: (B, C, R, R)}. These keep train_sdflow.py and
# evaluate_sdflow.py from each writing their own version of the same loop
# (they previously each had a private copy of the norm-clip block, which is
# how a norm cap can silently end up applied in training but not in eval).

def skips_norm_per_sample(skips):
    """(B,) tensor: total L2 norm across every injected resolution.

    For LOGGING and for the hard cap. Do not use this for the L2 penalty --
    see skips_reg_per_sample, which is scale-free in the number of bands.
    """
    norms = [t.reshape(t.size(0), -1).pow(2).sum(dim=1) for t in skips.values()]
    return torch.stack(norms, dim=0).sum(dim=0).clamp(min=1e-12).sqrt()


def skips_reg_per_sample(skips):
    """(B,) tensor: MEAN squared norm per band -- the quantity
    --controlnet_reg_weight multiplies.

    Averaged over bands, not summed, so the penalty measures "how hard is
    each band pushing" rather than "how many bands are there". Summing (the
    total norm squared) makes the penalty scale linearly with band count:
    two bands sitting at gain 1.0 would be charged 2.0 where a single band
    at the same per-band strength is charged 1.0, so merely ADDING a
    resolution doubles the pressure to shrink every gain. That turns
    --controlnet_reg_weight into a penalty on using multi-resolution
    injection at all, which is the opposite of what the flag is for, and it
    is not hypothetical: the first multi-resolution run (v12) decayed both
    control_skip_norm_r64 and control_skip_norm_r128 from 1.0 to ~0.2 along
    near-identical curves over 50k steps -- the signature of a penalty
    dominating the task gradient uniformly, not of the model judging one
    band more useful than the other.

    With a single band this is identical to the old definition, so
    single-resolution runs keep their calibration for --controlnet_reg_weight.
    """
    sq = [t.reshape(t.size(0), -1).pow(2).sum(dim=1) for t in skips.values()]
    return torch.stack(sq, dim=0).mean(dim=0)


def clip_skips(skips, max_norm):
    """Cap each resolution's per-sample norm at max_norm, independently.

    Per-resolution rather than on the combined norm: each band has its own
    learnable gain and its own natural scale against the generator feature
    map it is added to, so a single combined cap would let a large coarse
    band eat the whole budget and silently zero out the fine band that the
    synthesis edits actually need.
    """
    if max_norm is None or max_norm <= 0:
        return skips
    out = {}
    for res, t in skips.items():
        n = t.reshape(t.size(0), -1).norm(dim=1)
        scale = (max_norm / n.clamp(min=1e-8)).clamp(max=1.0)
        out[res] = t * scale.view(-1, 1, 1, 1)
    return out


def add_skips(a, b):
    """Sum two skip dicts (used when stacking several attribute edits)."""
    if a is None:
        return b
    if b is None:
        return a
    out = dict(a)
    for res, t in b.items():
        out[res] = out[res] + t if res in out else t
    return out


def is_legacy_state_dict(state):
    """True when a saved control_encoder came from LegacySingleResControlEncoder.

    The two architectures store different parameter names ('attr_heads.*' vs
    'stages.*'/'heads.*'), so a checkpoint can be identified without knowing
    which flags produced it -- which matters because the runs that need
    comparing (a single-resolution baseline against a multi-resolution
    challenger) are by construction on opposite sides of this change.
    """
    return any(k.startswith('attr_heads.') for k in state)


class _PerDirectionSlots:
    """Maps (attribute, direction) onto a slot index.

    Shared by both encoders below, which had kept byte-identical copies of
    this method -- to the point that the copy in the legacy class still
    named AttributeControlEncoder in its error message. With
    --controlnet_per_direction each attribute owns two slots (add and rm)
    instead of one, so 'add glasses' and 'remove glasses' stop sharing a
    single set of injection weights. Adds no parameters and no state, so it
    does not affect either class's state_dict.
    """

    def slot_index(self, attr_idx, is_rm=None):
        if not self.per_direction:
            return attr_idx
        if is_rm is None:
            raise ValueError(f'per_direction {type(self).__name__} needs is_rm per sample')
        return attr_idx * 2 + is_rm.long()


class LegacySingleResControlEncoder(_PerDirectionSlots, nn.Module):
    """The original single-resolution encoder, kept verbatim so checkpoints
    trained before multi-resolution injection stay loadable.

    This is not the architecture to train new runs with -- it exists so the
    single-resolution baseline can still be evaluated with the same command
    as the run being compared against it. Deleting it would make every
    pre-existing control_encoder checkpoint unloadable, which is exactly the
    baseline an A/B needs.

    Structure and parameter names are unchanged from the version that trained
    those checkpoints: shared trunk -> seed, one full ConvTranspose2d stack
    per slot ('attr_heads'), one gain per slot ('log_gain'), single output
    tensor at out_res.
    """

    def __init__(self, num_attrs, out_channels=512, out_res=64, seed_res=4, hidden_dim=256,
                 init_gain=1.0, per_direction=False, latent_cond=False, latent_dim=512):
        super().__init__()
        self.num_attrs = int(num_attrs)
        self.out_channels = int(out_channels)
        self.out_res = int(out_res)
        self.seed_res = int(seed_res)
        self.per_direction = bool(per_direction)
        self.latent_cond = bool(latent_cond)
        self.latent_dim = int(latent_dim)
        self.num_slots = self.num_attrs * 2 if self.per_direction else self.num_attrs
        assert self.out_res % self.seed_res == 0 and \
            (self.out_res // self.seed_res) & (self.out_res // self.seed_res - 1) == 0, \
            "out_res / seed_res must be a power of 2"
        num_upsamples = int(math.log2(self.out_res // self.seed_res))

        if self.latent_cond:
            latent_feat_dim = min(hidden_dim, self.latent_dim)
            self.latent_proj = nn.Sequential(
                nn.LayerNorm(self.latent_dim),
                nn.Linear(self.latent_dim, latent_feat_dim),
                nn.ReLU(inplace=True),
            )
        else:
            latent_feat_dim = 0
            self.latent_proj = None
        trunk_in = self.num_attrs + latent_feat_dim
        self.fc = nn.Sequential(
            nn.Linear(trunk_in, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.seed_proj = nn.Linear(hidden_dim, self.out_channels * self.seed_res * self.seed_res)
        self.attr_heads = nn.ModuleList([
            self._build_head(self.out_channels, num_upsamples) for _ in range(self.num_slots)
        ])
        init_gain = max(float(init_gain), 1e-4)
        raw = math.log(math.expm1(init_gain))
        self.log_gain = nn.Parameter(torch.full((self.num_slots,), raw))
        # Mirrors the multi-resolution attribute so callers can report the
        # injected width the same way for either architecture.
        self.res_channels = {self.out_res: self.out_channels}
        self.out_res_list = [self.out_res]

    def _build_head(self, channels, num_upsamples):
        layers = []
        for _ in range(num_upsamples):
            layers += [
                nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        return nn.Sequential(*layers)

    def forward(self, attr_delta, attr_idx, is_rm=None, latent=None):
        """Returns {out_res: (B, C, out_res, out_res)} -- a one-entry dict, so
        callers handle either architecture through the same interface. The
        generator accepts a dict at any number of resolutions, including one."""
        B = attr_delta.size(0)
        device, dtype = attr_delta.device, attr_delta.dtype
        trunk_in = attr_delta
        if self.latent_cond:
            if latent is None:
                raise ValueError(
                    'LegacySingleResControlEncoder(latent_cond=True) needs the source '
                    'latent; pass latent=<W+ tensor>.')
            w = latent.mean(dim=1).to(device=device, dtype=dtype)
            trunk_in = torch.cat([attr_delta, self.latent_proj(w)], dim=1)
        hidden = self.fc(trunk_in)
        seed = self.seed_proj(hidden).view(B, self.out_channels, self.seed_res, self.seed_res)

        out = torch.zeros(B, self.out_channels, self.out_res, self.out_res,
                          device=device, dtype=dtype)
        attr_idx = attr_idx.view(-1).long()
        slot_idx = self.slot_index(attr_idx, is_rm)
        for s in range(self.num_slots):
            mask = slot_idx == s
            if mask.any():
                out[mask] = self.attr_heads[s](seed[mask])

        out = F.normalize(out.reshape(B, -1), dim=1).view_as(out)
        gain = F.softplus(self.log_gain)[slot_idx].to(device=device, dtype=dtype)
        return {self.out_res: out * gain.view(-1, 1, 1, 1)}


class AttributeControlEncoder(_PerDirectionSlots, nn.Module):
    """Predicts additive feature-map corrections at one or more StyleGAN2
    resolutions, from the attribute being edited and (optionally) the source
    latent.

    Magnitude is decoupled from direction on purpose. The network only
    decides the SHAPE of the injected signal; its output is L2-normalized
    per sample per resolution and then scaled by a learnable gain. Letting
    the conv stack set the magnitude directly turned out to be unusable in
    both directions: at the shared base LR the zero-initialized stack never
    grew past ~0.5 against a 512x64x64 feature map whose own norm measures
    ~2931 (a 0.017% perturbation -- --disable_controlnet reproduced identical
    accuracy, ID, LPIPS and leakage, so the module was inert), and at 20x LR
    with no penalty it diverged to ~6e7 within 70 steps. Stacked 512-channel
    ConvTranspose2d layers compound weight growth multiplicatively, so the
    output norm responds to LR far too sharply to tune by trial.

    With this split, control_skip_norm in the training log IS the gain:
    directly observable, settable via --controlnet_init_gain, and bounded by
    --controlnet_max_norm. The gain stays learnable so the model can still
    choose how much to lean on each band.

    per_direction (default off, opt-in via --controlnet_per_direction) gives
    add and rm their own output convs and their own gains, instead of one
    shared set handling both directions. Sharing forces the SAME weights to
    learn two very different tasks at once -- e.g. eyeglasses add must
    synthesize frame structure from nothing (source has no glasses pixels at
    all) while rm only has to erase an existing one, an easier task. A
    128x128 run (v41) showed exactly the predicted failure mode:
    judge_celeb_acc/attr_15_add climbed slowly from ~0 while
    judge_celeb_acc/attr_15_rm drifted down over the same steps -- the harder
    add task pulling shared capacity away from rm. Splitting removes the
    shared capacity these two are competing for.

    region_cond (default off, opt-in via --controlnet_region_cond) feeds a
    per-pixel region prior (e.g. a BiSeNet skin mask for age) into the
    upsampling ladder as an extra input channel at every stage, concatenated
    before each ConvTranspose2d. This is the DC-ControlNet-style fix for the
    gap region_gate_concentration_loss (common/region_stat_loss.py) only
    approximates after the fact: without it, the decoder never SEES where a
    region is, so it either paints uniformly (a global attribute's gate stays
    at its wide-open init, per that loss's own docstring) or has to be told
    after the fact where to gate. With region_cond, the network is shown the
    region as it decodes, the way the original ControlNet paper's own
    conditions (edges, depth, pose) are all spatially-dense maps fed directly
    into the network -- not something the network has to be supervised into
    discovering. The new input channel's weight is zero-initialized, so at
    step 0 it contributes nothing and the module is otherwise identical to
    region_cond=False; from there it must LEARN to read the channel.
    Changes every stage's input channel count, so a checkpoint saved with
    region_cond=False cannot --resume_dir into a run with this flag set (same
    class of incompatibility as --controlnet_per_direction changing the
    per-slot layout -- see that flag's help).
    """

    def __init__(self, num_attrs, out_channels=None, out_res=(64,), seed_res=4,
                 hidden_dim=256, init_gain=1.0, per_direction=False,
                 latent_cond=False, latent_dim=512, channel_multiplier=2,
                 region_cond=False):
        """
        out_res: resolution(s) to inject at. An int is accepted for the
            single-resolution behaviour this module started with; a list or
            tuple turns on multi-resolution injection. Each entry must be a
            power-of-two multiple of seed_res and a resolution the generator
            actually produces.

        out_channels: legacy override for the SINGLE-resolution case only.
            With several resolutions the channel width is not a free choice
            -- it has to equal the generator's own width at each resolution
            (512 at 64, 256 at 128, 128 at 256 for channel_multiplier=2) or
            the addition inside Generator.forward() will not broadcast.

        latent_cond: condition the injected feature maps on the SOURCE LATENT
            (this face) in addition to attr_delta.

            WHY THIS EXISTS: without it, forward() sees only attr_delta -- a
            (B, num_attrs) vector that, for a given attribute and direction, is
            essentially the SAME value for every sample in the batch. The seed,
            the decoder, and therefore the entire injected feature map are then
            identical for every face: one fixed, face-agnostic spatial pattern
            added to a generator feature map whose content differs for every
            identity, pose and framing.

            For a local, position-critical structure like eyeglasses that is
            close to the worst case: the "glasses" perturbation lands at fixed
            spatial coordinates rather than on THIS face's eyes. It can still
            produce enough glasses-like texture for a detector to fire (which
            is how a parser/CLIP judge can score it as present) while never
            forming a correctly placed, well-shaped frame -- exactly the
            "score is high but the glasses are not properly rendered" symptom.
            The real ControlNet is conditioned on a spatial input (edges, pose,
            depth) for precisely this reason; this module had no spatial
            conditioning at all.

            With latent_cond=True the shared trunk additionally consumes the
            per-sample W+ latent (mean-pooled over layers), so the injected
            maps can be placed and shaped per face. Off by default because it
            adds parameters: a checkpoint trained without it cannot be loaded
            into a model built with it (and vice versa). Turn it on for new
            runs.
        """
        super().__init__()
        self.num_attrs = int(num_attrs)
        self.seed_res = int(seed_res)
        self.per_direction = bool(per_direction)
        self.latent_cond = bool(latent_cond)
        self.latent_dim = int(latent_dim)
        self.region_cond = bool(region_cond)
        self.num_slots = self.num_attrs * 2 if self.per_direction else self.num_attrs

        if isinstance(out_res, int):
            out_res = [out_res]
        self.out_res = sorted(int(r) for r in out_res)
        if not self.out_res:
            raise ValueError('out_res must name at least one resolution')

        chan = stylegan2_channels(channel_multiplier)
        for r in self.out_res:
            if r not in chan:
                raise ValueError(f'resolution {r} is not a StyleGAN2 resolution ({sorted(chan)})')
            ratio = r // self.seed_res
            if r <= self.seed_res or r % self.seed_res != 0 or ratio & (ratio - 1) != 0:
                raise ValueError(
                    f'out_res {r} must be a power-of-two multiple of seed_res '
                    f'{self.seed_res}, strictly larger than it. (Generator.forward() only '
                    f'exposes feature maps from 8x8 up, so seed_res itself is never a valid '
                    f'injection point.)')
        if out_channels is not None and len(self.out_res) == 1:
            # Legacy single-resolution call sites pass the width explicitly.
            chan = dict(chan)
            chan[self.out_res[0]] = int(out_channels)
        self.res_channels = {r: chan[r] for r in self.out_res}
        self.max_res = self.out_res[-1]

        # ── shared trunk: (attr_delta [, latent]) -> a small spatial seed ──
        # The latent goes through its own LayerNorm + projection before being
        # concatenated: raw StyleGAN W+ coordinates are far larger in scale
        # than attr_delta (which lives in [-1, 1]), so concatenating them
        # directly would let the latent dominate the trunk's first layer and
        # effectively drown out the attribute signal -- the module would
        # condition on "which face" while barely reacting to "which edit".
        # Normalizing puts the two inputs on a comparable footing.
        if self.latent_cond:
            latent_feat_dim = min(hidden_dim, self.latent_dim)
            self.latent_proj = nn.Sequential(
                nn.LayerNorm(self.latent_dim),
                nn.Linear(self.latent_dim, latent_feat_dim),
                nn.ReLU(inplace=True),
            )
        else:
            latent_feat_dim = 0
            self.latent_proj = None
        trunk_in = self.num_attrs + latent_feat_dim
        self.fc = nn.Sequential(
            nn.Linear(trunk_in, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.seed_channels = chan[self.seed_res]
        self.seed_proj = nn.Linear(
            hidden_dim, self.seed_channels * self.seed_res * self.seed_res)

        # ── shared upsampling ladder, seed_res -> max_res ──────────────────
        # One stage per doubling, each narrowing to the generator's own width
        # at that resolution so the tap points below need no extra projection.
        # With region_cond, each stage takes one EXTRA input channel: the
        # region prior, resized to that stage's input resolution and
        # concatenated in forward() before the ConvTranspose2d runs -- so the
        # decoder sees WHERE the region is at every step of decoding, not
        # only as a post-hoc penalty on its output (see class docstring).
        self.stage_res = []
        stages = []
        r = self.seed_res
        c_in = self.seed_channels
        while r < self.max_res:
            c_out = chan[r * 2]
            stage_in = c_in + (1 if self.region_cond else 0)
            stages.append(nn.Sequential(
                nn.ConvTranspose2d(stage_in, c_out, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            r *= 2
            c_in = c_out
            self.stage_res.append(r)
        self.stages = nn.ModuleList(stages)
        if self.region_cond:
            # Zero-init the region channel's weight slice: at step 0 this
            # input contributes nothing and the stack matches region_cond's
            # behavior with an all-zero region map, giving training a stable
            # starting point to learn to read the channel FROM, rather than
            # starting with a random-weighted extra input fighting the rest
            # of the stack from step 0 (same convention as attr_lora_B and
            # mag_w_cond in direction_bank.py).
            with torch.no_grad():
                for stage in self.stages:
                    # ConvTranspose2d.weight is (in_channels, out_channels,
                    # kH, kW) -- the OPPOSITE layout from Conv2d -- so the
                    # newly-appended input channel (index stage_in - 1) is
                    # sliced on dim 0, not dim 1.
                    stage[0].weight[-1:, :, :, :].zero_()

        # ── per-slot output conv at each injected resolution ───────────────
        # Slot layout is attr-major when per_direction: slot 2a is attribute
        # a's add direction, 2a+1 its rm direction (same convention as
        # direction_bank.py / JudgePeakDeclineBalancer / LearnableAttributeScales).
        # NOT zero-initialized: the output is L2-normalized in forward(), so a
        # zero conv would divide by zero. Magnitude safety comes from the gain
        # (and --controlnet_max_norm) instead.
        #
        # +1 output channel: a per-pixel spatial gate logit, sigmoid'd and
        # multiplied into the content BEFORE the L2 normalize in _tap(). Without
        # it, injection has no mechanism telling it WHERE the face/eyes/glasses
        # are -- the seed is a spatial pattern grown purely from a pooled
        # attribute+latent vector, so nothing rewards concentrating energy in
        # one region over spreading it across the whole map, and for local
        # attributes (currently eyeglasses, see LOCAL_REGION_CLASSES in
        # train_sdflow.py) that fights --local_region_loss_weight: it penalizes
        # injected content landing outside the allowed region, but the encoder
        # had no cheap way to comply short of shrinking its output everywhere,
        # spending its whole magnitude budget to satisfy a spatial constraint.
        # The gate gives it a direct lever -- local_region_loss's existing
        # gradient now teaches THIS gate where to close, instead of fighting
        # through the whole conv stack. No new loss needed: local_region_loss
        # already supplies the signal, this just gives it something cheap to
        # act on. Bias-initialized wide open (see below) so a global attribute
        # (Male/Young), which local_region_loss never touches, has no gradient
        # pushing its gate anywhere and stays effectively full-coverage.
        self.heads = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(self.res_channels[r], self.res_channels[r] + 1, kernel_size=3, padding=1)
                for r in self.out_res
            ])
            for _ in range(self.num_slots)
        ])
        with torch.no_grad():
            for _slot_heads in self.heads:
                for _head in _slot_heads:
                    _head.bias[-1] = 4.0          # sigmoid(4) ~= 0.982: starts near fully-open
                    _head.weight[-1].mul_(0.01)   # low input-sensitivity at step 0

        # One learnable magnitude per (slot, resolution), through softplus so
        # it stays positive. Stored inverted from init_gain so that at step 0
        # each injected band has exactly norm == init_gain.
        init_gain = max(float(init_gain), 1e-4)
        raw = math.log(math.expm1(init_gain))
        self.log_gain = nn.Parameter(torch.full((self.num_slots, len(self.out_res)), raw))

    def forward(self, attr_delta, attr_idx, is_rm=None, latent=None, region_mask=None):
        """
        attr_delta: (B, num_attrs) -- same tensor passed to AttributeDirectionBank.
        attr_idx:   (B,) long -- which attribute is being edited, per sample.
        is_rm:      (B,) bool -- True where the source already HAS the
                    attribute (this edit removes it). Required when
                    per_direction=True, same convention as
                    AttributeDirectionBank/JudgePeakDeclineBalancer/
                    LearnableAttributeScales; ignored otherwise.
        latent:     (B, num_layers, latent_dim) source W+ latent. REQUIRED when
                    latent_cond=True -- it is what makes the injected maps
                    depend on THIS face rather than being one fixed pattern
                    shared by every sample (see __init__ docstring). Ignored
                    when latent_cond=False.
        region_mask: (B, 1, H, W) in [0, 1], any spatial size -- REQUIRED when
                    region_cond=True (see class docstring), resized to each
                    stage's resolution before being concatenated as an extra
                    input channel. A caller with nothing meaningful to
                    condition on for a given sample (no defined region for
                    that attribute) should pass an all-ones mask for it, not
                    omit the tensor -- region_cond changes the conv shape
                    unconditionally, so every sample needs a value here once
                    it is on. Ignored when region_cond=False.

        Returns: {resolution: (B, C_res, res, res)}, to pass as StyleGAN2
                 Generator's `skips=` argument.
        """
        B = attr_delta.size(0)
        device, dtype = attr_delta.device, attr_delta.dtype
        trunk_in = attr_delta
        if self.latent_cond:
            if latent is None:
                raise ValueError(
                    'AttributeControlEncoder(latent_cond=True) needs the source latent; '
                    'pass latent=<W+ tensor>.')
            w = latent.mean(dim=1).to(device=device, dtype=dtype)   # (B, latent_dim)
            trunk_in = torch.cat([attr_delta, self.latent_proj(w)], dim=1)
        hidden = self.fc(trunk_in)
        feat = self.seed_proj(hidden).view(
            B, self.seed_channels, self.seed_res, self.seed_res)

        attr_idx = attr_idx.view(-1).long()
        slot_idx = self.slot_index(attr_idx, is_rm)
        gains = F.softplus(self.log_gain).to(device=device, dtype=dtype)   # (slots, R)

        # Climb the shared ladder once for the whole batch, tapping off a
        # per-slot output conv wherever the current resolution is one we inject at.
        # last_gate_mean: {res: scalar} mean spatial-gate value this forward call,
        # logging only (see heads' +1-channel comment in __init__) -- read this
        # after a forward pass to watch a local attribute's gate close in on its
        # allowed region over training, and confirm a global attribute's gate
        # stays near its ~0.98 init.
        #
        # last_gate: {res: (B,1,res,res)} the same gate WITH gradient intact
        # (last_gate_mean is detached, for cheap logging only). This is what
        # a caller needs to put a spatial loss directly on the gate itself --
        # see common/region_stat_loss.py's region_gate_concentration_loss,
        # written for exactly this: local attributes get gate guidance for
        # free through --local_region_loss_weight's pixel-difference penalty,
        # but nothing analogous exists for a global attribute (age), so its
        # gate has no gradient pushing it anywhere and stays at its
        # full-coverage init indefinitely (see this class's own docstring).
        # Overwritten every forward() call; safe to read only within the same
        # training step that produced it, same lifetime as last_gate_mean.
        self.last_gate_mean = {}
        self.last_gate = {}
        if self.region_cond and region_mask is None:
            raise ValueError(
                'AttributeControlEncoder(region_cond=True) needs region_mask; '
                'pass region_mask=<(B,1,H,W) tensor in [0,1]> (all-ones where no '
                'region is defined for a sample\'s attribute).')
        wanted = set(self.out_res)
        skips = {}
        for stage, res in zip(self.stages, self.stage_res):
            if self.region_cond:
                # Resized to feat's CURRENT (pre-stage) resolution -- this
                # stage doubles it, so the concatenated map matches feat's
                # spatial size going in, not the resolution it produces.
                r_mask = F.interpolate(region_mask.to(device=device, dtype=dtype),
                                       feat.shape[-2:], mode='bilinear', align_corners=False)
                feat = torch.cat([feat, r_mask], dim=1)
            feat = stage(feat)
            if res in wanted:
                skips[res] = self._tap(feat, res, slot_idx, gains, B, device, dtype)
        return skips

    def _tap(self, feat, res, slot_idx, gains, B, device, dtype):
        """Per-slot output conv at one resolution: split into content + a
        spatial gate logit, sigmoid the gate and apply it to the content BEFORE
        the L2 normalize, then scale by that (slot, resolution)'s learnable
        gain. Gating before normalize is what lets the gate actually
        concentrate the fixed energy budget into the gated region instead of
        just attenuating the whole map uniformly (which normalize would undo)."""
        r_i = self.out_res.index(res)
        C = self.res_channels[res]
        out = torch.zeros(B, C, res, res, device=device, dtype=dtype)
        gate_logit = torch.zeros(B, 1, res, res, device=device, dtype=dtype)
        for s in range(self.num_slots):
            mask = slot_idx == s
            if mask.any():
                raw = self.heads[s][r_i](feat[mask])       # (n, C+1, res, res)
                out[mask] = raw[:, :C]
                gate_logit[mask] = raw[:, C:]
        gate = torch.sigmoid(gate_logit)                     # (B, 1, res, res)
        with torch.no_grad():
            self.last_gate_mean[res] = gate.mean().detach()
        self.last_gate[res] = gate
        out = out * gate
        out = F.normalize(out.reshape(B, -1), dim=1).view_as(out)
        return out * gains[slot_idx, r_i].view(-1, 1, 1, 1)

    def region_channel_weight_norm(self):
        """{stage_res: scalar} L2 norm of each stage's region-channel weight
        slice (the one zero-initialized in __init__ when region_cond=True).

        The only direct signal that the network is actually LEARNING to read
        the region input: region_cond changes the network's INPUT, not its
        supervised output the way region_gate_concentration_loss's gate does,
        so there is nothing else in the existing logs (last_gate_mean
        included) that moves in response to this specific mechanism. Starts
        at exactly 0.0 every stage by construction; a run where these stay at
        0.0 for many thousands of steps means the region channel is being
        ignored, not just slow to help -- gradient is reaching it (it is not
        detached), so a truly stuck 0.0 points at the region signal being
        too weak or too uncorrelated with the losses actually driving
        training to be worth the network's attention, not at a wiring bug.
        Returns {} when region_cond=False (nothing to report).
        """
        if not self.region_cond:
            return {}
        with torch.no_grad():
            return {
                res: stage[0].weight[-1:, :, :, :].norm().item()
                for stage, res in zip(self.stages, self.stage_res)
            }
