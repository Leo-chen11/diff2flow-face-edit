"""Direction-symmetric region-statistic auxiliary loss.

Generalizes what was a one-off, removal-direction-only computation inline
in training/train_sdflow.py (--hair_gray_loss_weight): it pushed the hair
region's saturation DOWN for age(39) REMOVAL edits only (src young -> edit
old), grounded in a dump_attr_failures.py audit that found Brown_Hair the
strongest failure correlate on that direction. It had no mirror for the
ADD direction (src old -> edit young). A later failure-vs-success audit on
THAT direction found the same shape of problem, just flipped: Gray_Hair
present in the SOURCE is the single strongest correlate of SUCCESS
(success=0.25 vs fail=0.04) -- the model's only reliable lever for "look
younger" is a hair-color change it was never asked to make on this
direction, only stumbled into as a side effect of general skin smoothing.
Sources without that lever (already dark hair, or feminine-coded features
per the same audit) have nothing to exploit and stay stuck below the
success threshold.

directional_region_saturation_loss() is that mechanism written once,
parameterized by direction instead of hardcoded to one. push=-1
reproduces the original hair_gray_loss formula exactly; push=+1 is its
mirror for the add direction. Both are hinge losses (only penalize the
wrong side), so a sample that already satisfies the target costs nothing.

region_gate_concentration_loss() addresses a different, later-discovered
gap: models/control_encoder.py's AttributeControlEncoder already predicts
a per-pixel spatial gate for its injected feature-map content, but that
gate only ever gets a training signal for LOCAL attributes, through
--local_region_loss_weight's pixel-difference penalty (see that flag and
LOCAL_REGION_CLASSES in train_sdflow.py). AttributeControlEncoder's own
docstring says as much: the gate is "bias-initialized wide open ... so a
global attribute (Male/Young), which local_region_loss never touches, has
no gradient pushing its gate anywhere and stays effectively
full-coverage." A gate with no spatial preference is exactly the failure
mode the published ControlNet literature documents for this class of
mechanism -- conditioning built for spatially-dense, pixel-aligned inputs
(edges, depth, pose) generalizes poorly to a diffuse, non-spatial
condition like "is this face old", because nothing in the architecture
tells it WHERE in the frame that condition should manifest (DC-ControlNet,
ICCV 2025, addresses the same gap for multi-element scene composition by
decoupling a global condition into per-element, region-aware control).
This loss is the minimal version of that fix for age: reward the gate for
concentrating inside a texture-relevant face region (skin, where wrinkles/
skin-texture aging actually lives) instead of leaving it a flat, spatially
uninformed multiplier.

directional_region_frequency_loss() closes the gap that the gate loss
above, by itself, turned out not to close. Conditioning the injection on
the skin region made the edit more localised -- and the model spent that
localisation on smoothing the skin more precisely, not on adding texture
to it (measured: skin HF energy fell further than in the runs without it).
The same thing happened when the fine-layer reg floor was lowered and when
the DDS fine-layer block was lifted. Three different mechanisms, three
times the same outcome, because every existing loss in this project is
blind to fine texture and "smoother, darker, higher contrast" is the
cheapest way to satisfy all of them at once. This term measures the one
quantity none of the others can see and is one-sided like the rest, so a
sample that already keeps its texture pays nothing.
"""
import torch
import torch.nn.functional as F


def region_mean_saturation(img_pm1, mask):
    """Mask-weighted mean HSV-style saturation over a batch.

    img_pm1: (N, 3, H, W) in [-1, 1] (generator output range).
    mask:    (N, 1, H, W) or broadcastable, typically a FaceParser region
             mask (e.g. FaceParser.get_region_mask output).
    Returns a single scalar tensor -- the mean over the whole batch, not
    per-sample, matching what the original inline version computed.
    """
    img01 = img_pm1 * 0.5 + 0.5
    mx = img01.max(dim=1, keepdim=True).values
    mn = img01.min(dim=1, keepdim=True).values
    sat = (mx - mn) / mx.clamp(min=1e-4)
    return (sat * mask).sum() / mask.sum().clamp(min=1e-4)


def directional_region_saturation_loss(src_img, edit_img, src_mask, edit_mask,
                                       push, relative_ratio, bound):
    """One-sided hinge loss on a region's saturation, relative to its OWN
    source value -- works uniformly regardless of the source's actual hair
    color instead of assuming one fixed numerical definition of "old" or
    "young" hair.

    push=-1 (removal / aging direction): penalize the edited region for
        staying MORE saturated than `relative_ratio` of the source (capped
        ABOVE by `bound`) -- moves toward gray. Identical to the original
        --hair_gray_loss_weight formula with bound=--hair_gray_abs_cap.
    push=+1 (add / de-aging direction): penalize the edited region for
        being LESS saturated than the source raised `relative_ratio` of the
        way toward full saturation (floored BELOW by `bound`) -- moves
        toward natural color.

    src_sat is detached (it is the fixed reference point the edit is
    scored against, not something this loss should push on).
    """
    with torch.no_grad():
        src_sat = region_mean_saturation(src_img, src_mask)
    edit_sat = region_mean_saturation(edit_img, edit_mask)
    if push < 0:
        target = (src_sat * relative_ratio).clamp(max=bound)
        return torch.relu(edit_sat - target)
    target = (src_sat + (1.0 - src_sat) * relative_ratio).clamp(min=bound)
    return torch.relu(target - edit_sat)


def region_area_fraction(prob):
    """Mask-weighted area of a region as a fraction of the frame.

    prob MUST come from a differentiable source -- FaceParser.region_prob,
    not FaceParser.get_region_mask. The latter is @torch.no_grad() and
    argmax'd, so a loss on the value returned here would have exactly zero
    gradient and would silently do nothing.
    """
    return prob.sum() / prob.numel()


def directional_region_area_loss(src_prob, edit_prob, push, relative_change, bound):
    """One-sided hinge on how much of the frame a region covers, relative to
    the source's own coverage.

    push=+1: the edited region must cover at least (1 + relative_change) x
        the source's area, capped above by `bound`.
    push=-1: at most (1 - relative_change) x, floored below by `bound`.

    The target is MULTIPLICATIVE here, unlike
    directional_region_saturation_loss's "move a fraction of the way toward
    the extreme". Saturation is a [0,1] quantity where 1.0 is a meaningful
    ceiling, so moving partway toward it makes sense. An area fraction is
    also in [0,1] but a face's hair only ever covers roughly 0.05-0.30 of the
    frame -- moving "halfway to 1.0" would demand hair over half the image.
    Scaling the source's own coverage keeps the ask proportionate to how much
    hair the person started with, and `bound` stops it running away.
    """
    with torch.no_grad():
        src_a = region_area_fraction(src_prob)
    edit_a = region_area_fraction(edit_prob)
    if push > 0:
        target = (src_a * (1.0 + relative_change)).clamp(max=bound)
        return torch.relu(target - edit_a)
    target = (src_a * (1.0 - relative_change)).clamp(min=bound)
    return torch.relu(edit_a - target)


def _gaussian_blur(x, sigma):
    """Separable Gaussian blur, DIFFERENTIABLE.

    Deliberately the same kernel construction as
    scripts/probe_noise_texture.py's _gaussian_blur, so the quantity this
    file's loss optimises and the quantity that probe reports are the same
    number, not two similar-looking definitions that drift apart. The one
    difference is that this one is not wrapped in @torch.no_grad() -- a
    blur inside a loss has to carry gradient back to the image.
    """
    ks = int(sigma * 6) | 1                      # odd kernel
    coords = torch.arange(ks, device=x.device, dtype=x.dtype) - ks // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, 1, -1)
    c = x.shape[1]
    x = F.conv2d(x, g.expand(c, 1, 1, ks), padding=(0, ks // 2), groups=c)
    x = F.conv2d(x, g.transpose(-1, -2).expand(c, 1, ks, 1), padding=(ks // 2, 0), groups=c)
    return x


def region_hf_energy(img_pm1, mask, sigma=2.0):
    """Mask-weighted mean squared high-frequency residual.

    img - blur(img) keeps exactly what a Gaussian removes: pores, creases,
    fine shading. Restricted to `mask` (in practice BiSeNet skin) because
    hair carries far more high-frequency energy than any wrinkle and would
    swamp the measurement.

    Matches scripts/probe_noise_texture.py's skin_hf_energy formula, including
    its denominator convention (mask pixel count x channel count), so a value
    logged from training is directly comparable to one the probe prints.

    Returns a single scalar over the whole batch, matching
    region_mean_saturation's convention above.
    """
    hf = img_pm1 - _gaussian_blur(img_pm1, sigma)
    denom = (mask.sum() * img_pm1.shape[1]).clamp(min=1.0)
    return ((hf ** 2) * mask).sum() / denom


def directional_region_frequency_loss(src_img, edit_img, src_mask, edit_mask,
                                      push, relative_change, bound, sigma=2.0):
    """One-sided hinge on a region's HIGH-FREQUENCY energy, relative to its
    own source value.

    WHY THIS EXISTS. Every loss in this project measures the edit through
    something that is blind to fine texture: the attribute teachers are
    classifiers reading mostly low-frequency evidence (a 256px r34/CelebA
    head is happy to call a darkened, higher-contrast face "old"), id_loss
    reads an ArcFace embedding that is trained to be texture-invariant on
    purpose, reg_loss counts W+ displacement, DDS scores a latent-space
    noise residual. None of them can tell a face that grew wrinkles from a
    face that was darkened and SMOOTHED. Three separate mechanism changes
    (lowering the fine reg floor, unblocking DDS above layer 12, and
    conditioning ControlNet on the skin region) were each measured, and
    each one was spent on MORE smoothing rather than more texture --
    scripts/probe_noise_texture.py reported skin HF energy going the wrong
    way in all three. That is not three coincidences; it is what happens
    when nothing in the objective pays for texture. This is the term that
    pays for it.

    push=+1 (aging direction, age REMOVAL edits): the edited skin must carry
        at least (1 + relative_change) x the source's own HF energy, capped
        ABOVE by `bound`.
    push=-1 (de-aging direction, age ADD edits): at most
        (1 - relative_change) x, floored BELOW by `bound`.

    The target is MULTIPLICATIVE, following directional_region_area_loss
    rather than directional_region_saturation_loss: HF energy is a small
    positive quantity with no meaningful ceiling (~7e-4 on a typical FFHQ
    reconstruction at 512), so "move a fraction of the way toward 1.0" would
    be meaningless, while scaling the source's own measured value keeps the
    ask proportionate to how much texture the person started with. `bound`
    stops it running away on either side.

    src energy is detached: it is the fixed reference the edit is scored
    against, the same role src_sat/src_a play above.

    TWO KNOWN LIMITS, stated here rather than discovered later:

    1. This is NOT the removed --color_shift_loss_weight (see its note in
       training/train_sdflow.py) repeated. That one failed structurally: it
       averaged SIGNED mean RGB over the whole face, so a local change
       cancelled to ~0 and the loss could never see it. Here the residual is
       SQUARED before averaging, so every pixel's contribution is
       non-negative and local texture cannot cancel against other local
       texture.
    2. Mean energy is location-agnostic. It says how much fine detail the
       skin carries, not whether that detail is arranged as plausible
       wrinkles -- uniform grain over the whole cheek satisfies it as well
       as a nasolabial fold does. The discriminator (--disc_realism_weight)
       and the diffusion prior (--age_diffusion_weight) are what push
       against added grain that does not look like a photograph; this term
       is only the one that stops texture being DESTROYED. Any run using it
       has to be checked visually, not just by the number going up.
    """
    with torch.no_grad():
        src_e = region_hf_energy(src_img, src_mask, sigma)
    edit_e = region_hf_energy(edit_img, edit_mask, sigma)
    if push > 0:
        target = (src_e * (1.0 + relative_change)).clamp(max=bound)
        return torch.relu(target - edit_e)
    target = (src_e * (1.0 - relative_change)).clamp(min=bound)
    return torch.relu(edit_e - target)


def region_gate_concentration_loss(gate, region_mask, margin=0.15):
    """One-sided hinge pulling a ControlNet-style spatial gate to concentrate
    inside `region_mask`, instead of leaving it spatially flat.

    gate: (B,1,H,W) in [0,1] -- AttributeControlEncoder._tap's per-pixel gate
        (see models/control_encoder.py's AttributeControlEncoder.last_gate),
        taken WITH gradient, not the detached last_gate_mean used for logging.
    region_mask: (B,1,h,w) in [0,1], any spatial size -- resized to match
        `gate` if needed. Pass a FaceParser.get_region_mask() (no_grad,
        argmax'd) output, not region_prob(): the region here is a fixed
        reference the gate is scored against, the same role src_sat/src_a
        play above, not the quantity being optimised.

    Hinge on the GAP between the gate's mean value inside the region and its
    mean value outside: pays nothing once inside already exceeds outside by
    `margin`, so a gate that already concentrates in the right place is not
    pushed to full binary open/closed -- it only has to prefer the region,
    not consume it. Mirrors directional_region_saturation_loss/
    directional_region_area_loss's convention of a one-sided penalty that a
    satisfying sample pays nothing for.
    """
    if region_mask.shape[-2:] != gate.shape[-2:]:
        region_mask = F.interpolate(region_mask, gate.shape[-2:],
                                    mode='bilinear', align_corners=False)
    inside = (gate * region_mask).sum(dim=(1, 2, 3)) \
        / region_mask.sum(dim=(1, 2, 3)).clamp(min=1e-6)
    outside_mask = 1.0 - region_mask
    outside = (gate * outside_mask).sum(dim=(1, 2, 3)) \
        / outside_mask.sum(dim=(1, 2, 3)).clamp(min=1e-6)
    return torch.relu(margin - (inside - outside)).mean()
