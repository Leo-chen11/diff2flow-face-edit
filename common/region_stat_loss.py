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
