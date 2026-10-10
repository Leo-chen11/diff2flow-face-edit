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

outside_region_preservation_loss() holds the pixels outside a region to the
source reconstruction (the eyeglasses locality loss).
"""
import torch


def region_mean_saturation(img_pm1, mask):
    """Mask-weighted mean HSV-style saturation, per sample.

    img_pm1: (N, 3, H, W) in [-1, 1] (generator output range).
    mask:    (N, 1, H, W), typically a FaceParser region mask (e.g.
             FaceParser.get_region_mask output).
    Returns (N,): each sample's own mean (it used to be one value pooled over
    the batch, so a sample's target depended on the others in it).
    """
    img01 = img_pm1 * 0.5 + 0.5
    mx = img01.max(dim=1, keepdim=True).values
    mn = img01.min(dim=1, keepdim=True).values
    sat = (mx - mn) / mx.clamp(min=1e-4)
    return (sat * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp(min=1e-4)


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
    scored against, not something this loss should push on). The hinge is
    taken per sample against that sample's own source, then averaged.
    """
    with torch.no_grad():
        src_sat = region_mean_saturation(src_img, src_mask)
    edit_sat = region_mean_saturation(edit_img, edit_mask)
    if push < 0:
        target = (src_sat * relative_ratio).clamp(max=bound)
        return torch.relu(edit_sat - target).mean()
    target = (src_sat + (1.0 - src_sat) * relative_ratio).clamp(min=bound)
    return torch.relu(target - edit_sat).mean()


def region_area_fraction(prob):
    """Area of a region as a fraction of the frame, per sample: (N,).

    prob MUST come from a differentiable source -- FaceParser.region_prob,
    not FaceParser.get_region_mask. The latter is @torch.no_grad() and
    argmax'd, so a loss on the value returned here would have exactly zero
    gradient and would silently do nothing.
    """
    return prob.flatten(1).mean(1)


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
        return torch.relu(target - edit_a).mean()
    target = (src_a * (1.0 - relative_change)).clamp(min=bound)
    return torch.relu(edit_a - target).mean()


def outside_region_preservation_loss(edit_img, ref_img, allowed):
    """Mean squared pixel change OUTSIDE `allowed`, normalized by the outside
    area -- the same formula training/train_sdflow.py's local_region_loss
    has always used for eyeglasses, factored out so it can be applied to
    attributes that have no tight region of their own.

    WHY THIS EXISTS. DC-ControlNet, the method --controlnet_region_cond
    adapts, is a GENERATION method: it composes an image from layout and
    content conditions and has no notion of "leave everything else as it
    was", because generation never needs one. Editing does. The adaptation
    brought over the input side (WHERE the network may act, via the region
    channel) but not an output side (WHERE it must NOT act), and a v26->v27
    comparison inside one run isolates exactly that gap: Eyeglasses, the
    only attribute with an output-side constraint (local_region_loss),
    stayed flat (LPIPS 0.116->0.128, ID_ind 0.808->0.807), while Male and
    Young, sharing the same trunk, the same all-ones/BiSeNet region channel
    and the same schedule, degraded sharply (Male LeakCLIP 0.038->0.133,
    Young LPIPS 0.199->0.319) -- and Young did so despite HAVING a real
    input-side region (BiSeNet skin), so input-side region information does
    not substitute for an output-side constraint.

    For a global attribute there is no tight region to protect, but there
    is still a well-defined one to leave alone: everything that is not the
    face or hair (background, clothing, hat).

    allowed: (B,1,H,W) in [0,1], 1 = may change. ref_img should be the
    source RECONSTRUCTION, not the real photo, for the same inversion-gap
    reason local_region_loss uses it.
    """
    outside = 1.0 - allowed
    diff_sq = (edit_img - ref_img).pow(2)
    return (diff_sq * outside).sum() / (outside.sum() * diff_sq.shape[1]).clamp(min=1e-6)
