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
"""
import torch


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
