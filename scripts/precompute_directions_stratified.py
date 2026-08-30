"""
Direction Bank v2: Stratified Multi-Direction Computation

Computes K=4 semantically meaningful directions per attribute by conditioning
on the other two binary attributes. All three attributes use 2x2 binary
strata (built from AttributeClassifier's thresholded 0/1 predictions).

Glasses  K=4: male x young | male x old | female x young | female x old
Gender   K=4: young x no-glasses | young x glasses | old x no-glasses | old x glasses
Age      K=4: male x no-glasses | male x glasses | female x no-glasses | female x glasses

Within each stratum, the high/low split for the attribute actually being
edited no longer uses a fixed 0.5 decision boundary. It uses the top/bottom
`--extreme_pct` percent of a *continuous* confidence score (see
scripts/extract_continuous_attrs.py), which avoids polluting the direction
with samples the classifier itself is unsure about.

The direction itself defaults to a per-layer LDA / whitened estimate
(Ledoit-Wolf shrinkage covariance) instead of a plain mean-difference vector,
so the direction accounts for the latent space's covariance structure
instead of only comparing group means. The whitened direction is rescaled to
the plain mean-diff's per-layer norm, so `layer_norms` (used to initialize
AttributeDirectionBank.magnitude_net) keeps the same physically meaningful
scale as before -- only the *angle* of the direction changes.

Both upgrades (percentile splits, LDA/whitened direction) apply uniformly to
all three attributes; there is no attribute-specific special-casing.

Two further precision knobs, off by default (reproduce the old behavior
unless passed):
  --strata_margin: the stratification conditioning variables (e.g. gender/
    glasses used to define strata for the age direction) used to accept the
    classifier's binary 0/1 predictions at face value -- a sample the
    classifier was 51% sure of counted the same as one it was 99% sure of
    when deciding which STRATUM computes a direction, even though only the
    TARGET attribute's own split got the confidence-margin treatment
    (--extreme_pct). --strata_margin extends the same idea to the
    conditioning side: a sample must sit at least this far from 0.5 on the
    conditioning attribute's own continuous score to count toward any
    stratum; ambiguous ones are dropped rather than guessed into a bucket.
  --group_center trimmed_mean: replace each high/low group's plain mean
    with a coordinate-wise trimmed mean before differencing, so a handful
    of mislabeled or atypical samples in a small stratum can't single-
    handedly drag the direction toward them.

A third knob, --substyle_k, targets a different problem: even a perfectly
confident, perfectly stratified "has this attribute" group is not visually
homogeneous for attributes that have more than one common APPEARANCE (thin
vs thick vs rimless glasses; closed-mouth vs open-mouth smiling; straight
vs loosely-wavy hair). A single mean-difference direction across the whole
group averages every style together -- for a high-frequency, spatially
precise attribute that washes out the very detail that makes the edit look
real, rather than reinforcing it (validate_direction_bank.py confirms this
concretely: eyeglasses' raw direction is far weaker than gender/age's at
the same alpha). --substyle_k > 1 runs k-means on each stratum's "high"
group BEFORE computing a direction, splitting it into that many visually
coherent sub-clusters, and returns one direction per sub-cluster instead of
one for the whole group -- letting the DATA'S OWN structure supply extra
K, on top of (not instead of) the hand-picked gender/age conditioning
strata, uniformly for every attribute in --attribute_index (not just
eyeglasses). Same spirit as GANSpace/SeFa's "find structure unsupervised
instead of only from labels", applied as a light augmentation of this
file's existing LDA/percentile pipeline rather than a move to a different
latent space (see StyleSpace/StyleCLIP's Global Directions for that
heavier alternative). Off by default (1 = old behavior); --K and --age_k
(when not 1) are auto-multiplied by --substyle_k, see main().
"""

import argparse
import os
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_tensor(path, key_candidates):
    obj = torch.load(path, map_location="cpu")
    if torch.is_tensor(obj):
        return obj.float()
    if isinstance(obj, dict):
        for key in key_candidates:
            if key in obj and torch.is_tensor(obj[key]):
                return obj[key].float()
        tensors = {k: v for k, v in obj.items() if torch.is_tensor(v)}
        if len(tensors) == 1:
            return list(tensors.values())[0].float()
        raise ValueError(f"Ambiguous dict keys: {list(obj.keys())}")
    raise ValueError(f"Unexpected type: {type(obj)}")


def load_latents(path):
    t = _load_tensor(path, ["latents", "latent", "w", "w_plus", "wplus", "values"])
    if t.dim() == 4 and t.shape[1] == 1:
        t = t.squeeze(1)
    if t.dim() != 3 or t.shape[1:] != (18, 512):
        raise ValueError(f"Expected [N,18,512], got {tuple(t.shape)}")
    return t


def load_preds(path):
    t = _load_tensor(path, ["preds", "values", "predictions", "attrs", "labels"])
    if t.dim() != 2:
        raise ValueError(f"Expected [N, num_attrs], got {tuple(t.shape)}")
    return t


def load_paths(path):
    """Return the 'paths' list saved alongside a tensor, or None if absent.

    Used to catch the case where --preds_file and --continuous_preds_file
    were built from different or differently-ordered image lists: matching
    row counts alone would silently pass while actually pairing each row
    with the wrong image.
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "paths" in obj:
        return list(obj["paths"])
    return None


# ---------------------------------------------------------------------------
# Direction computation
# ---------------------------------------------------------------------------

def extreme_masks(scores, pct=20.0, cross_scores=None):
    """Boolean high/low masks from the top/bottom `pct` percent of a
    continuous score.

    Replaces a fixed 0.5 decision-boundary split so the direction is
    computed from confidently-labeled samples instead of ones near the
    classifier's most ambiguous region, where label noise is worst.

    cross_scores: optional independent-judge continuous score (same shape
    as `scores`, e.g. from extract_continuous_attr.py --cross_judge clip).
    When given, a sample only counts as 'high' if the cross-judge ALSO puts
    it above 0.5, and 'low' only if the cross-judge ALSO puts it below --
    i.e. the percentile-extreme selection is additionally cross-validated
    against an architecturally different judge. WHY: `scores` typically
    comes from the same r34 classifier used as the frozen training-time
    teacher, so a sample it is confidently (but wrongly) extreme on would
    otherwise flow straight into the direction geometry with no check --
    the exact "same-classifier blind spot" failure mode this project has
    documented a large gap for on eyeglasses specifically (r34 ~91% vs an
    independent CLIP judge far lower). See --require_cross_judge_agree.
    """
    if scores.numel() == 0:
        empty = torch.zeros_like(scores, dtype=torch.bool)
        return empty, empty
    hi_thresh = torch.quantile(scores, 1.0 - pct / 100.0)
    lo_thresh = torch.quantile(scores, pct / 100.0)
    mask_high = scores >= hi_thresh
    mask_low = scores <= lo_thresh
    if cross_scores is not None:
        mask_high = mask_high & (cross_scores >= 0.5)
        mask_low = mask_low & (cross_scores < 0.5)
    return mask_high, mask_low


def confident_strata_masks(cond_a_cont, cond_b_cont, margin=0.0):
    """2x2 stratification masks from two continuous conditioning scores,
    each required to sit confidently on one side of 0.5.

    margin=0.0 (default) reproduces the old behavior of thresholding each
    conditioning score at 0.5 -- equivalent to the binary predictions the
    strata used to be built from directly. margin>0 additionally REQUIRES
    the conditioning attribute's own score to sit at least `margin` away
    from 0.5 before a sample counts toward a stratum, instead of trusting a
    razor-thin >0.5-vs-<0.5 call.

    WHY THIS MATTERS: only the TARGET attribute's own high/low split used a
    confidence margin (--extreme_pct on its continuous score, see
    extreme_masks). The CONDITIONING attributes -- e.g. gender/age when
    stratifying the glasses direction -- still used a bare 0.5 threshold on
    the same, possibly noisy, classifier output. A sample the classifier is
    51% sure is "male" got treated identically to one it is 99% sure about
    when deciding which STRATUM computes the glasses direction, even though
    that's the exact same label-noise problem --extreme_pct exists to avoid
    on the target side. Ambiguous conditioning samples are simply DROPPED
    (excluded from all four strata) rather than forced into a bucket they
    may not really belong in -- same philosophy as extreme_masks.

    Returns list of 4 (label_suffix, mask) tuples in order:
        (a=hi,b=hi), (a=hi,b=lo), (a=lo,b=hi), (a=lo,b=lo)
    """
    a_hi = cond_a_cont >= 0.5 + margin
    a_lo = cond_a_cont <= 0.5 - margin
    b_hi = cond_b_cont >= 0.5 + margin
    b_lo = cond_b_cont <= 0.5 - margin
    return [
        ("hi_hi", a_hi & b_hi),
        ("hi_lo", a_hi & b_lo),
        ("lo_hi", a_lo & b_hi),
        ("lo_lo", a_lo & b_lo),
    ]


def trimmed_mean(x, trim_frac=0.1, dim=0):
    """Coordinate-wise trimmed mean along `dim`: drop the top/bottom
    `trim_frac` fraction of samples per-coordinate before averaging.

    A plain mean is unboundedly sensitive to a single outlier (one
    mislabeled or atypical sample can drag the whole group centroid, and
    therefore the direction, toward it). Trimming is the standard cheap fix:
    still an average, just no longer letting the most extreme handful of
    samples on EACH coordinate dominate it. Falls back to a plain mean if
    trimming would remove too much of the group.
    """
    n = x.shape[dim]
    k = int(n * trim_frac)
    if k <= 0 or n - 2 * k <= 0:
        return x.mean(dim=dim)
    sorted_x, _ = torch.sort(x, dim=dim)
    return sorted_x.narrow(dim, k, n - 2 * k).mean(dim=dim)


def shrinkage_covariance(X, shrinkage=None):
    """Ledoit-Wolf shrinkage-to-scaled-identity covariance estimate.

    X: (N, D), not necessarily centered. Returns (D, D).

    Vectorized (no per-sample Python loop) using the identity
        sum_i ||x_i x_i^T - S||_F^2 = sum_i ||x_i||^4 - N * ||S||_F^2
    so this stays fast for N in the tens of thousands and D=512.

    Pass an explicit `shrinkage` in [0, 1] to skip the automatic estimate.
    """
    N, D = X.shape
    Xc = X - X.mean(dim=0, keepdim=True)
    # Biased (1/N) normalization, matching the Ledoit-Wolf convention this
    # vectorized shrinkage-intensity identity assumes. The final direction is
    # rescaled to the plain mean-diff's norm afterward regardless, so this
    # only affects the (already heuristic) shrinkage-intensity estimate, not
    # the returned direction's overall magnitude.
    S = (Xc.t() @ Xc) / max(N, 1)

    if shrinkage is None:
        mu = torch.diagonal(S).mean()
        target = mu * torch.eye(D, dtype=X.dtype, device=X.device)
        d2 = (S - target).pow(2).sum()
        norms4 = Xc.pow(2).sum(dim=1).pow(2).sum()
        frob_S2 = (S * S).sum()
        b_bar2 = ((norms4 - N * frob_S2) / (N ** 2)).clamp(min=0.0)
        b2 = torch.minimum(b_bar2, d2)
        shrinkage = (b2 / d2).item() if d2.item() > 1e-12 else 0.0
        shrinkage = float(min(max(shrinkage, 0.0), 1.0))

    mu = torch.diagonal(S).mean()
    target = mu * torch.eye(D, dtype=X.dtype, device=X.device)
    return (1.0 - shrinkage) * S + shrinkage * target


def compute_direction(latents, mask_high, mask_low, min_samples=50,
                       method="lda", shrinkage=None, group_center="mean",
                       group_trim_frac=0.1):
    """Direction between two subsets of latents.

    method="mean_diff": high_mean - low_mean (legacy behaviour).
    method="lda": per-layer Fisher/whitened direction using a Ledoit-Wolf
        shrinkage covariance, rescaled to the plain mean-diff's per-layer
        norm. Whitening accounts for the latent space's covariance
        structure instead of only comparing group means, which is the main
        source of imprecision in a raw mean-diff direction; rescaling keeps
        the physical displacement magnitude (used by layer_norms /
        magnitude_net) unchanged so only the angle is affected.
    group_center="mean" (default) or "trimmed_mean": how each group's
        centroid is computed before differencing. trimmed_mean drops the
        most extreme group_trim_frac of samples per-coordinate first (see
        trimmed_mean()), reducing sensitivity to any single mislabeled or
        atypical sample -- worth trying if a stratum is small enough that
        one outlier can visibly shift its direction.
    """
    n_high = int(mask_high.sum().item())
    n_low = int(mask_low.sum().item())
    if n_high < min_samples or n_low < min_samples:
        return None, n_high, n_low

    high = latents[mask_high]
    low = latents[mask_low]
    if group_center == "trimmed_mean":
        center_high = trimmed_mean(high, trim_frac=group_trim_frac, dim=0)
        center_low = trimmed_mean(low, trim_frac=group_trim_frac, dim=0)
    elif group_center == "mean":
        center_high = high.mean(dim=0)
        center_low = low.mean(dim=0)
    else:
        raise ValueError(f"Unknown group_center: {group_center}")
    mean_diff = center_high - center_low   # (L, D)
    if mean_diff.norm() < 1e-6:
        return None, n_high, n_low

    if method == "mean_diff":
        return mean_diff, n_high, n_low
    if method != "lda":
        raise ValueError(f"Unknown direction method: {method}")

    num_layers, dim = mean_diff.shape
    lda_dir = torch.zeros_like(mean_diff)
    for layer in range(num_layers):
        pooled = torch.cat([
            high[:, layer, :] - high[:, layer, :].mean(dim=0, keepdim=True),
            low[:, layer, :] - low[:, layer, :].mean(dim=0, keepdim=True),
        ], dim=0)
        cov = shrinkage_covariance(pooled, shrinkage=shrinkage)
        # Ridge floor is an ABSOLUTE term (1e-4), not just proportional to
        # cov's own scale. A small/tightly-clustered group (e.g. a substyle_k
        # sub-cluster with few samples) can have a near-zero covariance
        # diagonal, in which case the old purely-proportional ridge
        # (1e-4 * diag.mean(), clamped only at 1e-8) also shrinks toward
        # zero and stops regularizing right when the solve needs it most --
        # torch.linalg.solve on a near-singular matrix can then return
        # enormous or non-finite values that used to propagate straight into
        # the saved bank undetected. Keeping an absolute floor alongside the
        # proportional one bounds the solve's condition number regardless of
        # how small the group's own variance happens to be.
        ridge = torch.clamp(1e-4 * cov.diagonal().mean(), min=1e-4)
        cov = cov + ridge * torch.eye(dim, dtype=cov.dtype, device=cov.device)
        w = torch.linalg.solve(cov, mean_diff[layer].unsqueeze(1)).squeeze(1)
        lda_dir[layer] = w

    lda_norm = lda_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    mean_norm = mean_diff.norm(dim=-1, keepdim=True)
    rescaled = lda_dir / lda_norm * mean_norm

    # Safety net: never let a non-finite (NaN/Inf) direction reach the
    # caller silently. This is what actually produced the eyeglasses
    # ID_ind/Leak == nan and the Male ID-collapse-with-flat-AccCLIP pattern
    # (garbage direction, not a real edit) seen after enabling --substyle_k
    # on an already-small stratum -- an ill-conditioned covariance solve on
    # too few samples slipped through uncaught. Treat it as a failed
    # direction (same signal as too-few-samples) so the normal fallback
    # path handles it instead of corrupting the bank.
    if not torch.isfinite(rescaled).all():
        return None, n_high, n_low
    return rescaled, n_high, n_low


# ---------------------------------------------------------------------------
# Unsupervised sub-style clustering (--substyle_k)
# ---------------------------------------------------------------------------

def kmeans_latents(x, k, n_iter=30, seed=0):
    """Plain Lloyd's-algorithm k-means over flattened per-sample latents.

    x: (N, L, D). Returns (N,) long cluster assignment in [0, k).

    Random init (not k-means++) + enough iterations is adequate at the group
    sizes this runs on (tens to low thousands of samples) and keeps this
    dependency-free (no sklearn). Deterministic given `seed` so a rerun with
    the same data reproduces the same clusters.
    """
    N = x.shape[0]
    flat = x.reshape(N, -1)
    g = torch.Generator().manual_seed(seed)
    init_idx = torch.randperm(N, generator=g)[:k]
    centers = flat[init_idx].clone()
    assign = torch.full((N,), -1, dtype=torch.long)
    for it in range(n_iter):
        dists = torch.cdist(flat, centers)          # (N, k)
        new_assign = dists.argmin(dim=1)
        if it > 0 and torch.equal(new_assign, assign):
            assign = new_assign
            break
        assign = new_assign
        for c in range(k):
            sel = assign == c
            if sel.any():
                centers[c] = flat[sel].mean(dim=0)
            # An empty cluster keeps its last center; --substyle_min-sample
            # filtering downstream drops any sub-cluster too small to trust
            # anyway, so a temporarily-empty cluster during iteration isn't
            # a correctness problem, just wasted capacity for that round.
    return assign


def _pad_directions(dirs, k, noise_std=0.01):
    """Tile + tiny noise to stretch `dirs` (non-empty list of (18,512)
    tensors) up to exactly `k` entries. Mirrors AttributeDirectionBank's own
    'bank K < requested K' padding convention (models/direction_bank.py), so
    a stratum/sub-cluster that came up short never breaks the fixed K shape
    every attribute in the bank must share.
    """
    if not dirs:
        raise ValueError("_pad_directions: need at least one direction to pad from")
    out = list(dirs)
    i = 0
    while len(out) < k:
        base = dirs[i % len(dirs)]
        out.append(base + torch.randn_like(base) * noise_std * base.norm())
        i += 1
    return out[:k]


def compute_direction_substyle(latents, mask_high, mask_low, min_samples=50,
                                method="lda", shrinkage=None, group_center="mean",
                                group_trim_frac=0.1, substyle_k=1):
    """Like compute_direction, but when substyle_k > 1, first splits the
    HIGH group into that many k-means sub-clusters (see kmeans_latents) and
    computes one direction per sub-cluster against the shared LOW group,
    instead of one direction averaging every visual style in HIGH together.

    Returns None if the whole group is too small (same failure signal as
    compute_direction returning d=None), otherwise a list of EXACTLY
    substyle_k (direction, n_high, n_low) tuples -- short results are padded
    via _pad_directions so callers never need to branch on how many actually
    came out of the clustering.
    """
    if substyle_k <= 1:
        d, nh, nl = compute_direction(latents, mask_high, mask_low, min_samples,
                                      method=method, shrinkage=shrinkage,
                                      group_center=group_center, group_trim_frac=group_trim_frac)
        return None if d is None else [(d, nh, nl)]

    n_high_total = int(mask_high.sum().item())
    n_low_total = int(mask_low.sum().item())
    if n_high_total < min_samples or n_low_total < min_samples:
        return None

    per_cluster_min = max(8, min_samples // substyle_k)
    results = []
    if n_high_total >= substyle_k * per_cluster_min:
        high_idx = mask_high.nonzero(as_tuple=True)[0]
        assign = kmeans_latents(latents[high_idx], substyle_k)
        for c in range(substyle_k):
            sub_high_idx = high_idx[assign == c]
            if sub_high_idx.numel() < per_cluster_min:
                continue
            sub_mask_high = torch.zeros_like(mask_high)
            sub_mask_high[sub_high_idx] = True
            d, nh, nl = compute_direction(latents, sub_mask_high, mask_low, per_cluster_min,
                                          method=method, shrinkage=shrinkage,
                                          group_center=group_center, group_trim_frac=group_trim_frac)
            if d is not None:
                results.append((d, nh, nl))

    if not results:
        # Too few samples to sub-cluster reliably, or every cluster came up
        # under per_cluster_min -- fall back to one direction for the whole
        # group rather than producing noise.
        d, nh, nl = compute_direction(latents, mask_high, mask_low, min_samples,
                                      method=method, shrinkage=shrinkage,
                                      group_center=group_center, group_trim_frac=group_trim_frac)
        if d is None:
            return None
        results = [(d, nh, nl)]

    padded_dirs = _pad_directions([r[0] for r in results], substyle_k)
    counts = [(r[1], r[2]) for r in results]
    while len(counts) < substyle_k:
        counts.append(counts[-1])   # informational only; padded copies reuse the last real count
    return [(d, c[0], c[1]) for d, c in zip(padded_dirs, counts)]


def _fallback_direction(latents, attr_scores, pct=20.0, method="lda", shrinkage=None,
                         group_center="mean", group_trim_frac=0.1, cross_scores=None):
    """Unconditional direction as fallback, using percentile extremes of a
    continuous score instead of a fixed 0.5 split."""
    mask_high, mask_low = extreme_masks(attr_scores, pct=pct, cross_scores=cross_scores)
    d, _, _ = compute_direction(latents, mask_high, mask_low, method=method, shrinkage=shrinkage,
                                group_center=group_center, group_trim_frac=group_trim_frac)
    return d


# ---------------------------------------------------------------------------
# Per-attribute stratified directions
# ---------------------------------------------------------------------------

def compute_glasses_directions(latents, preds, continuous, K=4, min_samples=50,
                                pct=20.0, method="lda", shrinkage=None,
                                strata_margin=0.0, group_center="mean", group_trim_frac=0.1,
                                cross_scores=None, substyle_k=1):
    """
    Attr 15 (Eyeglasses), conditioned on gender x age.
      K0: male   x young    K1: male   x old
      K2: female x young    K3: female x old
    (each x substyle_k sub-clusters when substyle_k > 1 -- see module docstring)
    High/low split uses the top/bottom `pct`% of the continuous glasses score.
    Conditioning (gender/age) strata require confidence >= strata_margin
    away from 0.5 (see confident_strata_masks); ambiguous samples on either
    conditioning attribute are dropped from all four strata.
    cross_scores: optional independent-judge score (see extreme_masks).
    """
    glasses_cont = continuous[:, 15]
    gender_cont = continuous[:, 20]
    young_cont = continuous[:, 39]

    labels = ["male_young", "male_old", "female_young", "female_old"]
    strata = [(labels[i], m) for i, (_, m) in
              enumerate(confident_strata_masks(gender_cont, young_cont, margin=strata_margin))]

    directions = []
    for name, sub_mask in strata:
        sub_lat = latents[sub_mask]
        sub_scores = glasses_cont[sub_mask]
        sub_cross = cross_scores[sub_mask] if cross_scores is not None else None
        mask_high, mask_low = extreme_masks(sub_scores, pct=pct, cross_scores=sub_cross)
        results = compute_direction_substyle(sub_lat, mask_high, mask_low, min_samples,
                                             method=method, shrinkage=shrinkage,
                                             group_center=group_center, group_trim_frac=group_trim_frac,
                                             substyle_k=substyle_k)
        if results is not None:
            for d, nh, nl in results:
                print(f"  glasses/{name}: high={nh}, low={nl}, norm={d.norm():.3f} "
                      f"(stratum n={int(sub_mask.sum())})")
            directions.extend(r[0] for r in results)
        else:
            print(f"  glasses/{name}: FAILED (stratum n={int(sub_mask.sum())}) -> using fallback"
                  + (f" (x{substyle_k})" if substyle_k > 1 else ""))
            fb = _fallback_direction(latents, glasses_cont, pct, method, shrinkage,
                                     group_center, group_trim_frac, cross_scores)
            directions.extend(_pad_directions([fb], substyle_k))

    # Pad up to K if this attribute produced fewer directions than the bank's
    # fixed K (e.g. --substyle_k_attrs excluded it, so it only clustered
    # substyle_k=1 per stratum while other attributes were multiplied up) --
    # every attribute in the bank must share exactly the same K.
    if len(directions) < K:
        directions = _pad_directions(directions, K)
    return torch.stack(directions[:K])   # (K, 18, 512)


def compute_gender_directions(latents, preds, continuous, K=4, min_samples=50,
                               pct=20.0, method="lda", shrinkage=None,
                               strata_margin=0.0, group_center="mean", group_trim_frac=0.1,
                               cross_scores=None, substyle_k=1):
    """
    Attr 20 (Male), conditioned on age x glasses.
      K0: young x no-glasses    K1: young x glasses
      K2: old   x no-glasses    K3: old   x glasses
    (each x substyle_k sub-clusters when substyle_k > 1 -- see module docstring)
    High/low split uses the top/bottom `pct`% of the continuous gender score.
    See compute_glasses_directions for strata_margin/group_center/cross_scores semantics.
    """
    gender_cont = continuous[:, 20]
    young_cont = continuous[:, 39]
    glasses_cont = continuous[:, 15]

    labels = ["young_noglasses", "young_glasses", "old_noglasses", "old_glasses"]
    strata = [(labels[i], m) for i, (_, m) in
              enumerate(confident_strata_masks(young_cont, glasses_cont, margin=strata_margin))]

    directions = []
    for name, sub_mask in strata:
        sub_lat = latents[sub_mask]
        sub_scores = gender_cont[sub_mask]
        sub_cross = cross_scores[sub_mask] if cross_scores is not None else None
        mask_high, mask_low = extreme_masks(sub_scores, pct=pct, cross_scores=sub_cross)
        results = compute_direction_substyle(sub_lat, mask_high, mask_low, min_samples,
                                             method=method, shrinkage=shrinkage,
                                             group_center=group_center, group_trim_frac=group_trim_frac,
                                             substyle_k=substyle_k)
        if results is not None:
            for d, nh, nl in results:
                print(f"  gender/{name}: high={nh}, low={nl}, norm={d.norm():.3f} "
                      f"(stratum n={int(sub_mask.sum())})")
            directions.extend(r[0] for r in results)
        else:
            print(f"  gender/{name}: FAILED (stratum n={int(sub_mask.sum())}) -> using fallback"
                  + (f" (x{substyle_k})" if substyle_k > 1 else ""))
            fb = _fallback_direction(latents, gender_cont, pct, method, shrinkage,
                                     group_center, group_trim_frac, cross_scores)
            directions.extend(_pad_directions([fb], substyle_k))

    if len(directions) < K:
        directions = _pad_directions(directions, K)
    return torch.stack(directions[:K])


def compute_generic_directions(attr_idx, latents, preds, continuous, K=4, min_samples=50,
                                pct=20.0, method="lda", shrinkage=None,
                                strata_margin=0.0, group_center="mean", group_trim_frac=0.1,
                                cross_scores=None, substyle_k=1):
    """
    Generic stratified direction for any CelebA attribute not given a
    dedicated hand-tuned conditioning scheme (glasses/gender/age). Stratifies
    on gender x age (same conditioning as glasses), since those two are
    always present in the base attribute set and are the strongest known
    visual confounds. Falls back per-stratum like the specialized functions.
    (each stratum x substyle_k sub-clusters when substyle_k > 1 -- see module docstring)
    """
    attr_cont = continuous[:, attr_idx]
    gender_cont = continuous[:, 20]
    young_cont = continuous[:, 39]

    labels = ["male_young", "male_old", "female_young", "female_old"]
    strata = [(labels[i], m) for i, (_, m) in
              enumerate(confident_strata_masks(gender_cont, young_cont, margin=strata_margin))]

    directions = []
    for name, sub_mask in strata:
        sub_lat = latents[sub_mask]
        sub_scores = attr_cont[sub_mask]
        sub_cross = cross_scores[sub_mask] if cross_scores is not None else None
        mask_high, mask_low = extreme_masks(sub_scores, pct=pct, cross_scores=sub_cross)
        results = compute_direction_substyle(sub_lat, mask_high, mask_low, min_samples,
                                             method=method, shrinkage=shrinkage,
                                             group_center=group_center, group_trim_frac=group_trim_frac,
                                             substyle_k=substyle_k)
        if results is not None:
            for d, nh, nl in results:
                print(f"  attr{attr_idx}/{name}: high={nh}, low={nl}, norm={d.norm():.3f} "
                      f"(stratum n={int(sub_mask.sum())})")
            directions.extend(r[0] for r in results)
        else:
            print(f"  attr{attr_idx}/{name}: FAILED (stratum n={int(sub_mask.sum())}) -> using fallback"
                  + (f" (x{substyle_k})" if substyle_k > 1 else ""))
            fb = _fallback_direction(latents, attr_cont, pct, method, shrinkage,
                                     group_center, group_trim_frac, cross_scores)
            directions.extend(_pad_directions([fb], substyle_k))

    if len(directions) < K:
        directions = _pad_directions(directions, K)
    return torch.stack(directions[:K])


def compute_age_directions(latents, preds, continuous, K=4, min_samples=50,
                            pct=20.0, method="lda", shrinkage=None,
                            strata_margin=0.0, group_center="mean", group_trim_frac=0.1,
                            cross_scores=None, substyle_k=1):
    """
    Attr 39 (Young), conditioned on gender x glasses.
      K0: male   x no-glasses    K1: male   x glasses
      K2: female x no-glasses    K3: female x glasses
    (each x substyle_k sub-clusters when substyle_k > 1 -- see module docstring)
    High/low split uses the top/bottom `pct`% of the continuous age score.
    See compute_glasses_directions for strata_margin/group_center/cross_scores semantics.
    """
    age_cont = continuous[:, 39]
    gender_cont = continuous[:, 20]
    glasses_cont = continuous[:, 15]

    labels = ["male_noglasses", "male_glasses", "female_noglasses", "female_glasses"]
    strata = [(labels[i], m) for i, (_, m) in
              enumerate(confident_strata_masks(gender_cont, glasses_cont, margin=strata_margin))]

    directions = []
    for name, sub_mask in strata:
        sub_lat = latents[sub_mask]
        sub_scores = age_cont[sub_mask]
        sub_cross = cross_scores[sub_mask] if cross_scores is not None else None
        mask_high, mask_low = extreme_masks(sub_scores, pct=pct, cross_scores=sub_cross)
        results = compute_direction_substyle(sub_lat, mask_high, mask_low, min_samples,
                                             method=method, shrinkage=shrinkage,
                                             group_center=group_center, group_trim_frac=group_trim_frac,
                                             substyle_k=substyle_k)
        if results is not None:
            for d, nh, nl in results:
                print(f"  age/{name}: high={nh}, low={nl}, norm={d.norm():.3f} "
                      f"(stratum n={int(sub_mask.sum())})")
            directions.extend(r[0] for r in results)
        else:
            print(f"  age/{name}: FAILED (stratum n={int(sub_mask.sum())}) -> using fallback"
                  + (f" (x{substyle_k})" if substyle_k > 1 else ""))
            fb = _fallback_direction(latents, age_cont, pct, method, shrinkage,
                                     group_center, group_trim_frac, cross_scores)
            directions.extend(_pad_directions([fb], substyle_k))

    if len(directions) < K:
        directions = _pad_directions(directions, K)
    return torch.stack(directions[:K])


def compute_age_k1_stratified(latents, preds, continuous, min_samples=50,
                               pct=20.0, method="lda", shrinkage=None,
                               strata_margin=0.0, group_center="mean", group_trim_frac=0.1,
                               cross_scores=None, substyle_k=1):
    """Debiased K=1 age direction via stratum-size-weighted average.

    Computes 4 sub-directions conditioned on gender x glasses, then averages
    them weighted by stratum size. This removes gender/glasses leakage
    without the norm collapse caused by intra-attr orthogonalization.

    Uses the same generic direction primitive (LDA/whitened + percentile
    split) as glasses/gender -- no age-only special-casing beyond the
    stratum-size-weighted averaging strategy itself, which is a K-selection
    choice independent of how each sub-direction is computed.

    substyle_k is accepted for call-site uniformity with the other compute_*
    functions but INTENTIONALLY IGNORED here: this function's whole point is
    collapsing 4 sub-directions down to 1 debiased direction, the opposite
    of what substyle_k's job (produce MORE, more homogeneous directions) is
    for. Pass --age_k 4 (compute_age_directions instead of this function) to
    get substyle_k sub-clustering applied to age.

    Returns: (1, 18, 512) -- single direction, ready to be tiled to fill K slots.
    """
    if substyle_k > 1:
        print(f"  [substyle_k] ignored for age_k=1 (debiased collapse-to-one path); "
              f"use --age_k 4 to apply sub-clustering to age.")
    age_cont = continuous[:, 39]
    gender_cont = continuous[:, 20]
    glasses_cont = continuous[:, 15]

    labels = ["male_noglasses", "male_glasses", "female_noglasses", "female_glasses"]
    strata = [(labels[i], m) for i, (_, m) in
              enumerate(confident_strata_masks(gender_cont, glasses_cont, margin=strata_margin))]

    sub_dirs, weights = [], []
    for name, sub_mask in strata:
        sub_lat = latents[sub_mask]
        sub_scores = age_cont[sub_mask]
        sub_cross = cross_scores[sub_mask] if cross_scores is not None else None
        mask_high, mask_low = extreme_masks(sub_scores, pct=pct, cross_scores=sub_cross)
        d, nh, nl = compute_direction(sub_lat, mask_high, mask_low, min_samples,
                                       method=method, shrinkage=shrinkage,
                                       group_center=group_center, group_trim_frac=group_trim_frac)
        n_total = int(sub_mask.sum().item())
        if d is not None:
            print(f"  age/{name}: high={nh}, low={nl}, norm={d.norm():.4f}, stratum_size={n_total}")
            sub_dirs.append(d)
            weights.append(float(n_total))
        else:
            print(f"  age/{name}: SKIPPED (high={nh}, low={nl} — below min_samples={min_samples})")

    if not sub_dirs:
        raise ValueError("No valid age sub-directions — check min_samples/extreme_pct or data")

    w = torch.tensor(weights)
    w = w / w.sum()
    age_dir = sum(wi.item() * d for wi, d in zip(w, sub_dirs))   # (18, 512)
    print(f"  → weighted average norm: {age_dir.norm():.4f}  (from {len(sub_dirs)} strata)")
    return age_dir.unsqueeze(0)   # (1, 18, 512)


# ---------------------------------------------------------------------------
# Color-confound decorrelation (age direction only)
# ---------------------------------------------------------------------------
#
# Diagnosis (verified at inference time with two direction-agnostic checks):
#   1. --override_residual_scale 0.0 (pure Direction Bank output, zero flow
#      contribution) on a trained checkpoint reproduced the exact same
#      age-edit color-cast artifact (skin turning blue / orange / magenta at
#      high edit strength) as every training configuration tested -- proving
#      the artifact lives in the precomputed age direction_units themselves,
#      not in the flow, gate, residual, train_scale, or any training loss.
#   2. Zeroing the age direction's fine W+ layers (index >=4) removed the
#      color cast but also collapsed real editing accuracy (CLIP-judged
#      young->old accuracy 76%->17%); narrowing the cut to only the last 4
#      layers preserved accuracy but left the color cast untouched. The
#      confound and the genuine aging signal occupy the SAME fine layers --
#      no single layer-range boundary separates them.
#
# This targets the root cause directly: FFHQ photos labeled "old" vs "young"
# likely differ systematically in color grading/lighting (photo era, camera,
# skin-tone distribution) independent of true facial aging. LDA/whitening
# only suppresses NOISY (attribute-uncorrelated) dimensions -- a confound
# that is consistently correlated with the attribute in the training data is
# indistinguishable from true signal to the direction-fitting procedure, and
# gets encoded (or even amplified by whitening) right along with it.
#
# Fix: independently estimate a "color confound direction" per W+ layer from
# real dataset images (average warm/cool tone, R-B), the same LDA/percentile-
# split machinery already used for attributes, then remove (per-layer
# orthogonal projection) only that specific direction from the age vector --
# a surgical single-direction removal, not a whole-layer blanket cut, so it
# should damage the real aging signal far less than the layer-truncation
# experiments did.

def compute_image_color_scores(image_root, paths, resize=32, face_crop=True):
    """Per-image warm/cool tone score: mean(R) - mean(B) at low resolution.

    Cheap proxy for color grading (the observed artifacts were specifically
    warm/orange/red or cool/blue casts, i.e. exactly this axis). Downsampling
    to `resize`x`resize` keeps this fast even for tens of thousands of images
    -- only the coarse tone is needed, not detail.

    face_crop=True (default) restricts the average to the face region, using
    the same crop convention as common/id_loss.py's ArcFace preprocessing
    (scale(188) square crop at row-offset scale(35), col-offset scale(32),
    all relative to image width/256). A first attempt averaged the WHOLE
    photo (background, hair, clothing included) and found the resulting
    confound direction nearly orthogonal to the age direction (cos
    ~0.03-0.10) -- too diluted by attribute-irrelevant regions to explain
    the observed artifact. The artifact is about skin/facial tone, not
    whole-photo color grading, so the color score should look where the
    artifact actually shows up.
    """
    from PIL import Image
    import numpy as np

    scores = torch.zeros(len(paths), dtype=torch.float32)
    for i, rel_path in enumerate(paths):
        img_path = os.path.join(image_root, rel_path)
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if face_crop:
                w, h = img.size
                crop_h = int(188 * w / 256)
                row_off = int(35 * w / 256)
                col_off = int(32 * w / 256)
                img = img.crop((col_off, row_off, col_off + crop_h, row_off + crop_h))
            img = img.resize((resize, resize))
            arr = torch.from_numpy(np.asarray(img)).float()  # (H, W, 3)
        mean_rgb = arr.mean(dim=(0, 1))  # (3,)
        scores[i] = mean_rgb[0] - mean_rgb[2]  # R - B: positive=warm, negative=cool
        if (i + 1) % 5000 == 0:
            print(f"  color scores: {i + 1}/{len(paths)}")
    return scores


def compute_color_confound_direction(latents, color_scores, min_samples=50,
                                      pct=20.0, method="lda", shrinkage=None,
                                      group_center="mean", group_trim_frac=0.1):
    """Direction in W+ that explains color-tone variation, independent of any
    attribute label. Same primitive as attribute directions (percentile
    split + LDA/whitened), just conditioned on the color score instead."""
    mask_high, mask_low = extreme_masks(color_scores, pct=pct)
    d, n_high, n_low = compute_direction(
        latents, mask_high, mask_low, min_samples, method=method, shrinkage=shrinkage,
        group_center=group_center, group_trim_frac=group_trim_frac,
    )
    if d is None:
        raise ValueError(
            f"Color confound direction failed (high={n_high}, low={n_low}, "
            f"min_samples={min_samples}); lower --min_samples or check --image_root."
        )
    print(f"  color confound: high={n_high}, low={n_low}, norm={d.norm():.3f}")
    return d  # (18, 512)


def project_out_direction(direction, confound_dir):
    """Per-layer orthogonal projection: remove confound_dir's component from
    direction. Unlike a layer-range cut, this only removes the specific
    linear subspace aligned with the confound at each layer, leaving
    everything orthogonal to it (including most of the true signal, as long
    as it isn't collinear with the confound) untouched.

    direction:    (K, 18, 512) or (18, 512)
    confound_dir: (18, 512)
    """
    conf_unit = F.normalize(confound_dir, dim=-1, eps=1e-8)  # (18, 512)
    squeeze = direction.dim() == 2
    d = direction.unsqueeze(0) if squeeze else direction     # (K, 18, 512)
    dot = (d * conf_unit.unsqueeze(0)).sum(dim=-1, keepdim=True)  # (K, 18, 1)
    cleaned = d - dot * conf_unit.unsqueeze(0)
    return cleaned.squeeze(0) if squeeze else cleaned


# ---------------------------------------------------------------------------
# Residual helpers
# ---------------------------------------------------------------------------

def representative_direction(dirs_k):
    """Uniform average over K directions -> single (L, D) representative."""
    return dirs_k.mean(dim=0)


def remove_direction_components(w_all, unit_directions):
    """Per-layer projection removal.

    w_all:           (N, L, D)
    unit_directions: list of (L, D) unit-normalised tensors
    Returns cleaned (N, L, D).
    """
    w_clean = w_all.clone()
    for d_unit in unit_directions:
        for layer in range(w_all.shape[1]):
            d_hat = F.normalize(d_unit[layer], dim=0)             # (D,)
            proj = (w_clean[:, layer] @ d_hat).unsqueeze(1) * d_hat  # (N, D)
            w_clean[:, layer] = w_clean[:, layer] - proj
    return w_clean


# ---------------------------------------------------------------------------
# Orthogonalization
# ---------------------------------------------------------------------------

def intra_attr_orthogonalize_safe(directions, iters=5, min_norm_ratio=0.1, max_norm_ratio=5.0):
    """Symmetric (Löwdin/Newton-Schulz) orthogonalization with fallback for
    near-degenerate cases.

    If a direction's norm ends up outside [min_norm_ratio, max_norm_ratio] x
    its original norm after orthogonalization, fall back to the original
    (unorthogonalized) direction instead. Two distinct failure modes need
    two distinct checks here, not one:
      - COLLAPSE (norm -> ~0): happens when two strata/sub-clusters produce
        nearly identical direction vectors -- this is the case the
        min_norm_ratio check has always caught.
      - EXPLOSION (norm -> huge but finite, e.g. ~1e12): the iteration
        `D_unit = 1.5*D_unit - 0.5*(D_unit @ D_unit.t()) @ D_unit` is
        Newton-Schulz for the orthogonal polar factor, which only CONVERGES
        when D_unit's singular values lie in (0, sqrt(3)). K unit vectors
        that are highly correlated/near-duplicate (e.g. --substyle_k
        sub-clusters of an already-narrow stratum that didn't have much
        real visual diversity to split on, or several strata falling back
        to the same padded direction) push the largest singular value past
        sqrt(3), and the iteration then DIVERGES -- doubling its error each
        pass instead of shrinking it, so 5 iterations is enough to reach
        an astronomical but still torch.isfinite() norm. A finite-but-huge
        result silently passes the isfinite() guard in compute_direction
        (that guard runs on a different, earlier stage) and previously had
        no check here at all, so it reached the saved bank undetected --
        this is what produced the ~2e12 mean_norm seen on the gender
        direction after enabling --substyle_k_attrs on it.
    Both checks compare against the PRE-orthogonalization norm, so this is a
    "did this operation do something numerically sane" test, not a fixed
    absolute threshold.

    Args:
        directions: (K, 18, 512)
    Returns:
        (K, 18, 512)
    """
    K, num_layers, dim = directions.shape
    original = directions.clone()
    result = directions.clone()
    for layer in range(num_layers):
        D = result[:, layer]                                # (K, 512)
        orig_norms = D.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        D_unit = D / orig_norms
        for _ in range(iters):
            D_unit = 1.5 * D_unit - 0.5 * (D_unit @ D_unit.t()) @ D_unit
        orth = D_unit * orig_norms
        # Fallback: if the norm collapsed OR exploded, or went non-finite,
        # keep the original (unorthogonalized) direction for those rows.
        new_norms = orth.norm(dim=-1)
        orig_norms_flat = orig_norms.squeeze(-1)
        bad = (
            (new_norms < min_norm_ratio * orig_norms_flat)
            | (new_norms > max_norm_ratio * orig_norms_flat)
            | ~torch.isfinite(new_norms)
        )
        if bad.any():
            orth[bad] = original[:, layer][bad]
        result[:, layer] = orth
    return result


def sanitize_non_finite_directions(all_dirs, attribute_index, latents, continuous, pct, method,
                                    shrinkage, group_center, group_trim_frac):
    """Final, unconditional safety net -- replace any STILL-non-finite (attr,
    K) direction slot with a freshly computed unconditional fallback, no
    matter which upstream stage produced it.

    compute_direction() and intra_attr_orthogonalize_safe() each already
    reject non-finite output at their OWN stage, but a large enough
    --substyle_k (K=16 from --substyle_k 4, observed in practice) can push
    the orthogonalization's Newton-Schulz iteration to genuine float32
    overflow. Because that iteration's update is a shared (K, D) @ (D, K)
    matrix product across ALL K rows of one attribute at once, a single row
    overflowing to inf/nan during an iteration can spread to every OTHER row
    through that same matrix multiply before the per-row isfinite check at
    the end of the function ever gets a chance to isolate just the one bad
    row -- so entire attributes can come back fully non-finite despite that
    guard. --decorrelate_cross_attr then compounds this: an attribute with a
    now-corrupted representative direction can carry that corruption into
    every OTHER attribute that projects against it (observed: an attribute
    that never used --substyle_k still came back fully non-finite after
    decorrelating against one that did).

    This runs on `all_dirs` right before it is turned into layer_norms/
    direction_units and saved -- the one chokepoint every direction passes
    through regardless of which upstream stage actually failed in a given
    run -- so it is the last line of defense rather than a replacement for
    the earlier, stage-specific guards.
    """
    num_attrs, K = all_dirs.shape[0], all_dirs.shape[1]
    for a_i, attr in enumerate(attribute_index):
        finite_per_k = torch.isfinite(all_dirs[a_i]).reshape(K, -1).all(dim=-1)
        bad_k = (~finite_per_k).nonzero(as_tuple=True)[0].tolist()
        if not bad_k:
            continue
        print(f"[SANITIZE] attr {attr}: non-finite at K-slots {bad_k} after decorrelation/"
              f"orthogonalization -- recomputing an unconditional fallback direction for them "
              f"instead of saving inf/nan into the bank.")
        fb = _fallback_direction(latents, continuous[:, attr], pct=pct, method=method,
                                 shrinkage=shrinkage, group_center=group_center,
                                 group_trim_frac=group_trim_frac)
        if fb is None or not torch.isfinite(fb).all():
            raise RuntimeError(
                f"attr {attr}: even the unconditional fallback direction is non-finite -- this is "
                f"not a K-slot-specific issue, check --latent_file/--continuous_preds_file for this "
                f"attribute directly."
            )
        for k in bad_k:
            all_dirs[a_i, k] = fb
    return all_dirs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stratified direction bank (LDA/whitened, percentile splits)")
    parser.add_argument("--latent_file", default="./data/ffhq_e4e_latents.pth")
    parser.add_argument("--preds_file", default="./data/ffhq_e4e_preds.pth",
                        help="Binary 0/1 predictions, used to define strata (gender/glasses groups).")
    parser.add_argument("--continuous_preds_file", default="./data/ffhq_e4e_preds_continuous.pth",
                        help="Continuous [0,1] scores from scripts/extract_continuous_attrs.py, "
                             "used for the target attribute's own high/low split within each stratum.")
    parser.add_argument("--direction_method", choices=["lda", "mean_diff"], default="lda",
                        help="lda: per-layer whitened direction (recommended). "
                             "mean_diff: legacy high_mean - low_mean.")
    parser.add_argument("--extreme_pct", type=float, default=20.0,
                        help="Use the top/bottom this-percent of the continuous score as the "
                             "high/low groups, instead of a fixed 0.5 threshold. Must be < 50.")
    parser.add_argument("--shrinkage", type=float, default=None,
                        help="Fixed Ledoit-Wolf shrinkage in [0,1]. Default: estimated automatically per layer.")
    parser.add_argument("--K", type=int, default=4,
                        help="K for glasses and gender. Age uses --age_k.")
    parser.add_argument("--age_k", type=int, default=1,
                        help="K for age. Default 1 = debiased weighted-average direction "
                             "(no intra-attr orthogonalization). Use 4 to match original behaviour.")
    parser.add_argument("--residual_age", action="store_true",
                        help="After stratified age extraction, project out representative "
                             "glasses/gender directions from W+. Reduces cross-attr overlap "
                             "(dir_orth). Only effective when --age_k 1.")
    parser.add_argument("--decorrelate_age_color", action="store_true",
                        help="Estimate a color-tone confound direction from real dataset images "
                             "(mean R-B) and orthogonally project it out of the age direction. "
                             "Fixes a confirmed color-cast artifact in the age direction (see "
                             "code comment above compute_image_color_scores) without the "
                             "accuracy collapse caused by layer-range truncation. Requires "
                             "--image_root to point at the actual image files.")
    parser.add_argument("--image_root", default="data/FFHQ",
                        help="Root directory for --decorrelate_age_color; combined with the "
                             "'paths' saved in --preds_file.")
    parser.add_argument("--color_confound_pct", type=float, default=None,
                        help="Percentile split for the color confound direction. Defaults to "
                             "--extreme_pct if not set.")
    parser.add_argument("--color_resize", type=int, default=32,
                        help="Downsample images to this size before averaging color for "
                             "--decorrelate_age_color. Only global tone matters, so this can "
                             "stay small for speed.")
    parser.add_argument("--color_face_crop", action=argparse.BooleanOptionalAction, default=True,
                        help="Restrict the color-confound score to the face region (same crop "
                             "as common/id_loss.py) instead of the whole photo. Default on: a "
                             "whole-photo version measured cos~0.03-0.10 against the age "
                             "direction, too diluted by background/hair/clothing to matter.")
    parser.add_argument("--output", default="./data/direction_bank_k4_stratified.pth")
    parser.add_argument("--min_samples", type=int, default=50,
                        help="Minimum samples per high/low group. NOTE: with --extreme_pct, each "
                             "stratum's usable pool shrinks to roughly 2*extreme_pct%% of its size, "
                             "so small strata may need a lower --min_samples (e.g. 20-30) or a "
                             "larger --extreme_pct to avoid falling back to the unconditional direction.")
    parser.add_argument("--strata_margin", type=float, default=0.0,
                        help="Confidence margin (see confident_strata_masks) required on each "
                             "CONDITIONING attribute's own continuous score before a sample counts "
                             "toward a stratum, instead of a bare 0.5 threshold. 0.0 (default) "
                             "reproduces the old behavior. Try 0.15-0.25 to drop samples the "
                             "classifier itself is unsure about on the conditioning side too --  "
                             "only the target attribute's split got this treatment before "
                             "(--extreme_pct); this extends the same idea to the strata that decide "
                             "which K-slot a sample's direction gets computed from.")
    parser.add_argument("--group_center", choices=["mean", "trimmed_mean"], default="mean",
                        help="How each high/low group's centroid is computed. trimmed_mean drops "
                             "the most extreme --group_trim_frac fraction of samples per-coordinate "
                             "first, reducing sensitivity to a handful of mislabeled/atypical "
                             "samples in small strata. mean = old behavior.")
    parser.add_argument("--group_trim_frac", type=float, default=0.1,
                        help="Trim fraction for --group_center trimmed_mean (per side, per "
                             "coordinate). 0.1 = drop the most extreme 10%% on each end.")
    parser.add_argument("--require_cross_judge_agree", nargs="*", type=int, default=None,
                        help="Attribute indices to require an independent judge's agreement for, "
                             "before a sample counts as confidently high/low (see extreme_masks). "
                             "PER-ATTRIBUTE, not a blanket on/off switch: the attributes this "
                             "project has documented the LARGEST r34-vs-independent-judge gap on "
                             "(eyeglasses specifically, ~91%% r34 vs ~44%% CLIP) are exactly the "
                             "ones where requiring agreement throws away the most samples -- every "
                             "stratum can starve below --min_samples and fall back to the same weak "
                             "unconditional direction for all K slots, which is WORSE than plain "
                             "r34-only filtering, not better (confirmed empirically: applying this "
                             "blanket-on to eyeglasses collapsed its validate_direction_bank.py "
                             "AccCLIP to a flat ~1%% at every alpha, while gender/age -- which don't "
                             "have that large a judge gap -- improved). Pass the attributes where "
                             "the two judges are known to mostly agree (e.g. 20 39 for gender/age); "
                             "leave attributes with a large documented judge gap (e.g. 15 for "
                             "eyeglasses) OUT of this list. Needs --continuous_preds_file built with "
                             "extract_continuous_attr.py --cross_judge clip; attributes not covered "
                             "by that file's --cross_judge_attrs fall back to r34-only with a "
                             "warning regardless of this flag. Omit entirely for the old default "
                             "(r34-only for everything).")
    parser.add_argument("--substyle_k", type=int, default=1,
                        help="Split each stratum's 'high' (attribute-present) group into this "
                             "many k-means sub-clusters BEFORE computing a direction, instead of "
                             "one direction averaging every visual style together (thin/thick/"
                             "rimless glasses, closed/open-mouth smiling, etc.) -- see module "
                             "docstring. Applies to every attribute in --substyle_k_attrs (default: "
                             "every attribute in --attribute_index), except age when --age_k 1 (the "
                             "debiased collapse-to-one path is intentionally incompatible -- use "
                             "--age_k 4 if you want this applied to age too). 1 (default) = old "
                             "behavior. --K and --age_k (when not 1) are auto-multiplied by this.")
    parser.add_argument("--substyle_k_attrs", nargs="*", type=int, default=None,
                        help="Restrict --substyle_k to these attribute indices; attributes left out "
                             "keep substyle_k=1 (plain per-stratum direction). Default (omitted): "
                             "every attribute in --attribute_index, i.e. no restriction. WHY THIS "
                             "MATTERS: substyle_k splits an ALREADY-stratified, ALREADY-extreme_pct-"
                             "filtered group into even smaller k-means sub-clusters -- fine for a "
                             "common attribute with plenty of samples on both sides (gender, age), "
                             "but for a RARE attribute (eyeglasses is a small minority of FFHQ) it "
                             "can push per-cluster sample counts low enough that the LDA covariance "
                             "solve becomes poorly conditioned even with the isfinite() safety net "
                             "in compute_direction (that net stops NaN from reaching the bank, it "
                             "doesn't make a barely-adequate sample size well-conditioned). Pass e.g. "
                             "--substyle_k 2 --substyle_k_attrs 20 39 to apply sub-clustering only "
                             "where the sample count comfortably supports it.")
    parser.add_argument("--attribute_index", nargs="*", type=int, default=[15, 20, 39])
    parser.add_argument("--decorrelate_cross_attr", action="store_true",
                         help="Generic pairwise decorrelation: for every attribute in "
                              "--attribute_index, project out every OTHER attribute's "
                              "representative direction. Applies uniformly regardless of which "
                              "specific pair turns out to be entangled (beard vs gender, age vs "
                              "gender, or any future pair) -- no per-pair flag needed. Confirmed "
                              "cases so far: beard direction inheriting 'more male' (beards rare "
                              "on female training faces), age direction dragging Male toward 1.0 "
                              "even with age_k=4 gender-stratified sub-directions (see "
                              "dump_attr_failures.py --watch_attrs 20).")
    args = parser.parse_args()

    if not (0 < args.extreme_pct < 50):
        parser.error("--extreme_pct must be between 0 and 50 (exclusive) so high/low groups don't overlap.")

    K = args.K
    age_k = args.age_k
    if age_k > K:
        parser.error(f"--age_k ({age_k}) cannot exceed --K ({K}); age_dirs would end up with more "
                     "rows than glasses_dirs/gender_dirs and torch.stack(...) would fail.")

    # --substyle_k multiplies the direction count each stratify function
    # actually produces (see compute_direction_substyle) -- auto-derive K/
    # age_k here instead of making the caller do the multiplication by hand,
    # so torch.stack(...) shapes stay consistent by construction. age_k==1
    # (the debiased collapse-to-one path) is intentionally left alone.
    if args.substyle_k > 1:
        new_K = K * args.substyle_k
        print(f"[substyle_k] --K auto-derived: {K} strata x substyle_k {args.substyle_k} = {new_K}")
        K = new_K
        if age_k > 1:
            new_age_k = age_k * args.substyle_k
            print(f"[substyle_k] --age_k auto-derived: {age_k} strata x substyle_k "
                  f"{args.substyle_k} = {new_age_k}")
            age_k = new_age_k

    print("Loading latents ...")
    latents = load_latents(args.latent_file)
    print(f"  shape: {tuple(latents.shape)}")

    print("Loading binary predictions (used for stratum grouping) ...")
    preds = load_preds(args.preds_file)
    print(f"  shape: {tuple(preds.shape)}")

    print("Loading continuous predictions (used for high/low splits) ...")
    continuous = load_preds(args.continuous_preds_file)
    print(f"  shape: {tuple(continuous.shape)}")
    if continuous.shape[0] != preds.shape[0]:
        parser.error(
            f"--preds_file has {preds.shape[0]} rows but --continuous_preds_file has "
            f"{continuous.shape[0]} rows; they must come from the same image list/order."
        )
    preds_paths = load_paths(args.preds_file)
    continuous_paths = load_paths(args.continuous_preds_file)
    if preds_paths is not None and continuous_paths is not None and preds_paths != continuous_paths:
        parser.error(
            "--preds_file and --continuous_preds_file have matching row counts but different "
            "'paths' order/content -- they were built from different or differently-ordered image "
            "lists. Re-run scripts/extract_continuous_attrs.py with the same --img_list used for "
            "--preds_file, otherwise every row pairs a prediction with the wrong image."
        )

    for attr_name, idx in [("glasses", 15), ("gender", 20), ("young", 39)]:
        uniq = preds[:, idx].unique().tolist()
        cont = continuous[:, idx]
        print(f"  attr {idx} ({attr_name}): binary unique={uniq}, "
              f"continuous mean={cont.mean():.3f} std={cont.std():.3f}")

    cross_by_attr = {}
    if args.require_cross_judge_agree is not None:
        want_attrs = [int(a) for a in args.require_cross_judge_agree]
        _cross_raw = torch.load(args.continuous_preds_file, map_location="cpu")
        if not (isinstance(_cross_raw, dict) and "values_cross_clip" in _cross_raw):
            parser.error(
                "--require_cross_judge_agree needs --continuous_preds_file built with "
                "extract_continuous_attr.py --cross_judge clip (missing 'values_cross_clip' key)."
            )
        _cross_vals = _cross_raw["values_cross_clip"].float()
        _cross_attrs = [int(a) for a in _cross_raw["cross_attribute_index"]]
        available = {a: _cross_vals[:, i] for i, a in enumerate(_cross_attrs)}
        # Only apply agreement-filtering to attributes explicitly requested --
        # NOT every attribute the continuous file happens to have cross scores
        # for. Blanket-applying this to an attribute with a large documented
        # r34-vs-CLIP gap (eyeglasses) starves its sample pool per-stratum and
        # produces a WORSE direction than plain r34-only filtering -- see the
        # help text above for the empirical confirmation.
        cross_by_attr = {a: available[a] for a in want_attrs if a in available}
        missing = [a for a in want_attrs if a not in available]
        skipped = [a for a in args.attribute_index if a not in want_attrs]
        print(f"[CrossJudge] agreement required for attrs {sorted(cross_by_attr)}"
              + (f"; requested but unavailable (not in --cross_judge_attrs when the continuous "
                 f"file was built, falling back to r34-only) {missing}" if missing else "")
              + (f"; r34-only by choice (not in --require_cross_judge_agree) {skipped}"
                 if skipped else ""))

    # --substyle_k_attrs restricts ACTUAL k-means sub-clustering (the
    # substyle_k kwarg passed into each compute_*_directions call) to a
    # subset of attributes; K/age_k above (the TARGET row count every
    # attribute's direction tensor must have so torch.stack(...) across
    # attributes works) stay uniform regardless -- an excluded attribute
    # just produces its plain num_strata directions and gets padded up to
    # that target count by the guard now at the end of every stratify
    # function. Default (--substyle_k_attrs omitted): apply to every
    # attribute in --attribute_index, i.e. old uniform behavior.
    def _substyle_k_for(attr):
        if args.substyle_k <= 1:
            return args.substyle_k
        if args.substyle_k_attrs is None or attr in args.substyle_k_attrs:
            return args.substyle_k
        return 1

    print(f"\nmethod={args.direction_method}, extreme_pct={args.extreme_pct}, "
          f"K glasses/gender={K}, K age={age_k}, residual_age={args.residual_age}, "
          f"strata_margin={args.strata_margin}, group_center={args.group_center}"
          + (f" (trim_frac={args.group_trim_frac})" if args.group_center == "trimmed_mean" else "")
          + (f", substyle_k={args.substyle_k} restricted to attrs {args.substyle_k_attrs}"
             if args.substyle_k > 1 else ""))

    common_kwargs = dict(pct=args.extreme_pct, method=args.direction_method, shrinkage=args.shrinkage,
                         strata_margin=args.strata_margin, group_center=args.group_center,
                         group_trim_frac=args.group_trim_frac)

    print(f"\n=== Eyeglasses (attr 15), K={K}, substyle_k={_substyle_k_for(15)} ===")
    glasses_dirs = compute_glasses_directions(latents, preds, continuous, K, args.min_samples,
                                              cross_scores=cross_by_attr.get(15),
                                              substyle_k=_substyle_k_for(15), **common_kwargs)

    print(f"\n=== Gender / Male (attr 20), K={K}, substyle_k={_substyle_k_for(20)} ===")
    gender_dirs = compute_gender_directions(latents, preds, continuous, K, args.min_samples,
                                            cross_scores=cross_by_attr.get(20),
                                            substyle_k=_substyle_k_for(20), **common_kwargs)

    # ── Age direction ──────────────────────────────────────────────────────────
    # When --residual_age: build a W+ where the representative glasses and gender
    # directions have been projected out before computing the age direction.
    # This ensures cross-attribute orthogonality (dir_orth) is near-zero by
    # construction rather than relying on post-hoc Gram-Schmidt.
    latents_for_age = latents
    if args.residual_age and age_k == 1:
        print("\n=== Removing glasses/gender projections from W+ for age ===")
        glasses_rep = representative_direction(glasses_dirs)   # (L, D) uniform avg
        gender_rep = representative_direction(gender_dirs)     # (L, D) uniform avg
        glasses_rep_unit = F.normalize(glasses_rep, dim=-1, eps=1e-8)
        gender_rep_unit = F.normalize(gender_rep, dim=-1, eps=1e-8)
        latents_for_age = remove_direction_components(
            latents, [glasses_rep_unit, gender_rep_unit]
        )
        print(f"  projected out glasses and gender from {latents.shape[0]} latents")

    print(f"\n=== Age / Young (attr 39), K={age_k}, substyle_k={_substyle_k_for(39)} ===")
    if age_k == 1:
        # compute_age_k1_stratified ignores substyle_k>1 by design (see its
        # own note), so this is passed for consistency/logging only.
        age_dir_k1 = compute_age_k1_stratified(latents_for_age, preds, continuous, args.min_samples,
                                               cross_scores=cross_by_attr.get(39),
                                               substyle_k=_substyle_k_for(39), **common_kwargs)
        age_dirs = age_dir_k1.expand(K, -1, -1).clone()   # (K, L, D) — tiled
        skip_age_orth = True
    else:
        age_dirs = compute_age_directions(latents_for_age, preds, continuous, age_k, args.min_samples,
                                          cross_scores=cross_by_attr.get(39),
                                          substyle_k=_substyle_k_for(39), **common_kwargs)
        if age_k < K:
            pad = age_dirs[-1:].expand(K - age_k, -1, -1).clone()
            age_dirs = torch.cat([age_dirs, pad], dim=0)
        skip_age_orth = False

    if args.decorrelate_age_color:
        print("\n=== Removing color-tone confound from age direction ===")
        paths_for_color = load_paths(args.preds_file)
        if paths_for_color is None:
            parser.error(
                "--decorrelate_age_color requires --preds_file to have saved 'paths' "
                "(image filenames matching latents row order)."
            )
        color_scores = compute_image_color_scores(
            args.image_root, paths_for_color, resize=args.color_resize,
            face_crop=args.color_face_crop,
        )
        print(f"  color score (R-B): mean={color_scores.mean():.3f} std={color_scores.std():.3f}")
        confound_dir = compute_color_confound_direction(
            latents, color_scores, args.min_samples,
            pct=(args.color_confound_pct if args.color_confound_pct is not None else args.extreme_pct),
            method=args.direction_method, shrinkage=args.shrinkage,
            group_center=args.group_center, group_trim_frac=args.group_trim_frac,
        )

        # Report overlap BEFORE removal so the fix's effect size is visible,
        # not just asserted.
        confound_unit = F.normalize(confound_dir, dim=-1, eps=1e-8)
        age_unit_before = F.normalize(age_dirs, dim=-1, eps=1e-8)
        cos_before = (age_unit_before * confound_unit.unsqueeze(0)).sum(dim=-1)  # (K, 18)
        print(f"  |cos(age_dir, color_confound)| before removal: "
              f"mean={cos_before.abs().mean():.4f} max={cos_before.abs().max():.4f} "
              f"(per-layer, averaged over K)")

        age_dirs = project_out_direction(age_dirs, confound_dir)

        age_unit_after = F.normalize(age_dirs, dim=-1, eps=1e-8)
        cos_after = (age_unit_after * confound_unit.unsqueeze(0)).sum(dim=-1)
        print(f"  |cos(age_dir, color_confound)| after removal:  "
              f"mean={cos_after.abs().mean():.4f} max={cos_after.abs().max():.4f} "
              f"(should be ~0)")

    # Any attributes beyond the hand-tuned glasses/gender/age trio get the
    # generic gender x age stratified treatment.
    extra_attr_ids = [a for a in args.attribute_index if a not in (15, 20, 39)]
    extra_dirs = {}
    for attr_idx in extra_attr_ids:
        print(f"\n=== Attr {attr_idx} (generic), K={K}, substyle_k={_substyle_k_for(attr_idx)} ===")
        extra_dirs[attr_idx] = compute_generic_directions(
            attr_idx, latents, preds, continuous, K, args.min_samples,
            cross_scores=cross_by_attr.get(attr_idx),
            substyle_k=_substyle_k_for(attr_idx), **common_kwargs
        )

    # Stack in the order given by args.attribute_index.
    dirs_by_idx = {15: glasses_dirs, 20: gender_dirs, 39: age_dirs, **extra_dirs}

    if args.decorrelate_cross_attr:
        # Generic pairwise decorrelation: for EVERY attribute in the bank,
        # project out every OTHER attribute's representative direction.
        # Supersedes hand-written per-pair fixes (an earlier version of this
        # script had a --decorrelate_extra_attrs flag that only covered
        # extra-attrs-vs-{glasses,gender,age}, and a --decorrelate_age_gender
        # flag added just for the age-vs-gender leak found via
        # dump_attr_failures.py --watch_attrs 20). Both were one-off patches
        # for whichever pair had just been diagnosed; this runs the same
        # projection for all pairs automatically, so a newly added attribute
        # doesn't need its own manually-diagnosed flag before it's known to
        # need one.
        print("\n=== Removing cross-attribute confounds (all pairs) ===")
        reps = {a: representative_direction(dirs_by_idx[a]) for a in args.attribute_index}
        for a in args.attribute_index:
            d = dirs_by_idx[a]
            for b in args.attribute_index:
                if b == a:
                    continue
                conf_dir = reps[b]
                conf_unit = F.normalize(conf_dir, dim=-1, eps=1e-8)
                d_unit_before = F.normalize(d, dim=-1, eps=1e-8)
                cos_before = (d_unit_before * conf_unit.unsqueeze(0)).sum(dim=-1)
                d = project_out_direction(d, conf_dir)
                d_unit_after = F.normalize(d, dim=-1, eps=1e-8)
                cos_after = (d_unit_after * conf_unit.unsqueeze(0)).sum(dim=-1)
                if cos_before.abs().mean() > 0.02:
                    print(f"  attr{a} vs attr{b}: |cos| before mean={cos_before.abs().mean():.4f} "
                          f"max={cos_before.abs().max():.4f}  -> after "
                          f"mean={cos_after.abs().mean():.4f} max={cos_after.abs().max():.4f}")
            dirs_by_idx[a] = d
        # Recompute representative directions are stale after this loop
        # (each attr was cleaned using the ORIGINAL reps of the others,
        # standard one-pass approximate decorrelation -- a fully joint
        # solve would require iterating to convergence, not needed here
        # since residual cross-cosines after one pass are already ~0, see
        # printed diagnostics above).

    all_dirs = torch.stack([dirs_by_idx[a] for a in args.attribute_index], dim=0)
    print(f"\nAll directions shape: {tuple(all_dirs.shape)}")

    # Intra-attribute orthogonalization for glasses, gender, and any generic
    # extra attributes. Age is skipped when age_k=1 (tiled identical
    # directions would collapse to zero).
    print("Intra-attribute orthogonalization (glasses, gender, extra attrs) ...")
    attr_index_map = {idx: i for i, idx in enumerate(args.attribute_index)}
    for attr_idx in [15, 20] + extra_attr_ids:
        if attr_idx in attr_index_map:
            a = attr_index_map[attr_idx]
            all_dirs[a] = intra_attr_orthogonalize_safe(all_dirs[a])
    if not skip_age_orth and 39 in attr_index_map:
        a = attr_index_map[39]
        all_dirs[a] = intra_attr_orthogonalize_safe(all_dirs[a])
        print("Intra-attribute orthogonalization (age) ...")
    else:
        print("Age orthogonalization: SKIPPED (age_k=1, tiled direction)")

    all_dirs = sanitize_non_finite_directions(
        all_dirs, args.attribute_index, latents, continuous,
        pct=args.extreme_pct, method=args.direction_method, shrinkage=args.shrinkage,
        group_center=args.group_center, group_trim_frac=args.group_trim_frac,
    )

    # Norms and unit vectors
    layer_norms = all_dirs.norm(dim=-1)                    # (num_attrs, K, 18)
    direction_units = F.normalize(all_dirs, dim=-1, eps=1e-8)  # (num_attrs, K, 18, 512)

    # Intra-attr cosine similarity check (glasses, gender, extra attrs)
    print("\n=== Intra-attribute cosine similarity after orthogonalization ===")
    for attr_idx in [15, 20] + extra_attr_ids:
        if attr_idx not in attr_index_map:
            continue
        a = attr_index_map[attr_idx]
        for layer in [0, 8, 17]:
            for i in range(K):
                for j in range(i + 1, K):
                    cos = F.cosine_similarity(
                        direction_units[a, i, layer].unsqueeze(0),
                        direction_units[a, j, layer].unsqueeze(0),
                    ).item()
                    if abs(cos) > 0.1:
                        print(f"  attr {attr_idx} layer {layer}: K{i} vs K{j}: {cos:.4f}")

    bank = {
        "direction_units": direction_units,   # (num_attrs, K, 18, 512)
        "layer_norms": layer_norms,           # (num_attrs, K, 18)
        "num_k": K,
        "attribute_index": args.attribute_index,
        "age_k": age_k,
        "direction_method": args.direction_method,
        "extreme_pct": args.extreme_pct,
        "decorrelate_cross_attr": bool(args.decorrelate_cross_attr),
        "stratification": {
            15: ["male_young", "male_old", "female_young", "female_old"],
            20: ["young_noglasses", "young_glasses", "old_noglasses", "old_glasses"],
            39: ["weighted_avg_k1"] * K if age_k == 1 else
                ["male_noglasses", "male_glasses", "female_noglasses", "female_glasses"],
            **{a: ["male_young", "male_old", "female_young", "female_old"] for a in extra_attr_ids},
        },
    }
    # Cross-attribute cosine similarity (K0 representative vs K0 representative)
    num_attrs = len(args.attribute_index)
    print("\n=== Cross-attribute cosine similarity (representative K0, avg over layers) ===")
    reps = [direction_units[a, 0] for a in range(num_attrs)]   # list of (L, D)
    for i in range(num_attrs):
        for j in range(i + 1, num_attrs):
            cos_per_layer = (reps[i] * reps[j]).sum(dim=-1)    # (L,)
            print(
                f"  attr {args.attribute_index[i]} vs attr {args.attribute_index[j]}: "
                f"mean={cos_per_layer.mean():.4f}  max={cos_per_layer.abs().max():.4f}"
            )

    torch.save(bank, args.output)
    print(f"\nSaved → {args.output}")

    print("\n=== Summary ===")
    attr_names = {15: "glasses", 20: "gender", 39: "age/young",
                  24: "no_beard", 31: "smiling", 33: "wavy_hair"}
    strat = bank["stratification"]
    for a, attr_idx in enumerate(args.attribute_index):
        norms_k = layer_norms[a].mean(dim=-1)   # (K,) mean over 18 layers
        labels = strat.get(attr_idx, [f"K{k}" for k in range(K)])
        print(f"\nAttr {attr_idx} ({attr_names.get(attr_idx, '?')}):")
        for k in range(K):
            label = labels[k] if k < len(labels) else f"K{k}"
            print(f"  K{k} [{label}]: mean_norm={norms_k[k]:.4f}")


if __name__ == "__main__":
    main()
