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

The stratification conditioning variables (e.g. gender/glasses used to
define strata for the age direction) still use the classifier's binary 0/1
predictions -- those are meant to be discrete groups, and only the target
attribute's own high/low split benefits from continuous percentile
selection.
"""

import argparse
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

def extreme_masks(scores, pct=20.0):
    """Boolean high/low masks from the top/bottom `pct` percent of a
    continuous score.

    Replaces a fixed 0.5 decision-boundary split so the direction is
    computed from confidently-labeled samples instead of ones near the
    classifier's most ambiguous region, where label noise is worst.
    """
    if scores.numel() == 0:
        empty = torch.zeros_like(scores, dtype=torch.bool)
        return empty, empty
    hi_thresh = torch.quantile(scores, 1.0 - pct / 100.0)
    lo_thresh = torch.quantile(scores, pct / 100.0)
    mask_high = scores >= hi_thresh
    mask_low = scores <= lo_thresh
    return mask_high, mask_low


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
                       method="lda", shrinkage=None):
    """Direction between two subsets of latents.

    method="mean_diff": high_mean - low_mean (legacy behaviour).
    method="lda": per-layer Fisher/whitened direction using a Ledoit-Wolf
        shrinkage covariance, rescaled to the plain mean-diff's per-layer
        norm. Whitening accounts for the latent space's covariance
        structure instead of only comparing group means, which is the main
        source of imprecision in a raw mean-diff direction; rescaling keeps
        the physical displacement magnitude (used by layer_norms /
        magnitude_net) unchanged so only the angle is affected.
    """
    n_high = int(mask_high.sum().item())
    n_low = int(mask_low.sum().item())
    if n_high < min_samples or n_low < min_samples:
        return None, n_high, n_low

    high = latents[mask_high]
    low = latents[mask_low]
    mean_diff = high.mean(dim=0) - low.mean(dim=0)   # (L, D)
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
        ridge = 1e-4 * cov.diagonal().mean().clamp(min=1e-8)
        cov = cov + ridge * torch.eye(dim, dtype=cov.dtype, device=cov.device)
        w = torch.linalg.solve(cov, mean_diff[layer].unsqueeze(1)).squeeze(1)
        lda_dir[layer] = w

    lda_norm = lda_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    mean_norm = mean_diff.norm(dim=-1, keepdim=True)
    rescaled = lda_dir / lda_norm * mean_norm
    return rescaled, n_high, n_low


def _fallback_direction(latents, attr_scores, pct=20.0, method="lda", shrinkage=None):
    """Unconditional direction as fallback, using percentile extremes of a
    continuous score instead of a fixed 0.5 split."""
    mask_high, mask_low = extreme_masks(attr_scores, pct=pct)
    d, _, _ = compute_direction(latents, mask_high, mask_low, method=method, shrinkage=shrinkage)
    return d


# ---------------------------------------------------------------------------
# Per-attribute stratified directions
# ---------------------------------------------------------------------------

def compute_glasses_directions(latents, preds, continuous, K=4, min_samples=50,
                                pct=20.0, method="lda", shrinkage=None):
    """
    Attr 15 (Eyeglasses), conditioned on gender x age (binary strata).
      K0: male   x young    (gender=1, young=1)
      K1: male   x old      (gender=1, young=0)
      K2: female x young    (gender=0, young=1)
      K3: female x old      (gender=0, young=0)
    High/low split uses the top/bottom `pct`% of the continuous glasses score.
    """
    glasses_cont = continuous[:, 15]
    gender = preds[:, 20]
    young = preds[:, 39]

    strata = [
        ("male_young",   (gender == 1) & (young == 1)),
        ("male_old",     (gender == 1) & (young == 0)),
        ("female_young", (gender == 0) & (young == 1)),
        ("female_old",   (gender == 0) & (young == 0)),
    ]

    directions = []
    for name, sub_mask in strata:
        sub_lat = latents[sub_mask]
        sub_scores = glasses_cont[sub_mask]
        mask_high, mask_low = extreme_masks(sub_scores, pct=pct)
        d, nh, nl = compute_direction(sub_lat, mask_high, mask_low, min_samples,
                                       method=method, shrinkage=shrinkage)
        if d is not None:
            print(f"  glasses/{name}: high={nh}, low={nl}, norm={d.norm():.3f}")
            directions.append(d)
        else:
            print(f"  glasses/{name}: FAILED (high={nh}, low={nl}) -> using fallback")
            directions.append(_fallback_direction(latents, glasses_cont, pct, method, shrinkage))

    return torch.stack(directions[:K])   # (K, 18, 512)


def compute_gender_directions(latents, preds, continuous, K=4, min_samples=50,
                               pct=20.0, method="lda", shrinkage=None):
    """
    Attr 20 (Male), conditioned on age x glasses (binary strata).
      K0: young x no-glasses  (young=1, glasses=0)
      K1: young x glasses     (young=1, glasses=1)
      K2: old   x no-glasses  (young=0, glasses=0)
      K3: old   x glasses     (young=0, glasses=1)
    High/low split uses the top/bottom `pct`% of the continuous gender score.
    """
    gender_cont = continuous[:, 20]
    young = preds[:, 39]
    glasses = preds[:, 15]

    strata = [
        ("young_noglasses", (young == 1) & (glasses == 0)),
        ("young_glasses",   (young == 1) & (glasses == 1)),
        ("old_noglasses",   (young == 0) & (glasses == 0)),
        ("old_glasses",     (young == 0) & (glasses == 1)),
    ]

    directions = []
    for name, sub_mask in strata:
        sub_lat = latents[sub_mask]
        sub_scores = gender_cont[sub_mask]
        mask_high, mask_low = extreme_masks(sub_scores, pct=pct)
        d, nh, nl = compute_direction(sub_lat, mask_high, mask_low, min_samples,
                                       method=method, shrinkage=shrinkage)
        if d is not None:
            print(f"  gender/{name}: high={nh}, low={nl}, norm={d.norm():.3f}")
            directions.append(d)
        else:
            print(f"  gender/{name}: FAILED (high={nh}, low={nl}) -> using fallback")
            directions.append(_fallback_direction(latents, gender_cont, pct, method, shrinkage))

    return torch.stack(directions[:K])


def compute_age_directions(latents, preds, continuous, K=4, min_samples=50,
                            pct=20.0, method="lda", shrinkage=None):
    """
    Attr 39 (Young), conditioned on gender x glasses (binary strata).
      K0: male   x no-glasses  (gender=1, glasses=0)
      K1: male   x glasses     (gender=1, glasses=1)
      K2: female x no-glasses  (gender=0, glasses=0)
      K3: female x glasses     (gender=0, glasses=1)
    High/low split uses the top/bottom `pct`% of the continuous age score.
    """
    age_cont = continuous[:, 39]
    gender = preds[:, 20]
    glasses = preds[:, 15]

    strata = [
        ("male_noglasses",   (gender == 1) & (glasses == 0)),
        ("male_glasses",     (gender == 1) & (glasses == 1)),
        ("female_noglasses", (gender == 0) & (glasses == 0)),
        ("female_glasses",   (gender == 0) & (glasses == 1)),
    ]

    directions = []
    for name, sub_mask in strata:
        sub_lat = latents[sub_mask]
        sub_scores = age_cont[sub_mask]
        mask_high, mask_low = extreme_masks(sub_scores, pct=pct)
        d, nh, nl = compute_direction(sub_lat, mask_high, mask_low, min_samples,
                                       method=method, shrinkage=shrinkage)
        if d is not None:
            print(f"  age/{name}: high={nh}, low={nl}, norm={d.norm():.3f}")
            directions.append(d)
        else:
            print(f"  age/{name}: FAILED (high={nh}, low={nl}) -> using fallback")
            directions.append(_fallback_direction(latents, age_cont, pct, method, shrinkage))

    return torch.stack(directions[:K])


def compute_age_k1_stratified(latents, preds, continuous, min_samples=50,
                               pct=20.0, method="lda", shrinkage=None):
    """Debiased K=1 age direction via stratum-size-weighted average.

    Computes 4 sub-directions conditioned on gender x glasses, then averages
    them weighted by stratum size. This removes gender/glasses leakage
    without the norm collapse caused by intra-attr orthogonalization.

    Uses the same generic direction primitive (LDA/whitened + percentile
    split) as glasses/gender -- no age-only special-casing beyond the
    stratum-size-weighted averaging strategy itself, which is a K-selection
    choice independent of how each sub-direction is computed.

    Returns: (1, 18, 512) -- single direction, ready to be tiled to fill K slots.
    """
    age_cont = continuous[:, 39]
    gender = preds[:, 20]
    glasses = preds[:, 15]

    strata = [
        ("male_noglasses",   (gender == 1) & (glasses == 0)),
        ("male_glasses",     (gender == 1) & (glasses == 1)),
        ("female_noglasses", (gender == 0) & (glasses == 0)),
        ("female_glasses",   (gender == 0) & (glasses == 1)),
    ]

    sub_dirs, weights = [], []
    for name, sub_mask in strata:
        sub_lat = latents[sub_mask]
        sub_scores = age_cont[sub_mask]
        mask_high, mask_low = extreme_masks(sub_scores, pct=pct)
        d, nh, nl = compute_direction(sub_lat, mask_high, mask_low, min_samples,
                                       method=method, shrinkage=shrinkage)
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

def intra_attr_orthogonalize_safe(directions, iters=5, min_norm_ratio=0.1):
    """Symmetric orthogonalization with fallback for near-degenerate cases.

    If a direction collapses below min_norm_ratio of its original norm after
    orthogonalization, fall back to the original (unorthogonalized) direction.
    This happens when two strata produce nearly identical direction vectors.

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
        # Fallback: if norm collapsed, keep original direction
        new_norms = orth.norm(dim=-1)
        orig_norms_flat = orig_norms.squeeze(-1)
        collapsed = new_norms < min_norm_ratio * orig_norms_flat
        if collapsed.any():
            orth[collapsed] = original[:, layer][collapsed]
        result[:, layer] = orth
    return result


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
    parser.add_argument("--output", default="./data/direction_bank_k4_stratified.pth")
    parser.add_argument("--min_samples", type=int, default=50,
                        help="Minimum samples per high/low group. NOTE: with --extreme_pct, each "
                             "stratum's usable pool shrinks to roughly 2*extreme_pct%% of its size, "
                             "so small strata may need a lower --min_samples (e.g. 20-30) or a "
                             "larger --extreme_pct to avoid falling back to the unconditional direction.")
    parser.add_argument("--attribute_index", nargs="*", type=int, default=[15, 20, 39])
    args = parser.parse_args()

    if not (0 < args.extreme_pct < 50):
        parser.error("--extreme_pct must be between 0 and 50 (exclusive) so high/low groups don't overlap.")

    K = args.K
    age_k = args.age_k
    if age_k > K:
        parser.error(f"--age_k ({age_k}) cannot exceed --K ({K}); age_dirs would end up with more "
                     "rows than glasses_dirs/gender_dirs and torch.stack(...) would fail.")

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

    print(f"\nmethod={args.direction_method}, extreme_pct={args.extreme_pct}, "
          f"K glasses/gender={K}, K age={age_k}, residual_age={args.residual_age}")

    common_kwargs = dict(pct=args.extreme_pct, method=args.direction_method, shrinkage=args.shrinkage)

    print(f"\n=== Eyeglasses (attr 15), K={K} ===")
    glasses_dirs = compute_glasses_directions(latents, preds, continuous, K, args.min_samples, **common_kwargs)

    print(f"\n=== Gender / Male (attr 20), K={K} ===")
    gender_dirs = compute_gender_directions(latents, preds, continuous, K, args.min_samples, **common_kwargs)

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

    print(f"\n=== Age / Young (attr 39), K={age_k} ===")
    if age_k == 1:
        age_dir_k1 = compute_age_k1_stratified(latents_for_age, preds, continuous, args.min_samples, **common_kwargs)
        age_dirs = age_dir_k1.expand(K, -1, -1).clone()   # (K, L, D) — tiled
        skip_age_orth = True
    else:
        age_dirs = compute_age_directions(latents_for_age, preds, continuous, age_k, args.min_samples, **common_kwargs)
        if age_k < K:
            pad = age_dirs[-1:].expand(K - age_k, -1, -1).clone()
            age_dirs = torch.cat([age_dirs, pad], dim=0)
        skip_age_orth = False

    # Stack: (num_attrs, K, 18, 512)
    all_dirs = torch.stack([glasses_dirs, gender_dirs, age_dirs], dim=0)
    print(f"\nAll directions shape: {tuple(all_dirs.shape)}")

    # Intra-attribute orthogonalization for glasses and gender only.
    # Age is skipped when age_k=1 (tiled identical directions would collapse to zero).
    print("Intra-attribute orthogonalization (glasses, gender) ...")
    attr_index_map = {idx: i for i, idx in enumerate(args.attribute_index)}
    for attr_idx in [15, 20]:
        if attr_idx in attr_index_map:
            a = attr_index_map[attr_idx]
            all_dirs[a] = intra_attr_orthogonalize_safe(all_dirs[a])
    if not skip_age_orth and 39 in attr_index_map:
        a = attr_index_map[39]
        all_dirs[a] = intra_attr_orthogonalize_safe(all_dirs[a])
        print("Intra-attribute orthogonalization (age) ...")
    else:
        print("Age orthogonalization: SKIPPED (age_k=1, tiled direction)")

    # Norms and unit vectors
    layer_norms = all_dirs.norm(dim=-1)                    # (num_attrs, K, 18)
    direction_units = F.normalize(all_dirs, dim=-1, eps=1e-8)  # (num_attrs, K, 18, 512)

    # Intra-attr cosine similarity check (glasses and gender only)
    print("\n=== Intra-attribute cosine similarity after orthogonalization ===")
    for attr_idx in [15, 20]:
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
        "stratification": {
            15: ["male_young", "male_old", "female_young", "female_old"],
            20: ["young_noglasses", "young_glasses", "old_noglasses", "old_glasses"],
            39: ["weighted_avg_k1"] * K if age_k == 1 else
                ["male_noglasses", "male_glasses", "female_noglasses", "female_glasses"],
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
    attr_names = {15: "glasses", 20: "gender", 39: "age/young"}
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
