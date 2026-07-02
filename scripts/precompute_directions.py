import argparse

import numpy as np
import torch
import torch.nn.functional as F


def load_tensor_from_file(path):
    if path.endswith(".npy"):
        return torch.from_numpy(np.load(path))
    return torch.load(path, map_location="cpu")


def extract_tensor(obj, keys, name):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                value = obj[key]
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value)
                if torch.is_tensor(value):
                    return value
    raise ValueError(f"Could not find {name} tensor. Tried keys: {keys}")


def extract_latents(obj):
    latents = extract_tensor(obj, ["latents", "latent", "w", "w_plus", "values"], "latents").float()
    if latents.dim() == 4 and latents.size(1) == 1:
        latents = latents.squeeze(1)
    if latents.dim() != 3 or latents.shape[1:] != (18, 512):
        raise ValueError(f"Expected latents [N,18,512], got {tuple(latents.shape)}")
    return latents


def extract_preds(obj):
    preds = extract_tensor(obj, ["preds", "predictions", "labels", "attrs", "attributes", "values"], "preds").float()
    if preds.dim() != 2:
        raise ValueError(f"Expected predictions [N,num_attrs], got {tuple(preds.shape)}")
    return preds


def _split_high_low(latents, scores, low_thresh, high_thresh, min_count):
    """Return (high_latents, low_mean) splitting by score threshold."""
    high_mask = scores > high_thresh
    low_mask = scores < low_thresh
    if high_mask.sum() < min_count or low_mask.sum() < min_count:
        k = max(min_count, int(0.2 * len(scores)))
        _, high_idx = scores.topk(k)
        _, low_idx = scores.topk(k, largest=False)
        high_mask = torch.zeros_like(scores, dtype=torch.bool)
        low_mask = torch.zeros_like(scores, dtype=torch.bool)
        high_mask[high_idx] = True
        low_mask[low_idx] = True

    high_count = int(high_mask.sum().item())
    low_count = int(low_mask.sum().item())
    if high_count < min_count or low_count < min_count:
        raise ValueError(f"Not enough samples: high={high_count}, low={low_count}, min={min_count}")

    return latents[high_mask], latents[low_mask].mean(dim=0), high_count, low_count


def compute_raw_direction(latents, scores, low_thresh=0.3, high_thresh=0.7, min_count=10):
    """Single direction: high-mean minus low-mean. Returns (18, 512)."""
    high_latents, low_mean, high_count, low_count = _split_high_low(
        latents, scores, low_thresh, high_thresh, min_count
    )
    return high_latents.mean(dim=0) - low_mean, high_count, low_count


def align_continuous_age_direction_to_young_attr(direction, attr_idx, continuous_age_scores, invert=True):
    """Convert continuous-age old-minus-young direction to CelebA Young semantics.

    Continuous age scores are higher for older faces, so high-mean minus low-mean
    gives an old direction. Attribute 39 in this project is CelebA Young, where
    positive attr_delta means "more young". Direction Bank multiplies directions
    by attr_delta, so attr 39 must store young-minus-old to stay consistent.
    """
    if attr_idx == 39 and continuous_age_scores is not None and invert:
        print("  [age] inverted continuous age direction to match CelebA Young attr semantics")
        return -direction
    return direction


def compute_k_directions(latents, scores, num_k, low_thresh=0.3, high_thresh=0.7, min_count=10):
    """K directions per attribute via K-means on the high-score latent pool.

    Returns:
        directions: (K, 18, 512) float tensor
        high_count: int
        low_count: int
    """
    from sklearn.cluster import KMeans

    high_latents, low_mean, high_count, low_count = _split_high_low(
        latents, scores, low_thresh, high_thresh, min_count
    )

    # Use mean-pooled latent for clustering to keep dimensionality manageable
    X = high_latents.mean(dim=1).numpy()   # (N_high, 512)

    effective_k = min(num_k, high_count)
    if effective_k < num_k:
        print(f"  warning: only {high_count} high samples, reducing K from {num_k} to {effective_k}")

    km = KMeans(n_clusters=effective_k, n_init=10, random_state=0)
    labels = km.fit_predict(X)

    dirs = []
    for k in range(effective_k):
        cluster_mask = labels == k
        if cluster_mask.sum() == 0:
            dirs.append(high_latents.mean(dim=0) - low_mean)
        else:
            cluster_mean = high_latents[cluster_mask].mean(dim=0)
            dirs.append(cluster_mean - low_mean)
    dirs = torch.stack(dirs, dim=0)   # (K, 18, 512)

    # Pad to num_k if effective_k < num_k (shouldn't happen normally)
    if effective_k < num_k:
        pad = dirs[:1].repeat(num_k - effective_k, 1, 1)
        dirs = torch.cat([dirs, pad], dim=0)

    return dirs, high_count, low_count


def orthogonalize_directions(directions):
    """Gram-Schmidt orthogonalization per layer.

    Accepts either:
        (num_dirs, num_layers, latent_dim)   — flat list of directions
        (num_attrs, num_k, num_layers, latent_dim) — will be flattened and reshaped back
    """
    original_shape = directions.shape
    if directions.ndim == 4:
        num_attrs, num_k, num_layers, latent_dim = directions.shape
        flat = directions.reshape(num_attrs * num_k, num_layers, latent_dim)
    else:
        flat = directions

    num_dirs, num_layers, _ = flat.shape
    result = flat.clone()
    for layer in range(num_layers):
        for i in range(num_dirs):
            v = result[i, layer].clone()
            for j in range(i):
                u = result[j, layer]
                v = v - ((v @ u) / (u @ u + 1e-8)) * u
            result[i, layer] = v

    if directions.ndim == 4:
        return result.reshape(original_shape)
    return result


def remove_direction_components(w_all, unit_directions):
    """Remove each direction's projection from w_all, per layer.

    w_all:           (N, n_layers, 512)
    unit_directions: list of (n_layers, 512) unit-normalized tensors
    Returns cleaned (N, n_layers, 512).
    """
    w_clean = w_all.clone()
    for d_unit in unit_directions:
        for layer_idx in range(w_all.shape[1]):
            d_hat = d_unit[layer_idx]                                          # (512,)
            proj = (w_clean[:, layer_idx] @ d_hat).unsqueeze(1) * d_hat       # (N, 512)
            w_clean[:, layer_idx] = w_clean[:, layer_idx] - proj
    return w_clean


def print_direction_stats(label, dirs_k, attr_idx):
    norms_k = dirs_k.norm(dim=-1)   # (K, 18)
    print(
        f"  {label} attr {attr_idx}: K={dirs_k.shape[0]}, "
        f"norm_mean={norms_k.mean():.4f}  "
        f"[coarse={norms_k[:, :4].mean():.4f}  "
        f"mid={norms_k[:, 4:10].mean():.4f}  "
        f"fine={norms_k[:, 10:].mean():.4f}]"
    )


def print_cosine_sim_matrix(direction_units, attribute_index):
    """Print avg-over-layers cosine similarity matrix between all attribute pairs."""
    num_attrs = len(attribute_index)
    # direction_units: (num_attrs, K, 18, 512)
    # Use K=0 slice for K=1 case
    units = direction_units[:, 0]   # (num_attrs, 18, 512)
    print("\nCross-attribute cosine similarity (avg over layers):")
    header = "         " + "".join(f"  attr{a:<4}" for a in attribute_index)
    print(header)
    for i in range(num_attrs):
        row = f"  attr{attribute_index[i]:<4}"
        for j in range(num_attrs):
            cos = F.cosine_similarity(
                units[i].reshape(-1), units[j].reshape(-1), dim=0
            ).item()
            row += f"  {cos:+.4f}  "
        print(row)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_file", default="./data/ffhq_e4e_latents.pth")
    parser.add_argument("--preds_file", default="./data/ffhq_e4e_preds.pth")
    parser.add_argument("--attribute_index", nargs="*", type=int, default=[15, 20, 39])
    parser.add_argument("--output", default="./data/direction_bank_init.pth")
    parser.add_argument("--low_thresh", type=float, default=0.3)
    parser.add_argument("--high_thresh", type=float, default=0.7)
    parser.add_argument("--min_count", type=int, default=10)
    parser.add_argument("--num_k", type=int, default=1,
                        help="Number of directions per attribute. K=1 is the original mean-diff direction.")
    parser.add_argument("--residual_age", action="store_true",
                        help="Extract age (attr 39) direction in the residual space after removing "
                             "glasses and gender components. Fixes the 60%% norm loss from Gram-Schmidt.")
    parser.add_argument("--residual_all", action="store_true",
                        help="Extract each attribute direction sequentially in the residual W+ space "
                             "of all previously computed attributes (in attribute_index order). "
                             "Produces fully orthogonal directions by construction. Supersedes --residual_age.")
    parser.add_argument("--age_scores_file", type=str, default=None,
                        help="Path to continuous age scores .pth (shape (N,), higher = older). "
                             "Overrides binary preds[:,39] for attr 39 when provided.")
    parser.add_argument("--age_low_pct", type=float, default=0.15,
                        help="Percentile below which samples are treated as young (used with --age_scores_file).")
    parser.add_argument("--age_high_pct", type=float, default=0.85,
                        help="Percentile above which samples are treated as old (used with --age_scores_file).")
    parser.add_argument("--invert_continuous_age_direction",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="When using --age_scores_file for attr 39, invert old-young into "
                             "young-old so the direction matches CelebA Young semantics.")
    args = parser.parse_args()

    latents = extract_latents(load_tensor_from_file(args.latent_file))
    preds = extract_preds(load_tensor_from_file(args.preds_file))
    if latents.size(0) != preds.size(0):
        raise ValueError(f"Latents/preds length mismatch: {latents.size(0)} vs {preds.size(0)}")

    continuous_age_scores = None
    if args.age_scores_file is not None:
        _age_data = torch.load(args.age_scores_file, map_location="cpu")
        continuous_age_scores = _age_data["values"] if isinstance(_age_data, dict) else _age_data
        if continuous_age_scores.shape[0] != latents.shape[0]:
            raise ValueError(
                f"age_scores length mismatch: {continuous_age_scores.shape[0]} vs {latents.shape[0]}"
            )
        _lo = torch.quantile(continuous_age_scores, args.age_low_pct).item()
        _hi = torch.quantile(continuous_age_scores, args.age_high_pct).item()
        _n_young = (continuous_age_scores < _lo).sum().item()
        _n_old   = (continuous_age_scores > _hi).sum().item()
        print(f"[age] continuous scores: min={continuous_age_scores.min():.3f}  "
              f"max={continuous_age_scores.max():.3f}  "
              f"lo={_lo:.3f} (p{args.age_low_pct*100:.0f}, {_n_young} young)  "
              f"hi={_hi:.3f} (p{args.age_high_pct*100:.0f}, {_n_old} old)")

    print(f"latents: {tuple(latents.shape)}")
    print(f"preds:   {tuple(preds.shape)}")
    print(f"attrs:   {args.attribute_index}  K={args.num_k}")
    if args.residual_all:
        print("mode:    residual_all (each attr extracted in residual W+ of all prior attrs)")
    elif args.residual_age:
        print("mode:    residual_age (attr 39 extracted in anchor-cleaned W+ space)")
    print()

    use_residual_all = args.residual_all
    use_residual_age = args.residual_age and not use_residual_all and 39 in args.attribute_index
    anchor_attrs = [a for a in args.attribute_index if a != 39]  # [15, 20]

    if use_residual_all and args.num_k != 1:
        raise ValueError("--residual_all only supports --num_k 1")
    if use_residual_age and args.num_k != 1:
        raise ValueError("--residual_age only supports --num_k 1")

    raw_dirs_map = {}   # attr_idx -> (K, 18, 512)
    counts = {}

    if use_residual_all:
        # ── Residual-all: each attr computed in W+ with all prior attrs projected out ─
        print("=== Residual-all: sequential W+ cleaning ===")
        accumulated_units = []   # (18, 512) unit directions accumulated so far
        for attr_idx in args.attribute_index:
            if accumulated_units:
                latents_to_use = remove_direction_components(latents, accumulated_units)
                print(f"  attr {attr_idx}: projecting out {len(accumulated_units)} prior direction(s)")
            else:
                latents_to_use = latents
                print(f"  attr {attr_idx}: using original W+")

            if attr_idx == 39 and continuous_age_scores is not None:
                scores = continuous_age_scores
                lo = torch.quantile(scores, args.age_low_pct).item()
                hi = torch.quantile(scores, args.age_high_pct).item()
            else:
                scores = preds[:, attr_idx]
                lo, hi = args.low_thresh, args.high_thresh
            direction, high_count, low_count = compute_raw_direction(
                latents_to_use, scores,
                low_thresh=lo,
                high_thresh=hi,
                min_count=args.min_count,
            )
            direction = align_continuous_age_direction_to_young_attr(
                direction,
                attr_idx,
                continuous_age_scores,
                invert=args.invert_continuous_age_direction,
            )
            dirs_k = direction.unsqueeze(0)   # (1, 18, 512)
            raw_dirs_map[attr_idx] = dirs_k
            counts[int(attr_idx)] = {"high": high_count, "low": low_count}
            print(f"  high={high_count}  low={low_count}")
            print_direction_stats("residual", dirs_k, attr_idx)
            accumulated_units.append(F.normalize(direction, dim=-1, eps=1e-8))

        # Compare to Gram-Schmidt for reference
        all_raw_ref = torch.stack([raw_dirs_map[a] for a in args.attribute_index], dim=0)
        naive_orth = orthogonalize_directions(all_raw_ref)
        print("\nGram-Schmidt reference norms (for comparison):")
        for i, attr_idx in enumerate(args.attribute_index):
            print_direction_stats("gram-schmidt", naive_orth[i], attr_idx)

    else:
        # ── Step 1: Extract all non-age directions from original W+ ───────────────
        attrs_first_pass = anchor_attrs if use_residual_age else args.attribute_index
        print("=== Pass 1: anchor directions (original W+) ===")
        for attr_idx in attrs_first_pass:
            if attr_idx == 39 and continuous_age_scores is not None:
                scores = continuous_age_scores
                lo = torch.quantile(scores, args.age_low_pct).item()
                hi = torch.quantile(scores, args.age_high_pct).item()
            else:
                scores = preds[:, attr_idx]
                lo, hi = args.low_thresh, args.high_thresh

            if args.num_k == 1:
                direction, high_count, low_count = compute_raw_direction(
                    latents, scores,
                    low_thresh=lo,
                    high_thresh=hi,
                    min_count=args.min_count,
                )
                direction = align_continuous_age_direction_to_young_attr(
                    direction,
                    attr_idx,
                    continuous_age_scores,
                    invert=args.invert_continuous_age_direction,
                )
                dirs_k = direction.unsqueeze(0)   # (1, 18, 512)
            else:
                dirs_k, high_count, low_count = compute_k_directions(
                    latents, scores,
                    num_k=args.num_k,
                    low_thresh=lo,
                    high_thresh=hi,
                    min_count=args.min_count,
                )
                dirs_k = align_continuous_age_direction_to_young_attr(
                    dirs_k,
                    attr_idx,
                    continuous_age_scores,
                    invert=args.invert_continuous_age_direction,
                )

            raw_dirs_map[attr_idx] = dirs_k
            counts[int(attr_idx)] = {"high": high_count, "low": low_count}
            print(f"  high={high_count}  low={low_count}")
            print_direction_stats("raw", dirs_k, attr_idx)

        # ── Step 2 (residual_age only): clean W+ and extract age in residual space ─
        if use_residual_age:
            print("\n=== Pass 2: age residual direction (anchor-cleaned W+) ===")

            anchor_tensor = torch.stack(
                [raw_dirs_map[a] for a in anchor_attrs], dim=0
            )   # (num_anchors, 1, 18, 512)
            anchor_orth = orthogonalize_directions(anchor_tensor)
            anchor_units = F.normalize(anchor_orth, dim=-1, eps=1e-8)
            anchor_unit_list = [anchor_units[i, 0] for i in range(len(anchor_attrs))]

            print(f"  projecting out {anchor_attrs} from {latents.shape[0]} latents...")
            latents_clean = remove_direction_components(latents, anchor_unit_list)

            if continuous_age_scores is not None:
                scores_age = continuous_age_scores
                lo_age = torch.quantile(scores_age, args.age_low_pct).item()
                hi_age = torch.quantile(scores_age, args.age_high_pct).item()
            else:
                scores_age = preds[:, 39]
                lo_age, hi_age = args.low_thresh, args.high_thresh
            direction_age, high_count, low_count = compute_raw_direction(
                latents_clean, scores_age,
                low_thresh=lo_age,
                high_thresh=hi_age,
                min_count=args.min_count,
            )
            direction_age = align_continuous_age_direction_to_young_attr(
                direction_age,
                39,
                continuous_age_scores,
                invert=args.invert_continuous_age_direction,
            )
            age_dirs_k = direction_age.unsqueeze(0)   # (1, 18, 512)
            raw_dirs_map[39] = age_dirs_k
            counts[39] = {"high": high_count, "low": low_count}
            print(f"  high={high_count}  low={low_count}")
            print_direction_stats("residual", age_dirs_k, 39)

            all_raw = torch.stack([raw_dirs_map[a] for a in args.attribute_index], dim=0)
            naive_orth = orthogonalize_directions(all_raw)
            age_pos = args.attribute_index.index(39)
            print_direction_stats("gram-schmidt (reference)", naive_orth[age_pos], 39)

    # ── Step 3: Assemble final directions in attribute_index order ─────────────
    raw_dirs = torch.stack(
        [raw_dirs_map[a] for a in args.attribute_index], dim=0
    )   # (num_attrs, K, 18, 512)

    # Residual directions (residual_all / residual_age) are near-orthogonal by construction;
    # run Gram-Schmidt to guarantee exact orthogonality across all modes.
    orth_dirs = orthogonalize_directions(raw_dirs)

    layer_norms = orth_dirs.norm(dim=-1)              # (num_attrs, K, 18)
    direction_units = F.normalize(orth_dirs, dim=-1, eps=1e-8)

    # ── Step 4: Diagnostics ───────────────────────────────────────────────────
    print("\n=== Final direction norms (after orthogonalization) ===")
    for i, attr_idx in enumerate(args.attribute_index):
        print_direction_stats("final", orth_dirs[i], attr_idx)

    print_cosine_sim_matrix(direction_units, args.attribute_index)

    # Warn if any pair is still significantly non-orthogonal
    num_attrs = len(args.attribute_index)
    for layer in range(18):
        for i in range(num_attrs):
            for j in range(i + 1, num_attrs):
                for ki in range(args.num_k):
                    for kj in range(args.num_k):
                        cos = F.cosine_similarity(
                            direction_units[i, ki, layer],
                            direction_units[j, kj, layer],
                            dim=0,
                        ).item()
                        if abs(cos) > 0.05:
                            print(
                                f"warning: layer {layer}, attr {args.attribute_index[i]}[k={ki}] vs "
                                f"attr {args.attribute_index[j]}[k={kj}] cos={cos:.4f}"
                            )

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    bank = {
        "direction_units": direction_units,   # (num_attrs, K, 18, 512)
        "layer_norms": layer_norms,            # (num_attrs, K, 18)
        "num_k": args.num_k,
        "attribute_index": [int(x) for x in args.attribute_index],
        "raw_directions": raw_dirs,
        "counts": counts,
        "low_thresh": args.low_thresh,
        "high_thresh": args.high_thresh,
        "source_latent_file": args.latent_file,
        "source_preds_file": args.preds_file,
        "continuous_age_scores_file": args.age_scores_file,
        "age_low_pct": args.age_low_pct if args.age_scores_file else None,
        "age_high_pct": args.age_high_pct if args.age_scores_file else None,
        "invert_continuous_age_direction": (
            args.invert_continuous_age_direction if args.age_scores_file else None
        ),
    }
    torch.save(bank, args.output)
    print(f"saved: {args.output}  shape={tuple(direction_units.shape)}")


if __name__ == "__main__":
    main()
