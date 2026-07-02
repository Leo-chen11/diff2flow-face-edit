"""
Compare strength-sweep CSV outputs from original SDFlow and newer runs.

Example:
    python evaluation/compare_strength_sweeps.py \
        --baseline Original=./output/compare/original/metrics_summary.csv \
        --run Ours=./output/compare/v20/metrics_summary.csv \
        --output_dir ./output/compare/original_vs_v20

Inputs are the metrics_summary.csv files produced by:
  - evaluation/eval_original_sdflow_strength_sweep.py
  - evaluation/eval_strength_sweep.py
"""

import argparse
import csv
import os
from collections import defaultdict


METRICS = [
    "target_success",
    "effective_success",
    "target_gain",
    "target_gain_norm",
    "leakage_l1",
    "preserve_acc",
    "id_sim_real",
    "delta_rms",
    "delta_coarse_rms",
    "balanced_score",
    "practical_score",
]


HIGHER_IS_BETTER = {
    "target_success",
    "effective_success",
    "target_gain",
    "target_gain_norm",
    "preserve_acc",
    "id_sim_real",
    "balanced_score",
    "practical_score",
}


def parse_named_path(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=path/to/metrics_summary.csv")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise argparse.ArgumentTypeError("Run name cannot be empty.")
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return name, path


def read_csv(path, run_name):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out = {"run": run_name}
            for key, value in row.items():
                if key in ("attribute", "selection_mode"):
                    out[key] = value
                elif key:
                    try:
                        out[key] = float(value)
                    except (TypeError, ValueError):
                        out[key] = value
            rows.append(out)
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_key(row):
    return row["attribute"], round(float(row["strength"]), 6)


def metric_delta(metric, value, baseline_value):
    delta = float(value) - float(baseline_value)
    if metric in HIGHER_IS_BETTER:
        improved = delta > 0
    else:
        improved = delta < 0
    return delta, improved


def select_best(rows):
    by_attr = defaultdict(list)
    for row in rows:
        by_attr[row["attribute"]].append(row)

    selected = []
    for attr, attr_rows in sorted(by_attr.items()):
        strict = [r for r in attr_rows if float(r.get("passes_strict_best_filter", 0.0)) >= 0.5]
        fallback = [r for r in attr_rows if float(r.get("passes_fallback_best_filter", 0.0)) >= 0.5]
        if strict:
            pool = strict
            mode = "strict"
        elif fallback:
            pool = fallback
            mode = "fallback"
        else:
            pool = attr_rows
            mode = "unconstrained"
        best = max(pool, key=lambda r: float(r.get("practical_score", 0.0)))
        best = dict(best)
        best["selection_mode_compare"] = mode
        selected.append(best)
    return selected


def build_delta_rows(baseline_name, baseline_rows, run_rows):
    baseline_by_key = {row_key(row): row for row in baseline_rows}
    deltas = []
    for row in run_rows:
        key = row_key(row)
        base = baseline_by_key.get(key)
        if base is None:
            continue
        out = {
            "baseline": baseline_name,
            "run": row["run"],
            "attribute": row["attribute"],
            "strength": row["strength"],
        }
        for metric in METRICS:
            if metric not in row or metric not in base:
                continue
            delta, improved = metric_delta(metric, row[metric], base[metric])
            out[f"{metric}_baseline"] = base[metric]
            out[f"{metric}_run"] = row[metric]
            out[f"{metric}_delta"] = delta
            out[f"{metric}_improved"] = int(improved)
        deltas.append(out)
    return deltas


def fmt(value, digits=4):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_report(path, baseline_name, all_best_rows, delta_rows):
    by_run = defaultdict(list)
    for row in all_best_rows:
        by_run[row["run"]].append(row)

    with open(path, "w") as f:
        f.write("# SDFlow Strength Sweep Comparison\n\n")
        f.write(f"Baseline: **{baseline_name}**\n\n")

        f.write("## Recommended Strengths\n\n")
        f.write("| Run | Attribute | Strength | pScore | Eff | GainN | ID | Leak | Delta | Mode |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for run, rows in sorted(by_run.items()):
            for row in rows:
                f.write(
                    f"| {run} | {row['attribute']} | {fmt(row['strength'], 1)} "
                    f"| {fmt(row.get('practical_score'))} "
                    f"| {fmt(row.get('effective_success'), 3)} "
                    f"| {fmt(row.get('target_gain_norm'), 3)} "
                    f"| {fmt(row.get('id_sim_real'))} "
                    f"| {fmt(row.get('leakage_l1'))} "
                    f"| {fmt(row.get('delta_rms'))} "
                    f"| {row.get('selection_mode_compare', '')} |\n"
                )

        f.write("\n## Largest Practical-Score Improvements\n\n")
        ranked = sorted(
            [
                r for r in delta_rows
                if "practical_score_delta" in r and r["run"] != baseline_name
            ],
            key=lambda r: float(r["practical_score_delta"]),
            reverse=True,
        )[:12]
        f.write("| Run | Attribute | Strength | ΔpScore | ΔEff | ΔGainN | ΔID | ΔLeak | ΔDelta |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in ranked:
            f.write(
                f"| {row['run']} | {row['attribute']} | {fmt(row['strength'], 1)} "
                f"| {fmt(row.get('practical_score_delta'))} "
                f"| {fmt(row.get('effective_success_delta'), 3)} "
                f"| {fmt(row.get('target_gain_norm_delta'), 3)} "
                f"| {fmt(row.get('id_sim_real_delta'))} "
                f"| {fmt(row.get('leakage_l1_delta'))} "
                f"| {fmt(row.get('delta_rms_delta'))} |\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Compare SDFlow strength-sweep metrics_summary.csv files.")
    parser.add_argument("--baseline", required=True, type=parse_named_path,
                        help="Baseline in NAME=path format, e.g. Original=.../metrics_summary.csv")
    parser.add_argument("--run", action="append", required=True, type=parse_named_path,
                        help="Run to compare in NAME=path format. Can be repeated.")
    parser.add_argument("--output_dir", default="./output/eval_compare")
    args = parser.parse_args()

    baseline_name, baseline_path = args.baseline
    baseline_rows = read_csv(baseline_path, baseline_name)
    run_sets = [(baseline_name, baseline_rows)]
    for run_name, run_path in args.run:
        run_sets.append((run_name, read_csv(run_path, run_name)))

    all_rows = [row for _, rows in run_sets for row in rows]
    all_best = []
    for _, rows in run_sets:
        all_best.extend(select_best(rows))

    delta_rows = []
    for run_name, rows in run_sets:
        if run_name == baseline_name:
            continue
        delta_rows.extend(build_delta_rows(baseline_name, baseline_rows, rows))

    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "all_summary.csv"), all_rows)
    write_csv(os.path.join(args.output_dir, "best_by_model.csv"), all_best)
    write_csv(os.path.join(args.output_dir, "delta_vs_baseline.csv"), delta_rows)
    write_report(os.path.join(args.output_dir, "comparison_report.md"), baseline_name, all_best, delta_rows)

    print(f"Wrote {os.path.join(args.output_dir, 'all_summary.csv')}")
    print(f"Wrote {os.path.join(args.output_dir, 'best_by_model.csv')}")
    print(f"Wrote {os.path.join(args.output_dir, 'delta_vs_baseline.csv')}")
    print(f"Wrote {os.path.join(args.output_dir, 'comparison_report.md')}")


if __name__ == "__main__":
    main()
