"""List what each training run changed, from the config.json every run saves.

For a series of runs (in the order given), print each run's starting point
(from scratch, or --resume_dir @ --resume_step) and the flags whose value
differs from the previous run in the list -- i.e. what that experiment
changed. Keys that only name the run itself are skipped.

Usage:
    python -m scripts.diff_run_configs ./output/SDFlow/substyle3_glasses_v2[4-9]* \
        ./output/SDFlow/substyle3_glasses_v3*
    python -m scripts.diff_run_configs --base ./output/SDFlow/substyle3_glasses_v24 \
        ./output/SDFlow/substyle3_glasses_v3*          # every run vs one baseline
"""
import argparse
import json
import os
import re

SKIP = {'run_name', 'resume_dir', 'resume_step', 'wandb_name', 'wandb_id', 'wandb_run_id'}


def load(run):
    path = run if run.endswith('.json') else os.path.join(run, 'config.json')
    with open(path) as f:
        return json.load(f)


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


def fmt(v):
    s = json.dumps(v) if not isinstance(v, str) else v
    return s if len(s) <= 80 else s[:77] + '...'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', help='Run directories (or config.json files).')
    p.add_argument('--base', default=None, help='Compare every run to this one instead of the previous.')
    p.add_argument('--no_sort', action='store_true', help='Keep the given order (default: natural sort).')
    args = p.parse_args()

    runs = [r.rstrip('/') for r in args.runs
            if os.path.exists(r if r.endswith('.json') else os.path.join(r, 'config.json'))]
    missing = [r for r in args.runs if r.rstrip('/') not in runs]
    if missing:
        print(f'(no config.json, skipped: {" ".join(missing)})\n')
    if not args.no_sort:
        runs.sort(key=natural_key)
    base = load(args.base) if args.base else None
    prev, prev_name = None, None
    for run in runs:
        cfg = load(run)
        name = cfg.get('run_name') or os.path.basename(run)
        start = (f'resume {os.path.basename(str(cfg["resume_dir"]).rstrip("/"))} @ {cfg.get("resume_step")}'
                 if cfg.get('resume_dir') else 'from scratch')
        steps = cfg.get('max_steps')
        print(f'=== {name}  ({start}{f", max_steps {steps}" if steps else ""})')
        ref, ref_name = (base, os.path.basename(args.base.rstrip('/'))) if base else (prev, prev_name)
        if ref is None:
            print('  (first run in the list: no reference to diff against)\n')
        else:
            keys = sorted((set(cfg) | set(ref)) - SKIP)
            diffs = [(k, ref.get(k), cfg.get(k)) for k in keys if ref.get(k) != cfg.get(k)]
            if not diffs:
                print(f'  same flags as {ref_name}')
            for k, a, b in diffs:
                print(f'  {k}: {fmt(a)} -> {fmt(b)}')
            print(f'  (vs {ref_name})\n')
        prev, prev_name = cfg, name


if __name__ == '__main__':
    main()
