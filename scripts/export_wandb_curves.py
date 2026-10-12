"""Export training curves of named wandb runs to one long-format CSV
(run, step, key, value) at full resolution -- the panel "Export CSV" button
downsamples long runs.

    python scripts/export_wandb_curves.py \
        --runs SDFlow_multi_v1_anneal_seed1 SDFlow_multi_v1_replace_cont_seed1 \
        --out figures/anneal_curves.csv

Needs `wandb login` on this machine. --runs takes the run names shown in the
wandb UI (display names) or run ids.
"""
import argparse
import csv
import os

DEFAULT_KEYS = ['dir_bank_flow_bank_cos', 'dir_bank_mix', 'dir_bank_flow_share',
                'loss_bank_prior', 'bank_prior_weight']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--keys', nargs='+', default=DEFAULT_KEYS)
    p.add_argument('--project', default='SDFlow')
    p.add_argument('--entity', default=None, help='default: your wandb default entity')
    p.add_argument('--out', default='figures/anneal_curves.csv')
    args = p.parse_args()

    import wandb
    api = wandb.Api()
    entity = args.entity or api.default_entity
    path = f'{entity}/{args.project}'
    all_runs = list(api.runs(path))

    rows = []
    for name in args.runs:
        hits = [r for r in all_runs if r.name == name or r.id == name]
        if not hits:
            near = sorted(r.name for r in all_runs if name.split('_')[-1] in r.name)[:10]
            raise SystemExit(f'no run named {name!r} in {path}. Similar: {near}')
        if len(hits) > 1:
            print(f'[warn] {len(hits)} runs named {name!r}; using the most recent ({hits[0].id})')
        run = hits[0]
        for key in args.keys:
            n = 0
            for row in run.scan_history(keys=['_step', key]):
                v = row.get(key)
                if v is None:
                    continue
                rows.append((name, int(row['_step']), key, float(v)))
                n += 1
            print(f'{name}: {key} {n} points')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'step', 'key', 'value'])
        w.writerows(rows)
    print(f'wrote {len(rows)} rows -> {args.out}')


if __name__ == '__main__':
    main()
