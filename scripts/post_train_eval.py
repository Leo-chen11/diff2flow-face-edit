"""Evaluate finished runs as soon as their training is done, then compare them.

Waits until every listed run has its final checkpoint (save_models/prior-<step>)
-- the GPU is shared, so nothing starts while any of them is still training --
then, for each run:
  eval      evaluate_sdflow at --scales (the development faces, offset 0)
  adaptive  the per-face adaptive ladder (scripts/run_inference_variants.py)
  preview   render_preview at scale 1.0
and finally scripts.compare_runs_matched_id for every --compare a:b pair
(a run name, or a path to an eval JSON of a run that lives elsewhere).
Steps whose output exists are skipped, so the script can be re-run.

Usage (start it now; it waits for the training queue):
    python scripts/post_train_eval.py \
        --runs multi_v1_scratch_noaux_seed1 multi_v1_scratch_male_seed1 \
        --compare multi_v1_scratch_noaux_seed1:multi_v1_scratch_male_seed1
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True, help='Run names under --output_root.')
    p.add_argument('--output_root', default='./output/SDFlow')
    p.add_argument('--step', type=int, default=50000)
    p.add_argument('--scales', nargs='+', default=['0.5', '0.6', '0.7', '0.85', '1.0'])
    p.add_argument('--num_samples', type=int, default=500)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--judge', default='./data/r50_celebahq_eval.pth')
    p.add_argument('--compare', nargs='*', default=[],
                   help='a:b pairs; each side a run name or an eval JSON path.')
    p.add_argument('--no_wait', action='store_true')
    p.add_argument('--poll', type=int, default=600, help='Seconds between checks while waiting.')
    a = p.parse_args()

    os.chdir(ROOT)
    py = sys.executable
    run_dir = {r: os.path.join(a.output_root, r) for r in a.runs}
    eval_json = {r: os.path.join(d, f'eval_s{a.step}_clamp02.json') for r, d in run_dir.items()}

    def ready(d):
        return os.path.exists(os.path.join(d, 'save_models', f'prior-{a.step:07d}'))

    while not a.no_wait and not all(ready(d) for d in run_dir.values()):
        todo = [r for r, d in run_dir.items() if not ready(d)]
        print(f'[{time.strftime("%m-%d %H:%M")}] waiting for {todo}', flush=True)
        time.sleep(a.poll)
    time.sleep(0 if a.no_wait else 120)        # let the last checkpoint finish writing

    common = ['--edit_direction', 'indep', '--independent_attr_weights', a.judge,
              '--independent_attr_backbone', 'r50', '--celeba_attr_judge_weights', '',
              '--age_fine_layer_scale', '1.0', '--src_cond_clamp', '0.2']

    def run(log, cmd):
        print(f'[{time.strftime("%m-%d %H:%M")}] {os.path.basename(log)}', flush=True)
        with open(log, 'w') as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        print(f'    exit {rc}', flush=True)

    for r, d in run_dir.items():
        if not ready(d):
            print(f'{r}: no step-{a.step} checkpoint, skipped')
            continue
        if not os.path.exists(eval_json[r]):
            run(os.path.join(d, f'eval_s{a.step}_clamp02.log'),
                [py, 'evaluation/evaluate_sdflow.py', '--checkpoint_dir', d, '--step', str(a.step)]
                + common + ['--leak40', '--num_samples', str(a.num_samples), '--batch', str(a.batch),
                            '--eval_scales'] + a.scales + ['--out_json', eval_json[r]])
        if not os.path.exists(os.path.join(d, 'eval_adaptive.json')):
            run(os.path.join(d, 'post_adaptive.log'),
                [py, 'scripts/run_inference_variants.py', '--checkpoint_dir', d, '--step', str(a.step),
                 '--only', 'adaptive', '--base_json', eval_json[r]])
        if not any(f.startswith(f'preview_step{a.step}_scale1.0') for f in os.listdir(d)):
            run(os.path.join(d, 'post_preview.log'),
                [py, 'scripts/render_preview.py', '--checkpoint_dir', d, '--step', str(a.step),
                 '--scale', '1.0', '--num_faces', '10', '--edit_direction', 'indep',
                 '--independent_attr_weights', a.judge, '--independent_attr_backbone', 'r50',
                 '--src_cond_clamp', '0.2', '--age_fine_layer_scale', '1.0'])

    for pair in a.compare:
        x, y = pair.split(':')
        jx = eval_json.get(x, x)
        jy = eval_json.get(y, y)
        if not (os.path.exists(jx) and os.path.exists(jy)):
            print(f'compare {pair}: missing {jx if not os.path.exists(jx) else jy}, skipped')
            continue
        nx, ny = (os.path.basename(os.path.dirname(j)) or j for j in (jx, jy))
        out = os.path.join(os.path.dirname(jy), f'cmp_{nx}_vs_{ny}.txt')
        with open(out, 'w') as f:
            subprocess.call([py, '-m', 'scripts.compare_runs_matched_id', '--a', jx, '--b', jy,
                             '--names', nx[-12:], ny[-12:], '--points', '3'],
                            stdout=f, stderr=subprocess.STDOUT)
        print(f'compare {pair} -> {out}', flush=True)
    print(f'[{time.strftime("%m-%d %H:%M")}] all done', flush=True)


if __name__ == '__main__':
    main()
