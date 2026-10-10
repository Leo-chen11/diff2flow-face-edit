"""Queue one more training run behind the ones on the GPU, then evaluate.

1. waits until every --wait_for run has its final checkpoint (prior-<step>),
2. trains --run_name with the exact flags of --base_config (a run's
   config.json, via scripts/rebuild_train_cmd.py) except those named by --set,
3. runs scripts/post_train_eval.py --no_wait on --eval_runs with --compare.

Usage:
    python scripts/queue_after.py \
        --wait_for multi_v1_scratch_noaux_seed1 multi_v1_scratch_male_seed1 \
        --base_config ./output/SDFlow/multi_v1_scratch_noaux_seed1/config.json \
        --run_name multi_v1_scratch_maleid_seed1 \
        --set id_hinge_threshold_override=39:0.68,20:0.72 \
        --eval_runs multi_v1_scratch_noaux_seed1 multi_v1_scratch_male_seed1 multi_v1_scratch_maleid_seed1 \
        --compare multi_v1_scratch_noaux_seed1:multi_v1_scratch_maleid_seed1
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wait_for', nargs='*', default=[])
    p.add_argument('--output_root', default='./output/SDFlow')
    p.add_argument('--step', type=int, default=50000)
    p.add_argument('--base_config', required=True)
    p.add_argument('--run_name', required=True)
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    p.add_argument('--eval_runs', nargs='*', default=[])
    p.add_argument('--compare', nargs='*', default=[])
    p.add_argument('--poll', type=int, default=600)
    p.add_argument('--dry_run', action='store_true', help='Print the training command only.')
    a = p.parse_args()

    os.chdir(ROOT)
    py = sys.executable

    def ready(run):
        return os.path.exists(os.path.join(a.output_root, run, 'save_models', f'prior-{a.step:07d}'))

    if not a.dry_run:
        while not all(ready(r) for r in a.wait_for):
            print(f'[{time.strftime("%m-%d %H:%M")}] waiting for '
                  f'{[r for r in a.wait_for if not ready(r)]}', flush=True)
            time.sleep(a.poll)
        if a.wait_for:
            time.sleep(120)            # the last checkpoint finishes writing, the GPU frees up

    rebuild = [py, '-m', 'scripts.rebuild_train_cmd', '--config', a.base_config,
               '--set', f'run_name={a.run_name}', '--prefix', 'TRAIN']
    for s in a.set:
        rebuild += ['--set', s]
    text = subprocess.check_output(rebuild, text=True)
    tokens = shlex.split(text.replace('\\\n', ' '))
    assert tokens and tokens[0] == 'TRAIN', text
    cmd = [py, 'training/train_sdflow.py'] + tokens[1:]
    print(' '.join(shlex.quote(t) for t in cmd), flush=True)
    if a.dry_run:
        return

    os.makedirs('logs', exist_ok=True)
    if not ready(a.run_name):
        print(f'[{time.strftime("%m-%d %H:%M")}] training {a.run_name}', flush=True)
        with open(os.path.join('logs', f'{a.run_name}.log'), 'w') as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        print(f'    exit {rc}', flush=True)
        if rc != 0:
            sys.exit(rc)

    if a.eval_runs:
        subprocess.call([py, 'scripts/post_train_eval.py', '--no_wait', '--output_root', a.output_root,
                         '--step', str(a.step), '--runs'] + a.eval_runs
                        + (['--compare'] + a.compare if a.compare else []))
    print(f'[{time.strftime("%m-%d %H:%M")}] all done', flush=True)


if __name__ == '__main__':
    main()
