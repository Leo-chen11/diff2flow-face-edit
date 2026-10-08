"""Run the inference-time variants of one checkpoint, one after another, and
compare each with the checkpoint's own base eval at matched identity.

Variants (all without retraining):
  nocaps    evaluate_sdflow --no_train_caps (edit-norm caps lifted)
  male      scripts.diagnose_attr_edit --attr 20 with larger --clamp_margins
  ema       the EMA weights (save_models_ema)
  pres      --preserve_boundaries (boundaries fitted first if missing)
  adaptive  --adaptive_ladder 0.5 0.7 0.85 1.0

Every variant writes <name>.log (and .json) into the checkpoint dir, and a
matched-ID comparison against --base_json as cmp_<name>.txt. A variant whose
JSON already exists is skipped, so the script can be re-run after a stop.

Usage:
    python scripts/run_inference_variants.py --checkpoint_dir ./output/SDFlow/multi_v1_scratch_k2
    python scripts/run_inference_variants.py --checkpoint_dir ... --only nocaps ema
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ORDER = ['nocaps', 'male', 'ema', 'pres', 'adaptive']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--step', type=int, default=50000)
    p.add_argument('--base_json', default=None,
                   help='Default: <checkpoint_dir>/eval_s<step>_clamp02.json')
    p.add_argument('--only', nargs='+', choices=ORDER, default=ORDER)
    p.add_argument('--num_samples', type=int, default=500)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--judge', default='./data/r50_celebahq_eval.pth')
    p.add_argument('--boundaries', default='./data/attr_boundaries_wplus.pth')
    p.add_argument('--dry_run', action='store_true', help='Print the commands only.')
    a = p.parse_args()

    os.chdir(ROOT)
    ck = os.path.abspath(a.checkpoint_dir)
    base = a.base_json or os.path.join(ck, f'eval_s{a.step}_clamp02.json')
    py = sys.executable
    judge = ['--independent_attr_weights', a.judge, '--independent_attr_backbone', 'r50',
             '--celeba_attr_judge_weights', '', '--age_fine_layer_scale', '1.0']
    common = ['--edit_direction', 'indep'] + judge + [
        '--src_cond_clamp', '0.2', '--leak40', '--num_samples', str(a.num_samples),
        '--batch', str(a.batch)]
    scales = ['--eval_scales', '0.7', '0.85', '1.0']

    def ev(ckdir, out, extra):
        return [py, 'evaluation/evaluate_sdflow.py', '--checkpoint_dir', ckdir,
                '--step', str(a.step)] + common + extra + ['--out_json', out]

    def run(name, cmd):
        log = os.path.join(ck, f'{name}.log')
        print(f'[{time.strftime("%H:%M:%S")}] {name}: ' + ' '.join(repr(c) if ' ' in c or not c else c
                                                                for c in cmd), flush=True)
        if a.dry_run:
            return 0
        with open(log, 'w') as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        print(f'    -> exit {rc}, log {log}', flush=True)
        return rc

    def compare(name, out):
        if a.dry_run or not os.path.exists(out) or not os.path.exists(base):
            if not a.dry_run and not os.path.exists(base):
                print(f'    (no base eval {base}; comparison skipped)')
            return
        with open(os.path.join(ck, f'cmp_{name}.txt'), 'w') as f:
            subprocess.call([py, '-m', 'scripts.compare_runs_matched_id', '--a', base, '--b', out,
                             '--names', 'base', name, '--points', '3'],
                            stdout=f, stderr=subprocess.STDOUT)
        print(f'    -> cmp_{name}.txt')

    for name in [n for n in ORDER if n in a.only]:
        out = os.path.join(ck, f'eval_{name}.json')
        if name != 'male' and os.path.exists(out):
            print(f'{name}: {out} exists, skipped')
            compare(name, out)
            continue
        if name == 'nocaps':
            run('eval_nocaps', ev(ck, out, scales + ['--no_train_caps']))
        elif name == 'male':
            run('diagnose_male_margins', [
                py, '-m', 'scripts.diagnose_attr_edit', '--checkpoint_dir', ck, '--step', str(a.step),
                '--attr', '20', '--num_samples', str(a.num_samples)] + judge + [
                '--clamp_margins', '0.2', '0.35', '0.45',
                '--out_dir', os.path.join(ck, 'diagnose_Male_margins')])
            continue
        elif name == 'ema':
            ema_src = os.path.join(ck, 'save_models_ema')
            if not os.path.isdir(ema_src):
                print(f'ema: {ema_src} not found, skipped')
                continue
            ema_dir = ck.rstrip('/') + '_ema_view'
            if not a.dry_run:
                os.makedirs(ema_dir, exist_ok=True)
                link = os.path.join(ema_dir, 'save_models')
                if os.path.islink(link):
                    os.remove(link)
                if not os.path.exists(link):
                    os.symlink(ema_src, link)
                with open(os.path.join(ck, 'config.json')) as s, \
                        open(os.path.join(ema_dir, 'config.json'), 'w') as d:
                    d.write(s.read())
            run('eval_ema', ev(ema_dir, out, scales))
        elif name == 'pres':
            if not os.path.exists(a.boundaries):
                run('fit_boundaries', [py, '-m', 'scripts.fit_attr_boundaries', '--out', a.boundaries])
            run('eval_pres', ev(ck, out, scales + ['--preserve_boundaries', a.boundaries]))
        elif name == 'adaptive':
            run('eval_adaptive', ev(ck, out, ['--adaptive_ladder', '0.5', '0.7', '0.85', '1.0']))
            continue   # results keyed by margin, not scale: compared by hand
        compare(name, out)
    print(f'[{time.strftime("%H:%M:%S")}] all done')


if __name__ == '__main__':
    main()
