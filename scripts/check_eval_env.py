"""Check that a machine can run evaluate_sdflow.py: packages, local data files,
and the pretrained weights that are downloaded on first use.

Run it on the new machine BEFORE copying checkpoints, with internet on, so the
downloads land in the cache once; afterwards the eval also runs offline.

    python -m scripts.check_eval_env                 # report only
    python -m scripts.check_eval_env --download      # also fetch the missing weights
    python -m scripts.check_eval_env --run ./output/SDFlow/multi_v1_smile_bangs --step 80000

What it checks (each line OK / MISSING / FAIL):
  packages   torch, torchvision, lpips, facenet_pytorch, clip, numpy, PIL, tqdm
  data       the files evaluate_sdflow.py reads by default (+ the bank the run's
             config.json names, and the run's checkpoint files with --run)
  weights    torchvision resnet18/34/50 and alexnet (ImageNet, used as backbones and
             by LPIPS), facenet vggface2 (training IDLoss) and casia-webface (ID_ind
             judge), CLIP ViT-L/14 (AccCLIP judge). LPIPS ships its own linear
             weights inside the package.
"""
import argparse
import importlib
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

DATA_FILES = [
    'data/stylegan2-ffhq-config-f.pt', 'data/ffhq_e4e_latents.pth', 'data/ffhq_e4e_preds.pth',
    'data/ffhq.txt', 'data/FFHQ', 'data/r34_a40_age_256_classifier.pth',
    'data/r50_celebahq_eval.pth', 'data/parsing_bisenet.pth',
]
PACKAGES = [('torch', 'torch'), ('torchvision', 'torchvision'), ('numpy', 'numpy'),
            ('PIL', 'pillow'), ('tqdm', 'tqdm'), ('lpips', 'pip install lpips'),
            ('facenet_pytorch', 'pip install facenet-pytorch'),
            ('clip', 'pip install git+https://github.com/openai/CLIP.git')]


def line(status, what, extra=''):
    print(f'  {status:<8} {what}' + (f'  {extra}' if extra else ''))
    return status == 'OK'


def check_packages():
    print('packages')
    ok = True
    for mod, hint in PACKAGES:
        try:
            m = importlib.import_module(mod)
            line('OK', mod, getattr(m, '__version__', ''))
        except Exception as e:                       # noqa: BLE001
            ok = False
            line('MISSING', mod, f'({type(e).__name__}) install: {hint}')
    try:
        import torch
        print(f'  cuda available: {torch.cuda.is_available()}'
              + (f' ({torch.cuda.get_device_name(0)})' if torch.cuda.is_available() else ''))
        if not torch.cuda.is_available():
            print('  WARNING  evaluate_sdflow.py puts models on cuda; it will not run without a GPU')
    except Exception:                                # noqa: BLE001
        pass
    return ok


def check_data(run=None, step=None):
    print('data files (relative to the project root)')
    ok = True
    for rel in DATA_FILES:
        p = os.path.join(PROJECT_ROOT, rel)
        if os.path.isdir(p):
            n = sum(len(f) for _, _, f in os.walk(p))
            ok &= line('OK' if n else 'EMPTY', rel, f'{n} files')
        else:
            ok &= line('OK' if os.path.exists(p) else 'MISSING', rel,
                       f'{os.path.getsize(p) / 1e6:.0f} MB' if os.path.exists(p) else '')
    # image list consistency: a few listed images must exist
    lst = os.path.join(PROJECT_ROOT, 'data/ffhq.txt')
    if os.path.exists(lst):
        try:
            import pandas as pd
            df = pd.read_csv(lst)
            col = 'path' if 'path' in df.columns else df.columns[0]
            sample = list(df[col].values[:3]) + list(df[col].values[-3:])
            miss = [s for s in sample if not os.path.exists(os.path.join(PROJECT_ROOT, 'data/FFHQ', str(s)))]
            ok &= line('OK' if not miss else 'MISSING', 'images named in ffhq.txt',
                       f'{len(df)} listed; first/last 3 checked' + (f'; absent: {miss}' if miss else ''))
        except Exception as e:                       # noqa: BLE001
            ok &= line('FAIL', 'reading data/ffhq.txt', f'({type(e).__name__}: {e})')
    if run:
        cfg_path = os.path.join(run, 'config.json')
        if not os.path.exists(cfg_path):
            ok &= line('MISSING', cfg_path)
        else:
            cfg = json.load(open(cfg_path))
            line('OK', cfg_path)
            for key in ('direction_bank_path',):
                v = cfg.get(key)
                if v:
                    ok &= line('OK' if os.path.exists(os.path.join(PROJECT_ROOT, v)) else 'MISSING',
                               f'{key} = {v}')
        save = os.path.join(run, 'save_models')
        if not os.path.isdir(save):
            ok &= line('MISSING', save)
        else:
            names = sorted(os.listdir(save))
            if step is None:
                steps = sorted({int(n.rsplit('-', 1)[1]) for n in names if '-' in n and n.rsplit('-', 1)[1].isdigit()})
                step = steps[-1] if steps else None
            line('OK', f'{save}', f'{len(names)} files; latest step {step}')
            for mod in ('prior', 'conditioner', 'direction_bank', 'control_encoder'):
                f = f'{mod}-{int(step):07d}' if step is not None else None
                present = f and f in names
                if mod == 'control_encoder' and not present:
                    line('SKIP', f, '(only needed if the run used ControlNet)')
                else:
                    ok &= line('OK' if present else 'MISSING', f or f'{mod}-<step>')
    return ok


def check_weights(download):
    print('pretrained weights (downloaded on first use)')
    ok = True
    try:
        import torch
        hub = torch.hub.get_dir()
        ck = os.path.join(hub, 'checkpoints')
        have = set(os.listdir(ck)) if os.path.isdir(ck) else set()
        print(f'  torch hub cache: {ck}  ({len(have)} files)')
    except Exception:                                # noqa: BLE001
        have = set()

    def try_load(name, fn, needle):
        nonlocal ok
        cached = any(needle in h for h in have)
        if cached and not download:
            return line('OK', name, 'cached')
        if not download:
            ok = False
            return line('MISSING', name, 'not cached; rerun with --download (needs internet)')
        try:
            fn()
            return line('OK', name, 'downloaded' if not cached else 'cached')
        except Exception as e:                       # noqa: BLE001
            ok = False
            return line('FAIL', name, f'({type(e).__name__}: {str(e)[:90]})')

    try:
        import torchvision
        tv = torchvision.models
        try_load('torchvision resnet50 (R50 judge backbone)', lambda: tv.resnet50(weights='IMAGENET1K_V1'), 'resnet50-')
        try_load('torchvision resnet34 (r34 teacher backbone)', lambda: tv.resnet34(weights='IMAGENET1K_V1'), 'resnet34-')
        try_load('torchvision resnet18 (only some scripts)', lambda: tv.resnet18(weights='IMAGENET1K_V1'), 'resnet18-')
        try_load('torchvision alexnet (LPIPS backbone)', lambda: tv.alexnet(weights='IMAGENET1K_V1'), 'alexnet-')
    except Exception as e:                           # noqa: BLE001
        ok = False
        line('FAIL', 'torchvision', f'({e})')

    try:
        from facenet_pytorch import InceptionResnetV1
        for tag in ('vggface2', 'casia-webface'):
            try_load(f'facenet {tag} ({"training IDLoss" if tag == "vggface2" else "ID_ind judge"})',
                     lambda t=tag: InceptionResnetV1(pretrained=t), tag)
    except Exception as e:                           # noqa: BLE001
        ok = False
        line('MISSING', 'facenet_pytorch', f'({type(e).__name__})')

    try:
        import lpips
        lpips.LPIPS(net='alex', verbose=False)
        line('OK', 'lpips alex linear weights', 'ship inside the package')
    except Exception as e:                           # noqa: BLE001
        ok = False
        line('FAIL', 'lpips alex', f'({type(e).__name__}: {str(e)[:90]})')

    try:
        import clip
        cdir = os.path.expanduser('~/.cache/clip')
        cached = os.path.exists(os.path.join(cdir, 'ViT-L-14.pt'))
        if cached and not download:
            line('OK', 'CLIP ViT-L/14 (AccCLIP judge)', f'cached in {cdir}')
        elif not download:
            ok = False
            line('MISSING', 'CLIP ViT-L/14 (AccCLIP judge)', 'not cached (~890 MB); rerun with --download')
        else:
            try:
                clip.load('ViT-L/14', device='cpu', jit=False)
                line('OK', 'CLIP ViT-L/14 (AccCLIP judge)', 'ready')
            except Exception as e:                   # noqa: BLE001
                ok = False
                line('FAIL', 'CLIP ViT-L/14', f'({type(e).__name__}: {str(e)[:90]})')
    except Exception as e:                           # noqa: BLE001
        ok = False
        line('MISSING', 'clip', f'({type(e).__name__})')
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--download', action='store_true', help='Fetch missing pretrained weights (needs internet).')
    p.add_argument('--run', default=None, help='A run directory to check (config.json, bank, checkpoint files).')
    p.add_argument('--step', type=int, default=None)
    args = p.parse_args()
    results = [check_packages(), check_data(args.run, args.step), check_weights(args.download)]
    print()
    print('ALL OK: ready to run evaluate_sdflow.py' if all(results)
          else 'SOMETHING IS MISSING: fix the lines marked MISSING / FAIL / EMPTY above')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
