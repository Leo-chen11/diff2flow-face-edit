"""Option C calibration: where is the age trade-off between realism,
accuracy and identity?

The raw real-data age direction (bank slots, no model) looks right, but
costs identity: validate_direction_bank.py measured AccCLIP 70.5 / 85.8 /
95.3% at ID_ind 0.70 / 0.63 / 0.57 for alpha 1.0 / 1.25 / 1.5, over ALL 18
W+ layers. Option C keeps the direction for STRUCTURE and leaves texture to
the content condition, so what matters is how accuracy and identity move
when the direction is restricted to a subset of layers.

This sweeps layer sets x alpha on young (aging, rm) and old (add) sources and
reports, per setting: AccCLIP overall and per gender, and ID_ind. It then
lists the settings that reach a given ID floor, best accuracy first. Those
are the candidates for --bank_dir_layers and --id_hinge_threshold_override.
It also writes a montage of chosen settings, so the numbers can be checked by
eye.

Read-only. No checkpoint needed, only the bank file, G and the judges.

Usage:
    python -m scripts.probe_age_tradeoff \
        --direction_bank_path ./data/direction_bank_k4_stratified_v3.pth \
        --clip_calibration 39:0.517:0.132 --out_dir ./age_tradeoff

    # 2-D: structure-layer strength x detail-layer strength, best per ID floor
    python -m scripts.probe_age_tradeoff --split \
        --direction_bank_path ./data/direction_bank_k4_stratified_v3.pth \
        --clip_calibration 39:0.517:0.132 --out_dir ./age_tradeoff
"""
import argparse
import os
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'stylegan2'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torch.utils import data

AGE = 39
STRATA = ('male_glasses', 'male_noglasses', 'female_glasses', 'female_noglasses')

# W+ layer ranges (inclusive). 0-2 = 4-8px (head shape), 3-6 = 16-32px
# (jaw, cheeks, hairline), 7-10 = 64-128px (wrinkles, facial detail),
# 11-17 = 256-1024px (fine texture, colour).
LAYER_SETS = {
    'all':        (0, 17),
    'no_fine':    (0, 10),
    'structure':  (0, 6),
    'struct_mid': (3, 10),
    'texture':    (7, 17),
}


def load_age_dirs(path, mode):
    """Returns ('average' -> (18,512)) or per-stratum dict, raw magnitudes
    (layer_norms * unit), exactly as validate_direction_bank.py applies them."""
    bank = torch.load(path, map_location='cpu')
    attrs = [int(a) for a in bank['attribute_index']]
    row = attrs.index(AGE)
    du, ln = bank['direction_units'].float(), bank['layer_norms'].float()
    if du.ndim == 3:
        du, ln = du.unsqueeze(1), ln.unsqueeze(1)
    raw = ln[row].unsqueeze(-1) * F.normalize(du[row], dim=-1)          # (K, 18, 512)
    dirs = {'average': raw.mean(0)}
    strat = bank.get('stratification', {}).get(AGE)
    age_k = bank.get('age_k')
    if mode == 'routed':
        if not strat or list(strat[:4]) != list(STRATA) or not age_k or int(age_k) % 4:
            raise SystemExit('--dir_mode routed needs a bank built with --age_k 4 (or more)')
        per = int(age_k) // 4
        for s, n in enumerate(STRATA):
            dirs[n] = raw[s * per:(s + 1) * per].mean(0)
    return dirs


def layer_mask(lo, hi):
    m = torch.zeros(18, 1)
    m[lo:hi + 1] = 1.0
    return m


def to_pil(x, size):
    x = ((x.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x).resize((size, size))


def build_configs(args, device):
    """[(label, coef (18,1))]: per-layer strength of the age direction.

    Default: one strength over a layer set (--layer_sets x --alphas).
    --split: structure layers and detail layers get separate strengths
    (--struct_alphas x --detail_alphas), to find the per-band allocation that
    buys the most accuracy for a given identity cost."""
    configs = []
    if args.split:
        s_lo, s_hi = (int(x) for x in args.struct_layers.split('-'))
        d_lo, d_hi = (int(x) for x in args.detail_layers.split('-'))
        ms, md = layer_mask(s_lo, s_hi).to(device), layer_mask(d_lo, d_hi).to(device)
        for a_s in args.struct_alphas:
            for a_d in args.detail_alphas:
                configs.append((f's{a_s:g}+d{a_d:g}', a_s * ms + a_d * md))
    else:
        for sk in args.layer_sets:
            lo, hi = LAYER_SETS[sk]
            m = layer_mask(lo, hi).to(device)
            for a in args.alphas:
                configs.append((f'{sk} a{a:g}', a * m))
    return configs


@torch.no_grad()
def main(args):
    from evaluation.evaluate_sdflow import (CLIPAttributeJudge, IndependentIDJudge, is_clear,
                                            parse_clip_calibration)
    from models.dataset import SDFlowDataset
    from models.stylegan2.model import Generator

    device = args.device
    dirs = {k: v.to(device) for k, v in load_age_dirs(args.direction_bank_path, args.dir_mode).items()}
    configs = build_configs(args, device)

    ckpt = torch.load(args.stygan2_weights, map_location='cpu')
    G = Generator(size=1024, style_dim=512, n_mlp=8)
    G.load_state_dict(ckpt['g_ema'])
    G.to(device).eval()
    clip_judge = CLIPAttributeJudge([AGE], args.clip_judge_model, device,
                                    calibration=parse_clip_calibration(args.clip_calibration))
    id_judge = IndependentIDJudge(device, pretrained=args.id_indep_pretrained)

    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)),
                    T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file,
                       train=False, transform=tf)
    loader = data.DataLoader(ds, shuffle=False, batch_size=args.batch, num_workers=4)

    tile = 192
    stats = defaultdict(lambda: defaultdict(list))
    tiles = defaultdict(list)          # label -> PIL tiles of the first montage_n aging sources
    src_tiles = []
    seen = 0
    for _img, latent, pred in loader:
        if seen >= args.num_samples:
            break
        latent, pred = latent.to(device), pred.to(device)
        src = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
        src256 = F.interpolate(src, (256, 256))
        s_young = clip_judge.scores(src256)[:, 0]
        src_id = id_judge.extract(src256)
        male = pred[:, 20] >= 0.5
        glasses = pred[:, 15] >= 0.5
        clear = torch.tensor([is_clear(v.item()) for v in s_young], device=device)
        if not clear.any():
            continue
        B = latent.size(0)
        # aging (rm) for young sources: subtract (directions point toward young)
        sign = torch.where(s_young > 0.5, -torch.ones_like(s_young), torch.ones_like(s_young))
        if args.dir_mode == 'routed':
            names = [STRATA[(0 if m else 2) + (0 if g else 1)] for m, g in zip(male.tolist(), glasses.tolist())]
            d = torch.stack([dirs[n] for n in names])                  # (B, 18, 512)
        else:
            d = dirs['average'].unsqueeze(0).expand(B, -1, -1)
        mont_b = [b for b in range(B) if bool(clear[b]) and bool(s_young[b] > 0.5)]
        mont_b = mont_b[:max(0, args.montage_n - len(src_tiles))]
        for b in mont_b:
            src_tiles.append(to_pil(src256[b], tile))
        for label, coef in configs:
            w = latent + sign.view(B, 1, 1) * d * coef.unsqueeze(0)
            e = G([w], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
            e256 = F.interpolate(e, (256, 256))
            ey = clip_judge.scores(e256)[:, 0]
            idc = (src_id * id_judge.extract(e256)).sum(1)
            st = stats[label]
            for b in range(B):
                if not bool(clear[b]):
                    continue
                rm = bool(s_young[b] > 0.5)
                ok = float(ey[b] < 0.5) if rm else float(ey[b] > 0.5)
                st['ok'].append(ok)
                st['id'].append(idc[b].item())
                st['ok_rm' if rm else 'ok_add'].append(ok)
                if rm:
                    st['ok_rm_m' if bool(male[b]) else 'ok_rm_f'].append(ok)
                    st['id_rm'].append(idc[b].item())
            for b in mont_b:
                tiles[label].append(to_pil(e256[b], tile))
        seen += int(clear.sum())

    def mean(v):
        return sum(v) / len(v) if v else float('nan')

    print(f'\n{seen} clear sources, dir_mode={args.dir_mode}'
          + (f', split: structure {args.struct_layers} x detail {args.detail_layers}'
             if args.split else '') + '. rm = aging (young sources).')
    print(f'{"setting":<18}  {"AccCLIP":>8} {"rm":>7} {"rm_M":>7} {"rm_F":>7} {"add":>7}  '
          f'{"ID_ind":>7} {"ID_rm":>7}')
    rows = []
    for label, _ in configs:
        st = stats[label]
        r = dict(label=label, acc=mean(st['ok']), rm=mean(st['ok_rm']), rm_m=mean(st['ok_rm_m']),
                 rm_f=mean(st['ok_rm_f']), add=mean(st['ok_add']), id=mean(st['id']),
                 id_rm=mean(st['id_rm']))
        rows.append(r)
        print(f'{label:<18}  {r["acc"] * 100:7.1f}% {r["rm"] * 100:6.1f}% {r["rm_m"] * 100:6.1f}% '
              f'{r["rm_f"] * 100:6.1f}% {r["add"] * 100:6.1f}%  {r["id"]:7.4f} {r["id_rm"]:7.4f}')
    best = {}
    for floor in args.id_floors:
        ok = sorted([r for r in rows if r['id'] >= floor], key=lambda r: -r['acc'])[:5]
        print(f'\nBest AccCLIP with ID_ind >= {floor}:')
        if not ok:
            print('  (none)')
        for r in ok:
            print(f'  {r["label"]:<18} AccCLIP {r["acc"] * 100:.1f}% (men {r["rm_m"] * 100:.1f}%, '
                  f'women {r["rm_f"] * 100:.1f}%)  ID {r["id"]:.3f}')
        if ok:
            best[floor] = ok[0]['label']

    # Montage: the settings named by --montage, or in --split mode the best
    # setting at each ID floor.
    if args.split:
        show = list(dict.fromkeys(best[f] for f in sorted(best, reverse=True)))
    else:
        show = []
        for x in args.montage:
            k, a = x.split(':')
            show.append(f'{k} a{float(a):g}')
    show = [s for s in show if s in tiles]
    if src_tiles and show:
        os.makedirs(args.out_dir, exist_ok=True)
        labels = ['source'] + show
        canvas = Image.new('RGB', (tile * len(labels), tile * len(src_tiles) + 18), (20, 20, 20))
        dr = ImageDraw.Draw(canvas)
        for c, lab in enumerate(labels):
            dr.text((c * tile + 4, 3), lab, fill=(255, 255, 0))
        for r in range(len(src_tiles)):
            canvas.paste(src_tiles[r], (0, 18 + r * tile))
            for c, lab in enumerate(show, start=1):
                canvas.paste(tiles[lab][r], (c * tile, 18 + r * tile))
        path = os.path.join(args.out_dir, f'age_tradeoff_{args.dir_mode}'
                                          f'{"_split" if args.split else ""}.png')
        canvas.save(path)
        print(f'\nwrote {path}  (columns: {", ".join(show)})')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--direction_bank_path', required=True)
    p.add_argument('--dir_mode', default='average', choices=['average', 'routed'],
                   help='average: mean of all age slots (what an unrouted gate roughly gives); '
                        'routed: each source gets its own gender x glasses stratum '
                        '(what --age_gate_by_strata gives).')
    p.add_argument('--layer_sets', nargs='*', default=list(LAYER_SETS),
                   choices=list(LAYER_SETS))
    p.add_argument('--alphas', nargs='*', type=float, default=[0.75, 1.0, 1.25, 1.5])
    p.add_argument('--id_floors', nargs='*', type=float, default=[0.75, 0.78, 0.80, 0.82])
    p.add_argument('--device', default='cuda')
    p.add_argument('--montage', nargs='*',
                   default=['all:1.0', 'no_fine:1.25', 'structure:1.5', 'texture:1.0'],
                   help='set:alpha pairs to show side by side (alpha must be in --alphas).')
    p.add_argument('--montage_n', type=int, default=8)
    p.add_argument('--split', action='store_true',
                   help='2-D sweep: separate strengths for structure and detail layers '
                        '(--struct_alphas x --detail_alphas) instead of one strength per set.')
    p.add_argument('--struct_layers', default='0-6')
    p.add_argument('--detail_layers', default='7-10')
    p.add_argument('--struct_alphas', nargs='*', type=float, default=[1.0, 1.25, 1.5, 1.75, 2.0])
    p.add_argument('--detail_alphas', nargs='*', type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument('--num_samples', type=int, default=200)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--out_dir', default='./age_tradeoff')
    p.add_argument('--img_size', type=int, default=512)
    p.add_argument('--index_file', default='./data/ffhq.txt')
    p.add_argument('--image_root', default='data/FFHQ')
    p.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    p.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth')
    p.add_argument('--stygan2_weights', default='./data/stylegan2-ffhq-config-f.pt')
    p.add_argument('--clip_judge_model', default='ViT-L/14')
    p.add_argument('--clip_calibration', default=None)
    p.add_argument('--id_indep_pretrained', default='casia-webface',
                   choices=['casia-webface', 'vggface2'])
    main(p.parse_args())
