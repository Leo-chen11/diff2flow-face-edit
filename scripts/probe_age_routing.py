"""D3: why does MALE aging stay texture-only? Three questions about the
direction bank, answered without training.

  A. BANK GEOMETRY (bank file only). Are the K age slots real per-stratum
     directions, or one direction tiled K times? precompute_directions_
     stratified.py's default --age_k 1 averages the four gender x glasses
     strata into ONE direction and copies it into every slot. Then the gate
     has nothing to choose and every face ages along the same mostly
     population-weighted direction. Also: how do the male and female aging
     directions differ per W+ band?

  B. LEARNED ROUTING (checkpoint). For male vs female sources (by the
     conditioner's Male score, what --age_gate_by_strata would route on), how
     much gate mass lands on each stratum's slots? Nothing in training ties
     the gate to the stratum a slot was fit on. About 50% own-gender mass
     means routing ignores gender; close to 100% means it is right. Also: how
     is the model's actual age edit spread over W+ bands, per gender? Less
     energy in the 16-32px structure band for men would mean the model avoids
     structural change for them.

  C. RAW DIRECTIONS (bank + G + CLIP). Each stratum's direction applied
     alone to young male and female faces at several strengths, scored by
     CLIP, with a montage. If even the male direction at alpha 2 leaves men
     "young" to CLIP, the geometry itself lacks structural aging and no
     training change can add it: rebuild the bank.

Read-only: no training, no checkpoint writes.

Usage:
    # A only (seconds, no GPU model load)
    python -m scripts.probe_age_routing --bank_only \
        --direction_bank_path ./data/direction_bank_k4_stratified_v3.pth

    # A + B + C
    python -m scripts.probe_age_routing \
        --checkpoint_dir ./output/SDFlow/substyle3_glasses_v30_gbal --step 115000 \
        --clip_calibration 39:0.517:0.132 --out_dir ./age_routing_v30
"""
import argparse
import os
import sys

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
MALE_STRATA = (0, 1)

# StyleGAN2-1024 W+ layer -> synthesis resolution (same map as
# scripts/probe_wplus_band_occupancy.py).
def _layer_res(j):
    return 4 if j == 0 else min(2 ** (3 + (j - 1) // 2), 1024)

BANDS = [
    ('coarse 4-8px', [j for j in range(18) if _layer_res(j) <= 8]),
    ('STRUCTURE 16-32px', [j for j in range(18) if 16 <= _layer_res(j) <= 32]),
    ('mid 64-128px', [j for j in range(18) if 64 <= _layer_res(j) <= 128]),
    ('fine 256-1024px', [j for j in range(18) if _layer_res(j) >= 256]),
]


def band_shares(delta):
    """delta (..., 18, 512) -> (..., n_bands) share of squared norm per band."""
    e = delta.pow(2).sum(-1)                                  # (..., 18)
    tot = e.sum(-1, keepdim=True).clamp(min=1e-12)
    return torch.stack([e[..., idx].sum(-1) for _, idx in BANDS], dim=-1) / tot


def fmt_shares(sh):
    return '  '.join(f'{name.split()[0]}={v * 100:5.1f}%' for (name, _), v in zip(BANDS, sh.tolist()))


# ── A. bank geometry ────────────────────────────────────────────────────────
def analyse_bank(path):
    bank = torch.load(path, map_location='cpu')
    attrs = [int(a) for a in bank['attribute_index']]
    if AGE not in attrs:
        raise SystemExit(f'bank {path} has no attr {AGE}')
    row = attrs.index(AGE)
    du = bank['direction_units'].float()
    ln = bank['layer_norms'].float()
    if du.ndim == 3:
        du, ln = du.unsqueeze(1), ln.unsqueeze(1)
    units = F.normalize(du[row], dim=-1)                      # (K, 18, 512)
    raw = ln[row].unsqueeze(-1) * units                        # (K, 18, 512)
    K = units.shape[0]
    age_k = bank.get('age_k')
    strat = bank.get('stratification', {}).get(AGE) or bank.get('stratification', {}).get(str(AGE))
    print('=' * 72)
    print(f'A. BANK GEOMETRY  {path}')
    print('=' * 72)
    print(f'  K={K}  age_k={age_k}  age stratification={strat[:4] if strat else None}')
    print(f'  built with: direction_method={bank.get("direction_method")}  '
          f'extreme_pct={bank.get("extreme_pct")}  '
          f'decorrelate_cross_attr={bank.get("decorrelate_cross_attr")}  '
          f'(reuse these when rebuilding so only age_k changes)')

    cos = torch.einsum('ild,jld->ijl', units, units).abs().mean(-1)   # (K, K)
    off = cos[~torch.eye(K, dtype=torch.bool)]
    print(f'  pairwise mean|cos| between age slots: min={off.min():.3f} '
          f'median={off.median():.3f} max={off.max():.3f}' if K > 1 else '  single slot')

    tiled = (not strat) or str(strat[0]).startswith('weighted_avg') or (K > 1 and off.min() > 0.99)
    info = {'tiled': tiled, 'raw': raw, 'K': K, 'per': None, 'strata_raw': {}}
    info['strata_raw']['average'] = raw.mean(0)
    if tiled:
        print('\n  >>> VERDICT A: every age slot is the SAME direction (tiled --age_k 1 bank, a\n'
              '      population-weighted average of male and female aging). The gate has\n'
              '      nothing to route; male faces age along a direction dominated by\n'
              '      whichever gender FFHQ has more of. FIX: rebuild the bank with\n'
              '      --age_k 4 (keep --K / --substyle_k so K stays the same), then resume with\n'
              '      --refresh_bank_directions (+ --age_gate_by_strata).')
        print(f'  average direction band shares: {fmt_shares(band_shares(raw.mean(0)))}')
        return info

    if list(strat[:4]) != list(STRATA) or int(age_k) % 4:
        print(f'  [WARN] unexpected stratification layout {strat}; per-stratum analysis skipped')
        return info
    per = int(age_k) // 4
    info['per'] = per
    for s, name in enumerate(STRATA):
        d = raw[s * per:(s + 1) * per].mean(0)
        info['strata_raw'][name] = d
        print(f'  slots {s * per:2d}-{(s + 1) * per - 1:2d} {name:<17} norm={d.norm():6.2f}  '
              f'{fmt_shares(band_shares(d))}')
    m, f = info['strata_raw']['male_noglasses'], info['strata_raw']['female_noglasses']
    per_band = []
    for name, idx in BANDS:
        c = F.cosine_similarity(m[idx].reshape(-1), f[idx].reshape(-1), dim=0).item()
        per_band.append(f'{name.split()[0]}={c:+.2f}')
    print(f'  cos(male_noglasses, female_noglasses) per band: {"  ".join(per_band)}')
    print('  (low cos in STRUCTURE = men and women age differently there, which is what a\n'
          '   shared/averaged direction would wash out)')
    return info


# ── helpers ────────────────────────────────────────────────────────────────
def to_pil(x, size):
    x = ((x.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x).resize((size, size))


def montage(rows, labels, path, tile=192):
    if not rows:
        return
    ncol = len(rows[0])
    canvas = Image.new('RGB', (tile * ncol, tile * len(rows) + 18), (20, 20, 20))
    d = ImageDraw.Draw(canvas)
    for c, lab in enumerate(labels):
        d.text((c * tile + 4, 3), lab, fill=(255, 255, 0))
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            canvas.paste(to_pil(img, tile), (c * tile, 18 + r * tile))
    canvas.save(path)
    print(f'  wrote {path}')


@torch.no_grad()
def run_model_parts(args, info):
    from evaluation.evaluate_sdflow import (CLIPAttributeJudge, edited_attr_value,
                                            load_models, parse_clip_calibration)
    from models.dataset import SDFlowDataset

    prior, conditioner, G, id_criterion, _teacher, _ai, direction_bank, _ce = load_models(args)
    if direction_bank is None:
        raise SystemExit('checkpoint has no direction bank')
    ai = list(args.attribute_index)
    age_l, male_l = ai.index(AGE), ai.index(20)
    clip_judge = CLIPAttributeJudge([AGE], args.clip_judge_model, 'cuda',
                                    calibration=parse_clip_calibration(args.clip_calibration))

    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)),
                    T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file,
                       train=False, transform=tf)
    loader = data.DataLoader(ds, shuffle=False, batch_size=args.batch, num_workers=4)

    per = info['per']
    names = [n for n in ('male_noglasses', 'female_noglasses', 'male_glasses',
                         'female_glasses', 'average') if n in info['strata_raw']]
    dirs = {n: info['strata_raw'][n].cuda() for n in names}

    gate_mass = {True: [], False: []}          # male? -> list of (K,) alpha
    edit_bands = {True: [], False: []}
    clip_young = {g: {(n, a): [] for n in names for a in args.alphas} for g in (True, False)}
    clip_src = {True: [], False: []}
    mont = {True: [], False: []}
    count = {True: 0, False: 0}

    for img, latent, _pred in loader:
        if min(count.values()) >= args.num_per_gender:
            break
        img, latent = img.cuda(), latent.cuda()
        _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
        young = attr_cond[:, age_l] > 0.5          # rm (aging) sources only
        male = attr_cond[:, male_l] >= 0.5
        keep = young.clone()
        for b in range(img.size(0)):
            g = bool(male[b])
            if count[g] >= args.num_per_gender:
                keep[b] = False
        if not keep.any():
            continue
        img, latent, id_cond, attr_cond, male = (img[keep], latent[keep], id_cond[keep],
                                                 attr_cond[keep], male[keep])
        B = img.size(0)

        # ── B: the model's own age edit (flow -> bank), as edit_single_attribute
        zero_pad = torch.zeros(B, 18, 1, device=latent.device)
        mid_latent, _ = prior(latent, torch.cat([id_cond, attr_cond], 1), zero_pad)
        new_attr = attr_cond.clone()
        new_attr[:, age_l] = edited_attr_value(attr_cond[:, age_l], args.edit_scale, AGE,
                                               args.edit_target)
        new_raw, _ = prior(mid_latent, torch.cat([id_cond, new_attr], 1), zero_pad, reverse=True)
        idx = torch.full((B,), age_l, device=latent.device, dtype=torch.long)
        guided = direction_bank(new_raw - latent, new_attr - attr_cond, attr_idx=idx,
                                latent=latent, route_scores=attr_cond)
        alpha = direction_bank._last_alpha[:, age_l, :].float().cpu()
        shares = band_shares(guided.float()).cpu()
        src = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
        src256 = F.interpolate(src, (256, 256))
        cs = clip_judge.scores(src256)[:, 0].cpu()

        # ── C: raw stratum directions, aging sign (directions point toward young)
        outs = {}
        for n in names:
            for a in args.alphas:
                w = latent - a * dirs[n].unsqueeze(0)
                e = G([w], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
                e256 = F.interpolate(e, (256, 256))
                outs[(n, a)] = (e256, clip_judge.scores(e256)[:, 0].cpu())

        for b in range(B):
            g = bool(male[b])
            gate_mass[g].append(alpha[b])
            edit_bands[g].append(shares[b])
            clip_src[g].append(cs[b].item())
            for key, (_e, sc) in outs.items():
                clip_young[g][key].append(sc[b].item())
            if len(mont[g]) < args.montage_n:
                row = [src256[b]] + [outs[(n, args.montage_alpha)][0][b] for n in names
                                     if (n, args.montage_alpha) in outs]
                mont[g].append(row)
            count[g] += 1

    gname = {True: 'MALE', False: 'FEMALE'}
    print('\n' + '=' * 72)
    print(f'B. LEARNED ROUTING + ACTUAL EDIT  (young sources, edit_scale={args.edit_scale}, '
          f'target={args.edit_target})')
    print('=' * 72)
    for g in (True, False):
        if not gate_mass[g]:
            continue
        m = torch.stack(gate_mass[g]).mean(0)                   # (K,)
        line = f'  {gname[g]:<6} n={len(gate_mass[g]):3d}  '
        if per:
            st = [m[s * per:(s + 1) * per].sum().item() for s in range(4)]
            own = (st[0] + st[1]) if g else (st[2] + st[3])
            line += '  '.join(f'{n}={v * 100:4.1f}%' for n, v in zip(STRATA, st))
            line += f'   -> OWN-GENDER MASS {own * 100:.1f}%'
        else:
            line += '(tiled bank: gate choice has no effect)'
        print(line)
        print(f'         actual edit band shares: '
              f'{fmt_shares(torch.stack(edit_bands[g]).mean(0))}')
    if per:
        print('  Read: ~50% own-gender mass = routing ignores gender (use --age_gate_by_strata);\n'
              '        ~100% = routing is right, look at C for the geometry.')
    print('  Read: a clearly smaller STRUCTURE share for MALE than FEMALE = the trained model\n'
          '        keeps men\'s aging out of the 16-32px layers.')

    print('\n' + '=' * 72)
    print('C. RAW STRATUM DIRECTIONS  (CLIP Young prob after aging; fail = still >= 0.5)')
    print('=' * 72)
    for g in (True, False):
        if not clip_src[g]:
            continue
        n_src = len(clip_src[g])
        print(f'  {gname[g]} sources n={n_src}, source CLIP young mean '
              f'{sum(clip_src[g]) / n_src:.2f}')
        for n in names:
            cells = []
            for a in args.alphas:
                v = clip_young[g][(n, a)]
                fail = sum(x >= 0.5 for x in v) / max(len(v), 1)
                cells.append(f'a={a:g}: fail {fail * 100:5.1f}% (mean {sum(v) / max(len(v), 1):.2f})')
            print(f'    {n:<17} ' + '   '.join(cells))
    print('  Read: if MALE sources still fail often with male_noglasses at the largest alpha,\n'
          '        the direction itself has no structural aging for men -> rebuild the bank.\n'
          '        If male_noglasses beats average/female_noglasses on men, routing matters.')

    os.makedirs(args.out_dir, exist_ok=True)
    labels = ['source'] + [n for n in names]
    for g in (True, False):
        montage(mont[g], labels, os.path.join(
            args.out_dir, f'raw_dirs_{gname[g].lower()}_a{args.montage_alpha:g}.png'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bank_only', action='store_true', help='Part A only; no checkpoint needed.')
    p.add_argument('--checkpoint_dir', default=None)
    p.add_argument('--step', type=int, default=None)
    p.add_argument('--num_per_gender', type=int, default=100)
    p.add_argument('--alphas', nargs='*', type=float, default=[1.0, 1.5, 2.0])
    p.add_argument('--montage_alpha', type=float, default=1.5)
    p.add_argument('--montage_n', type=int, default=8)
    p.add_argument('--edit_scale', type=float, default=1.0)
    p.add_argument('--edit_target', default='mirror', choices=['mirror', 'train'])
    p.add_argument('--out_dir', default='./age_routing_probe')
    p.add_argument('--batch', type=int, default=4)

    p.add_argument('--index_file', default='./data/ffhq.txt')
    p.add_argument('--image_root', default='data/FFHQ')
    p.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    p.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth')
    p.add_argument('--stygan2_weights', default='./data/stylegan2-ffhq-config-f.pt')
    p.add_argument('--attribute_weights', default='./data/r34_a40_age_256_classifier.pth')
    p.add_argument('--face_parser_weights', default='./data/parsing_bisenet.pth')
    p.add_argument('--direction_bank_path', default=None)
    p.add_argument('--clip_judge_model', default='ViT-L/14')
    p.add_argument('--clip_calibration', default=None)

    # model-structure flags; auto-restored from the run's config.json
    p.add_argument('--img_size', type=int, default=512)
    p.add_argument('--attribute_index', nargs='*', type=int, default=[15, 20, 39])
    p.add_argument('--flow_modules', default='512-512-512-512-512')
    p.add_argument('--num_blocks', type=int, default=1)
    p.add_argument('--velocity_field', default='lag_dof')
    p.add_argument('--id_cond_dim', type=int, default=32)
    p.add_argument('--id_cond_scale', type=float, default=0.25)
    p.add_argument('--attr_backbone', default='resnet50')
    p.add_argument('--conditioner_backbone', default='resnet',
                   choices=['resnet', 'clip', 'resnet_clip'])
    p.add_argument('--clip_model', default='ViT-B/32')
    p.add_argument('--fused_hidden_dim', type=int, default=256)
    p.add_argument('--lag_gate_hidden_dim', type=int, default=64)
    p.add_argument('--lag_gate_init_bias', type=float, default=-0.5)
    p.add_argument('--direction_residual_scale', type=float, default=0.05)
    p.add_argument('--glasses_residual_scale', type=float, default=0.05)
    p.add_argument('--guided_delta_max_norm', type=float, default=0.0)
    p.add_argument('--override_residual_scale', type=float, default=None)
    p.add_argument('--age_fine_layer_scale', type=float, default=None)
    p.add_argument('--age_fine_layer_start', type=int, default=10)
    p.add_argument('--force_bank_directions', action='store_true')
    p.add_argument('--disable_controlnet', action='store_true')
    p.add_argument('--ignore_run_config', action='store_true')
    args = p.parse_args()
    if args.montage_alpha not in args.alphas:
        args.alphas.append(args.montage_alpha)

    if args.bank_only:
        if not args.direction_bank_path:
            p.error('--bank_only needs --direction_bank_path')
        analyse_bank(args.direction_bank_path)
        return

    if not args.checkpoint_dir:
        p.error('--checkpoint_dir is required unless --bank_only')
    from evaluation.evaluate_sdflow import _latest_step, apply_run_config
    args = apply_run_config(args)
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
    if not args.direction_bank_path:
        p.error('no --direction_bank_path (and none in the run config.json)')
    info = analyse_bank(args.direction_bank_path)
    run_model_parts(args, info)


if __name__ == '__main__':
    main()
