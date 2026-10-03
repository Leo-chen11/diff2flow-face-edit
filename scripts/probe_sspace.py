"""Does editing a few StyleSpace (S) channels beat a W+ direction on the
identity / accuracy trade-off?

S space: the per-channel style coefficients each StyleGAN2 layer computes
from w with its own affine (`modulation`). A W+ direction moves every S
channel of a layer at once (the affine is dense); S lets an edit move only
the channels that carry the attribute and leave the ones that carry
identity alone (StyleSpace, Wu et al. CVPR 2021).

For each attribute (default Eyeglasses 15, Male 20, Young 39):
  1. Statistics on the TRAINING split, from the e4e latents and the stored
     predictions only (no rendering): per channel, the mean difference
     between faces with and without the attribute, within strata of a
     confounder (gender for Young / Eyeglasses, Young for Male), divided by
     the within-group std. |z| high = the channel moves with the attribute
     but varies little between different people of the same group, i.e.
     attribute rather than identity. Young uses the classifier's age bins
     (index 40 of the preds) so "young" / "old" are clean groups.
  2. Edits on held-out TEST faces, every family on the SAME faces:
       wplus         mean W+ difference of the same groups, all 18 layers
       s_top{K}      mean S difference restricted to the K highest-|z|
                     conv channels (tRGB excluded unless --include_torgb)
     each at every --alphas strength (1.0 = the full average group
     difference). Direction per face from the evaluation classifier: an
     attribute the source has is removed, one it lacks is added.
     Sanity check: s_top{all} applies the same group difference as wplus
     through the affines, so the two curves should nearly coincide.
  3. Per family: accuracy (ResNet-50 evaluation classifier, and CLIP),
     ID_ind (facenet casia vs the reconstruction), leak (mean |change| of the
     other attributes' classifier probabilities), interpolated to accuracy
     at ID_ind 0.85 / 0.80 / 0.75, plus a montage per attribute with the
     setting of each family closest to ID_ind 0.80.

Read-only. Needs G, the latents / preds, the evaluation classifier, facenet
and CLIP. No SDFlow checkpoint.

Usage:
    python -m scripts.probe_sspace \
        --independent_attr_weights ./data/r50_celebahq_eval.pth \
        --clip_calibration 39:0.517:0.132 --out_dir ./sspace_probe
"""
import argparse
import json
import os
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'stylegan2'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

CELEBA_ATTRS = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
    'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
    'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
    'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
    'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
    'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
    'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
    'Wearing_Necklace', 'Wearing_Necktie', 'Young',
]
ATTR_NAMES = {i: n for i, n in enumerate(CELEBA_ATTRS)}
# Stratify each attribute's statistics by its main confounder.
# Default for any other attribute: gender.
STRATUM_ATTR = {39: 20, 15: 20, 20: 39}


def stratum_attr(attr):
    return STRATUM_ATTR.get(attr, 20)


# ── S space access ──────────────────────────────────────────────────────────

class StyleSpace:
    """The generator's modulation layers, in forward order, with the W+ index
    each one reads. styles(w) computes S directly (no rendering); set_delta
    makes the next G forward add a per-sample offset to chosen channels."""

    def __init__(self, G):
        mods = [(G.conv1.conv.modulation, 0, 'conv', 4), (G.to_rgb1.conv.modulation, 1, 'rgb', 4)]
        i, res = 1, 4
        for k in range(len(G.to_rgbs)):
            res *= 2
            mods.append((G.convs[2 * k].conv.modulation, i, 'conv', res))
            mods.append((G.convs[2 * k + 1].conv.modulation, i + 1, 'conv', res))
            mods.append((G.to_rgbs[k].conv.modulation, i + 2, 'rgb', res))
            i += 2
        self.mods = mods
        self.sizes = [m.weight.shape[0] for m, _, _, _ in mods]
        self.offsets = [0]
        for s in self.sizes:
            self.offsets.append(self.offsets[-1] + s)
        self.total = self.offsets[-1]
        self.kind = torch.cat([torch.full((s,), 0 if k == 'conv' else 1, dtype=torch.long)
                               for s, (_, _, k, _) in zip(self.sizes, mods)])
        self._delta = None
        self._captured = None
        self.handles = [m.register_forward_hook(self._hook(j)) for j, (m, _, _, _) in enumerate(mods)]

    def _hook(self, j):
        def fn(_module, _inp, out):
            if self._captured is not None:
                self._captured.append(out.detach())
            if self._delta is not None:
                lo, hi = self.offsets[j], self.offsets[j + 1]
                return out + self._delta[:, lo:hi].to(out.dtype)
            return None
        return fn

    @torch.no_grad()
    def styles(self, w):
        """w (B, 18, 512) -> S (B, total)."""
        return torch.cat([m(w[:, idx]) for m, idx, _, _ in self.mods], dim=1)

    def set_delta(self, delta):
        self._delta = delta

    def describe(self, flat_idx):
        j = next(j for j in range(len(self.mods)) if self.offsets[j] <= flat_idx < self.offsets[j + 1])
        _, widx, kind, res = self.mods[j]
        return f'{kind}{res}(w{widx})#{flat_idx - self.offsets[j]}'

    def res_of(self, flat_idx):
        j = next(j for j in range(len(self.mods)) if self.offsets[j] <= flat_idx < self.offsets[j + 1])
        return self.mods[j][3]


# ── statistics ──────────────────────────────────────────────────────────────

def _parse_bins(spec):
    lo, _, hi = spec.partition('-')
    return int(lo), int(hi or lo)


def group_of(pred, attr, args):
    """1 = attribute present, 0 = absent, None = excluded. Young uses the
    classifier age bin (index 40) when present, so the groups are clean."""
    if attr == 39 and pred.numel() > 40:
        ab = int(round(float(pred[40])))
        y = float(pred[39]) >= 0.5
        yl, yh = _parse_bins(args.young_bins)
        ol, oh = _parse_bins(args.old_bins)
        if y and yl <= ab <= yh:
            return 1
        if (not y) and ol <= ab <= oh:
            return 0
        return None
    return int(float(pred[attr]) >= 0.5)


@torch.no_grad()
def attribute_stats(ss, ds, attr, args, device):
    """Per stratum and group: running sums of S and W+. Returns
    (d_s (total,), z_s (total,), d_w (18, 512), counts)."""
    strat_attr = stratum_attr(attr)
    buckets = defaultdict(list)
    for f in ds.image_list:
        pred = ds._lookup_precomputed(ds.preds, f)
        g = group_of(pred, attr, args)
        if g is None:
            continue
        buckets[(int(float(pred[strat_attr]) >= 0.5), g)].append(f)
    gen = torch.Generator().manual_seed(args.seed)
    counts = {}
    sums = {}
    for key, files in buckets.items():
        if len(files) > args.num_stat:
            files = [files[i] for i in torch.randperm(len(files), generator=gen)[:args.num_stat].tolist()]
        counts[key] = len(files)
        s1 = torch.zeros(ss.total, device=device, dtype=torch.float64)
        s2 = torch.zeros_like(s1)
        w1 = torch.zeros(18, 512, device=device, dtype=torch.float64)
        for i in range(0, len(files), 256):
            w = torch.stack([ds._lookup_precomputed(ds.latents, f) for f in files[i:i + 256]]).to(device)
            s = ss.styles(w.float()).double()
            s1 += s.sum(0)
            s2 += (s * s).sum(0)
            w1 += w.double().sum(0)
        sums[key] = (s1, s2, w1)
    strata = sorted({k[0] for k in sums if (k[0], 0) in sums and (k[0], 1) in sums
                     and counts[(k[0], 0)] >= 20 and counts[(k[0], 1)] >= 20})
    if not strata:
        raise SystemExit(f'attr {attr}: no stratum has >= 20 faces in both groups')
    d_s, z_s, d_w = [], [], []
    for st in strata:
        (a1, a2, aw), (b1, b2, bw) = sums[(st, 1)], sums[(st, 0)]
        na, nb = counts[(st, 1)], counts[(st, 0)]
        ma, mb = a1 / na, b1 / nb
        va = (a2 / na - ma * ma).clamp(min=0)
        vb = (b2 / nb - mb * mb).clamp(min=0)
        pooled = ((na * va + nb * vb) / (na + nb)).sqrt().clamp(min=1e-6)
        d_s.append(ma - mb)
        z_s.append((ma - mb) / pooled)
        d_w.append(aw / na - bw / nb)
    per = {st: (d_s[i].float(), z_s[i].float(), d_w[i].float()) for i, st in enumerate(strata)}
    d_s = torch.stack(d_s).mean(0).float()
    z = torch.stack(z_s)
    # A channel whose sign flips between strata is not a consistent carrier.
    agree = (torch.sign(z) == torch.sign(z[0:1])).all(0)
    z_s = torch.where(agree, z.mean(0), torch.zeros_like(z[0])).float()
    d_w = torch.stack(d_w).mean(0).float()
    return (d_s, z_s, d_w, {f'{st}/{g}': counts[(st, g)] for st in strata for g in (0, 1)}, per)


# ── evaluation ──────────────────────────────────────────────────────────────

def to_pil(x, size):
    x = ((x.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x).resize((size, size))


def alpha_at_acc(accs, alphas, t):
    """Smallest strength reaching accuracy t: first grid point at or above t,
    linearly interpolated with the one before. None if never reached."""
    for i, (acc, a) in enumerate(zip(accs, alphas)):
        if acc >= t:
            if i == 0:
                return a
            a0, c0 = alphas[i - 1], accs[i - 1]
            return a0 + (t - c0) / max(acc - c0, 1e-9) * (a - a0)
    return None


def value_at_alpha(values, alphas, a):
    return interp_at(list(zip(alphas, values)), a)


def interp_at(points, x):
    pts = sorted(points)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return None


@torch.no_grad()
def run_attribute(attr, ss, G, test_ds, stats, judges, args, device):
    ind_judge, clip_judge, id_judge = judges
    d_s, z_s, d_w, _, per = stats
    # Groups whose statistics drive the edit: one shared set, or with
    # --per_stratum each stratum (e.g. men / women) its own directions and
    # its own top-K channels, picked per face from the classifier.
    groups = ({st: v for st, v in per.items()} if args.per_stratum
              else {'all': (d_s, z_s, d_w)})
    ks = [int(k) for k in args.k_list if k != 'all']
    fam_names = ['wplus'] + [f's_top{k}' for k in ks] + ['s_topall']
    fam_mask = defaultdict(dict)       # family -> group -> mask
    for g, (_, zg, _) in groups.items():
        score = zg.abs().clone()
        if not args.include_torgb:
            score[ss.kind.to(score.device) == 1] = 0
        og = torch.argsort(score, descending=True)
        n_valid = int((score > 0).sum())
        for k in ks:
            m = torch.zeros(ss.total, device=device)
            m[og[:min(k, n_valid)]] = 1
            fam_mask[f's_top{k}'][g] = m
        fam_mask['s_topall'][g] = (score > 0).float()
    families = [(f, None) for f in fam_names]
    configs = [(fam, a) for fam in fam_names for a in args.alphas]
    strat_attr = stratum_attr(attr)
    others = [a for a in args.attrs if a != attr]
    # CLIP only scores attributes it has a prompt pair for.
    if clip_judge is not None and attr not in clip_judge.attribute_index:
        clip_judge = None
    clip_idx = clip_judge.attribute_index.index(attr) if clip_judge is not None else None

    watch = [a for a in args.watch_attrs if a != attr]

    def render(fam, a, lat, sign, gkey):
        B = lat.size(0)
        if fam == 'wplus':
            ss.set_delta(None)
            dw = torch.stack([groups[k][2] for k in gkey])
            return G([lat + sign.view(B, 1, 1) * a * dw], input_is_latent=True,
                     randomize_noise=False)[0].clamp(-1, 1)
        ds = torch.stack([groups[k][0] for k in gkey])
        mask_b = torch.stack([fam_mask[fam][k] for k in gkey])
        ss.set_delta(sign.view(B, 1) * a * mask_b * ds)
        out = G([lat], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
        ss.set_delta(None)
        return out

    st = defaultdict(lambda: defaultdict(list))
    mont_lat, mont_sign, mont_key, src_tiles = [], [], [], []
    tile = 160
    seen = 0
    files = list(test_ds.image_list)
    for i in range(0, len(files), args.batch):
        if seen >= args.num_samples:
            break
        lat = torch.stack([test_ds._lookup_precomputed(test_ds.latents, f)
                           for f in files[i:i + args.batch]]).to(device).float()
        ss.set_delta(None)
        src = G([lat], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
        src256 = F.interpolate(src, (256, 256))
        p_src = torch.sigmoid(ind_judge(src256)[0][:, :40])
        pa = p_src[:, attr]
        clear = (pa > 0.65) | (pa < 0.35)
        if not clear.any():
            continue
        has = pa >= 0.5
        male = p_src[:, 20] >= 0.5
        sign = torch.where(has, -torch.ones_like(pa), torch.ones_like(pa))
        gkey = ([int(v) for v in (p_src[:, strat_attr] >= 0.5).tolist()] if args.per_stratum
                else ['all'] * lat.size(0))
        gkey = [k if k in groups else next(iter(groups)) for k in gkey]
        src_id = id_judge.extract(src256)
        B = lat.size(0)
        mont = [b for b in range(B) if bool(clear[b])]
        if attr == 39:
            mont = [b for b in mont if bool(has[b])]      # aging is the case that matters
        mont = mont[:max(0, args.montage_n - len(src_tiles))]
        for b in mont:
            src_tiles.append(to_pil(src256[b], tile))
            mont_lat.append(lat[b])
            mont_sign.append(sign[b])
            mont_key.append(gkey[b])
        for fam, a in configs:
            e = render(fam, a, lat, sign, gkey)
            e256 = F.interpolate(e, (256, 256))
            p_e = torch.sigmoid(ind_judge(e256)[0][:, :40])
            idc = (src_id * id_judge.extract(e256)).sum(1)
            pc = clip_judge.scores(e256)[:, clip_idx] if clip_judge is not None else None
            s = st[(fam, a)]
            for b in range(B):
                if not bool(clear[b]):
                    continue
                rm = bool(has[b])
                ok = float(p_e[b, attr] < 0.5) if rm else float(p_e[b, attr] >= 0.5)
                s['ok'].append(ok)
                s['ok_rm' if rm else 'ok_add'].append(ok)
                if rm:
                    s['ok_rm_m' if bool(male[b]) else 'ok_rm_f'].append(ok)
                s['id'].append(idc[b].item())
                if pc is not None:
                    s['clip'].append(float(pc[b] < 0.5) if rm else float(pc[b] >= 0.5))
                if others:
                    s['leak'].append(sum(abs(p_e[b, o] - p_src[b, o]).item() for o in others) / len(others))
                if watch:
                    s['watch'].append(sum(abs(p_e[b, o] - p_src[b, o]).item() for o in watch) / len(watch))
        seen += int(clear.sum())

    def mean(v):
        return sum(v) / len(v) if v else float('nan')

    name = ATTR_NAMES.get(attr, str(attr))
    rows = {}
    print(f'\n=== {name} ({attr}): {seen} clear test faces; rm = source has it'
          + (f'; per-stratum directions/channels by attr {strat_attr}' if args.per_stratum else '')
          + ' ===')
    hdr = f'{"family":<12}{"alpha":>6}  {"AccInd":>7} {"rm":>6} {"add":>6}'
    if attr == 39:
        hdr += f' {"rm_M":>6} {"rm_F":>6}'
    hdr += f' {"AccCLIP":>8} {"ID_ind":>7} {"leak":>6} {"watch":>6}'
    print(hdr)
    for fam, a in configs:
        s = st[(fam, a)]
        r = dict(acc=mean(s['ok']), rm=mean(s['ok_rm']), add=mean(s['ok_add']),
                 rm_m=mean(s['ok_rm_m']), rm_f=mean(s['ok_rm_f']), clip=mean(s['clip']),
                 id=mean(s['id']), leak=mean(s['leak']), watch=mean(s['watch']))
        rows[(fam, a)] = r
        line = (f'{fam:<12}{a:>6g}  {r["acc"] * 100:6.1f}% {r["rm"] * 100:5.1f}% '
                f'{r["add"] * 100:5.1f}%')
        if attr == 39:
            line += f' {r["rm_m"] * 100:5.1f}% {r["rm_f"] * 100:5.1f}%'
        line += f' {r["clip"] * 100:7.1f}% {r["id"]:7.4f} {r["leak"]:6.3f} {r["watch"]:6.3f}'
        print(line)

    print(f'  (leak = other edited attributes; watch = mean |change| of CelebA attrs {watch})')
    print(f'\n  {name}: accuracy at matched ID_ind (interpolated over alphas; -- = out of range)')
    frontier = {}
    for fam, _ in families:
        cells = []
        for t in args.id_targets:
            pts_i = [(rows[(fam, a)]['id'], rows[(fam, a)]['acc']) for a in args.alphas]
            pts_c = [(rows[(fam, a)]['id'], rows[(fam, a)]['clip']) for a in args.alphas]
            pts_l = [(rows[(fam, a)]['id'], rows[(fam, a)]['leak']) for a in args.alphas]
            pts_w = [(rows[(fam, a)]['id'], rows[(fam, a)]['watch']) for a in args.alphas]
            vi, vc = interp_at(pts_i, t), interp_at(pts_c, t)
            vl, vw = interp_at(pts_l, t), interp_at(pts_w, t)
            frontier.setdefault(fam, {})[f'{t:.2f}'] = dict(acc_ind=vi, acc_clip=vc, leak=vl, watch=vw)
            cells.append(f'@{t:.2f} ' + (f'Ind {vi * 100:5.1f}% CLIP {vc * 100:5.1f}% '
                                          f'leak {vl:.3f} watch {vw:.3f}'
                                          if None not in (vi, vc, vl, vw) else '      --      '))
        print(f'    {fam:<12} ' + '   '.join(cells))

    # Matched ACCURACY (the SDFlow / Latent Transformer protocol): the weakest
    # strength that reaches each target accuracy, and the identity / leak it
    # costs there. For easy attributes (e.g. Smiling reaches ~100% at ID 0.9)
    # this is the meaningful comparison; ID 0.80 would force over-editing.
    alphas_sorted = sorted(args.alphas)
    print(f'\n  {name}: identity and leak at matched AccInd (weakest strength reaching it)')
    for fam, _ in families:
        cells = []
        accs = [rows[(fam, a)]['acc'] for a in alphas_sorted]
        for t in args.acc_targets:
            a_t = alpha_at_acc(accs, alphas_sorted, t)
            if a_t is None:
                cells.append(f'@{t * 100:.0f}%        --        ')
                frontier.setdefault(fam, {})[f'acc{t:.2f}'] = None
                continue
            v = {k: value_at_alpha([rows[(fam, a)][k] for a in alphas_sorted], alphas_sorted, a_t)
                 for k in ('id', 'leak', 'watch', 'clip')}
            frontier.setdefault(fam, {})[f'acc{t:.2f}'] = dict(alpha=a_t, **v)
            cells.append(f'@{t * 100:.0f}% a{a_t:.2f} ID {v["id"]:.3f} leak {v["leak"]:.3f} '
                         f'watch {v["watch"]:.3f}')
        print(f'    {fam:<12} ' + '   '.join(cells))

    if src_tiles:
        # Every family rendered at the same operating point (--montage_at):
        # matched accuracy (acc:0.90) or matched identity (id:0.80).
        kind, _, val = args.montage_at.partition(':')
        val = float(val)
        lat_m = torch.stack(mont_lat)
        sign_m = torch.stack(mont_sign)
        cols, col_tiles = ['source'], []
        for fam, _ in families:
            if kind == 'acc':
                a_star = alpha_at_acc([rows[(fam, a)]['acc'] for a in alphas_sorted], alphas_sorted, val)
                if a_star is None:
                    a_star = alphas_sorted[-1]
            else:
                a_star = interp_at([(rows[(fam, a)]['id'], a) for a in args.alphas], val)
                if a_star is None:
                    a_star = min(args.alphas, key=lambda a: abs(rows[(fam, a)]['id'] - val))
            imgs = torch.cat([render(fam, a_star, lat_m[j:j + args.batch], sign_m[j:j + args.batch],
                                     mont_key[j:j + args.batch])
                              for j in range(0, lat_m.size(0), args.batch)])
            col_tiles.append([to_pil(F.interpolate(x[None], (256, 256))[0], tile) for x in imgs])
            cols.append(f'{fam} a{a_star:.2f}')
        os.makedirs(args.out_dir, exist_ok=True)
        canvas = Image.new('RGB', (tile * len(cols), tile * len(src_tiles) + 16), (20, 20, 20))
        dr = ImageDraw.Draw(canvas)
        for c, lab in enumerate(cols):
            dr.text((c * tile + 3, 2), lab, fill=(255, 255, 0))
        for r in range(len(src_tiles)):
            canvas.paste(src_tiles[r], (0, 16 + r * tile))
            for c, tl in enumerate(col_tiles, start=1):
                canvas.paste(tl[r], (c * tile, 16 + r * tile))
        path = os.path.join(args.out_dir, f'sspace_{name}.png')
        canvas.save(path)
        print(f'  montage -> {path}  (every family at {args.montage_at}: '
              + ('weakest strength reaching that AccInd)' if kind == 'acc'
                 else 'strength interpolated to that mean ID_ind)'))

    for g, (_, zg, _) in groups.items():
        score = zg.abs().clone()
        if not args.include_torgb:
            score[ss.kind.to(score.device) == 1] = 0
        og = torch.argsort(score, descending=True)
        tag = '' if g == 'all' else f' [stratum {strat_attr}={g}]'
        print(f'\n  {name}{tag}: top {args.show_top} channels by |z| '
              + ', '.join(f'{ss.describe(int(j))} z={zg[j]:+.2f}' for j in og[:args.show_top]))
        for kk in ks:
            res_hist = defaultdict(int)
            for j in og[:kk].tolist():
                res_hist[ss.res_of(j)] += 1
            print(f'  top{kk} by resolution: ' + ' '.join(f'{r}:{n}' for r, n in sorted(res_hist.items())))
    return {'rows': {f'{f}@{a:g}': r for (f, a), r in rows.items()}, 'frontier': frontier,
            'faces': seen}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--attrs', nargs='*', type=int, default=[39, 20, 15],
                   help='Any CelebA attribute ids (0-39); strata by gender unless listed in '
                        'STRATUM_ATTR.')
    p.add_argument('--k_list', nargs='*', default=['25', '50', '100', '200', '400', '800', 'all'])
    p.add_argument('--alphas', nargs='*', type=float,
                   default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0])
    p.add_argument('--acc_targets', nargs='*', type=float, default=[0.90, 0.95],
                   help='Report ID / leak at the weakest strength reaching these AccInd values.')
    p.add_argument('--montage_at', default='acc:0.90',
                   help="Montage operating point per family: 'acc:0.90' (weakest strength "
                        "reaching that AccInd) or 'id:0.80' (strength giving that mean ID_ind).")
    p.add_argument('--watch_attrs', nargs='*', type=int, default=[22, 24, 18, 36],
                   help='CelebA attrs whose unwanted change is reported (default Mustache, '
                        'No_Beard, Heavy_Makeup, Wearing_Lipstick).')
    p.add_argument('--id_targets', nargs='*', type=float, default=[0.85, 0.80, 0.75])
    p.add_argument('--per_stratum', action='store_true',
                   help='Separate directions and top-K channels per stratum (gender for Young / '
                        'Eyeglasses, Young for Male), chosen per face by the classifier.')
    p.add_argument('--include_torgb', action='store_true',
                   help='Allow tRGB (colour) channels in the top-K selection.')
    p.add_argument('--young_bins', default='2-2', help='Classifier age bins counted as young.')
    p.add_argument('--old_bins', default='6-6', help='Classifier age bins counted as old.')
    p.add_argument('--num_stat', type=int, default=3000, help='Max faces per stratum x group.')
    p.add_argument('--num_samples', type=int, default=150, help='Clear test faces per attribute.')
    p.add_argument('--batch', type=int, default=6)
    p.add_argument('--montage_n', type=int, default=8)
    p.add_argument('--show_top', type=int, default=12)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', default='./sspace_probe')
    p.add_argument('--independent_attr_weights', required=True,
                   help='Evaluation classifier (e.g. ./data/r50_celebahq_eval.pth).')
    p.add_argument('--independent_attr_backbone', default='r50')
    p.add_argument('--clip_judge_model', default='ViT-L/14')
    p.add_argument('--clip_calibration', default=None)
    p.add_argument('--no_clip', action='store_true')
    p.add_argument('--id_indep_pretrained', default='casia-webface')
    p.add_argument('--index_file', default='./data/ffhq.txt')
    p.add_argument('--image_root', default='data/FFHQ')
    p.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    p.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth')
    p.add_argument('--stygan2_weights', default='./data/stylegan2-ffhq-config-f.pt')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    from common.ops import load_network
    from evaluation.evaluate_sdflow import (CLIPAttributeJudge, IndependentIDJudge,
                                            parse_clip_calibration)
    from models.attribute_estimator import AttributeClassifier
    from models.dataset import SDFlowDataset
    from models.stylegan2.model import Generator

    device = args.device
    ckpt = torch.load(args.stygan2_weights, map_location='cpu')
    G = Generator(size=1024, style_dim=512, n_mlp=8)
    G.load_state_dict(ckpt['g_ema'])
    G.to(device).eval()
    ss = StyleSpace(G)
    print(f'S space: {ss.total} channels ({int((ss.kind == 0).sum())} conv, '
          f'{int((ss.kind == 1).sum())} tRGB) over {len(ss.mods)} layers')

    ind_judge = AttributeClassifier(backbone=args.independent_attr_backbone)
    ind_judge.load_state_dict(load_network(args.independent_attr_weights))
    ind_judge.to(device).eval()
    clip_attrs = [a for a in args.attrs if a in CLIPAttributeJudge.PROMPTS]
    clip_judge = None if (args.no_clip or not clip_attrs) else CLIPAttributeJudge(
        clip_attrs, args.clip_judge_model, device,
        calibration=parse_clip_calibration(args.clip_calibration))
    no_prompt = [a for a in args.attrs if a not in clip_attrs]
    if no_prompt and not args.no_clip:
        print(f'[note] no CLIP prompt for {no_prompt}: AccCLIP reported as nan for those')
    id_judge = IndependentIDJudge(device, pretrained=args.id_indep_pretrained)

    train_ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                             latents_file=args.latent_file, preds_file=args.preds_file, train=True)
    test_ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                            latents_file=args.latent_file, preds_file=args.preds_file, train=False)

    # Self-check: S computed from the affines == what the forward pass feeds the convs.
    w0 = torch.stack([test_ds._lookup_precomputed(test_ds.latents, f)
                      for f in test_ds.image_list[:2]]).to(device).float()
    ss._captured = []
    with torch.no_grad():
        G([w0], input_is_latent=True, randomize_noise=False)
    cap = torch.cat(ss._captured, dim=1)
    ss._captured = None
    err = (cap - ss.styles(w0)).abs().max().item()
    print(f'S self-check: max |forward - direct| = {err:.2e}')
    if err > 1e-3:
        raise SystemExit('S mapping does not match the generator forward pass')

    results = {}
    for attr in args.attrs:
        stats = attribute_stats(ss, train_ds, attr, args, device)
        print(f'\n{ATTR_NAMES.get(attr, attr)}: statistics from groups {stats[3]} '
              f'(stratum/present); |z|>1: {int((stats[1].abs() > 1).sum())} channels')
        results[ATTR_NAMES.get(attr, str(attr))] = run_attribute(
            attr, ss, G, test_ds, stats, (ind_judge, clip_judge, id_judge), args, device)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'sspace_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nresults -> {path}')


if __name__ == '__main__':
    main()
