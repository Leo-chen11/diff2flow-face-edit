"""Paper-style comparison table (editing accuracy, ID, SSIM, LPIPS, FID, CLIP ...)
with every method read at the SAME editing accuracy.

SSIM, LPIPS, FID and ID all reward editing less (an unedited face scores SSIM 1,
LPIPS 0, FID 0, ID 1), so a table of them is only meaningful once the methods
have edited equally much. Each method is therefore run at the strength where its
own curve (scripts/editing_curves.py CSV) reaches --target_acc editing accuracy,
per attribute, and everything else is measured there on the same faces:

  Acc      R50 judge: the edited attribute crossed 0.5 in the requested direction
           (r18: the CelebA ResNet-18, on faces both judges read clearly the same way)
  ID       cosine similarity, ArcFace IR-SE50 (facenet when --arcface_weights is missing)
  SSIM     Gaussian-window SSIM (11x11, sigma 1.5) on the 256 px RGB faces in [0, 1]
  LPIPS    AlexNet LPIPS on the 256 px faces
  Pres     share of the OTHER attributes whose R50 label did not change
  CLIP     cosine similarity between the edit and the prompt of the target state
           (evaluate_sdflow.CLIPAttributeJudge prompts); dCLIP = minus the same for the
           unedited face, the number to read since the raw value is narrow
  FID/KID  Inception features (torchmetrics' FID network) of the edited faces against the
           unedited reconstructions of the same faces, per attribute; KID is the polynomial-
           kernel MMD^2 (x1000) with the biased (V-statistic) estimator, which is 0 for two
           identical sets -- the unbiased one goes negative for paired, nearly identical sets.
           Both are biased upward for small --num_samples, the same way for every method:
           compare them, do not quote them as absolute.

Every number is measured against the unedited reconstruction G(e4e latent), not the real
photo, so the inversion error does not enter. Faces are the ones the R50 judge reads
clearly on the source (the same for all methods). Marks: bold = best (or tied with the best:
paired difference within 1.96 standard errors), underline = best of the rest; Acc is the
matched quantity and is not ranked. A "No-edit" row (kind noedit) shows what the
fidelity metrics give for doing nothing.

    python -m scripts.comparison_table --checkpoint_dir output/SDFlow/<run> --step 100000 \\
        --independent_attr_weights ./data/r50_celebahq_eval.pth --independent_attr_backbone r50 \\
        --only_attrs 15 20 39 --target_acc 60 80 --num_samples 1000 \\
        --method_spec "No-edit:noedit" \\
        --method_spec "InterFaceGAN:boundary:output/curves/interfacegan.csv" \\
        --method_spec "SDFlow:sdflow:output/curves/sdflow.csv" \\
        --method_spec "Ours:model:output/curves/ours.csv"

A method whose curve does not reach the target for an attribute gets "--" there.
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils import data
from tqdm import tqdm

import evaluation.evaluate_sdflow as E
from evaluation.evaluate_sdflow import (
    ATTR_NAMES, _latest_step, apply_run_config, build_optional_judges, build_parser,
    edit_single_attribute, load_models, resize_for_judge,
)
from models.arcface import ArcFaceJudge
from models.dataset import SDFlowDataset
from scripts.editing_curves import (
    CLEAR_HI, CLEAR_LO, OfficialSDFlow, boundary_directions, linear_directions,
)

# key, header, better (+1 higher, -1 lower, 0 = matched quantity, not ranked), scale, decimals
METRICS = [
    ('acc', 'Acc R50 %', 0, 100, 1),
    ('acc_celeb', 'Acc r18 %', 0, 100, 1),
    ('id', 'ID', +1, 1, 3),
    ('ssim', 'SSIM', +1, 1, 3),
    ('lpips', 'LPIPS', -1, 1, 3),
    ('fid', 'FID', -1, 1, 1),
    ('kid', 'KID x1e3', -1, 1000, 2),
    ('pres', 'Pres %', +1, 100, 1),
    ('clip', 'CLIP', +1, 1, 3),
    ('clip_d', 'dCLIP', +1, 1, 3),
]
PAPER_COLUMNS = ['acc', 'id', 'ssim', 'lpips', 'fid', 'clip_d']
PER_FACE = ('acc', 'acc_celeb', 'id', 'ssim', 'lpips', 'pres', 'clip', 'clip_d')
KINDS = ('model', 'boundary', 'linear', 'sdflow', 'noedit')


def name(g):
    return ATTR_NAMES.get(g, f'attr{g}')


# ── metrics ────────────────────────────────────────────────────────────────

def gaussian_ssim(x, y, win=11, sigma=1.5, data_range=1.0):
    """Per-image SSIM of (B, 3, H, W) tensors in [0, data_range]: Gaussian window,
    population covariance, 'valid' borders, mean over channels and pixels (the
    skimage settings gaussian_weights=True, sigma=1.5, use_sample_covariance=False)."""
    c = x.size(1)
    g = torch.exp(-(torch.arange(win, dtype=x.dtype, device=x.device) - win // 2) ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    k = (g[:, None] * g[None, :]).expand(c, 1, win, win).contiguous()

    def blur(t):
        return F.conv2d(t, k, groups=c)

    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return s.flatten(1).mean(1)


def fid_from_features(f1, f2):
    """Frechet distance of two (n, d) feature sets (float64 numpy)."""
    f1, f2 = np.asarray(f1, dtype=np.float64), np.asarray(f2, dtype=np.float64)
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    eig = np.linalg.eigvals(s1 @ s2)
    tr_covmean = np.sqrt(eig.astype(complex)).real.sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_covmean)


def kid_from_features(f1, f2):
    """Biased (V-statistic) MMD^2 with the polynomial kernel (x.y / d + 1)^3 of KID
    (Binkowski et al.). Unlike the unbiased estimator it is >= 0 and exactly 0 for
    identical sets, which suits an edit set paired with its own unedited faces."""
    f1, f2 = np.asarray(f1, dtype=np.float64), np.asarray(f2, dtype=np.float64)
    d = f1.shape[1]

    def k(a, b):
        return (a @ b.T / d + 1.0) ** 3

    return float(k(f1, f1).mean() + k(f2, f2).mean() - 2.0 * k(f1, f2).mean())


def build_inception(use):
    """features(x in [-1, 1], (B, 3, H, W)) -> (B, 2048) numpy, with torchmetrics' FID network
    (the one evaluate_sdflow's FID uses), or None."""
    if not use:
        return None
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        net = FrechetInceptionDistance(feature=2048, normalize=True).cuda().inception
    except Exception as exc:        # missing package or weights that cannot be fetched
        print(f'[WARN] FID / KID unavailable ({type(exc).__name__}: {exc})')
        return None

    @torch.no_grad()
    def features(x):
        return net(((x + 1.0) * 0.5).clamp(0, 1).mul(255).byte()).double().cpu().numpy()
    return features


# ── operating point ────────────────────────────────────────────────────────

def strength_at(csv_path, attr_name, target_pct, metric='target_success'):
    """The strength at which the method's measured curve crosses `target_pct` editing
    accuracy (linear interpolation between measured strengths, no extrapolation); None
    when the curve does not reach it."""
    pts = []
    with open(csv_path, newline='') as f:
        for r in csv.DictReader(f):
            if r['attribute'] == attr_name and r.get(metric) not in (None, '', 'nan'):
                pts.append((float(r[metric]) * 100.0, float(r['strength'])))
    if not pts:
        return None
    pts.sort()
    acc = np.array([p[0] for p in pts])
    if target_pct < acc.min() - 1e-9 or target_pct > acc.max() + 1e-9:
        return None
    return float(np.interp(target_pct, acc, [p[1] for p in pts]))


# ── table ──────────────────────────────────────────────────────────────────

def rank_marks(values, better, diff_se=None):
    """{method: 'best' | 'second' | ''} for {method: value or None}. A method tied with
    the best (|difference| <= 1.96 * diff_se(best, method)) is 'best' too."""
    have = {m: v for m, v in values.items() if v is not None and v == v}
    marks = {m: '' for m in values}
    if better == 0 or len(have) < 2:
        return marks
    order = sorted(have, key=lambda m: -better * have[m])
    best = order[0]
    marks[best] = 'best'
    rest = []
    for m in order[1:]:
        se = diff_se(best, m) if diff_se else None
        if se is not None and abs(have[m] - have[best]) <= 1.96 * se:
            marks[m] = 'best'
        else:
            rest.append(m)
    if rest:
        marks[rest[0]] = 'second'
    return marks


def cell(value, scale, dec, mark, style):
    if value is None or value != value:
        return '--'
    t = f'{value * scale:.{dec}f}'
    if float(t) == 0:
        t = t.lstrip('-')           # no '-0.0'
    if style == 'md':
        return {'best': f'**{t}**', 'second': f'<u>{t}</u>'}.get(mark, t)
    if style == 'tex':
        return {'best': f'\\textbf{{{t}}}', 'second': f'\\underline{{{t}}}'}.get(mark, t)
    return {'best': f'{t}*', 'second': f'{t}_'}.get(mark, t)


def render(rows, columns, marks, style):
    """rows: [(method, {metric: value})]; marks[metric][method]."""
    heads = {k: h for k, h, *_ in METRICS}
    spec = {k: (s, d) for k, _, b, s, d in METRICS}
    body = [[m] + [cell(v.get(c), *spec[c], marks[c].get(m, ''), style) for c in columns] for m, v in rows]
    head = ['Method'] + [heads[c].replace('%', '\\%') if style == 'tex' else heads[c] for c in columns]
    if style == 'md':
        out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
        out += ['| ' + ' | '.join(r) + ' |' for r in body]
    elif style == 'tex':
        out = ['\\begin{tabular}{l' + 'c' * len(columns) + '}', '\\hline',
               ' & '.join(head) + ' \\\\', '\\hline']
        out += [' & '.join(r) + ' \\\\' for r in body] + ['\\hline', '\\end{tabular}']
    else:
        w = [max(len(r[i]) for r in [head] + body) for i in range(len(head))]
        out = ['  '.join(x.ljust(w[i]) for i, x in enumerate(head))]
        out += ['  '.join(x.ljust(w[i]) for i, x in enumerate(r)) for r in body]
    return '\n'.join(out)


def summarize(rec, fid_kid, specs, attrs, target):
    """res[method][attr] = {metric: mean}; se_diff(a, b, attr, metric) = paired SE."""
    res = {s['name']: {} for s in specs}
    for s in specs:
        for g in attrs:
            r = rec.get((target, s['name'], g))
            if r is None:
                continue
            out = {k: float(np.nanmean(r[k])) for k in PER_FACE if np.isfinite(r[k]).any()}
            out['n'] = int(np.isfinite(r['acc']).sum())
            out.update(fid_kid.get((target, s['name'], g), {}))
            res[s['name']][g] = out

    def se_diff(a, b, g, metric):
        ra, rb = rec.get((target, a, g)), rec.get((target, b, g))
        if ra is None or rb is None or metric not in PER_FACE:
            return None
        d = ra[metric] - rb[metric]
        d = d[np.isfinite(d)]
        return float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 else None
    return res, se_diff


def tables(res, se_diff, specs, attrs, target, style):
    """(per-attribute tables {attr: text}, mean table text, mean rows)."""
    names = [s['name'] for s in specs]
    ranked = [m for m in names if not any(s['name'] == m and s['kind'] == 'noedit' for s in specs)]
    cols = [k for k, *_ in METRICS]
    out = {}
    for g in attrs + ['Mean']:
        rows, marks = [], {c: {} for c in cols}
        for m in names:
            if g == 'Mean':
                per = [res[m].get(a) for a in attrs]
                v = {c: (float(np.mean([p[c] for p in per])) if all(p is not None and c in p for p in per) else None)
                     for c in cols} if all(p is not None for p in per) else {}
            else:
                v = res[m].get(g, {})
            rows.append((m, v))
        for c in cols:
            better = next(b for k, _, b, *_ in METRICS if k == c)
            vals = {m: dict(rows)[m].get(c) for m in ranked}

            def dse(a, b, _c=c, _g=g):
                if _g == 'Mean':
                    ses = [se_diff(a, b, x, _c) for x in attrs]
                    return None if any(s is None for s in ses) else float(np.sqrt(sum(s * s for s in ses)) / len(attrs))
                return se_diff(a, b, _g, _c)
            marks[c] = rank_marks(vals, better, dse)
        out[g] = (rows, marks)
    texts = {g: render(r, cols, mk, style) for g, (r, mk) in out.items()}
    paper = {g: render(r, PAPER_COLUMNS, mk, style) for g, (r, mk) in out.items()}
    return texts, paper, out


# ── main ───────────────────────────────────────────────────────────────────

def parse_specs(raw):
    specs = []
    for item in raw:
        parts = item.split(':', 2)
        if len(parts) < 2 or parts[1] not in KINDS:
            raise SystemExit(f'--method_spec {item!r}: expected NAME:KIND[:curves.csv] with KIND in {KINDS}')
        if parts[1] != 'noedit' and (len(parts) < 3 or not os.path.exists(parts[2])):
            raise SystemExit(f'--method_spec {item!r}: needs the existing curves CSV of scripts/editing_curves.py')
        specs.append({'name': parts[0], 'kind': parts[1], 'csv': parts[2] if len(parts) > 2 else None})
    return specs


def main(argv=None):
    p = build_parser()
    p.add_argument('--method_spec', action='append', required=True, metavar='NAME:KIND[:CSV]',
                   help='A method: its name in the table, its kind (model, boundary, linear, sdflow, noedit) '
                        'and the curves CSV from scripts/editing_curves.py whose accuracy-vs-strength '
                        'curve picks its strength. Repeatable; order = row order.')
    p.add_argument('--target_acc', nargs='+', type=float, default=[80.0],
                   help='Editing accuracy (%%) every method is read at; one table per value.')
    p.add_argument('--match_metric', default='target_success', choices=['target_success', 'celeb_success'],
                   help='Which curve column the target accuracy refers to (R50 or CelebA r18).')
    p.add_argument('--only_attrs', nargs='+', type=int, default=None)
    p.add_argument('--arcface_weights', default='./data/model_ir_se50.pth')
    p.add_argument('--boundaries', default='./data/attr_boundaries_wplus.pth')
    p.add_argument('--sdflow_ckpt_dir', default='./data/ckpt')
    p.add_argument('--sdflow_step', type=int, default=10000)
    p.add_argument('--sdflow_attrs', nargs='+', type=int, default=[15, 20, 39])
    p.add_argument('--no_fid', action='store_true', help='Skip FID / KID (needs torchmetrics + torch-fidelity).')
    p.add_argument('--out_dir', default='./output/comparison')
    p.set_defaults(eval_resize='area', num_samples=1000)
    args = p.parse_args(argv)
    args = apply_run_config(args)
    E._RESIZE_MODE = args.eval_resize
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
    if not args.independent_attr_weights:
        raise SystemExit('--independent_attr_weights is required (R50 judge).')
    specs = parse_specs(args.method_spec)
    os.makedirs(args.out_dir, exist_ok=True)

    prior, conditioner, G, id_criterion, _, _, direction_bank = load_models(args)
    clip_judge, indep_id, lpips_fn, indep_teacher, _, celeb_judge = build_optional_judges(
        args, args.attribute_index, id_criterion)
    if indep_id is None or lpips_fn is None:
        raise SystemExit('facenet-pytorch and lpips are required')
    arc = ArcFaceJudge(args.arcface_weights) if os.path.exists(args.arcface_weights) else None
    if arc is None:
        print(f'[WARN] no ArcFace weights at {args.arcface_weights}: the ID column is facenet instead')
    inception = build_inception(not args.no_fid)

    kinds = {s['kind'] for s in specs}
    attrs = list(args.only_attrs or (args.sdflow_attrs if 'sdflow' in kinds else args.attribute_index))
    missing = [a for a in attrs if a not in args.attribute_index]
    if missing:
        raise SystemExit(f'--only_attrs {missing} are not edited attributes of {args.checkpoint_dir}')
    sdf = OfficialSDFlow(args.sdflow_ckpt_dir, args.sdflow_step, args.sdflow_attrs) if 'sdflow' in kinds else None

    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)), T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root, latents_file=args.latent_file,
                       preds_file=args.preds_file, train=False, transform=tf)
    lin = linear_directions(args.direction_bank_path, attrs) if 'linear' in kinds else None
    bnd = boundary_directions(args.boundaries, attrs, ds) if 'boundary' in kinds else None
    loader = data.DataLoader(ds, shuffle=False, batch_size=args.batch, num_workers=4)
    N = min(args.num_samples, len(ds))

    # the operating point of every (target, method, attribute)
    strength = {}
    for t in args.target_acc:
        for s in specs:
            for g in attrs:
                strength[(t, s['name'], g)] = 0.0 if s['kind'] == 'noedit' else \
                    strength_at(s['csv'], name(g), t, args.match_metric)
    for t in args.target_acc:
        print(f'\nstrengths at {t:g}% editing accuracy ({args.match_metric}):')
        for s in specs:
            print(f'  {s["name"]:<16} ' + '  '.join(
                f'{name(g)} ' + ('--' if strength[(t, s["name"], g)] is None else f'{strength[(t, s["name"], g)]:.3f}')
                for g in attrs))

    def edit(kind, img, latent, src, attr_cond, id_cond, g, li, d, sc):
        if kind == 'noedit':
            return src
        if kind == 'sdflow':
            return sdf.edit(G, img, latent, g, d, sc)
        if kind in ('linear', 'boundary'):
            dirs = lin if kind == 'linear' else bnd
            new_lat = latent + (d * sc).view(-1, 1, 1) * dirs[g].to(latent)
            return G([new_lat], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
        return edit_single_attribute(
            prior, conditioner, G, id_criterion, img, latent, attr_cond, id_cond, li, sc, direction_bank,
            attr_global_idx=g, bypass_glasses_direction_bank=args.bypass_glasses_direction_bank,
            composite=False, direction=d)

    rec = {}                                    # (t, method, g) -> {metric: (N,) array, NaN where not measured}
    feats = defaultdict(list)                   # (t, method, g) -> [(k, 2048)] edited features of the kept faces
    src_feats = defaultdict(list)               # g -> unedited features of the kept faces
    seen = 0
    with torch.no_grad():
        for img, latent, _ in tqdm(loader, desc='comparison'):
            if seen >= N:
                break
            take = min(img.size(0), N - seen)
            img, latent = img[:take].cuda(), latent[:take].cuda()
            _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
            src = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
            src_256 = resize_for_judge(src, (256, 256))
            sp = torch.sigmoid(indep_teacher(src_256)[0])[:, :40]
            sc_src = celeb_judge.scores(src_256)[:, :40] if celeb_judge is not None else None
            src_id_fn = indep_id.extract(src_256)
            src_id_arc = arc.extract(src_256) if arc is not None else None
            src_f = inception(src_256) if inception is not None else None
            for g in attrs:
                li = args.attribute_index.index(g)
                d = torch.where(sp[:, g] > 0.5, -1.0, 1.0)
                keep = (sp[:, g] > CLEAR_HI) | (sp[:, g] < CLEAR_LO)
                if not keep.any():
                    continue
                kidx = keep.nonzero().view(-1).cpu().numpy()
                src_clip = clip_judge.target_similarity(src_256, li, d > 0) if clip_judge is not None else None
                if src_f is not None:
                    src_feats[g].append(src_f[kidx])
                others = [j for j in attrs if j != g]
                so = sp[:, others]
                clear_o = (so > CLEAR_HI) | (so < CLEAR_LO)
                for t in args.target_acc:
                    for s in specs:
                        sc = strength[(t, s['name'], g)]
                        if sc is None:
                            continue
                        face = edit(s['kind'], img, latent, src, attr_cond, id_cond, g, li, d, sc)
                        f256 = resize_for_judge(face, (256, 256))
                        ep = torch.sigmoid(indep_teacher(f256)[0])[:, :40]
                        vals = {
                            'acc': torch.where(sp[:, g] > 0.5, ep[:, g] < 0.5 - args.success_margin,
                                               ep[:, g] > 0.5 + args.success_margin).float(),
                            'ssim': gaussian_ssim((src_256 + 1) * 0.5, (f256 + 1) * 0.5),
                            'lpips': lpips_fn(src_256, f256).flatten(),
                            'pres': (((so > 0.5) == (ep[:, others] > 0.5)) & clear_o).float().sum(1)
                            / clear_o.float().sum(1).clamp(min=1),
                        }
                        idf = (src_id_fn * indep_id.extract(f256)).sum(1)
                        vals['id'] = (src_id_arc * arc.extract(f256)).sum(1) if arc is not None else idf
                        if sc_src is not None:
                            ce, cs = celeb_judge.scores(f256)[:, g], sc_src[:, g]
                            both = (cs > CLEAR_HI) | (cs < CLEAR_LO)
                            both = both & ((cs > 0.5) == (sp[:, g] > 0.5))
                            ok = torch.where(cs > 0.5, ce < 0.5 - args.success_margin,
                                             ce > 0.5 + args.success_margin).float()
                            vals['acc_celeb'] = torch.where(both, ok, torch.full_like(ok, float('nan')))
                        if clip_judge is not None:
                            vals['clip'] = clip_judge.target_similarity(f256, li, d > 0)
                            vals['clip_d'] = vals['clip'] - src_clip
                        r = rec.setdefault((t, s['name'], g), {k: np.full(N, np.nan) for k in PER_FACE})
                        for k, v in vals.items():
                            r[k][seen + kidx] = v.double().cpu().numpy()[kidx]
                        if inception is not None:
                            feats[(t, s['name'], g)].append(inception(f256)[kidx])
            seen += take

    fid_kid = {}
    if inception is not None:
        for (t, m, g), fl in feats.items():
            fe, fs = np.concatenate(fl), np.concatenate(src_feats[g])
            if len(fe) > 2:
                fid_kid[(t, m, g)] = {'fid': fid_from_features(fe, fs), 'kid': kid_from_features(fe, fs)}

    all_out = {}
    for t in args.target_acc:
        res, se_diff = summarize(rec, fid_kid, specs, attrs, t)
        texts, paper, parts = tables(res, se_diff, specs, attrs, t, 'txt')
        print(f'\n{"=" * 78}\nAll methods read at {t:g}% editing accuracy ({args.match_metric}); {seen} faces; '
              f'* best (or tied), _ best of the rest; Acc is matched, not ranked')
        for g in attrs + ['Mean']:
            print(f'\n[{name(g) if g != "Mean" else "Mean over " + ", ".join(name(a) for a in attrs)}]')
            print(texts[g])
        print(f'\nPaper-style (mean over attributes):\n{paper["Mean"]}')
        tag = f'{t:g}'
        for style, ext in (('md', 'md'), ('tex', 'tex')):
            tx, pp, _ = tables(res, se_diff, specs, attrs, t, style)
            with open(os.path.join(args.out_dir, f'table_acc{tag}.{ext}'), 'w') as f:
                f.write(f'% paper-style, mean over attributes\n{pp["Mean"]}\n\n% full\n' if ext == 'tex'
                        else f'Paper-style, mean over attributes:\n\n{pp["Mean"]}\n\nFull:\n\n')
                for g in attrs + ['Mean']:
                    f.write(('\n% ' if ext == 'tex' else '\n**') + (name(g) if g != 'Mean' else 'Mean')
                            + ('\n' if ext == 'tex' else '**\n\n') + tx[g] + '\n')
        with open(os.path.join(args.out_dir, f'comparison_acc{tag}.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['target_acc', 'method', 'attribute', 'strength', 'n'] + [k for k, *_ in METRICS])
            for m in res:
                for g, v in res[m].items():
                    w.writerow([t, m, name(g), strength[(t, m, g)], v['n']] + [v.get(k, '') for k, *_ in METRICS])
        all_out[tag] = {m: {name(g): v for g, v in d.items()} for m, d in res.items()}
    with open(os.path.join(args.out_dir, 'comparison.json'), 'w') as f:
        json.dump({'config': {'checkpoint_dir': args.checkpoint_dir, 'step': args.step, 'num_faces': seen,
                              'target_acc': args.target_acc, 'match_metric': args.match_metric,
                              'attributes': [name(a) for a in attrs], 'eval_resize': args.eval_resize,
                              'id_model': 'ArcFace IR-SE50' if arc is not None else 'facenet casia-webface',
                              'methods': args.method_spec},
                   'results': all_out}, f, indent=2)
    print(f'\nsaved tables and CSV in {args.out_dir}')


if __name__ == '__main__':
    main()
