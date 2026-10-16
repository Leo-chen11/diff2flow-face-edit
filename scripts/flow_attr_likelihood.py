"""Can the conditional CNF itself tell whether a face has an attribute, and
does that add anything to the r34 training teacher? (Diagnostic only: no
training, no change to the model.)

The prior is a conditional CNF: dw/dt = v(w, t, c) with c = [id_cond, attr_cond],
and prior(w, c, 0) returns (z, delta_logp), so for any W+ latent
    log p(w | c) = sum log N(z) - sum delta_logp          (as the NLL term in
                                                          train_sdflow.py)
For attribute a, with every other condition held at the face's own values:
    s_a(w) = log p(w | c, a = hi) - log p(w | c, a = lo)
hi / lo are the 95th / 5th percentile of the conditioner's own reading of a
over the evaluated faces (in-distribution condition values, not 0/1).

Two details matter for the numbers:
  * the log-det comes from a Hutchinson trace estimate with one Gaussian probe
    per solve. Both conditions use the SAME probe (common random numbers) and
    --probes draws are averaged, so s_a is far less noisy than either term.
  * an edited latent is built as w' = f^-1(f(w; c_src); c_tgt): it reuses the
    source's base point z, which by construction is typical under c_tgt. So s_a
    on edits is expected to lean toward the target for EVERY edit; what matters
    is whether it separates the edits the judges accept from those they reject,
    not its sign.

Experiment 1 (real faces). Label = the judges' consensus reading of the source
  reconstruction (CLIP / BiSeNet-parser for glasses, and the CelebA ResNet-18;
  both clear, i.e. outside 0.35-0.65, and agreeing) -- the faces eval's
  --consensus keeps. AUC of s_a, of the r34 teacher's probability and of the
  conditioner's reading. No judge is trained or used to fit anything.
Experiment 2 (edits at --diag_scale, direction = the conditioner's reading, as
  the default eval). Label = both judges count the edit a success (strict
  0.5 crossing) on consensus sources. AUC of the signed flow score (s for adds,
  -s for removals), of the signed r34 probability, and of their rank average.
Experiment 3 (complementarity). Edits split by r34 success x flow success
  (flow threshold fitted on REAL faces from experiment 1, never on edit
  labels); judge success rate per cell. The cell to watch is r34-success /
  flow-fail: if its judge success is clearly lower than r34-success /
  flow-success, the flow sees false successes r34 does not.

    python scripts/flow_attr_likelihood.py --checkpoint_dir output/SDFlow/<run> \
        --step 80000 --bank_mode prior --num_samples 500
Output: <checkpoint_dir>/flow_attr_likelihood_s<step>.json (per-face scores
included, so thresholds can be revisited without rerunning).
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.flows.odefunc import ODEfunc
from models.flows.utils import standard_normal_logprob


class FixedProbe:
    """Make every ODEfunc in `prior` use the given Hutchinson probe (the CNF
    resets it to None before each solve, which would draw a fresh one)."""

    def __init__(self, prior):
        self.funcs = [m for m in prior.modules() if isinstance(m, ODEfunc)]
        self.e = None
        for f in self.funcs:
            orig = f.before_odeint

            def patched(e=None, _orig=orig, _self=self):
                _orig(e=_self.e if e is None else e)
            f.before_odeint = patched


@torch.no_grad()
def log_lik(prior, w, cond, probe, e):
    """log p(w | cond), summed over all W+ dims, with Hutchinson probe e."""
    probe.e = e
    zero = torch.zeros(w.size(0), w.size(1), 1, device=w.device, dtype=w.dtype)
    z, dlogp = prior(w, cond, zero)
    probe.e = None
    return (standard_normal_logprob(z).flatten(1).sum(1) - dlogp.flatten(1).sum(1))


@torch.no_grad()
def attr_score(prior, probe, w, id_cond, attr_cond, a, hi, lo, n_probes):
    """s_a(w), averaged over n_probes shared probes."""
    c_hi, c_lo = attr_cond.clone(), attr_cond.clone()
    c_hi[:, a], c_lo[:, a] = float(hi), float(lo)
    c_hi, c_lo = torch.cat([id_cond, c_hi], 1), torch.cat([id_cond, c_lo], 1)
    out = torch.zeros(w.size(0), device=w.device)
    for _ in range(n_probes):
        e = torch.randn_like(w)
        out += log_lik(prior, w, c_hi, probe, e) - log_lik(prior, w, c_lo, probe, e)
    return out / n_probes


def auc(score, label):
    """ROC-AUC via the Mann-Whitney statistic (ties count half)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind='mergesort')
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_ci(score, label, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    score, label = np.asarray(score, float), np.asarray(label, bool)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(score), len(score))
        v = auc(score[idx], label[idx])
        if v is not None:
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (None, None)


def rank01(x):
    x = np.asarray(x, float)
    return np.argsort(np.argsort(x)) / max(len(x) - 1, 1)


def youden_threshold(score, label):
    """Threshold maximizing TPR - FPR on (score, label)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    best, thr = -1, 0.0
    for t in np.unique(score):
        pred = score >= t
        j = pred[label].mean() - pred[~label].mean()
        if j > best:
            best, thr = j, float(t)
    return thr


def cell(c):
    js = '--' if c['judge_success'] is None else f"{c['judge_success'] * 100:.0f}%"
    return f"{js} (n={c['n']})"


def fmt_auc(v, ci):
    return '   --' if v is None else f'{v:.3f} [{ci[0]:.2f},{ci[1]:.2f}]'


def main():
    from torch.utils import data
    import torchvision.transforms as T
    from tqdm import tqdm
    from evaluation.evaluate_sdflow import (
        ATTR_NAMES, _latest_step, apply_run_config, bank_edit, build_optional_judges,
        build_parser, cap_delta_norm, consensus_label, load_models,
        strict_success, edited_attr_value)
    import evaluation.evaluate_sdflow as ev
    from models.dataset import SDFlowDataset

    p = build_parser()
    p.add_argument('--diag_scale', type=float, default=0.9, help='edit scale for experiments 2-3')
    p.add_argument('--probes', type=int, default=2, help='Hutchinson probes averaged per score')
    p.add_argument('--lo_hi_pct', type=float, nargs=2, default=[5.0, 95.0],
                   help='percentiles of the conditioner reading used as lo / hi')
    args = p.parse_args()
    if not os.path.isdir(args.checkpoint_dir):
        raise SystemExit(f'checkpoint_dir does not exist on this machine: {args.checkpoint_dir}')
    args = apply_run_config(args)
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)

    prior, conditioner, G, id_criterion, attr_teacher, _ai, bank = load_models(args)
    clip_judge, _iid, _lp, _it, glasses_parser, celeb_judge = build_optional_judges(
        args, args.attribute_index, id_criterion)
    if clip_judge is None or celeb_judge is None:
        raise SystemExit('needs both the CLIP judge and --celeba_attr_judge_weights')
    probe = FixedProbe(prior)
    names = [ATTR_NAMES.get(i, f'attr{i}') for i in args.attribute_index]
    gl = args.attribute_index.index(15) if 15 in args.attribute_index else None
    attr_index = list(args.attribute_index)

    def judges(img256):
        c = clip_judge.scores(img256)
        if glasses_parser is not None and gl is not None:
            c[:, gl] = glasses_parser.glasses_prob(img256)
        return c, celeb_judge.scores(img256)[:, attr_index]

    def r34(img256):
        return torch.sigmoid(attr_teacher(img256)[0])[:, attr_index]

    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)),
                    T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file,
                       train=False, transform=tf)
    if args.sample_offset:
        ds = data.Subset(ds, range(args.sample_offset, len(ds)))
    loader = data.DataLoader(ds, shuffle=False, batch_size=args.batch, num_workers=4)

    # ── pass 1: sources (latent, conditions, judge / teacher readings) ─────
    S = {k: [] for k in ('latent', 'id', 'attr', 'clip', 'celeb', 'r34')}
    seen = 0
    with torch.no_grad():
        for img, latent, _ in tqdm(loader, desc='sources'):
            if seen >= args.num_samples:
                break
            img, latent = img.cuda(), latent.cuda()
            _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
            face = F.interpolate(G([latent], input_is_latent=True, randomize_noise=False)[0]
                                 .clamp(-1, 1), (256, 256))
            c, e = judges(face)
            for k, v in (('latent', latent), ('id', id_cond), ('attr', attr_cond),
                         ('clip', c), ('celeb', e), ('r34', r34(face))):
                S[k].append(v.cpu())
            seen += img.size(0)
    S = {k: torch.cat(v)[:args.num_samples] for k, v in S.items()}
    N = S['latent'].size(0)
    lo = torch.quantile(S['attr'], args.lo_hi_pct[0] / 100, dim=0)
    hi = torch.quantile(S['attr'], args.lo_hi_pct[1] / 100, dim=0)
    print('condition lo / hi per attribute: ' + ', '.join(
        f'{n} {float(lo[a]):.2f}/{float(hi[a]):.2f}' for a, n in enumerate(names)))

    B = args.batch
    results = {'config': {'checkpoint_dir': args.checkpoint_dir, 'step': args.step,
                          'num_samples': N, 'sample_offset': args.sample_offset,
                          'diag_scale': args.diag_scale, 'probes': args.probes,
                          'lo': lo.tolist(), 'hi': hi.tolist(),
                          'bank_mode': getattr(args, 'bank_mode', None)},
               'attrs': {}}

    for a, name in enumerate(names):
        gidx = attr_index[a]
        # ── experiment 1: real faces ──────────────────────────────────────
        s_real, s_edit = [], []
        e_clip, e_celeb, e_r34 = [], [], []
        for i in tqdm(range(0, N, B), desc=f'{name}'):
            sl = slice(i, i + B)
            w = S['latent'][sl].cuda()
            idc, atc = S['id'][sl].cuda(), S['attr'][sl].cuda()
            s_real.append(attr_score(prior, probe, w, idc, atc, a, hi[a], lo[a], args.probes).cpu())
            # ── edit at diag_scale, conditioner's direction (default eval) ─
            with torch.no_grad():
                if bank is not None:
                    delta, _, _ = bank_edit(prior, bank, w, atc, idc, a, args.diag_scale, gidx, None)
                else:
                    zero = torch.zeros(w.size(0), 18, 1, device=w.device)
                    mid, _ = prior(w, torch.cat([idc, atc], 1), zero)
                    new = atc.clone()
                    new[:, a] = edited_attr_value(atc[:, a], args.diag_scale, gidx)
                    raw, _ = prior(mid, torch.cat([idc, new], 1), zero, reverse=True)
                    delta = cap_delta_norm(raw - w, ev._FINAL_DELTA_MAX_NORM)
                we = w + delta
                face = F.interpolate(G([we], input_is_latent=True, randomize_noise=False)[0]
                                     .clamp(-1, 1), (256, 256))
                c, e = judges(face)
                e_clip.append(c[:, a].cpu()); e_celeb.append(e[:, a].cpu())
                e_r34.append(r34(face)[:, a].cpu())
            # other conditions stay at the source's own reading
            s_edit.append(attr_score(prior, probe, we, idc, atc, a, hi[a], lo[a], args.probes).cpu())
        s_real, s_edit = torch.cat(s_real).numpy(), torch.cat(s_edit).numpy()
        e_clip, e_celeb, e_r34 = (torch.cat(x).numpy() for x in (e_clip, e_celeb, e_r34))
        src_clip, src_celeb = S['clip'][:, a].numpy(), S['celeb'][:, a].numpy()
        src_r34, src_cond = S['r34'][:, a].numpy(), S['attr'][:, a].numpy()

        lab = [consensus_label(float(x), float(y)) for x, y in zip(src_clip, src_celeb)]
        keep = np.array([l is not None for l in lab])
        has = np.array([l == 'rm' for l in lab])            # 'rm' = source HAS the attribute
        ex1 = {}
        for key, sc in (('flow', s_real), ('r34', src_r34), ('conditioner', src_cond)):
            v = auc(sc[keep], has[keep])
            ex1[key] = {'auc': v, 'ci': auc_ci(sc[keep], has[keep])}
        thr = youden_threshold(s_real[keep], has[keep])
        ex1['flow_threshold'] = thr
        ex1['n'] = int(keep.sum()); ex1['n_pos'] = int(has[keep].sum())

        # ── experiment 2: which edits do the judges accept? ─────────────
        adds = src_cond <= 0.5                      # conditioner's direction, as eval cond
        sign = np.where(adds, 1.0, -1.0)
        ok = np.array([bool(strict_success(float(sc_), float(ec_), 0.0)
                            and strict_success(float(se_), float(ee_), 0.0))
                       for sc_, ec_, se_, ee_ in zip(src_clip, e_clip, src_celeb, e_celeb)])
        flow_sig = sign * s_edit
        r34_sig = np.where(adds, e_r34, 1.0 - e_r34)
        combo = (rank01(flow_sig) + rank01(r34_sig)) / 2
        ex2 = {'n': int(keep.sum()), 'judge_success': float(ok[keep].mean()) if keep.any() else None}
        for key, sc in (('flow', flow_sig), ('r34', r34_sig), ('combo', combo)):
            ex2[key] = {'auc': auc(sc[keep], ok[keep]), 'ci': auc_ci(sc[keep], ok[keep])}

        # ── experiment 3: r34 success x flow success ───────────────────────
        r34_ok = np.array([bool(strict_success(float(x), float(y), 0.0))
                           for x, y in zip(src_r34, e_r34)])
        flow_has = s_edit >= thr
        flow_ok = np.where(adds, flow_has, ~flow_has)
        ex3 = {}
        for rk, rv in (('r34_ok', True), ('r34_fail', False)):
            for fk, fv in (('flow_ok', True), ('flow_fail', False)):
                m = keep & (r34_ok == rv) & (flow_ok == fv)
                ex3[f'{rk}/{fk}'] = {'n': int(m.sum()),
                                     'judge_success': float(ok[m].mean()) if m.any() else None}

        results['attrs'][name] = {
            'exp1_real': ex1, 'exp2_edit': ex2, 'exp3_quadrants': ex3,
            's_real_mean_has': float(s_real[keep & has].mean()) if (keep & has).any() else None,
            's_real_mean_lacks': float(s_real[keep & ~has].mean()) if (keep & ~has).any() else None,
            'per_face': {'s_real': s_real.tolist(), 's_edit': s_edit.tolist(),
                         'src_clip': src_clip.tolist(), 'src_celeb': src_celeb.tolist(),
                         'src_r34': src_r34.tolist(), 'src_cond': src_cond.tolist(),
                         'edit_clip': e_clip.tolist(), 'edit_celeb': e_celeb.tolist(),
                         'edit_r34': e_r34.tolist()},
        }

    # ── report ────────────────────────────────────────────────────────────
    print(f'\n{N} faces, step {args.step}, edits at scale {args.diag_scale}, '
          f'{args.probes} Hutchinson probes per score')
    print('\nExperiment 1 -- real faces, label = judges\' consensus reading (AUC [95% CI])')
    print(f'   {"":<12}{"n (has)":>10}  {"flow s_a":>20}  {"r34":>20}  {"conditioner":>20}   mean s has/lacks')
    for n_ in names:
        r = results['attrs'][n_]['exp1_real']
        R = results['attrs'][n_]
        print(f'   {n_:<12}{r["n"]:>5} ({r["n_pos"]:>3})  {fmt_auc(r["flow"]["auc"], r["flow"]["ci"]):>20}  '
              f'{fmt_auc(r["r34"]["auc"], r["r34"]["ci"]):>20}  '
              f'{fmt_auc(r["conditioner"]["auc"], r["conditioner"]["ci"]):>20}   '
              f'{R["s_real_mean_has"] or 0:+.1f} / {R["s_real_mean_lacks"] or 0:+.1f}')
    print('\nExperiment 2 -- edits, label = both judges accept (AUC [95% CI])')
    print(f'   {"":<12}{"n":>5} {"success":>8}  {"flow":>20}  {"r34":>20}  {"rank avg":>20}')
    for n_ in names:
        r = results['attrs'][n_]['exp2_edit']
        print(f'   {n_:<12}{r["n"]:>5} {(r["judge_success"] or 0) * 100:7.1f}%  '
              f'{fmt_auc(r["flow"]["auc"], r["flow"]["ci"]):>20}  '
              f'{fmt_auc(r["r34"]["auc"], r["r34"]["ci"]):>20}  '
              f'{fmt_auc(r["combo"]["auc"], r["combo"]["ci"]):>20}')
    print('\nExperiment 3 -- judge success rate by (r34, flow) verdict; flow threshold fitted on real faces')
    print(f'   {"":<12}{"r34 ok/flow ok":>18}{"r34 ok/flow fail":>18}{"r34 fail/flow ok":>18}{"r34 fail/flow fail":>20}')
    for n_ in names:
        q = results['attrs'][n_]['exp3_quadrants']
        cells = [q[k] for k in ('r34_ok/flow_ok', 'r34_ok/flow_fail', 'r34_fail/flow_ok', 'r34_fail/flow_fail')]
        print(f'   {n_:<12}' + ''.join(f'{cell(c):>18}' for c in cells))

    path = args.out_json or os.path.join(args.checkpoint_dir, f'flow_attr_likelihood_s{args.step}.json')
    with open(path, 'w') as f:
        json.dump(results, f)
    print(f'\nsaved {path}')


if __name__ == '__main__':
    main()
