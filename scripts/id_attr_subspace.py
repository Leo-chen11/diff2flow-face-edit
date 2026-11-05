"""Stage 1 of the identity plan: how much of an edit's identity loss is the
attribute itself?

Face-ID embeddings are not attribute-free: gender, age and glasses can be read
linearly off them. An edit that changes gender then MUST move the embedding along
those directions, and the identity loss charges that as lost identity. This script
measures how large that share is.

  1. FIT. On e4e reconstructions of TRAIN faces, one linear (shrinkage-LDA)
     direction per attribute in --subspace_attrs, labelled by the precomputed r34
     readings (the training teachers' labels; ambiguous 0.35-0.65 left out). Done in
     two embedding spaces:
       train_id  facenet VGGFace2 -- the identity model of the training loss
       arcface   IR-SE50          -- the evaluation identity metric
     The directions, orthonormalised, span the "attribute subspace" A. Held-out
     balanced accuracy of each direction says how much of the attribute the ID
     embedding carries.
  2. DECOMPOSE. For a checkpoint's edits of TEST faces (same edit and direction rule
     as scripts/editing_curves.py --method model), with e = source reconstruction and
     e' = edit (unit embeddings):
       cos_raw    <e, e'>                            what the ID numbers report
       cos_res    cosine after projecting A out      identity outside the attributes
       in_frac    |P_A (e' - e)|^2 / |e' - e|^2      share of the change inside A
     against a RANDOM subspace of the same dimension (mean of 5 draws): a random
     k-dim subspace holds ~k/512 of any change, so in_frac well above that, and
     cos_res well above the random cos_res, mean the loss really is in A.
  3. SAVE the train_id subspace (--out_subspace) for the stage-2 loss.

    python -m scripts.id_attr_subspace \\
        --checkpoint_dir output/SDFlow/multi_v1_min_2teach_seed1_bn_ctrl_from50k --step 100000 \\
        --independent_attr_weights ./data/r50_celebahq_eval.pth --independent_attr_backbone r50 \\
        --num_samples 300 --fit_faces 4000 --scales 0.5 0.8 0.95
"""
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
    ATTR_NAMES, _latest_step, apply_run_config, build_optional_judges,
    build_parser, edit_single_attribute, load_models, resize_for_judge,
)
from models.arcface import ArcFaceJudge
from models.dataset import SDFlowDataset

CLEAR_LO, CLEAR_HI = 0.35, 0.65
DEFAULT_SUBSPACE = [15, 20, 39, 31, 5,            # the edited attributes
                    24, 22, 16, 30, 0,            # beard family (moves with Male)
                    18, 36, 34,                   # makeup / lipstick / earrings (moves with Male)
                    17, 28, 4, 3, 13, 14]         # grey hair, hairline, bald, eye bags, chubby, chin (Young)


def name(a):
    return ATTR_NAMES.get(a, f'attr{a}')


def unit(x):
    return F.normalize(x.float(), dim=1)


def train_id_embed(id_criterion, x256):
    return unit(id_criterion.extract_features(x256))


# ── stage 1: the attribute subspace ──────────────────────────────────────────

def lda_direction(X, y, shrink):
    """Shrinkage LDA: w ~ (S_w + shrink * tr(S_w)/d * I)^-1 (mu1 - mu0), unit norm."""
    X1, X0 = X[y], X[~y]
    mu1, mu0 = X1.mean(0), X0.mean(0)
    Xc = torch.cat([X1 - mu1, X0 - mu0])
    S = Xc.T @ Xc / max(1, len(Xc) - 2)
    d = S.shape[0]
    S = S + shrink * S.trace() / d * torch.eye(d, dtype=S.dtype)
    w = torch.linalg.solve(S, mu1 - mu0)
    return w / w.norm()


def balanced_acc(score, y, thr):
    p = score > thr
    return 0.5 * ((p[y]).float().mean() + (~p[~y]).float().mean())


def fit_subspace(X, P, attrs, shrink, seed=0):
    """X (N, d) unit embeddings, P (N, 40) teacher probabilities.
    Returns basis Q (d, k), per-attribute directions and held-out balanced accuracy."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    n_tr = int(0.8 * len(X))
    tr, te = perm[:n_tr], perm[n_tr:]
    dirs, report = [], {}
    for a in attrs:
        p = P[:, a]
        clear = (p > CLEAR_HI) | (p < CLEAR_LO)
        y = p > 0.5
        trc, tec = tr[clear[tr]], te[clear[te]]
        n1, n0 = int(y[trc].sum()), int((~y[trc]).sum())
        if min(n1, n0) < 30:
            report[name(a)] = {'skipped': f'only {n1} positive / {n0} negative clear faces'}
            continue
        w = lda_direction(X[trc], y[trc], shrink)
        s_tr = X[trc] @ w
        cands = torch.quantile(s_tr, torch.linspace(0.02, 0.98, 97, dtype=s_tr.dtype))
        thr = max(cands.tolist(), key=lambda t: float(balanced_acc(s_tr, y[trc], t)))
        acc = float(balanced_acc(X[tec] @ w, y[tec], thr)) if len(tec) else float('nan')
        report[name(a)] = {'heldout_balanced_acc': acc, 'n_pos': n1, 'n_neg': n0}
        dirs.append(w)
    W = torch.stack(dirs, 1)                       # (d, k)
    Q, _ = torch.linalg.qr(W)
    return Q, W, report


def random_basis(d, k, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    Q, _ = torch.linalg.qr(torch.randn(d, k, generator=g))
    return Q


# ── stage 2: decomposition of the edits ──────────────────────────────────────

def decompose(e_src, e_edit, Q):
    """Per-sample cos_res and in_frac for one basis (Q on the same device)."""
    def proj_out(x):
        return x - (x @ Q) @ Q.T
    rs, re = proj_out(e_src), proj_out(e_edit)
    cos_res = (unit(rs) * unit(re)).sum(1)
    d = e_edit - e_src
    in_frac = ((d @ Q).pow(2).sum(1) / d.pow(2).sum(1).clamp(min=1e-12))
    return cos_res, in_frac


def main():
    p = build_parser()
    p.add_argument('--arcface_weights', default='./data/model_ir_se50.pth')
    p.add_argument('--scales', nargs='+', type=float, default=[0.5, 0.8, 0.95])
    p.add_argument('--only_attrs', nargs='+', type=int, default=None,
                   help='Edited attributes to decompose (default: all of the run).')
    p.add_argument('--subspace_attrs', nargs='+', type=int, default=DEFAULT_SUBSPACE,
                   help='CelebA ids whose directions span the attribute subspace.')
    p.add_argument('--fit_faces', type=int, default=4000, help='TRAIN faces the subspace is fit on.')
    p.add_argument('--shrink', type=float, default=0.1, help='LDA covariance shrinkage.')
    p.add_argument('--out_subspace', default='./data/id_attr_subspace.pt',
                   help='Where the train_id subspace is saved for the stage-2 loss.')
    p.set_defaults(eval_resize='area')
    args = p.parse_args()
    args = apply_run_config(args)
    E._RESIZE_MODE = args.eval_resize
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
    if not args.independent_attr_weights:
        raise SystemExit('--independent_attr_weights is required (R50 judge: edit direction and success).')
    if not os.path.exists(args.arcface_weights):
        raise SystemExit(f'no ArcFace weights at {args.arcface_weights}')

    prior, conditioner, G, id_criterion, _, _, direction_bank = load_models(args)
    _, _, _, indep_teacher, _, _ = build_optional_judges(args, args.attribute_index, id_criterion)
    arc = ArcFaceJudge(args.arcface_weights)
    spaces = ('train_id', 'arcface')

    def embed(x256):
        return {'train_id': train_id_embed(id_criterion, x256), 'arcface': unit(arc.extract(x256))}

    # 1. FIT on train reconstructions (latents + precomputed readings; no photos needed)
    ds_tr = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                          latents_file=args.latent_file, preds_file=args.preds_file, train=True)
    files = list(ds_tr.image_list)[:args.fit_faces]
    feats, probs = {s: [] for s in spaces}, []
    with torch.no_grad():
        for i in tqdm(range(0, len(files), args.batch), desc='fit: train reconstructions'):
            fs = files[i:i + args.batch]
            w = torch.stack([torch.as_tensor(ds_tr._lookup_precomputed(ds_tr.latents, f)).float().reshape(18, 512)
                             for f in fs]).cuda()
            pr = torch.stack([torch.as_tensor(ds_tr._lookup_precomputed(ds_tr.preds, f)).float().reshape(-1)[:40]
                              for f in fs])
            rec = G([w], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
            em = embed(resize_for_judge(rec, (256, 256)))
            for s in spaces:
                feats[s].append(em[s].cpu())
            probs.append(pr)
    P = torch.cat(probs)
    if P.min() < 0 or P.max() > 1:
        P = torch.sigmoid(P)
    bases, fit_report = {}, {}
    edited = list(args.only_attrs or args.attribute_index)
    for s in spaces:
        X = torch.cat(feats[s]).double()
        Q_all, W_all, rep = fit_subspace(X, P, args.subspace_attrs, args.shrink)
        Q_ed, _, _ = fit_subspace(X, P, [a for a in args.subspace_attrs if a in edited], args.shrink)
        k = Q_all.shape[1]
        bases[s] = {'attr_all': Q_all.float(), 'attr_edited': Q_ed.float(),
                    **{f'random{r}': random_basis(X.shape[1], k, r).float() for r in range(5)}}
        fit_report[s] = {'dim_all': k, 'dim_edited': Q_ed.shape[1], 'probes': rep}
        if s == 'train_id':
            os.makedirs(os.path.dirname(os.path.abspath(args.out_subspace)), exist_ok=True)
            used = [a for a in args.subspace_attrs if 'heldout_balanced_acc' in rep[name(a)]]
            torch.save({'basis': Q_all.float(), 'directions': W_all.float(), 'attrs': used,
                        'space': 'facenet-vggface2 (common/id_loss.IDLoss)', 'fit_faces': len(files),
                        'shrink': args.shrink}, args.out_subspace)
            print(f'saved train_id subspace ({k} dims) to {args.out_subspace}')

    print('\nheld-out balanced accuracy of one linear direction per attribute (0.5 = not encoded)')
    print(f'  {"attribute":<20}' + ''.join(f'{s:>12}' for s in spaces))
    for a in args.subspace_attrs:
        row = [fit_report[s]['probes'].get(name(a), {}).get('heldout_balanced_acc', float('nan')) for s in spaces]
        print(f'  {name(a):<20}' + ''.join(f'{v:12.3f}' for v in row))

    # 2. DECOMPOSE the checkpoint's edits on test faces
    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)), T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file,
                       train=False, transform=tf)
    loader = data.DataLoader(ds, shuffle=False, batch_size=args.batch, num_workers=4)
    gpu = {s: {b: Q.cuda() for b, Q in bases[s].items()} for s in spaces}
    rec = defaultdict(lambda: defaultdict(list))
    seen = 0
    with torch.no_grad():
        for img, latent, _ in tqdm(loader, desc='decompose: test edits'):
            if seen >= args.num_samples:
                break
            img, latent = img.cuda(), latent.cuda()
            _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
            src = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
            src256 = resize_for_judge(src, (256, 256))
            sp = torch.sigmoid(indep_teacher(src256)[0])[:, :40]
            e_src = embed(src256)
            for g in edited:
                li = args.attribute_index.index(g)
                d = torch.where(sp[:, g] > 0.5, -1.0, 1.0)
                keep = (sp[:, g] > CLEAR_HI) | (sp[:, g] < CLEAR_LO)
                if not keep.any():
                    continue
                for sc in args.scales:
                    face = edit_single_attribute(
                        prior, conditioner, G, id_criterion, img, latent, attr_cond, id_cond,
                        li, sc, direction_bank, attr_global_idx=g,
                        bypass_glasses_direction_bank=args.bypass_glasses_direction_bank,
                        composite=False, direction=d)
                    f256 = resize_for_judge(face, (256, 256))
                    ep = torch.sigmoid(indep_teacher(f256)[0])[:, :40]
                    ok = torch.where(sp[:, g] > 0.5, ep[:, g] < 0.5, ep[:, g] > 0.5)
                    e_ed = embed(f256)
                    r = rec[(g, sc)]
                    r['success'] += ok[keep].float().tolist()
                    for s in spaces:
                        a, b = e_src[s][keep], e_ed[s][keep]
                        r[f'{s}/cos_raw'] += (a * b).sum(1).tolist()
                        for bname, Q in gpu[s].items():
                            cr, fr = decompose(a, b, Q)
                            r[f'{s}/{bname}/cos_res'] += cr.tolist()
                            r[f'{s}/{bname}/in_frac'] += fr.tolist()
            seen += img.size(0)

    # 3. REPORT
    out = {'checkpoint_dir': args.checkpoint_dir, 'step': args.step, 'num_test_faces': seen,
           'fit_faces': len(files), 'subspace_attrs': args.subspace_attrs, 'fit': fit_report, 'edits': []}
    for (g, sc), r in sorted(rec.items()):
        row = {'attribute': name(g), 'scale': sc, 'n': len(r['success']),
               'r50_success': float(np.mean(r['success']))}
        for s in spaces:
            row[f'{s}/cos_raw'] = float(np.mean(r[f'{s}/cos_raw']))
            for b in ('attr_all', 'attr_edited'):
                row[f'{s}/{b}/cos_res'] = float(np.mean(r[f'{s}/{b}/cos_res']))
                row[f'{s}/{b}/in_frac'] = float(np.mean(r[f'{s}/{b}/in_frac']))
            row[f'{s}/random/cos_res'] = float(np.mean([np.mean(r[f'{s}/random{k}/cos_res']) for k in range(5)]))
            row[f'{s}/random/in_frac'] = float(np.mean([np.mean(r[f'{s}/random{k}/in_frac']) for k in range(5)]))
        out['edits'].append(row)

    for s in spaces:
        k = fit_report[s]['dim_all']
        print(f'\n[{s}] edit vs source reconstruction; attribute subspace = {k} dims '
              f'(a random {k}-dim subspace holds ~{k / 512:.1%} of any change)')
        print(f'  {"attribute":<11}{"scale":>6}{"R50":>7}{"cos_raw":>9}{"cos_res":>9}{"rand_res":>9}'
              f'{"in_frac":>9}{"rand_in":>9}{"ed_res":>8}')
        for row in out['edits']:
            print(f'  {row["attribute"]:<11}{row["scale"]:6.2f}{row["r50_success"] * 100:6.1f}%'
                  f'{row[f"{s}/cos_raw"]:9.3f}{row[f"{s}/attr_all/cos_res"]:9.3f}'
                  f'{row[f"{s}/random/cos_res"]:9.3f}{row[f"{s}/attr_all/in_frac"] * 100:8.1f}%'
                  f'{row[f"{s}/random/in_frac"] * 100:8.1f}%{row[f"{s}/attr_edited/cos_res"]:8.3f}')
    print('\n  cos_res  identity similarity with the attribute subspace projected out (ed_res: the '
          'edited attributes\' directions only)\n  in_frac  share of the embedding change that lies in '
          'the subspace; rand_* = same for a random subspace of equal size')

    path = args.out_json or os.path.join(args.checkpoint_dir, f'id_attr_subspace_s{args.step}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nsaved {path}')


if __name__ == '__main__':
    main()
