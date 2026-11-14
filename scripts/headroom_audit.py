"""Where is the headroom? A per-face audit that says which part of the editor is worth fixing.

Every R50-clear test face is edited at every strength of a ladder, each attribute, in the direction
the R50 judge's reading of the source asks for (as scripts/editing_curves.py). Per face and strength
it records the R50 judge's and the r34 training teacher's probability of the edited attribute and
the identity kept (ArcFace IR-SE50 and facenet, edit vs source reconstruction), under two readings
of the source:

    r    the conditioner's own reading                          (what the curves score today)
    dc   direction-consistent: a reading on the wrong side of 0.5 for the requested direction is
         moved to 0.5 -/+ --dc_margin, in the encoding and the target; other faces are untouched
         (editing_curves.py --consistent_reading; uses no judge's reading)

`report` then answers, per attribute and on their mean, three separate questions:

  A. READING     How much accuracy does a face lose because the conditioner misreads it?
                 dc minus r, at one fixed strength and as "succeeds at some strength".
  B. STRENGTH    How much identity is spent on faces that needed less? One strength for every face
                 (today) against a per-face strength:
                   teacher-stopped  the weakest strength at which the TRAINING teacher says the edit
                                    is done (by --margins); a face it never accepts gets the
                                    strongest. A method: the teacher is part of training, not a judge.
                   oracle           the weakest strength at which the R50 JUDGE says so. Not a
                                    method (it reads the judge): the accuracy no stop rule can
                                    exceed. Its identity is not a bound -- a face it never accepts
                                    runs to the strongest strength, where a teacher may stop early.
                 Both are compared with the fixed-strength curve AT THE SAME ACCURACY.
  C. WHAT IS LEFT  The faces no strength edits, split by why:
                   misread     the conditioner reads the source on the wrong side (A fixes these)
                   teacher>R50 the training teacher accepts the edit at some strength, R50 never
                               does: the editor satisfied what it was trained against -- a better
                               training signal is the fix, not more strength
                   unmoved     neither the teacher nor R50 moved by 0.2: the flow did not respond
                   partial     moved, not far enough at the strongest strength

and at which accuracy each way of choosing the strength still keeps ArcFace >= --id_goal (0.90).

    python -m scripts.headroom_audit --checkpoint_dir output/SDFlow/<run> --step 130000 \\
        --independent_attr_weights ./data/r50_celebahq_eval.pth --independent_attr_backbone r50 \\
        --num_samples 500 --out_csv output/audit/headroom_dev.csv
    python -m scripts.headroom_audit report output/audit/headroom_dev.csv

The run is ~2x scripts/editing_curves.py --method model (the dc pass is only rendered for batches
that hold a misread face). Use the development faces (no --sample_offset): this is a diagnosis.
"""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

CLEAR_LO, CLEAR_HI = 0.35, 0.65
READINGS = ('r', 'dc')
MOVED = 0.2          # a probability change below this counts as "did not move"


# ── report (numpy only) ────────────────────────────────────────────────────

def load(path):
    """{attribute: {'scales': [...], 'dir': (N,), 'cond_src': (N,), per reading v:
    v+'_p' / '_t' / '_arc' / '_id': (N, S)}}; p and t are the R50 / teacher probability of the
    edited attribute AFTER the edit, src values are before it."""
    rows = list(csv.DictReader(open(path, newline='')))
    if not rows:
        raise SystemExit(f'{path} has no rows')
    scales = sorted({float(k.split('@')[1]) for k in rows[0] if k.startswith('r_p@')})
    out = {}
    by_attr = defaultdict(list)
    for r in rows:
        by_attr[r['attribute']].append(r)
    for a, rs in by_attr.items():
        d = {'scales': scales}
        for k in ('dir', 'r50_src', 'cond_src', 't_src'):
            d[k] = np.array([float(r[k]) for r in rs])
        for v in READINGS:
            for m in ('p', 't', 'arc', 'id'):
                d[f'{v}_{m}'] = np.array([[float(r[f'{v}_{m}@{s:g}']) for s in scales] for r in rs])
        out[a] = d
    return out


def toward(prob, direction):
    """Probability of being on the REQUESTED side: prob when adding, 1 - prob when removing."""
    return np.where(direction[:, None] > 0, prob, 1.0 - prob) if prob.ndim == 2 \
        else np.where(direction > 0, prob, 1.0 - prob)


def stop_at_first(satisfied, values):
    """Per face, `values` at the weakest strength where `satisfied` holds, else at the strongest.
    satisfied, values: (N, S). Returns (N,) values and (N,) column indices."""
    idx = np.where(satisfied.any(1), satisfied.argmax(1), satisfied.shape[1] - 1)
    return values[np.arange(len(idx)), idx], idx


def fixed_id_at_acc(acc, ident, target):
    """Identity of the fixed-strength curve where it first reaches `target` accuracy (linear
    between the two strengths around the crossing); None if it never does or starts above it."""
    eps = 1e-9
    if acc[0] > target + eps:
        return None
    if abs(acc[0] - target) <= eps:
        return float(ident[0])
    for i in range(len(acc) - 1):
        a0, a1 = acc[i], acc[i + 1]
        if a0 - eps <= target <= a1 + eps and a1 > a0 - eps:
            if a1 - a0 <= eps:
                return float(ident[i])
            return float(ident[i] + (target - a0) / (a1 - a0) * (ident[i + 1] - ident[i]))
    return None


def fixed_acc_at_id(acc, ident, goal):
    """Best accuracy of the fixed-strength curve among its points (and the crossing between two
    of them) that keep identity >= goal; None if even the weakest strength is below it."""
    best = None
    for i in range(len(acc)):
        if ident[i] >= goal:
            best = acc[i] if best is None else max(best, acc[i])
        if i + 1 < len(acc) and ident[i] >= goal > ident[i + 1]:
            x = acc[i] + (ident[i] - goal) / (ident[i] - ident[i + 1]) * (acc[i + 1] - acc[i])
            best = x if best is None else max(best, x)
    return None if best is None else float(best)


def analyse(d, margins, id_key='arc'):
    """Everything `report` prints for one attribute, as numbers."""
    direction = d['dir']
    res = {'n': len(direction), 'scales': d['scales']}
    wrong_side = np.where(direction > 0, d['cond_src'] > 0.5, d['cond_src'] < 0.5)
    res['misread_share'] = float(wrong_side.mean())
    for v in READINGS:
        ok = toward(d[f'{v}_p'], direction) > 0.5                       # (N, S) R50 success
        ident = d[f'{v}_{id_key}']
        t_toward = toward(d[f'{v}_t'], direction)
        acc, idm = ok.mean(0), ident.mean(0)
        r = {'fixed_acc': acc, 'fixed_id': idm, 'any_ok': float(ok.any(1).mean())}
        # oracle: the weakest strength the judge accepts
        o_id, _ = stop_at_first(ok, ident)
        r['oracle'] = {'acc': r['any_ok'], 'id': float(o_id.mean()),
                       'fixed_id_same_acc': fixed_id_at_acc(acc, idm, r['any_ok'])}
        # teacher-stopped, one point per margin
        r['teacher'] = []
        for m in margins:
            done = t_toward > 0.5 + m
            s_ok, idx = stop_at_first(done, ok.astype(float))
            s_id, _ = stop_at_first(done, ident)
            a = float(s_ok.mean())
            r['teacher'].append({'margin': m, 'acc': a, 'id': float(s_id.mean()),
                                 'mean_scale': float(np.array(d['scales'])[idx].mean()),
                                 'never_done': float((~done.any(1)).mean()),
                                 'fixed_id_same_acc': fixed_id_at_acc(acc, idm, a)})
        # the faces no strength edits, by cause
        never = ~ok.any(1)
        t_ever = (t_toward > 0.5).any(1)
        t_src_toward = toward(d['t_src'], direction)
        r_src_toward = toward(d['r50_src'], direction)
        t_move = (t_toward.max(1) - t_src_toward)
        p_move = (toward(d[f'{v}_p'], direction).max(1) - r_src_toward)
        mis = never & wrong_side
        fooled = never & ~wrong_side & t_ever
        unmoved = never & ~wrong_side & ~t_ever & (t_move < MOVED) & (p_move < MOVED)
        partial = never & ~mis & ~fooled & ~unmoved
        r['left'] = {'never': float(never.mean()), 'misread': float(mis.mean()),
                     'teacher>R50': float(fooled.mean()), 'unmoved': float(unmoved.mean()),
                     'partial': float(partial.mean())}
        res[v] = r
    return res


def _pct(x):
    return '   -- ' if x is None else f'{x * 100:5.1f}%'


def _idf(x):
    return '  --  ' if x is None else f'{x:.3f}'


def _gain(a, b):
    return '   --  ' if a is None or b is None else f'{a - b:+.3f}'


def report(path, margins=(0.0, 0.2, 0.4), id_goal=0.90, fixed_scale=1.0, id_key='arc'):
    data = load(path)
    results = {a: analyse(d, margins, id_key) for a, d in data.items()}
    scales = next(iter(results.values()))['scales']
    fi = min(range(len(scales)), key=lambda i: abs(scales[i] - fixed_scale))
    id_name = 'ArcFace' if id_key == 'arc' else 'facenet'
    print(f'{path}: {len(results)} attributes, strengths {" ".join(f"{s:g}" for s in scales)}; '
          f'identity = {id_name} (edit vs source reconstruction); R50-clear faces, direction from R50')

    for a, res in results.items():
        print(f'\n{"=" * 78}\n{a}   ({res["n"]} faces; conditioner on the wrong side for the request: '
              f'{res["misread_share"]:.1%})')
        for v, title in (('r', "conditioner's reading (today)"), ('dc', 'direction-consistent reading')):
            r = res[v]
            print(f'\n  [{v}] {title}')
            print('    one strength for all   ' + '  '.join(f'{s:>5g}' for s in scales))
            print('      accuracy             ' + '  '.join(f'{x * 100:5.1f}' for x in r['fixed_acc']))
            print(f'      {id_name:<20} ' + '  '.join(f'{x:5.3f}' for x in r['fixed_id']))
            print(f'    per-face strength        accuracy  {id_name}   fixed-strength {id_name} at that accuracy   gain')
            for t in r['teacher']:
                print(f'      teacher-stopped m={t["margin"]:<4g} {_pct(t["acc"])}    {_idf(t["id"])}     '
                      f'{_idf(t["fixed_id_same_acc"])}                                {_gain(t["id"], t["fixed_id_same_acc"])}'
                      f'   (mean strength {t["mean_scale"]:.2f}, never accepted {t["never_done"]:.1%})')
            o = r['oracle']
            print(f'      oracle (R50 stops)     {_pct(o["acc"])}    {_idf(o["id"])}     '
                  f'{_idf(o["fixed_id_same_acc"])}                                {_gain(o["id"], o["fixed_id_same_acc"])}'
                  f'   (not a method: the accuracy no stop rule can exceed)')
            lf = r['left']
            print(f'    no strength edits {lf["never"]:.1%} of the faces:  misread {lf["misread"]:.1%}   '
                  f'teacher>R50 {lf["teacher>R50"]:.1%}   unmoved {lf["unmoved"]:.1%}   partial {lf["partial"]:.1%}')

    # ── the three answers, on the mean over attributes ─────────────────────
    A = list(results)
    mean = lambda f: float(np.mean([f(results[a]) for a in A]))

    def mean_opt(f):
        vals = [f(results[a]) for a in A]
        return None if any(x is None for x in vals) else float(np.mean(vals))

    print(f'\n{"=" * 78}\nMEAN OVER {len(A)} ATTRIBUTES ({", ".join(A)})\n')
    print('A. READING  (direction-consistent minus conditioner\'s)')
    print(f'     accuracy at strength {scales[fi]:g}:        {mean(lambda r: r["r"]["fixed_acc"][fi]) * 100:5.1f}% -> '
          f'{mean(lambda r: r["dc"]["fixed_acc"][fi]) * 100:5.1f}%   ({(mean(lambda r: r["dc"]["fixed_acc"][fi]) - mean(lambda r: r["r"]["fixed_acc"][fi])) * 100:+.1f} pp)')
    print(f'     succeeds at some strength:      {mean(lambda r: r["r"]["any_ok"]) * 100:5.1f}% -> '
          f'{mean(lambda r: r["dc"]["any_ok"]) * 100:5.1f}%   ({(mean(lambda r: r["dc"]["any_ok"]) - mean(lambda r: r["r"]["any_ok"])) * 100:+.1f} pp)')
    print(f'     {id_name} at strength {scales[fi]:g}:         {mean(lambda r: r["r"]["fixed_id"][fi]):.3f} -> '
          f'{mean(lambda r: r["dc"]["fixed_id"][fi]):.3f}')
    for v in READINGS:
        print(f'\nB. STRENGTH  [{v}]  identity gained by a per-face strength, at the SAME accuracy as one strength for all')
        for k, m in enumerate(margins):
            acc = mean(lambda r: r[v]['teacher'][k]['acc'])
            idv = mean(lambda r: r[v]['teacher'][k]['id'])
            fx = mean_opt(lambda r: r[v]['teacher'][k]['fixed_id_same_acc'])
            print(f'     teacher-stopped m={m:<4g}  accuracy {acc * 100:5.1f}%  {id_name} {idv:.3f}  '
                  f'fixed {_idf(fx)}  gain {_gain(idv, fx)}')
        acc, idv = mean(lambda r: r[v]['oracle']['acc']), mean(lambda r: r[v]['oracle']['id'])
        fx = mean_opt(lambda r: r[v]['oracle']['fixed_id_same_acc'])
        print(f'     oracle (accuracy bound) accuracy {acc * 100:5.1f}%  {id_name} {idv:.3f}  fixed {_idf(fx)}  gain {_gain(idv, fx)}')
    print('\n     (fixed "--": the fixed-strength curve never reaches that accuracy -- the per-face rule is then '
          'better on accuracy itself)')
    for v in READINGS:
        lf = {k: mean(lambda r, k=k: r[v]['left'][k]) for k in ('never', 'misread', 'teacher>R50', 'unmoved', 'partial')}
        print(f'\nC. WHAT IS LEFT  [{v}]  no strength edits {lf["never"]:.1%} of the faces:  misread {lf["misread"]:.1%}   '
              f'teacher>R50 {lf["teacher>R50"]:.1%}   unmoved {lf["unmoved"]:.1%}   partial {lf["partial"]:.1%}')

    print(f'\nACCURACY THAT STILL KEEPS {id_name} >= {id_goal:g}  (per attribute; -- = not reachable with these strengths)')
    print(f'     {"":<12}' + ''.join(f'{a[:10]:>11}' for a in A) + f'{"Mean":>11}')
    lines = [('one strength [r]', lambda r: fixed_acc_at_id(r['r']['fixed_acc'], r['r']['fixed_id'], id_goal)),
             ('one strength [dc]', lambda r: fixed_acc_at_id(r['dc']['fixed_acc'], r['dc']['fixed_id'], id_goal))]
    for v in READINGS:
        lines.append((f'teacher-stop [{v}]', lambda r, v=v: max(
            [t['acc'] for t in r[v]['teacher'] if t['id'] >= id_goal], default=None)))
        lines.append((f'oracle [{v}]', lambda r, v=v: r[v]['oracle']['acc'] if r[v]['oracle']['id'] >= id_goal else None))
    for label, f in lines:
        vals = [f(results[a]) for a in A]
        m = None if any(x is None for x in vals) else float(np.mean(vals))
        print(f'     {label:<20}' + ''.join(f'{_pct(x):>11}' for x in vals) + f'{_pct(m):>11}')
    print('\nRead it as: A large -> fix the source reading first. B gain large -> a per-face strength is the '
          'lever for identity.\nC teacher>R50 large -> the training signal (teachers), not the flow. C unmoved large -> '
          'the flow ignores the request on those faces.')
    return results


# ── run (GPU) ──────────────────────────────────────────────────────────────

def run():
    import torch
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

    p = build_parser()
    p.add_argument('--scales', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5])
    p.add_argument('--only_attrs', nargs='+', type=int, default=None)
    p.add_argument('--arcface_weights', default='./data/model_ir_se50.pth')
    p.add_argument('--dc_margin', type=float, default=0.1,
                   help='Reading dc: a source read on the wrong side for the request is put this far '
                        'on the right side of 0.5 (editing_curves.py --consistent_reading).')
    p.add_argument('--out_csv', required=True)
    p.set_defaults(eval_resize='area')
    args = p.parse_args()
    args = apply_run_config(args)
    E._RESIZE_MODE = args.eval_resize
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
    if not args.independent_attr_weights:
        raise SystemExit('--independent_attr_weights is required (R50 judge).')
    attrs = list(args.only_attrs or args.attribute_index)
    bad = [a for a in attrs if a not in args.attribute_index]
    if bad:
        raise SystemExit(f'--only_attrs {bad} are not edited attributes of this run ({args.attribute_index})')
    scales = sorted(args.scales)

    prior, conditioner, G, id_criterion, attr_teacher, _, direction_bank = load_models(args)
    _, indep_id, _, r50, _, _ = build_optional_judges(args, args.attribute_index, id_criterion)
    if indep_id is None:
        raise SystemExit('the independent face-ID model (facenet-pytorch) is required')
    arc = ArcFaceJudge(args.arcface_weights)

    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)), T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file, train=False, transform=tf)
    eval_ds = data.Subset(ds, range(args.sample_offset, len(ds))) if args.sample_offset else ds
    loader = data.DataLoader(eval_ds, shuffle=False, batch_size=args.batch, num_workers=4)

    rows, seen, dc_batches, batches = [], 0, 0, 0
    with torch.no_grad():
        for img, latent, _ in tqdm(loader, desc='headroom'):
            if seen >= args.num_samples:
                break
            img, latent = img.cuda(), latent.cuda()
            _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
            src = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
            s256 = resize_for_judge(src, (256, 256))
            sp = torch.sigmoid(r50(s256)[0])[:, :40]
            tp = torch.sigmoid(attr_teacher(s256)[0])[:, :40]
            s_arc, s_id = arc.extract(s256), indep_id.extract(s256)
            B = img.size(0)
            for g in attrs:
                li = args.attribute_index.index(g)
                keep = (sp[:, g] > CLEAR_HI) | (sp[:, g] < CLEAR_LO)
                if not keep.any():
                    continue
                d = torch.where(sp[:, g] > 0.5, -1.0, 1.0)
                cond_src = attr_cond[:, li]
                wrong = torch.where(d > 0, cond_src > 0.5, cond_src < 0.5) & keep
                dc_value = torch.where(d > 0, cond_src.clamp(max=0.5 - args.dc_margin),
                                       cond_src.clamp(min=0.5 + args.dc_margin))
                recs = [{'attribute': ATTR_NAMES.get(g, f'attr{g}'), 'face': seen + b, 'dir': float(d[b]),
                         'r50_src': float(sp[b, g]), 'cond_src': float(cond_src[b]), 't_src': float(tp[b, g])}
                        for b in range(B)]
                batches += 1
                dc_batches += int(bool(wrong.any()))
                for sc in scales:
                    measured = {}
                    for v in READINGS:
                        if v == 'dc' and not wrong.any():
                            measured[v] = measured['r']        # no face of this batch is misread: same edit
                            continue
                        face = edit_single_attribute(
                            prior, conditioner, G, id_criterion, img, latent, attr_cond, id_cond,
                            li, sc, direction_bank, attr_global_idx=g,
                            bypass_glasses_direction_bank=args.bypass_glasses_direction_bank,
                            composite=False, direction=d,
                            src_value=dc_value if v == 'dc' else None, src_value_in_encoding=(v == 'dc'))
                        f256 = resize_for_judge(face, (256, 256))
                        measured[v] = (torch.sigmoid(r50(f256)[0])[:, g], torch.sigmoid(attr_teacher(f256)[0])[:, g],
                                       (s_arc * arc.extract(f256)).sum(1), (s_id * indep_id.extract(f256)).sum(1))
                    for v in READINGS:
                        ep, et, a, i = measured[v]
                        for b in range(B):
                            recs[b][f'{v}_p@{sc:g}'] = float(ep[b])
                            recs[b][f'{v}_t@{sc:g}'] = float(et[b])
                            recs[b][f'{v}_arc@{sc:g}'] = float(a[b])
                            recs[b][f'{v}_id@{sc:g}'] = float(i[b])
                rows += [recs[b] for b in range(B) if bool(keep[b])]
            seen += B

    if not rows:
        raise SystemExit('no R50-clear face was found')
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'{seen} faces, {len(rows)} (face, attribute) rows; the dc pass was rendered for '
          f'{dc_batches}/{batches} batches\nsaved {args.out_csv}\n'
          f'report: python -m scripts.headroom_audit report {args.out_csv}')


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'report':
        import argparse
        ap = argparse.ArgumentParser(prog='headroom_audit report')
        ap.add_argument('csv')
        ap.add_argument('--margins', nargs='+', type=float, default=[0.0, 0.2, 0.4],
                        help='Teacher-stopped: the teacher must be this far past 0.5 on the requested side.')
        ap.add_argument('--id_goal', type=float, default=0.90)
        ap.add_argument('--fixed_scale', type=float, default=1.0, help='The strength section A is read at.')
        ap.add_argument('--id_metric', default='arc', choices=['arc', 'id'], help='arc: ArcFace IR-SE50; id: facenet.')
        a = ap.parse_args(sys.argv[2:])
        report(a.csv, tuple(a.margins), a.id_goal, a.fixed_scale, a.id_metric)
    else:
        run()
