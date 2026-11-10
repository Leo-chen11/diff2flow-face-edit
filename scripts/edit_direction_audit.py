"""Where does an eval gain come from? Per-face audit of one attribute's edits.

evaluate_sdflow.py (edit_direction cond) and scripts/editing_curves.py disagree on
eotny's Male: +3.3 pp R50 at matched ArcFace in the eval, none on the curves. The two
differ in two ways: the eval lets the model's CONDITIONER pick the edit direction and
scores every face, the curves take the direction from R50 and keep only faces R50
reads clearly (outside 0.35-0.65). This script edits the same test faces both ways
and records, per face:

  r50_src     R50 reading of the source reconstruction
  cond_src    the conditioner's reading (what picks the direction in the eval)
  clear       r50_src outside 0.35-0.65
  agree       conditioner and R50 put the source on the same side of 0.5
  t_src       the r34 TRAINING teacher's reading (--attribute_weights)
  per scale and per protocol (--protocols):
    c   direction and target from the conditioner          (= evaluate_sdflow, edit_direction cond)
    r   direction from R50, target from the conditioner    (= editing_curves.py)
    t   direction AND target value from the training teacher; the flow still
        encodes the source under the conditioner's reading
    te  as t, and the flow also ENCODES the source under the teacher's reading
    o   direction and target from R50 itself, the flow still encoding under the
        conditioner's reading (ORACLE: uses the evaluation judge, not a method)
    oe  as o, and the flow also ENCODES under R50's reading: the upper bound for
        what a correct reading of the source could give
  with
    ok        R50 success, judged from R50's own reading of the source
    arc       ArcFace of edit vs source reconstruction
    lip, mk   Wearing_Lipstick / Heavy_Makeup flipped (source read clearly)

t and te are fair: the teacher is the model training already uses, not a judge.
If they recover the faces c / r leave untouched, the misses come from the
conditioner's reading of the source, not from the flow's ability to edit.

Run it once per checkpoint, then compare two runs:

    python -m scripts.edit_direction_audit --checkpoint_dir output/SDFlow/<ctrl> --step 100000 \\
        --independent_attr_weights ./data/r50_celebahq_eval.pth --independent_attr_backbone r50 \\
        --attr 20 --scales 0.8 0.9 0.95 --num_samples 500 --out_csv output/audit/ctrl_male.csv
    python -m scripts.edit_direction_audit compare output/audit/ctrl_male.csv output/audit/eotny_male.csv
    python -m scripts.edit_direction_audit compare output/audit/ctrl_male.csv     # one run: every protocol

compare splits the faces into clear+agree / clear+disagree / ambiguous and gives, for
each group, its share of faces and both runs' success, ArcFace and lipstick/makeup flips
under each protocol -- so an eval gain can be traced to the group it comes from.
"""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

CLEAR_LO, CLEAR_HI = 0.35, 0.65
LIP, MAKEUP = 36, 18


def load_rows(p):
    rows = list(csv.DictReader(open(p)))
    return {int(r['face']): r for r in rows}


def group_of(r):
    if r['clear'] != '1':
        return 'ambiguous'
    return 'clear+agree' if r['agree'] == '1' else 'clear+disagree'


def fmean(rows, key):
    v = [float(r[key]) for r in rows if r.get(key, '') not in ('', 'nan')]
    return float(np.mean(v)) if v else float('nan')


PROTO_LABEL = {'c': 'conditioner direction and target (as evaluate_sdflow --edit_direction cond)',
               'r': 'R50 direction, conditioner target (as editing_curves.py)',
               't': 'training-teacher direction and target (encode with the conditioner)',
               'te': 'training-teacher direction and target, also in the encoding',
               'o': 'ORACLE: R50 direction and target, encoding unchanged (uses the eval judge; not a method)',
               'oe': 'ORACLE: R50 direction and target, also in the encoding (upper bound, not a method)'}


def report(path):
    A = load_rows(path)
    faces = sorted(A)
    keys = next(iter(A.values())).keys()
    protos = [p for p in PROTO_LABEL if any(k.startswith(p + '_ok@') for k in keys)]
    scales = sorted({k.split('@')[1] for k in keys if '@' in k}, key=float)
    groups = defaultdict(list)
    for f in faces:
        groups[group_of(A[f])].append(f)
    print(f'{os.path.basename(path)}: {len(faces)} faces')
    print('  share: ' + ', '.join(f'{g} {len(groups[g]) / len(faces):.1%}'
                                  for g in ('clear+agree', 'clear+disagree', 'ambiguous')))
    if 't_src' in keys:
        clear = [A[f] for f in faces if A[f]['clear'] == '1']
        t_agree = np.mean([(float(r['t_src']) > 0.5) == (float(r['r50_src']) > 0.5) for r in clear])
        c_agree = np.mean([r['agree'] == '1' for r in clear])
        print(f'  on R50-clear faces, agreement with R50: conditioner {c_agree:.1%}, training teacher {t_agree:.1%}')
        dis = [A[f] for f in groups['clear+disagree']]
        if dis:
            tc = np.mean([(float(r['t_src']) > 0.5) == (float(r['cond_src']) > 0.5) for r in dis])
            print(f'  on clear+disagree faces the training teacher sides with the conditioner on {tc:.1%} '
                  f'(high = R50 is the odd one out on these faces, not the model)')
    for sc in scales:
        print(f'\n  scale {sc}   (ok = R50 success %, arc = ArcFace)')
        print(f'    {"protocol":<9}' + ''.join(f'{g:>22}' for g in ('clear+agree', 'clear+disagree', 'ambiguous', 'all')))
        for p in protos:
            cells = []
            for g in ('clear+agree', 'clear+disagree', 'ambiguous', 'all'):
                rows = [A[f] for f in (faces if g == 'all' else groups[g])]
                ok, arc = fmean(rows, f'{p}_ok@{sc}'), fmean(rows, f'{p}_arc@{sc}')
                cells.append(f'{ok * 100:5.1f}% / {arc:.3f}' if rows else '--')
            print(f'    {p:<9}' + ''.join(f'{c:>22}' for c in cells))
    print('\n  protocols: ' + '; '.join(f'{p} = {PROTO_LABEL[p]}' for p in protos))
    print('  If t / te lift clear+disagree toward clear+agree, the misses are the conditioner\'s reading of\n'
          '  the source; o bounds what any correct reading can recover.')


def compare(path_a, path_b):
    A, B = load_rows(path_a), load_rows(path_b)
    faces = sorted(set(A) & set(B))
    scales = sorted({k.split('@')[1] for k in next(iter(A.values())) if '@' in k}, key=float)
    na, nb = os.path.basename(path_a), os.path.basename(path_b)
    print(f'{len(faces)} faces in both ({na}: {len(A)}, {nb}: {len(B)})')

    group = group_of
    # the group is a property of the SOURCE face under each run's conditioner; take run A's
    # R50 reading (identical in both: same reconstruction, same judge) and report both runs'
    # agreement separately
    groups = defaultdict(list)
    for f in faces:
        groups[group(A[f])].append(f)
    agree_a = np.mean([A[f]['agree'] == '1' for f in faces])
    agree_b = np.mean([B[f]['agree'] == '1' for f in faces])
    print(f'conditioner agrees with R50 on the source: {na} {agree_a:.1%}, {nb} {agree_b:.1%}')

    def mean(rows, key):
        v = [float(r[key]) for r in rows if r.get(key, '') not in ('', 'nan')]
        return (float(np.mean(v)), len(v)) if v else (float('nan'), 0)

    keys = next(iter(A.values())).keys()
    for proto, label in [(p, l) for p, l in PROTO_LABEL.items() if any(k.startswith(p + '_ok@') for k in keys)]:
        print(f'\n=== {label}')
        for sc in scales:
            print(f'  scale {sc}')
            print(f'    {"group":<16}{"share":>7}{"ok A":>8}{"ok B":>8}{"B-A":>7}{"arc A":>8}{"arc B":>8}'
                  f'{"lip A":>7}{"lip B":>7}{"mk A":>7}{"mk B":>7}')
            for g in ('clear+agree', 'clear+disagree', 'ambiguous', 'all'):
                fs = faces if g == 'all' else groups.get(g, [])
                if not fs:
                    continue
                ra, rb = [A[f] for f in fs], [B[f] for f in fs]
                oka, okb = mean(ra, f'{proto}_ok@{sc}')[0], mean(rb, f'{proto}_ok@{sc}')[0]
                print(f'    {g:<16}{len(fs) / len(faces):7.1%}{oka * 100:7.1f}%{okb * 100:7.1f}%'
                      f'{(okb - oka) * 100:+7.1f}'
                      f'{mean(ra, f"{proto}_arc@{sc}")[0]:8.3f}{mean(rb, f"{proto}_arc@{sc}")[0]:8.3f}'
                      f'{mean(ra, f"{proto}_lip@{sc}")[0] * 100:6.1f}%{mean(rb, f"{proto}_lip@{sc}")[0] * 100:6.1f}%'
                      f'{mean(ra, f"{proto}_mk@{sc}")[0] * 100:6.1f}%{mean(rb, f"{proto}_mk@{sc}")[0] * 100:6.1f}%')
    print('\n  ok   R50 success (from R50\'s own reading of the source)   arc  ArcFace vs reconstruction'
          '\n  lip / mk  share of faces whose Wearing_Lipstick / Heavy_Makeup reading (clear on the source) '
          'flipped\n  A group\'s contribution to the all-faces difference = share x (B-A).')


def run():
    import torch
    import torchvision.transforms as T
    from torch.utils import data
    from tqdm import tqdm

    import evaluation.evaluate_sdflow as E
    from evaluation.evaluate_sdflow import (
        _latest_step, apply_run_config, build_optional_judges, build_parser, edit_single_attribute,
        load_models, resize_for_judge,
    )
    from models.arcface import ArcFaceJudge
    from models.dataset import SDFlowDataset

    p = build_parser()
    p.add_argument('--attr', type=int, default=20, help='CelebA id of the edited attribute.')
    p.add_argument('--scales', nargs='+', type=float, default=[0.8, 0.9, 0.95])
    p.add_argument('--arcface_weights', default='./data/model_ir_se50.pth')
    p.add_argument('--protocols', nargs='+', default=['c', 'r', 't', 'te', 'o', 'oe'],
                   choices=list(PROTO_LABEL), help='Which ways of setting direction and target to run.')
    p.add_argument('--out_csv', required=True)
    p.set_defaults(eval_resize='area')
    args = p.parse_args()
    args = apply_run_config(args)
    E._RESIZE_MODE = args.eval_resize
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
    if not args.independent_attr_weights:
        raise SystemExit('--independent_attr_weights is required (R50 judge).')
    if args.attr not in args.attribute_index:
        raise SystemExit(f'--attr {args.attr} is not an edited attribute of this run ({args.attribute_index})')

    prior, conditioner, G, id_criterion, attr_teacher, _, direction_bank = load_models(args)
    _, _, _, r50, _, _ = build_optional_judges(args, args.attribute_index, id_criterion)
    arc = ArcFaceJudge(args.arcface_weights)
    g, li = args.attr, args.attribute_index.index(args.attr)

    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)), T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file, train=False, transform=tf)
    loader = data.DataLoader(ds, shuffle=False, batch_size=args.batch, num_workers=4)

    def flipped(sp, ep, j):
        clear = (sp[:, j] > CLEAR_HI) | (sp[:, j] < CLEAR_LO)
        flip = ((sp[:, j] > 0.5) != (ep[:, j] > 0.5)) & clear
        return [float(f) if c else float('nan') for f, c in zip(flip.tolist(), clear.tolist())]

    rows, seen = [], 0
    with torch.no_grad():
        for img, latent, _ in tqdm(loader, desc='audit'):
            if seen >= args.num_samples:
                break
            img, latent = img.cuda(), latent.cuda()
            _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
            src = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
            s256 = resize_for_judge(src, (256, 256))
            sp = torch.sigmoid(r50(s256)[0])[:, :40]
            tp = torch.sigmoid(attr_teacher(s256)[0])[:, :40]
            s_arc = arc.extract(s256)
            cond_src = attr_cond[:, li]
            d_r50 = torch.where(sp[:, g] > 0.5, -1.0, 1.0)
            d_t = torch.where(tp[:, g] > 0.5, -1.0, 1.0)
            # protocol -> (direction, source value for the target, also in the encoding)
            setups = {'c': (None, None, False), 'r': (d_r50, None, False),
                      't': (d_t, tp[:, g], False), 'te': (d_t, tp[:, g], True),
                      'o': (d_r50, sp[:, g], False), 'oe': (d_r50, sp[:, g], True)}
            B = img.size(0)
            recs = [{'face': seen + b, 'r50_src': float(sp[b, g]), 'cond_src': float(cond_src[b]),
                     't_src': float(tp[b, g]),
                     'clear': int((sp[b, g] > CLEAR_HI) or (sp[b, g] < CLEAR_LO)),
                     'agree': int((sp[b, g] > 0.5) == (cond_src[b] > 0.5))} for b in range(B)]
            for sc in args.scales:
                for proto in args.protocols:
                    d, sv, enc = setups[proto]
                    face = edit_single_attribute(
                        prior, conditioner, G, id_criterion, img, latent, attr_cond, id_cond,
                        li, sc, direction_bank, attr_global_idx=g,
                        bypass_glasses_direction_bank=args.bypass_glasses_direction_bank,
                        composite=False, direction=d, src_value=sv, src_value_in_encoding=enc)
                    f256 = resize_for_judge(face, (256, 256))
                    ep = torch.sigmoid(r50(f256)[0])[:, :40]
                    ok = torch.where(sp[:, g] > 0.5, ep[:, g] < 0.5, ep[:, g] > 0.5)
                    a = (s_arc * arc.extract(f256)).sum(1)
                    lip, mk = flipped(sp, ep, LIP), flipped(sp, ep, MAKEUP)
                    for b in range(B):
                        recs[b][f'{proto}_ok@{sc:g}'] = int(ok[b])
                        recs[b][f'{proto}_arc@{sc:g}'] = float(a[b])
                        recs[b][f'{proto}_lip@{sc:g}'] = lip[b]
                        recs[b][f'{proto}_mk@{sc:g}'] = mk[b]
            rows += recs
            seen += B

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    agree = np.mean([r['agree'] for r in rows])
    clear = np.mean([r['clear'] for r in rows])
    print(f'{len(rows)} faces; R50 clear on {clear:.1%}; conditioner agrees with R50 on {agree:.1%}')
    print(f'saved {args.out_csv}')


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'compare':
        if len(sys.argv) == 3:
            report(sys.argv[2])
        elif len(sys.argv) == 4:
            compare(sys.argv[2], sys.argv[3])
        else:
            raise SystemExit('usage: python -m scripts.edit_direction_audit compare <run csv> [<run B csv>]')
    else:
        run()
