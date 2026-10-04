"""Why do some edits of one attribute fail? Four hypotheses, tested in one pass.

Built for Young "add" (old -> young): ~1/3 fail at scale 1.0 and most of those
barely move (R50 dP < 0.05), while "rm" (young -> old) succeeds ~93%.

  H1 condition mismatch   the edit direction comes from the R50 judge, but how
                          far the flow is asked to move comes from the
                          conditioner's own score: attr_delta = scale * (end - cond).
                          A face R50 calls old but the conditioner calls young
                          (0.8) gets attr_delta 0.2 and barely changes.
                          Tested by C : the attribute's condition replaced by
                                        the R50 reading (diagnostic only: R50
                                        is the judge), and
                                    C': replaced by the training r34 reading
                                        (usable at inference).
  H2 too weak             right direction, not enough push.  B: scale 1.5/2/3.
  H3 locked fine layers   the bank edit is restricted to W+ layers 0-10
                          (bank_dir_layers 39:0-10); un-greying hair or smoothing
                          skin may need 11-17.  D: layers 0-17 for this edit.
  H4 judge threshold      very old sources may need a drastic change before R50
                          says "young", or the face changes but R50 disagrees.
                          Look at the source readings and the montage.

Sources: the eval's test faces; direction from R50 on the source (as
evaluate_sdflow --edit_direction indep), unclear sources (0.35-0.65) skipped.

Outputs (in --out_dir): a table per direction (add / rm) x outcome at scale 1.0
(condition values, attr_delta, edit norm, dP), the share of baseline failures
each intervention rescues with its accuracy / ID cost, failures.png
(rows: failed add faces; columns: source, A, B@2, C, C', D with the R50 score),
successes.png, and a JSON.

Usage:
    python -m scripts.diagnose_attr_edit --checkpoint_dir $CK --step 100000 --attr 39 \
        --independent_attr_weights ./data/r50_celebahq_eval.pth --independent_attr_backbone r50 \
        --celeba_attr_judge_weights "" --age_fine_layer_scale 1.0 --num_samples 500
"""
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torch.utils import data
from tqdm import tqdm

from evaluation.evaluate_sdflow import (
    ATTR_NAMES, _latest_step, apply_run_config, build_optional_judges, build_parser,
    edit_single_attribute, load_models, resolve_controlnet_disable_attrs,
)
from models.dataset import SDFlowDataset
from scripts.analyze_residual import bank_edit

CLEAR_LO, CLEAR_HI = 0.35, 0.65


def to_pil(x, size=160):
    x = F.interpolate(x.unsqueeze(0), (size, size))[0]
    return Image.fromarray(((x.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy())


def montage(rows, labels, path, size=160):
    """rows: list of (list of (img tensor, caption))."""
    if not rows:
        return
    head = 18
    W, H = size * len(labels), (size + head) * len(rows) + head
    canvas = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(canvas)
    for j, lab in enumerate(labels):
        d.text((j * size + 4, 2), lab, fill='black')
    for i, row in enumerate(rows):
        y = head + i * (size + head)
        for j, (img, cap) in enumerate(row):
            canvas.paste(to_pil(img, size), (j * size, y))
            d.text((j * size + 4, y + size + 2), cap, fill='black')
    canvas.save(path)


def main():
    p = build_parser()
    p.add_argument('--attr', type=int, default=39)
    p.add_argument('--scales_extra', nargs='+', type=float, default=[1.5, 2.0, 3.0])
    p.add_argument('--montage_rows', type=int, default=24)
    p.add_argument('--out_dir', default=None)
    args = p.parse_args()
    args = apply_run_config(args)
    args = resolve_controlnet_disable_attrs(args)
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
    if not args.independent_attr_weights:
        raise SystemExit('--independent_attr_weights is required (R50 judge).')
    g = int(args.attr)
    if g not in args.attribute_index:
        raise SystemExit(f'attr {g} not in this run\'s attribute_index {args.attribute_index}')
    li = args.attribute_index.index(g)
    name = ATTR_NAMES.get(g, f'attr{g}')
    out_dir = args.out_dir or os.path.join(args.checkpoint_dir, f'diagnose_{name}_s{args.step}')
    os.makedirs(out_dir, exist_ok=True)

    prior, conditioner, G, id_criterion, attr_teacher, _, direction_bank, control_encoder = load_models(args)
    _, indep_id, _, indep_teacher, _, _ = build_optional_judges(args, args.attribute_index, id_criterion)
    ce_kw = dict(control_encoder=control_encoder,
                 controlnet_max_norm=getattr(args, 'controlnet_max_norm', 0.0),
                 controlnet_disable_attrs=getattr(args, 'controlnet_disable_attrs', None),
                 controlnet_embed_res=getattr(args, 'controlnet_embed_res', 64))
    b2 = 2.0 if 2.0 in args.scales_extra else args.scales_extra[-1]
    conds = ['A'] + [f'B{s:g}' for s in args.scales_extra] + ['C', "C'", 'D']

    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)), T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file,
                       train=False, transform=tf)
    loader = data.DataLoader(ds, shuffle=False, batch_size=args.batch, num_workers=4)

    recs = []                       # one dict per clear source
    imgs = {}                       # index in recs -> {cond: image}, kept for montage candidates
    keep_imgs = args.montage_rows * 4
    seen = 0
    with torch.no_grad():
        for img, latent, _ in tqdm(loader, desc=f'diagnose {name}'):
            if seen >= args.num_samples:
                break
            img, latent = img.cuda(), latent.cuda()
            B = img.size(0)
            seen += B
            _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
            src = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
            s256 = F.interpolate(src, (256, 256))
            r50 = torch.sigmoid(indep_teacher(s256)[0])[:, g]
            r34 = torch.sigmoid(attr_teacher(s256)[0])[:, g]
            cond = attr_cond[:, li]
            src_id = indep_id.extract(s256) if indep_id is not None else None
            keep = (r50 > CLEAR_HI) | (r50 < CLEAR_LO)
            if not keep.any():
                continue
            d = torch.where(r50 > 0.5, -1.0, 1.0)
            delta, _, ad = bank_edit(prior, direction_bank, latent, attr_cond, id_cond, li, 1.0, g, d)

            def run(scale=1.0, acond=attr_cond):
                face = edit_single_attribute(
                    prior, conditioner, G, id_criterion, img, latent, acond, id_cond, li, scale,
                    direction_bank, attr_global_idx=g,
                    bypass_glasses_direction_bank=args.bypass_glasses_direction_bank,
                    composite=False, direction=d, **ce_kw)
                f256 = F.interpolate(face, (256, 256))
                ep = torch.sigmoid(indep_teacher(f256)[0])[:, g]
                idv = (src_id * indep_id.extract(f256)).sum(1) if src_id is not None else torch.zeros(B)
                return face, ep, idv

            out = {'A': run()}
            for s in args.scales_extra:
                out[f'B{s:g}'] = run(scale=s)
            ac = attr_cond.clone()
            ac[:, li] = r50
            out['C'] = run(acond=ac)
            ac = attr_cond.clone()
            ac[:, li] = r34
            out["C'"] = run(acond=ac)
            old_mask = direction_bank._dir_layer_mask.get(li)
            direction_bank.set_dir_layers(li, 0, direction_bank.num_layers - 1)
            try:
                out['D'] = run()
            finally:
                if old_mask is None:
                    direction_bank._dir_layer_mask.pop(li, None)
                else:
                    direction_bank._dir_layer_mask[li] = old_mask

            for b in range(B):
                if not keep[b]:
                    continue
                add = bool(d[b] > 0)
                rec = {'dir': 'add' if add else 'rm', 'r50': float(r50[b]), 'r34': float(r34[b]),
                       'cond': float(cond[b]), 'attr_delta': float(ad[b]),
                       'edit_norm': float(delta[b].flatten().norm())}
                for c, (_, ep, idv) in out.items():
                    e = float(ep[b])
                    rec[f'ok_{c}'] = (e > 0.5) if add else (e < 0.5)
                    rec[f'dp_{c}'] = (e - rec['r50']) if add else (rec['r50'] - e)
                    rec[f'p_{c}'] = e
                    rec[f'id_{c}'] = float(idv[b])
                if add and len(imgs) < keep_imgs:
                    small = lambda x: F.interpolate(x[b:b + 1], (160, 160), mode='area')[0].cpu()
                    imgs[len(recs)] = {'src': small(src),
                                       **{c: small(out[c][0]) for c in ['A', f'B{b2:g}', 'C', "C'", 'D']}}
                recs.append(rec)

    def mean(xs):
        return float(np.mean(xs)) if len(xs) else float('nan')

    report = {'config': {'checkpoint_dir': args.checkpoint_dir, 'step': args.step, 'attr': g,
                         'num_faces': seen, 'scales_extra': args.scales_extra}, 'groups': {}, 'rescue': {}}
    print(f'\n{name}: {len(recs)} clear sources of {seen} faces. Direction from R50; '
          f'"cond" = the conditioner\'s score the flow edits from.\n')
    print('Baseline (scale 1.0) by direction and outcome:')
    print(f'  {"group":<14} {"n":>4} {"cond":>6} {"cond wrong side":>15} {"R50 src":>8} {"r34 src":>8} '
          f'{"|attr_delta|":>12} {"|edit|":>7} {"R50 dP":>7}')
    for dr in ('add', 'rm'):
        for okv, lab in ((False, 'fail'), (True, 'success')):
            rs = [r for r in recs if r['dir'] == dr and r['ok_A'] == okv]
            if not rs:
                continue
            wrong = [(r['cond'] > 0.5) if dr == 'add' else (r['cond'] < 0.5) for r in rs]
            row = dict(n=len(rs), cond=mean([r['cond'] for r in rs]), cond_wrong_side=mean(wrong),
                       r50=mean([r['r50'] for r in rs]), r34=mean([r['r34'] for r in rs]),
                       attr_delta=mean([abs(r['attr_delta']) for r in rs]),
                       edit_norm=mean([r['edit_norm'] for r in rs]), dp=mean([r['dp_A'] for r in rs]))
            report['groups'][f'{dr}_{lab}'] = row
            print(f'  {dr + " " + lab:<14} {row["n"]:4d} {row["cond"]:6.2f} {row["cond_wrong_side"] * 100:14.0f}% '
                  f'{row["r50"]:8.2f} {row["r34"]:8.2f} {row["attr_delta"]:12.3f} {row["edit_norm"]:7.2f} '
                  f'{row["dp"]:+7.2f}')

    print('\nInterventions (same faces):')
    print(f'  {"cond":<7} {"add acc":>8} {"add ID":>7} {"rm acc":>7} {"rm ID":>6} {"rescued add fails":>18} '
          f'{"rescued rm fails":>17}')
    for c in conds:
        row = {}
        for dr in ('add', 'rm'):
            rs = [r for r in recs if r['dir'] == dr]
            fails = [r for r in rs if not r['ok_A']]
            row[dr] = dict(acc=mean([r[f'ok_{c}'] for r in rs]), id=mean([r[f'id_{c}'] for r in rs]),
                           rescued=mean([r[f'ok_{c}'] for r in fails]) if c != 'A' else 0.0,
                           n_fail=len(fails))
        report['rescue'][c] = row
        def resc(r):
            return f'{r["rescued"] * 100:3.0f}% of {r["n_fail"]:<3d}' if r['n_fail'] and c != 'A' else '--'
        print(f'  {c:<7} {row["add"]["acc"] * 100:7.1f}% {row["add"]["id"]:7.3f} {row["rm"]["acc"] * 100:6.1f}% '
              f'{row["rm"]["id"]:6.3f} {resc(row["add"]):>18} {resc(row["rm"]):>17}')
    print('\nA = scale 1.0; B<s> = scale s; C = condition set to the R50 reading (diagnostic: R50 is the '
          'judge); C\' = condition set to the training r34 reading (usable at inference); D = W+ layers '
          '0-17 unlocked.\n"cond wrong side" = the conditioner reads the source on the target side already '
          '(add: > 0.5), so the flow is asked for a small change.')

    labels = ['source', 'A 1.0', f'B {b2:g}', 'C R50', "C' r34", 'D 0-17']
    keys = ['A', f'B{b2:g}', 'C', "C'", 'D']

    def rows_for(okv, n):
        out_rows = []
        for i, r in enumerate(recs):
            if i in imgs and r['dir'] == 'add' and r['ok_A'] == okv:
                im = imgs[i]
                row = [(im['src'], f'R50 {r["r50"]:.2f} c{r["cond"]:.2f}')]
                row += [(im[k], f'{r["p_" + k]:.2f}' + (' ok' if r['ok_' + k] else '')) for k in keys]
                out_rows.append(row)
            if len(out_rows) >= n:
                break
        return out_rows

    montage(rows_for(False, args.montage_rows), labels, os.path.join(out_dir, 'failures.png'))
    montage(rows_for(True, 8), labels, os.path.join(out_dir, 'successes.png'))
    with open(os.path.join(out_dir, 'diagnose.json'), 'w') as f:
        json.dump({**report, 'samples': recs}, f, indent=1)
    print(f'\nsaved {out_dir}/failures.png, successes.png, diagnose.json '
          f'(montage: add faces; source caption = R50 reading and conditioner value c)')


if __name__ == '__main__':
    main()
