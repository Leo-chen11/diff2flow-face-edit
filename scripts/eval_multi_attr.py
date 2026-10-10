"""Stacked edits: change 2-3 attributes of the same face at once, and score it.

For every combination (pairs and triples of the run's attributes) and every
composition rule, each attribute's edit direction comes from the independent
R50 judge's reading of the source (as evaluate_sdflow.py --edit_direction
indep). A sample counts for a combination only when R50 reads EVERY edited
attribute clearly on the source (< 0.35 or > 0.65).

Composition rules (--compose, several allowed):
  sum   one flow pass per attribute from the same source, calibrated W+ deltas
        and ControlNet skips added (evaluate_sdflow.edit_multi_attribute).
  orth  as sum, but each delta first loses, per W+ layer, its component in the
        span of the other attributes' deltas (shared directions applied once).
  seq   one attribute after the other: edit, regenerate, re-read the
        conditions from the edited face, edit the next. ControlNet skips
        accumulate. The usual "sequential editing" of StyleFlow / Latent
        Transformer; errors can accumulate.

Reported per combination and rule:
  Acc       R50 success rate of each edited attribute
  all       share of faces where EVERY edited attribute succeeded
  single    the same faces edited one attribute at a time (the stacking-free
            reference): per-attribute rate and the share where all single
            edits succeed. Stacking loss = single - stacked.
  ID_ind / LPIPS vs the source reconstruction
  others    mean |dP| of the run's attributes that were NOT edited

Usage (all evaluate_sdflow.py model / judge options apply):
    python -m scripts.eval_multi_attr \
        --checkpoint_dir ./output/SDFlow/multi_v1_cont20k_ctrl --step 90000 \
        --independent_attr_weights ./data/r50_celebahq_eval.pth \
        --independent_attr_backbone r50 --celeba_attr_judge_weights "" \
        --age_fine_layer_scale 1.0 --multi_scale 1.0 --num_samples 200 \
        --compose sum orth seq
"""
import itertools
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils import data
from tqdm import tqdm

from evaluation.evaluate_sdflow import (
    ATTR_NAMES, _latest_step, apply_run_config, build_optional_judges, build_parser,
    edit_multi_attribute, edit_single_attribute, load_models,
    resolve_controlnet_disable_attrs,
)
from models.control_encoder import add_skips
from models.dataset import SDFlowDataset

DEFAULT_TRIPLES = [(15, 31, 5), (20, 39, 31), (20, 15, 39)]   # local x local, global x global x expression, mixed


def name(g):
    return ATTR_NAMES.get(g, f'attr{g}')


def parse_combos(specs, attribute_index):
    if specs:
        combos = [tuple(int(x) for x in s.split(',')) for s in specs]
    else:
        combos = list(itertools.combinations(attribute_index, 2))
        combos += [t for t in DEFAULT_TRIPLES if all(a in attribute_index for a in t)]
    for c in combos:
        if not 2 <= len(c) <= 3 or len(set(c)) != len(c):
            raise SystemExit(f'combo {c}: give 2 or 3 distinct attributes')
        missing = [a for a in c if a not in attribute_index]
        if missing:
            raise SystemExit(f'combo {c}: {missing} not in this run\'s attribute_index {attribute_index}')
    return combos


def success(src, edit, margin):
    """Vectorised strict_success: the edited score crossed 0.5 (by margin)."""
    return torch.where(src > 0.5, edit < 0.5 - margin, edit > 0.5 + margin)


def clear(p):
    return (p > 0.65) | (p < 0.35)


def mean(v):
    return float(np.mean(v)) if len(v) else None


def main():
    p = build_parser()
    p.add_argument('--combos', nargs='*', default=None,
                   help='Attribute combinations as comma lists of CelebA indices, e.g. 15,31 20,39,31. '
                        'Default: every pair of the run\'s attributes plus '
                        f'{DEFAULT_TRIPLES}.')
    p.add_argument('--compose', nargs='+', default=['sum', 'orth', 'seq'],
                   choices=['sum', 'orth', 'seq'])
    p.add_argument('--multi_scale', type=float, default=1.0,
                   help='edit_scale of every attribute in a combination (and of the single-edit reference).')
    args = p.parse_args()
    args = apply_run_config(args)
    args = resolve_controlnet_disable_attrs(args)
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
    if not args.independent_attr_weights:
        raise SystemExit('--independent_attr_weights is required: it sets each edit\'s direction '
                         'and scores the result.')
    if args.composite_face_region:
        print('[WARN] --composite_face_region is ignored here (stacked edits are not composited).')

    prior, conditioner, G, id_criterion, attr_teacher, \
        attribute_index, direction_bank, control_encoder = load_models(args)
    _, indep_id, lpips_fn, indep_teacher, _, _ = \
        build_optional_judges(args, args.attribute_index, id_criterion)
    combos = parse_combos(args.combos, list(args.attribute_index))
    local = {g: args.attribute_index.index(g) for g in args.attribute_index}
    ce_kw = dict(control_encoder=control_encoder,
                 controlnet_max_norm=getattr(args, 'controlnet_max_norm', 0.0),
                 controlnet_disable_attrs=getattr(args, 'controlnet_disable_attrs', None),
                 controlnet_embed_res=getattr(args, 'controlnet_embed_res', 64))
    print(f'{len(combos)} combinations x compose {args.compose} at scale {args.multi_scale}: '
          + ', '.join('+'.join(name(a) for a in c) for c in combos))

    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)), T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file,
                       train=False, transform=tf)
    loader = data.DataLoader(ds, shuffle=False, batch_size=args.batch, num_workers=4)

    # rec[(combo, mode)][key] -> list of per-sample values
    rec = defaultdict(lambda: defaultdict(list))
    seen = 0
    with torch.no_grad():
        for img, latent, _ in tqdm(loader, desc='stacked edits'):
            if seen >= args.num_samples:
                break
            img, latent = img.cuda(), latent.cuda()
            B = img.size(0)
            _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
            src_face = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
            src_256 = F.interpolate(src_face, (256, 256))
            sp = torch.sigmoid(indep_teacher(src_256)[0])[:, :40]
            src_id = indep_id.extract(src_256) if indep_id is not None else None
            direction = {g: torch.where(sp[:, g] > 0.5, -1.0, 1.0) for g in args.attribute_index}

            def score(face):
                f256 = F.interpolate(face, (256, 256))
                ep = torch.sigmoid(indep_teacher(f256)[0])[:, :40]
                idv = (src_id * indep_id.extract(f256)).sum(1) if src_id is not None else None
                lp = lpips_fn(src_256, f256).flatten() if lpips_fn is not None else None
                return ep, idv, lp

            # Single-attribute reference, once per attribute.
            single_ok = {}
            for g in sorted({a for c in combos for a in c}):
                face = edit_single_attribute(
                    prior, conditioner, G, id_criterion, img, latent, attr_cond, id_cond,
                    local[g], args.multi_scale, direction_bank, attr_global_idx=g,
                    bypass_glasses_direction_bank=args.bypass_glasses_direction_bank,
                    composite=False, direction=direction[g], **ce_kw)
                ep, _, _ = score(face)
                single_ok[g] = success(sp[:, g], ep[:, g], args.success_margin)

            for combo in combos:
                keep = torch.stack([clear(sp[:, g]) for g in combo], 0).all(0)
                if not keep.any():
                    continue
                dirs = [direction[g] for g in combo]
                for mode in args.compose:
                    if mode == 'seq':
                        cur_img, cur_lat, skips, face = img, latent, None, None
                        for g, d in zip(combo, dirs):
                            _, idc, atc = conditioner.make_condition(cur_img, cur_lat, id_criterion)
                            _, cur_lat, sk = edit_multi_attribute(
                                prior, conditioner, G, id_criterion, cur_img, cur_lat, atc, idc,
                                [local[g]], [args.multi_scale], direction_bank, attr_global_idxs=[g],
                                directions=[d], return_parts=True, **ce_kw)
                            skips = add_skips(skips, sk)
                            face = G([cur_lat], skips=skips, embed_res=ce_kw['controlnet_embed_res'],
                                     input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
                            cur_img = F.interpolate(face, (args.img_size, args.img_size))
                    else:
                        face = edit_multi_attribute(
                            prior, conditioner, G, id_criterion, img, latent, attr_cond, id_cond,
                            [local[g] for g in combo], [args.multi_scale] * len(combo), direction_bank,
                            attr_global_idxs=list(combo), directions=dirs, compose=mode, **ce_kw)
                    ep, idv, lp = score(face)
                    r = rec[(combo, mode)]
                    oks = []
                    for g in combo:
                        ok = success(sp[:, g], ep[:, g], args.success_margin)
                        oks.append(ok)
                        r[f'acc_{g}'] += ok[keep].float().tolist()
                        r[f'single_{g}'] += single_ok[g][keep].float().tolist()
                    r['all'] += torch.stack(oks, 0).all(0)[keep].float().tolist()
                    r['single_all'] += torch.stack([single_ok[g] for g in combo], 0).all(0)[keep].float().tolist()
                    others = [g for g in args.attribute_index if g not in combo]
                    if others:
                        r['others'] += (ep[:, others] - sp[:, others]).abs().mean(1)[keep].tolist()
                    if idv is not None:
                        r['id'] += idv[keep].tolist()
                    if lp is not None:
                        r['lpips'] += lp[keep].tolist()
            seen += B

    def pc(v):
        return f'{v * 100:5.1f}' if v is not None else '   --'

    def f3(v):
        return f'{v:.3f}' if v is not None else '   --'

    out = {'config': {'checkpoint_dir': args.checkpoint_dir, 'step': args.step,
                      'num_samples': args.num_samples, 'multi_scale': args.multi_scale,
                      'compose': args.compose, 'success_margin': args.success_margin,
                      'independent_attr_weights': args.independent_attr_weights},
           'combos': {}}
    summary = defaultdict(lambda: defaultdict(list))
    for combo in combos:
        cname = ' + '.join(name(g) for g in combo)
        rows = {m: rec.get((combo, m)) for m in args.compose}
        n = len(next((r['all'] for r in rows.values() if r), []))
        print(f'\n=== {cname} (n={n} faces with every source attribute clear) ===')
        if not n:
            continue
        hdr = ''.join(f' {name(g)[:10]:>10}' for g in combo)
        print(f'  {"compose":<8} {"ID_ind":>7} {"LPIPS":>6}{hdr} {"all":>6} {"others|dP|":>11}')
        ref = next(r for r in rows.values() if r)
        print(f'  {"single":<8} {"--":>7} {"--":>6}'
              + ''.join(f' {pc(mean(ref[f"single_{g}"])):>10}' for g in combo)
              + f' {pc(mean(ref["single_all"])):>6} {"--":>11}')
        cj = out['combos'][cname] = {'n': n, 'single': {name(g): mean(ref[f'single_{g}']) for g in combo},
                                     'single_all': mean(ref['single_all'])}
        for m, r in rows.items():
            if not r:
                continue
            print(f'  {m:<8} {f3(mean(r["id"])):>7} {f3(mean(r["lpips"])):>6}'
                  + ''.join(f' {pc(mean(r[f"acc_{g}"])):>10}' for g in combo)
                  + f' {pc(mean(r["all"])):>6} {f3(mean(r["others"])):>11}')
            cj[m] = {'acc': {name(g): mean(r[f'acc_{g}']) for g in combo}, 'all': mean(r['all']),
                     'id': mean(r['id']), 'lpips': mean(r['lpips']), 'others_abs_dp': mean(r['others'])}
            k = f'{len(combo)} attrs'
            summary[k][m].append((mean(r['all']), mean(ref['single_all']), mean(r['id']),
                                  np.mean([mean(r[f'single_{g}']) - mean(r[f'acc_{g}']) for g in combo])))

    print('\n=== Summary (mean over combinations) ===')
    print(f'  {"":<8} {"compose":<8} {"all":>6} {"single all":>11} {"ID_ind":>7} {"stack loss/attr":>16}')
    out['summary'] = {}
    for k, per in summary.items():
        for m, vals in per.items():
            a = np.array([[x if x is not None else np.nan for x in v] for v in vals], dtype=float)
            al, sa, idv, sl = np.nanmean(a, 0)
            print(f'  {k:<8} {m:<8} {al * 100:5.1f}% {sa * 100:10.1f}% {idv:7.3f} {sl * 100:+15.1f}')
            out['summary'].setdefault(k, {})[m] = dict(all=al, single_all=sa, id=idv, stack_loss_per_attr=sl)
    print('\nall = every edited attribute succeeded (R50). single all = the same faces edited one '
          'attribute at a time. stack loss/attr = single - stacked accuracy, averaged over the edited '
          'attributes (positive = stacking costs accuracy).')

    path = args.out_json or os.path.join(
        args.checkpoint_dir, f'eval_multi_attr_step{args.step}_n{args.num_samples}.json')
    if os.path.exists(path):
        stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(os.path.getmtime(path)))
        os.replace(path, f'{os.path.splitext(path)[0]}_prev{stamp}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved {path}')


if __name__ == '__main__':
    main()
