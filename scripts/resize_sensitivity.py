"""How much do the 1024 -> 256 downsampling mode and the scores it feeds the
judges move the reported success rates?

Every judge and the training teacher see F.interpolate(face, (256, 256)),
i.e. mode='nearest', which aliases: two renderings of the SAME edit that differ
by a sub-pixel shift can land on different sides of 0.5. This script edits once
per face and attribute, then scores the SAME generated images after
downsampling them three ways:

    nearest      what every evaluation so far did
    area         box average of the source pixels (what --eval_resize area would do)
    bilinear_aa  bilinear with antialiasing

Both the source reconstruction and the edit go through the same mode, so each
column is a complete, self-consistent evaluation. Per attribute and judge it
prints

    success   strict success (crosses 0.5), under each mode
    |dp|      mean |p_edit(nearest) - p_edit(area)|, the size of the aliasing noise
    flip      share of faces whose success differs between nearest and area

Reading it: flip within a few percent and success within about a point = the
resize mode is not worth touching. Flip above ~10% or success moving several
points = old numbers carry aliasing noise and the mode should be fixed before
anything is trained on top of it.

Judges: the training teacher (r34), the independent R50, and the CelebA R18.
CLIP is left out: it does its own bicubic resize to 224 from whatever it is fed
and needs its prompt setup, which is not part of this question.

    python -m scripts.resize_sensitivity \
        --checkpoint_dir ./output/SDFlow/multi_v1_min_head_seed1 --step 80000 \
        --bank_mode prior --edit_scale 0.9 --num_samples 200 \
        --independent_attr_weights ./data/r50_celebahq_eval.pth \
        --independent_attr_backbone r50
"""
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils import data
from tqdm import tqdm

from common.ops import load_network
from models.attribute_estimator import AttributeClassifier
from models.dataset import SDFlowDataset
from evaluation.evaluate_sdflow import (CelebAAttrClassifierJudge, _latest_step,
                                        apply_run_config, build_parser,
                                        edit_single_attribute, is_clear, load_models,
                                        resolve_controlnet_disable_attrs, strict_success)

MODES = ['nearest', 'area', 'bilinear_aa']
JUDGES = ['teacher', 'r50', 'r18']
NAMES = {15: 'Eyeglasses', 20: 'Male', 39: 'Young', 31: 'Smiling', 5: 'Bangs'}


def downsample(x, mode, size=256):
    """x: (B,3,H,W). 'nearest' is F.interpolate's default, the existing behaviour."""
    if mode == 'nearest':
        return F.interpolate(x, (size, size))
    if mode == 'area':
        return F.interpolate(x, (size, size), mode='area')
    if mode == 'bilinear_aa':
        return F.interpolate(x, (size, size), mode='bilinear', align_corners=False,
                             antialias=True)
    raise ValueError(mode)


def summarize(records, margin=0.0):
    """records: list of {judge: {mode: (p_src, p_edit)}} -> {judge: stats}."""
    out = {}
    for j in JUDGES:
        ok = {m: np.array([strict_success(r[j][m][0], r[j][m][1], margin) for r in records])
              for m in MODES}
        p_n = np.array([r[j]['nearest'][1] for r in records])
        p_a = np.array([r[j]['area'][1] for r in records])
        out[j] = {'n': len(records),
                  'success': {m: float(ok[m].mean()) for m in MODES},
                  'abs_dp': float(np.abs(p_n - p_a).mean()),
                  'flip_area': float((ok['nearest'] != ok['area']).mean()),
                  'flip_bilinear_aa': float((ok['nearest'] != ok['bilinear_aa']).mean())}
    return out


def main():
    p = build_parser()
    p.add_argument('--edit_scale', type=float, default=0.9)
    args = p.parse_args()
    args = apply_run_config(args)
    args = resolve_controlnet_disable_attrs(args)
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
    if not args.independent_attr_weights:
        raise SystemExit('--independent_attr_weights is required (the R50 judge picks the '
                         'faces and the edit direction, as in structure_vs_shortcut)')
    args.eval_scales = [args.edit_scale]
    torch.set_grad_enabled(False)

    prior, conditioner, G, id_criterion, attr_teacher, attribute_index, direction_bank, _ = \
        load_models(args)
    r50 = AttributeClassifier(backbone=args.independent_attr_backbone)
    r50.load_state_dict(load_network(args.independent_attr_weights))
    r50 = r50.cuda().eval()
    r18 = CelebAAttrClassifierJudge(args.celeba_attr_judge_weights)

    def score(judge, x):
        if judge == 'teacher':
            return torch.sigmoid(attr_teacher(x)[0])
        if judge == 'r50':
            return torch.sigmoid(r50(x)[0])
        return r18.scores(x)

    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)),
                    T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file,
                       train=False, transform=tf)
    if args.sample_offset:
        ds = data.Subset(ds, range(args.sample_offset, len(ds)))
    loader = data.DataLoader(ds, shuffle=False, batch_size=args.batch, num_workers=4)

    attrs = [g for g in NAMES if g in args.attribute_index]
    records = {g: [] for g in attrs}
    seen = 0
    for img, latent, _pred in tqdm(loader, desc='faces'):
        if seen >= args.num_samples:
            break
        img, latent = img.cuda(), latent.cuda()
        B = img.size(0)
        _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
        src = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
        src_p = {(j, m): score(j, downsample(src, m)) for j in JUDGES for m in MODES}

        for gidx in attrs:
            local = args.attribute_index.index(gidx)
            # faces and direction fixed by the R50 under the CURRENT (nearest) reading,
            # so every mode scores the same faces and the same edit
            s_ref = src_p[('r50', 'nearest')][:, gidx]
            keep = torch.tensor([is_clear(float(s)) for s in s_ref], device=img.device)
            if not keep.any():
                continue
            direction = torch.where(s_ref < 0.5, 1.0, -1.0).to(img.device)
            edited = edit_single_attribute(
                prior, conditioner, G, id_criterion, img, latent, attr_cond, id_cond,
                local, args.edit_scale, direction_bank, attr_global_idx=gidx,
                composite=False, direction=direction)
            edit_p = {(j, m): score(j, downsample(edited, m)) for j in JUDGES for m in MODES}
            for b in range(B):
                if bool(keep[b]):
                    records[gidx].append({j: {m: (float(src_p[(j, m)][b, gidx]),
                                                  float(edit_p[(j, m)][b, gidx]))
                                              for m in MODES} for j in JUDGES})
        seen += B

    report = {}
    print(f'\n{seen} faces, edit_scale {args.edit_scale:g}; strict success, same mode for '
          f'source and edit.\n')
    print(f'{"attribute":<11}{"judge":<9}{"n":>5}  ' + ''.join(f'{m:>13}' for m in MODES)
          + f'{"|dp|":>8}{"flip area":>11}{"flip bil":>10}')
    for g in attrs:
        if not records[g]:
            continue
        s = summarize(records[g], args.success_margin)
        report[NAMES[g]] = s
        for j in JUDGES:
            r = s[j]
            print(f'{NAMES[g]:<11}{j:<9}{r["n"]:>5}  '
                  + ''.join(f'{100 * r["success"][m]:12.1f}%' for m in MODES)
                  + f'{r["abs_dp"]:8.3f}{100 * r["flip_area"]:10.1f}%'
                    f'{100 * r["flip_bilinear_aa"]:9.1f}%')
        print()
    out = args.out_json or os.path.join(args.checkpoint_dir,
                                        f'resize_sensitivity_s{args.edit_scale:g}.json')
    with open(out, 'w') as f:
        json.dump({'edit_scale': args.edit_scale, 'n_faces': seen, 'step': args.step,
                   'checkpoint_dir': args.checkpoint_dir, 'results': report}, f, indent=2)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
