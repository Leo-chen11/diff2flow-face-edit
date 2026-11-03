"""The identity ceiling the e4e inversion puts on every edit.

Our identity numbers compare an edit with the RECONSTRUCTION G(e4e(photo)), so the
inversion error does not enter them. This script measures that error itself: the
identity similarity and LPIPS between each real photo and its unedited reconstruction
(edit strength 0), on the test faces the other evaluations use. An editor that
changes nothing at all scores this much against the real photo, so no method
editing in e4e's latent space can score higher there, however it is trained.

Reported per face and summarised (mean, p10, p50, p90):
  ArcFace   IR-SE50 cosine, the SDFlow paper's identity metric
  facenet   the independent identity model the evaluations use
  LPIPS     AlexNet perceptual distance
and, with --edited_csv, the same real-photo similarity of the edits themselves is
NOT measured (that needs the edited images); the ceiling is compared instead with
the edit-vs-reconstruction values in a curves CSV of scripts/editing_curves.py, so
"how much of the remaining loss is the inversion's and how much the edit's" reads
off one table:  real->edit identity ~ real->recon identity x recon->edit identity
(a first-order estimate, labelled as such).

    python -m scripts.inversion_ceiling --checkpoint_dir output/SDFlow/<run> --step 100000 \\
        --independent_attr_weights ./data/r50_celebahq_eval.pth --independent_attr_backbone r50 \\
        --num_samples 500 [--edited_csv output/curves3/ours.csv --at_accuracy 80]
"""
import csv
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils import data

import evaluation.evaluate_sdflow as E
from evaluation.evaluate_sdflow import (
    _latest_step, apply_run_config, build_optional_judges, build_parser, load_models, resize_for_judge,
)
from models.arcface import ArcFaceJudge
from models.dataset import SDFlowDataset


def summarize(v):
    v = np.asarray(v, dtype=float)
    return {'mean': float(v.mean()), 'p10': float(np.percentile(v, 10)), 'p50': float(np.percentile(v, 50)),
            'p90': float(np.percentile(v, 90)), 'n': int(len(v))}


def main():
    p = build_parser()
    p.add_argument('--arcface_weights', default='./data/model_ir_se50.pth')
    p.add_argument('--edited_csv', default=None,
                   help='A scripts/editing_curves.py CSV (id_arc = edit vs reconstruction): adds a '
                        'first-order real->edit estimate at --at_accuracy.')
    p.add_argument('--at_accuracy', type=float, default=80.0)
    p.add_argument('--out_json', default=None)
    p.set_defaults(eval_resize='area')
    args = p.parse_args()
    args = apply_run_config(args)
    E._RESIZE_MODE = args.eval_resize
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)

    _, _, G, id_criterion, _, _, _ = load_models(args)
    _, indep_id, lpips_fn, _, _, _ = build_optional_judges(args, args.attribute_index, id_criterion)
    arc = ArcFaceJudge(args.arcface_weights)
    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)), T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file,
                       train=False, transform=tf)
    loader = data.DataLoader(ds, shuffle=False, batch_size=args.batch, num_workers=4)

    res = {'arcface': [], 'facenet': [], 'lpips': []}
    seen = 0
    with torch.no_grad():
        for img, latent, _ in loader:
            if seen >= args.num_samples:
                break
            img, latent = img.cuda(), latent.cuda()
            recon = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
            r256, s256 = resize_for_judge(recon, (256, 256)), resize_for_judge(img, (256, 256))
            res['arcface'] += (arc.extract(s256) * arc.extract(r256)).sum(1).cpu().tolist()
            if indep_id is not None:
                res['facenet'] += (indep_id.extract(s256) * indep_id.extract(r256)).sum(1).cpu().tolist()
            if lpips_fn is not None:
                res['lpips'] += lpips_fn(s256, r256).flatten().cpu().tolist()
            seen += img.shape[0]

    out = {k: summarize(v) for k, v in res.items() if v}
    print(f'\nreal photo vs its e4e reconstruction (edit strength 0), {seen} test faces')
    print(f'  {"metric":<10}{"mean":>8}{"p10":>8}{"p50":>8}{"p90":>8}')
    for k, s in out.items():
        print(f'  {k:<10}{s["mean"]:8.3f}{s["p10"]:8.3f}{s["p50"]:8.3f}{s["p90"]:8.3f}')
    print('  An editor that changes nothing scores this much against the real photo; identity '
          'similarity there cannot exceed it for any method editing in e4e\'s W+.')

    if args.edited_csv:
        rows = [r for r in csv.DictReader(open(args.edited_csv)) if r['attribute'] != 'All']
        ceiling = out['arcface']['mean']
        print(f'\nfirst-order real->edit identity at editing accuracy {args.at_accuracy:g}% '
              f'(= {ceiling:.3f} x edit->reconstruction ArcFace):')
        for attr in sorted({r['attribute'] for r in rows}):
            ar = sorted((float(r['target_success']) * 100, float(r['id_arc'])) for r in rows
                        if r['attribute'] == attr and r['id_arc'])
            x = np.array([t[0] for t in ar])
            if not (x.min() <= args.at_accuracy <= x.max()):
                print(f'  {attr:<11} the curve does not reach {args.at_accuracy:g}%')
                continue
            idv = float(np.interp(args.at_accuracy, x, [t[1] for t in ar]))
            print(f'  {attr:<11} edit->recon {idv:.3f}   real->edit ~ {ceiling * idv:.3f}')
        out['first_order_note'] = 'real->edit ~ real->recon x recon->edit'

    path = args.out_json or os.path.join(args.checkpoint_dir, f'inversion_ceiling_s{args.step}.json')
    with open(path, 'w') as f:
        json.dump({'num_faces': seen, 'real_vs_reconstruction': out}, f, indent=2)
    print(f'\nsaved {path}')


if __name__ == '__main__':
    main()
