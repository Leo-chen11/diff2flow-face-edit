"""Do the edits the training teacher accepts but the judge rejects actually
change STRUCTURE, or only texture?

The failure tables say 60-75% of the failures in Male-rm, Bangs-add and
Young-add are "still": the JUDGE's probability barely moved. That is measured
on the judge, not on the image -- the r34 teacher scores those same edits
93-95% successful and their LPIPS is about half the inversion gap, so the
image IS changing. Two readings fit:

  shortcut   the edit changes texture (skin, colour) enough to move r34, but
             never grows the structure the attribute needs (hair over the
             forehead, a different jaw and hairline, smoothed wrinkles)
  judge gap  the structure does change and the judge simply does not agree

This script separates them. For each edit it splits the faces into

  both      the teacher and the judge both call the edit a success
  teacher   the teacher calls it a success, the judge does not   <- the question
  judge     the judge calls it a success, the teacher does not
  neither   both call it a failure
  pre       the teacher already reads the SOURCE on the target side (the
            judge does not, or the face would not be in the cell). The
            conditioner shares the teacher's reading, so without
            --src_cond_clamp the flow is asked for ~no change; and
            strict_success would score the teacher in the opposite direction.

and reports, per group, how much the STRUCTURE moved between the source
reconstruction and the edit, measured from BiSeNet's segmentation (no
landmarks, no extra weights):

  hair_frac      hair pixels / all pixels                    (Male, Bangs)
  forehead_hair  hair pixels inside the forehead band        (Bangs)
  jaw_width      width of the skin mask low on the face      (Male)
  skin_hf        Laplacian energy inside the skin mask       (Young: wrinkles)
  hair_sat       mean saturation of the hair region          (Young: the
                 "grey hair = old" shortcut the robust-teacher run showed)
  glasses_frac   glasses pixels / all pixels                 (Eyeglasses)

Eyeglasses-add is the CONTROL: that cell works (91-93%), so its structure
must move in the `both` group. If it does not, the measurement is at fault,
not the model.

Reading it: in the `teacher` group, structure change close to the `neither`
group means shortcut; close to the `both` group means the judge disagrees
about structure that is really there.

    python -m scripts.structure_vs_shortcut \
        --checkpoint_dir ./output/SDFlow/multi_v1_min_head_seed1 --step 80000 \
        --bank_mode prior --edit_scale 0.9 --num_samples 500 \
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
import torchvision.utils as vutils
from torch.utils import data
from tqdm import tqdm

from common.face_parser import FaceParser
from common.ops import load_network
from models.attribute_estimator import AttributeClassifier
from models.dataset import SDFlowDataset
from evaluation.evaluate_sdflow import (CelebAAttrClassifierJudge, _latest_step,
                                        apply_run_config, build_parser,
                                        edit_single_attribute, is_clear, load_models,
                                        resolve_controlnet_disable_attrs, strict_success)

# (absolute attribute index, direction, label). The three weak cells, plus
# Eyeglasses-add as the control that is known to work.
CELLS = [(5, 'add', 'Bangs add'),
         (20, 'rm', 'Male rm'),
         (39, 'add', 'Young add'),
         (15, 'add', 'Eyeglasses add (control)')]
# BiSeNet classes
SKIN, GLASSES, HAIR = 1, 6, 17
# brows, eyes AND glasses: adding glasses overwrites the eye labels, which
# would otherwise leave the forehead band undefined on exactly the control cell
EYE_LEVEL = [2, 3, 4, 5, 6]
MEASURES = ['hair_frac', 'forehead_hair', 'jaw_width', 'skin_hf', 'hair_sat', 'glasses_frac']
GROUPS = ['both', 'teacher', 'judge', 'neither', 'pre']
# Attributes CelebA ties to the edited ones (Bangs with not-Male, Eyeglasses
# with Male and not-Young). The dumps showed the teacher-only group full of
# adult men for Bangs-add and glasses for Male-rm / Young-add: if the judge
# inherited those label correlations, its success rate splits on them.
CO_ATTRS = [(20, 'Male'), (15, 'Eyeglasses'), (39, 'Young'), (35, 'Wearing_Hat')]


@torch.no_grad()
def segment(parser, x):
    """x: [-1,1] (B,3,H,W) -> (B,512,512) long, BiSeNet's 19-class argmax.

    FaceParser.get_region_mask runs the net once per class list; this runs it
    once and keeps the labels, so all six measures cost one pass."""
    inp = F.interpolate(x, 512, mode='bilinear', align_corners=False)
    inp = (inp * 0.5 + 0.5 - parser.mean) / parser.std
    return parser.net(inp).argmax(dim=1)


@torch.no_grad()
def measure(parser, x, anchors=None):
    """Structural measures of a [-1,1] batch, each (B,) on the cpu.

    anchors: (forehead_top, forehead_bottom, jaw_row) per face, from the
    SOURCE. The same rows are then read in the edit, so the measure answers
    "did hair appear in THIS band" rather than "in a band that moved with the
    edit". Without it a successful bangs edit hides the eye labels it is
    anchored on and the face drops out as NaN -- exactly the faces the
    diagnostic is about. Returns (measures, anchors)."""
    seg = segment(parser, x)                                  # (B,512,512)
    B, H, W = seg.shape
    hair = (seg == HAIR).float()
    skin = (seg == SKIN).float()
    face = (((seg >= 1) & (seg <= 14)) | (seg == HAIR)).float()   # head, no background
    out = {}
    out['hair_frac'] = hair.mean((1, 2))
    out['glasses_frac'] = (seg == GLASSES).float().mean((1, 2))

    # Forehead band: between the top of the head and the top of the brows/eyes,
    # per face, so it follows the crop rather than a fixed box.
    eyes = torch.isin(seg, torch.tensor(EYE_LEVEL, device=seg.device)).float()
    rows = torch.arange(H, device=seg.device).view(1, H, 1).expand(B, H, W)
    big = H * 10
    head_top = torch.where(face > 0, rows, torch.full_like(rows, big)).amin((1, 2))
    eye_top = torch.where(eyes > 0, rows, torch.full_like(rows, big)).amin((1, 2))
    fh, jw, new_anchors = [], [], []
    for b in range(B):
        if anchors is not None:
            t, e, y = anchors[b]
        else:
            t, e = int(head_top[b]), int(eye_top[b])
            ys = torch.nonzero(face[b].amax(1)).flatten()
            if ys.numel() >= 2:
                y0, y1 = int(ys[0]), int(ys[-1])
                y = min(H - 1, y0 + int(0.85 * (y1 - y0)))
            else:
                y = -1
            if e >= big or e <= t:
                t, e = -1, -1
        new_anchors.append((t, e, y))
        fh.append(float(hair[b, t:e].mean()) if 0 <= t < e else float('nan'))
        if y < 0:
            jw.append(float('nan'))
        else:
            xs = torch.nonzero(skin[b, y]).flatten()
            jw.append(float(xs[-1] - xs[0]) / W if xs.numel() >= 2 else float('nan'))
    out['forehead_hair'] = torch.tensor(fh)
    out['jaw_width'] = torch.tensor(jw)

    # Wrinkles: Laplacian energy of the grey image inside the skin mask.
    g = F.interpolate(x, 512, mode='bilinear', align_corners=False)
    g = ((g * 0.5 + 0.5) * torch.tensor([0.299, 0.587, 0.114],
                                        device=x.device).view(1, 3, 1, 1)).sum(1, keepdim=True)
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                     device=x.device).view(1, 1, 3, 3)
    lap = F.conv2d(g, k, padding=1).abs()[:, 0]
    out['skin_hf'] = (lap * skin).sum((1, 2)) / skin.sum((1, 2)).clamp(min=1)

    # Hair saturation: max(rgb) - min(rgb), the grey-hair shortcut signal.
    rgb = F.interpolate(x, 512, mode='bilinear', align_corners=False) * 0.5 + 0.5
    sat = rgb.amax(1) - rgb.amin(1)
    out['hair_sat'] = (sat * hair).sum((1, 2)) / hair.sum((1, 2)).clamp(min=1)

    return {k: v.float().cpu() for k, v in out.items()}, (anchors or new_anchors)


def summarize(rows):
    """rows: list of per-face dicts -> {measure: (mean |delta|, mean signed delta)}."""
    out = {}
    for m in MEASURES:
        v = np.array([r[m] for r in rows], dtype=float)
        v = v[~np.isnan(v)]
        out[m] = (float(np.abs(v).mean()), float(v.mean())) if v.size else (float('nan'),) * 2
    return out


def main():
    p = build_parser()
    p.add_argument('--edit_scale', type=float, default=0.9,
                   help='One fixed edit strength; the groups are about WHAT changed, not how much.')
    p.add_argument('--dump_dir', default=None,
                   help='Also save a source-vs-edit grid per cell and group, so the numbers can be '
                        'checked by eye: the teacher group changes structure as much as the both '
                        'group, so what the judge rejects has to be visible, not measurable.')
    p.add_argument('--dump_n', type=int, default=8, help='Faces per grid (--dump_dir).')
    p.add_argument('--judge', choices=['indep', 'celeb'], default='indep',
                   help="Which judge defines the groups: 'indep' = --independent_attr_weights "
                        "(r50), 'celeb' = --celeba_attr_judge_weights (the CelebA r18 the "
                        "headline AccBoth also uses).")
    # --out_json already exists in build_parser; reuse it (evaluate() is never called here)
    args = p.parse_args()
    # evaluate_sdflow does these in its own __main__, not in load_models():
    # without them the flow is built from the argparse defaults (3 attributes)
    # instead of the run's own config.json (5), and the checkpoint will not load.
    args = apply_run_config(args)
    args = resolve_controlnet_disable_attrs(args)
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
    if args.judge == 'indep' and not args.independent_attr_weights:
        raise SystemExit('--independent_attr_weights is required: the judge defines the groups')
    judge_path = (args.independent_attr_weights if args.judge == 'indep'
                  else args.celeba_attr_judge_weights)
    args.eval_scales = [args.edit_scale]

    # this branch's load_models also returns control_encoder (None unless the
    # run used ControlNet); edit_single_attribute is called without it here
    prior, conditioner, G, id_criterion, attr_teacher, attribute_index, direction_bank, _ = \
        load_models(args)
    if args.judge == 'indep':
        judge_net = AttributeClassifier(backbone=args.independent_attr_backbone)
        judge_net.load_state_dict(load_network(args.independent_attr_weights))
        judge_net = judge_net.cuda().eval()

        def judge(x):
            return torch.sigmoid(judge_net(x)[0])
    else:
        judge = CelebAAttrClassifierJudge(args.celeba_attr_judge_weights).scores
    parser = FaceParser(weights_path=args.face_parser_weights).cuda().eval()

    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)),
                    T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file,
                       train=False, transform=tf)
    if args.sample_offset:
        ds = data.Subset(ds, range(args.sample_offset, len(ds)))
    loader = data.DataLoader(ds, shuffle=False, batch_size=args.batch, num_workers=4)

    # edit_single_attribute is not decorated no_grad (evaluate() wraps its own
    # call site), and nothing here needs gradients. Without this the generator
    # graph of every edit stays alive -- harmless while each batch was thrown
    # away, an OOM once --dump_dir keeps tensors that reference it.
    torch.set_grad_enabled(False)

    cells = [(g, d, lab) for g, d, lab in CELLS if g in args.attribute_index]
    buckets = {lab: {gr: [] for gr in GROUPS} for _, _, lab in cells}
    shots = {lab: {gr: [] for gr in GROUPS} for _, _, lab in cells}   # --dump_dir
    strata = {lab: [] for _, _, lab in cells}   # (ok_t, ok_j, {co-attr: source has it})
    seen = 0
    for img, latent, _pred in tqdm(loader, desc='faces'):
        if seen >= args.num_samples:
            break
        img, latent = img.cuda(), latent.cuda()
        B = img.size(0)
        _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
        src = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
        src256 = F.interpolate(src, (256, 256))
        src_t = torch.sigmoid(attr_teacher(src256)[0])
        src_j = judge(src256)
        src_m, anchors = measure(parser, src256)

        for gidx, dirn, lab in cells:
            local = args.attribute_index.index(gidx)
            add = dirn == 'add'
            # Only faces the JUDGE reads clearly on the side the edit starts
            # from -- the same subset AccInd is scored on.
            keep = torch.tensor([is_clear(float(s)) and (float(s) < 0.5) == add
                                 for s in src_j[:, gidx]], device=img.device)
            if not keep.any():
                continue
            direction = torch.full((B,), 1.0 if add else -1.0, device=img.device)
            edited = edit_single_attribute(
                prior, conditioner, G, id_criterion, img, latent, attr_cond, id_cond,
                local, args.edit_scale, direction_bank, attr_global_idx=gidx,
                composite=False, direction=direction)
            e256 = F.interpolate(edited, (256, 256))
            e_t = torch.sigmoid(attr_teacher(e256)[0])[:, gidx]
            e_j = judge(e256)[:, gidx]
            e_m, _ = measure(parser, e256, anchors)

            for b in range(B):
                if not bool(keep[b]):
                    continue
                pre = (float(src_t[b, gidx]) > 0.5) == add
                ok_t = strict_success(float(src_t[b, gidx]), float(e_t[b]), args.success_margin)
                ok_j = strict_success(float(src_j[b, gidx]), float(e_j[b]), args.success_margin)
                if pre:
                    grp = 'pre'
                else:
                    grp = ('both' if ok_j else 'teacher') if ok_t else ('judge' if ok_j else 'neither')
                buckets[lab][grp].append({m: float(e_m[m][b] - src_m[m][b]) for m in MEASURES})
                strata[lab].append((ok_t and not pre, ok_j, pre, {name: bool(src_j[b, k] > 0.5)
                                                 for k, name in CO_ATTRS if k != gidx}))
                if args.dump_dir and len(shots[lab][grp]) < args.dump_n:
                    shots[lab][grp].append((src256[b].detach().cpu(),
                                            e256[b].detach().cpu(),
                                            float(src_j[b, gidx]), float(e_j[b])))
        seen += B

    report = {}
    print(f'\n{seen} faces, edit_scale {args.edit_scale:g}, judge '
          f'{os.path.basename(judge_path)}, src_cond_clamp {args.src_cond_clamp}')
    print('Structure change between the source reconstruction and the edit.\n'
          'both = teacher and judge agree it worked; teacher = only the teacher; '
          'judge = only the judge; neither = both say it failed; pre = the teacher already\n'
          'reads the source as edited (a no-op unless --src_cond_clamp).\nEach cell: mean |delta| (mean signed delta).')
    for _, _, lab in cells:
        print(f'\n{lab}')
        print(f'  {"group":<8} {"n":>5} ' + ' '.join(f'{m:>20}' for m in MEASURES))
        report[lab] = {}
        for gr in GROUPS:
            rows = buckets[lab][gr]
            if not rows:
                print(f'  {gr:<8} {0:>5}')
                continue
            s = summarize(rows)
            report[lab][gr] = {'n': len(rows),
                               **{m: {'abs': s[m][0], 'signed': s[m][1]} for m in MEASURES}}
            print(f'  {gr:<8} {len(rows):>5} '
                  + ' '.join(f'{s[m][0]:9.4f} ({s[m][1]:+.4f})' for m in MEASURES))

    print('\nSuccess rate split by an attribute of the SOURCE (as the judge reads it).\n'
          'A judge that inherited a CelebA correlation shows a gap in judge%, not in teacher%.\n'
          'pre% = sources the teacher (and the conditioner) already read on the target side.')
    for _, _, lab in cells:
        print(f'\n{lab}')
        print(f'  {"source":<16} {"n":>5} {"teacher%":>9} {"judge%":>8} {"pre%":>7}')
        report[lab]['strata'] = {}
        for _, name in CO_ATTRS:
            for has in (True, False):
                rows = [r for r in strata[lab] if r[3].get(name) is has]
                if name not in (strata[lab][0][3] if strata[lab] else {}) or not rows:
                    continue
                t = 100 * np.mean([r[0] for r in rows])
                j = 100 * np.mean([r[1] for r in rows])
                pr = 100 * np.mean([r[2] for r in rows])
                key = f'{name}={int(has)}'
                report[lab]['strata'][key] = {'n': len(rows), 'teacher': t, 'judge': j, 'pre': pr}
                print(f'  {key:<16} {len(rows):>5} {t:8.1f}% {j:7.1f}% {pr:6.1f}%')

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        for _, _, lab in cells:
            for gr in GROUPS:
                rows = shots[lab][gr]
                if not rows:
                    continue
                # one column per face: source reconstruction on top, edit below
                grid = vutils.make_grid(torch.stack([r[0] for r in rows] + [r[1] for r in rows]),
                                        nrow=len(rows), normalize=True, value_range=(-1, 1))
                name = f'{lab.split(" (")[0].replace(" ", "_")}__{gr}.png'
                vutils.save_image(grid, os.path.join(args.dump_dir, name))
                probs = ' '.join(f'{r[2]:.2f}->{r[3]:.2f}' for r in rows)
                print(f'  [dump] {name}  judge prob per column: {probs}')
        print(f'  [dump] top row = source reconstruction, bottom row = edit; saved to {args.dump_dir}')

    tag = f'{args.judge}_s{args.edit_scale:g}' + (f'_clamp{args.src_cond_clamp:g}'
                                                  if args.src_cond_clamp is not None else '')
    out = args.out_json or os.path.join(args.checkpoint_dir, f'structure_vs_shortcut_{tag}.json')
    with open(out, 'w') as f:
        json.dump({'edit_scale': args.edit_scale, 'n_faces': seen,
                   'src_cond_clamp': args.src_cond_clamp,
                   'checkpoint_dir': args.checkpoint_dir, 'step': args.step,
                   'judge': judge_path, 'cells': report}, f, indent=2)
    print(f'\nsaved {out}')


if __name__ == '__main__':
    main()
