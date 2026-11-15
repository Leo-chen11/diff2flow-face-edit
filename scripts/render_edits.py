"""Render a checkpoint's edits as image grids, edited exactly as scripts/editing_curves.py scores them
(direction from the R50 judge's reading of the source, optional --consistent_reading), with every
cell labelled by what the judges say about it.

Modes:
  grid     one row per face: the source reconstruction, then each attribute's edit at its strength
           (--strength for all, --strengths 20:0.8,31:0.6 per attribute)
  sweep    one attribute (--attr), one row per face, one column per strength (--sweep)
  reading  one attribute (--attr), only faces whose conditioner reading is on the wrong side for the
           requested direction: source | edit with the conditioner's reading | edit with the
           direction-consistent reading (--consistent_reading, default 0.1 in this mode)

Which faces (--pick):
  first     the first R50-clear faces for the attribute(s)               (default)
  judges    (--attr) faces whose edit R50 accepts and the CelebA r18 judge rejects -- what the
            R50 / r18 accuracy gap is made of
  fail      (--attr) faces whose edit R50 rejects
Faces come from the test split from --sample_offset on (0: the development faces).

Labels, per edited cell: requested direction (+ add / - remove), R50 probability of the attribute
source -> edit with a mark (Y: crossed 0.5 the requested way), the same for r18 when its weights are
there, and ArcFace (edit vs source reconstruction) when ./data/model_ir_se50.pth is there.

    python -m scripts.render_edits --checkpoint_dir output/SDFlow/<run> --step 130000 \\
        --independent_attr_weights ./data/r50_celebahq_eval.pth --independent_attr_backbone r50 \\
        --consistent_reading 0.1 --strengths 15:0.96,20:0.79,39:0.8,31:0.59,5:0.94 --num_faces 8
    python -m scripts.render_edits ... --mode sweep --attr 20 --sweep 0.4 0.6 0.8 1.0 1.25
    python -m scripts.render_edits ... --mode reading --attr 39
    python -m scripts.render_edits ... --mode grid --attr 20 --pick judges
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

CLEAR_LO, CLEAR_HI = 0.35, 0.65


def parse_strengths(spec, default):
    out = {}
    for part in (spec or '').split(','):
        part = part.strip()
        if part:
            a, v = part.split(':')
            out[int(a)] = float(v)
    return lambda a: out.get(int(a), default)


def label_cells(img, cells, labels, cell):
    """Draw each label at the top-left of its cell. img: PIL image; cells: [(row, col)]."""
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for (r, c), text in zip(cells, labels):
        if not text:
            continue
        x, y = c * cell + 3, r * cell + 3
        lines = text.split('\n')
        w = max(draw.textlength(t, font=font) for t in lines) + 6
        h = 12 * len(lines) + 4
        draw.rectangle([x - 2, y - 2, x + w, y + h], fill=(0, 0, 0))
        for k, t in enumerate(lines):
            draw.text((x + 1, y + 12 * k), t, fill=(255, 255, 255), font=font)
    return img


def main():
    import torch
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as T
    from torch.utils import data

    import evaluation.evaluate_sdflow as E
    from evaluation.evaluate_sdflow import (ATTR_NAMES, _latest_step, apply_run_config, build_optional_judges,
                                            build_parser, edit_single_attribute, load_models, resize_for_judge)
    from models.dataset import SDFlowDataset

    p = build_parser()
    p.add_argument('--mode', default='grid', choices=['grid', 'sweep', 'reading'])
    p.add_argument('--attr', type=int, default=None, help='sweep / reading / --pick judges|fail: the CelebA id')
    p.add_argument('--only_attrs', nargs='+', type=int, default=None, help='grid: these attributes (default all)')
    p.add_argument('--strength', type=float, default=1.0, help='edit strength for every attribute')
    p.add_argument('--strengths', default=None, help="per attribute, e.g. '15:0.96,20:0.79' (others: --strength)")
    p.add_argument('--sweep', nargs='+', type=float, default=[0.4, 0.6, 0.8, 1.0, 1.25])
    p.add_argument('--consistent_reading', type=float, default=None, metavar='MARGIN',
                   help='as scripts/editing_curves.py; --mode reading uses 0.1 when not given')
    p.add_argument('--pick', default='first', choices=['first', 'judges', 'fail'])
    p.add_argument('--num_faces', type=int, default=8)
    p.add_argument('--scan', type=int, default=500, help='at most this many test faces are looked at')
    p.add_argument('--cell_size', type=int, default=256)
    p.add_argument('--arcface_weights', default='./data/model_ir_se50.pth')
    p.add_argument('--out', default=None)
    p.set_defaults(eval_resize='area', batch=4)
    args = p.parse_args()
    args = apply_run_config(args)
    E._RESIZE_MODE = args.eval_resize
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
    if not args.independent_attr_weights:
        raise SystemExit('--independent_attr_weights is required (R50 judge).')
    if (args.mode != 'grid' or args.pick != 'first') and args.attr is None:
        raise SystemExit(f'--mode {args.mode} --pick {args.pick} needs --attr')
    if args.attr is not None and args.attr not in args.attribute_index:
        raise SystemExit(f'--attr {args.attr} is not an edited attribute of this run ({args.attribute_index})')
    margin = args.consistent_reading
    if args.mode == 'reading' and margin is None:
        margin = 0.1
    strength_of = parse_strengths(args.strengths, args.strength)

    prior, conditioner, G, id_criterion, _, _, direction_bank = load_models(args)
    _, _, _, r50, _, celeb = build_optional_judges(args, args.attribute_index, id_criterion)
    arc = None
    if os.path.exists(args.arcface_weights):
        from models.arcface import ArcFaceJudge
        arc = ArcFaceJudge(args.arcface_weights)

    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)), T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root, latents_file=args.latent_file,
                       preds_file=args.preds_file, train=False, transform=tf)
    eval_ds = data.Subset(ds, range(args.sample_offset, len(ds))) if args.sample_offset else ds
    loader = data.DataLoader(eval_ds, shuffle=False, batch_size=args.batch, num_workers=2)

    if args.mode == 'grid':
        attrs = [args.attr] if (args.pick != 'first' and args.only_attrs is None) else \
            list(args.only_attrs or args.attribute_index)
        columns = [(g, strength_of(g), margin) for g in attrs]
    elif args.mode == 'sweep':
        columns = [(args.attr, s, margin) for s in args.sweep]
    else:
        columns = [(args.attr, strength_of(args.attr), None), (args.attr, strength_of(args.attr), margin)]
    key_attr = args.attr if args.attr is not None else None

    def edit(img, latent, attr_cond, id_cond, g, s, m, d):
        li = args.attribute_index.index(g)
        sv = None
        if m is not None:
            sv = torch.where(d > 0, attr_cond[:, li].clamp(max=0.5 - m), attr_cond[:, li].clamp(min=0.5 + m))
        return edit_single_attribute(prior, conditioner, G, id_criterion, img, latent, attr_cond, id_cond, li, s,
                                     direction_bank, attr_global_idx=g,
                                     bypass_glasses_direction_bank=args.bypass_glasses_direction_bank,
                                     composite=False, direction=d, src_value=sv, src_value_in_encoding=sv is not None)

    def judge(x256, g):
        out = {'r50': float(torch.sigmoid(r50(x256)[0])[0, g])}
        if celeb is not None:
            out['r18'] = float(celeb.scores(x256)[0, g])
        return out

    rows, labels, cells, seen, kept = [], [], [], 0, []
    with torch.no_grad():
        for img_b, lat_b, _ in loader:
            for b in range(img_b.size(0)):
                if len(rows) >= args.num_faces or seen >= args.scan:
                    break
                face_id = args.sample_offset + seen
                seen += 1
                img, latent = img_b[b:b + 1].cuda(), lat_b[b:b + 1].cuda()
                _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
                src = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
                s256 = resize_for_judge(src, (256, 256))
                sp = torch.sigmoid(r50(s256)[0])[0, :40]
                need = [key_attr] if key_attr is not None else [c[0] for c in columns]
                if any(CLEAR_LO <= float(sp[g]) <= CLEAR_HI for g in need):
                    continue                                   # as the curves: R50-clear sources only
                if args.mode == 'reading':
                    li = args.attribute_index.index(args.attr)
                    asked_add = float(sp[args.attr]) < 0.5
                    c_read = float(attr_cond[0, li])
                    if (asked_add and c_read <= 0.5) or (not asked_add and c_read >= 0.5):
                        continue                               # the conditioner reads it the requested way
                src_j = {g: judge(s256, g) for g in set(c[0] for c in columns)}
                row_imgs = [F.interpolate(src, (args.cell_size, args.cell_size), mode='area')]
                row_labels = [f'face {face_id}']
                s_arc = arc.extract(s256) if arc is not None else None
                verdicts = []
                for g, s, m in columns:
                    d = torch.tensor([-1.0 if float(sp[g]) > 0.5 else 1.0], device=src.device)
                    face = edit(img, latent, attr_cond, id_cond, g, s, m, d)
                    f256 = resize_for_judge(face, (256, 256))
                    ej = judge(f256, g)
                    sign = '+' if d.item() > 0 else '-'
                    okf = lambda before, after: (after > 0.5) if d.item() > 0 else (after < 0.5)
                    ok50 = okf(src_j[g]['r50'], ej['r50'])
                    text = f'{ATTR_NAMES.get(g, g)} {sign} s{s:g}' + (f' cr{m:g}' if m is not None else '')
                    text += f"\nR50 {src_j[g]['r50']:.2f}>{ej['r50']:.2f} {'Y' if ok50 else 'N'}"
                    ok18 = None
                    if 'r18' in ej:
                        ok18 = okf(src_j[g]['r18'], ej['r18'])
                        text += f"\nr18 {src_j[g]['r18']:.2f}>{ej['r18']:.2f} {'Y' if ok18 else 'N'}"
                    if s_arc is not None:
                        text += f"\nArc {float((s_arc * arc.extract(f256)).sum(1)):.3f}"
                    verdicts.append((g, ok50, ok18))
                    row_imgs.append(F.interpolate(face, (args.cell_size, args.cell_size), mode='area'))
                    row_labels.append(text)
                if args.pick != 'first':
                    v = [x for x in verdicts if x[0] == args.attr][-1]
                    if args.pick == 'judges' and not (v[1] and v[2] is False):
                        continue
                    if args.pick == 'fail' and v[1]:
                        continue
                r = len(rows)
                rows.append(torch.cat(row_imgs, dim=3))
                for c, t in enumerate(row_labels):
                    cells.append((r, c))
                    labels.append(t)
                kept.append(face_id)
            if len(rows) >= args.num_faces or seen >= args.scan:
                break

    if not rows:
        raise SystemExit(f'no face matched in the first {seen} test faces (--pick {args.pick}); raise --scan')
    grid = (torch.cat(rows, dim=2)[0].clamp(-1, 1) + 1) * 0.5
    from torchvision.transforms.functional import to_pil_image
    img = label_cells(to_pil_image(grid.cpu()), cells, labels, args.cell_size)
    out = args.out or os.path.join(
        args.checkpoint_dir, 'renders',
        f'{args.mode}_s{args.step}_off{args.sample_offset}_{args.pick}'
        + (f'_attr{args.attr}' if args.attr is not None else '')
        + (f'_cr{margin:g}' if margin is not None else '') + '.png')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    img.save(out)
    print(f'{len(rows)} faces {kept} (looked at {seen}); saved {out}')


if __name__ == '__main__':
    main()
