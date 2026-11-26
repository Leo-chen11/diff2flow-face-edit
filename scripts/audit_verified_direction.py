"""Check the verified-correction rule (common/verified_direction.py, train_sdflow.py
--train_edit_direction verified) on the TRAINING faces before trusting it, and measure how often it
overrules a conditioner that was right.

There is no ground truth for "young" or "male" in FFHQ, so whether a correction is right cannot be
computed; it has to be looked at. This script applies the exact function training uses, with the
conditioner of a given checkpoint (the stage-2 starting point, or a finished run), to every edit
training could draw (each face x each attribute), and writes:

  <out>/decisions.csv      one row per (face, attribute): the conditioner, every view, the decision
  <out>/summary.md         per attribute: how many edits are followed / corrected / abstain
  <out>/sheets/<attr>_<correct|abstain>_<k>.png
                           the faces themselves (photo | e4e reconstruction), numbered, with what the
                           conditioner and the views read
  <out>/labels.csv         one row per face on the sheets, column `human` empty: fill in
                              M  the conditioner was right (the correction / the views were wrong)
                              V  the views were right (the conditioner misread the face)
                              ?  cannot tell
then
  python -m scripts.audit_verified_direction --score <out>/labels.csv
prints, per attribute, the share of corrections that overruled a right conditioner (M among M + V)
with its 95% Wilson interval. That number is the rule's false-correction rate; quote it next to the
ablation. A rule whose rate is not clearly below the share of edits it fixes does more harm than good.

    python -m scripts.audit_verified_direction --checkpoint_dir <run> --step 50000 --num_faces 3000
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

NAMES = {15: 'Eyeglasses', 20: 'Male', 39: 'Young', 31: 'Smiling', 5: 'Bangs'}


def wilson(k, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def score(path):
    rows = list(csv.DictReader(open(path)))
    by = defaultdict(lambda: defaultdict(int))
    for r in rows:
        h = (r.get('human') or '').strip().upper()
        if h in ('M', 'V', '?'):
            by[(r['attribute'], r['decision'])][h] += 1
    if not by:
        raise SystemExit(f'no labelled row in {path} (column human: M / V / ?)')
    print(f'{"attribute":<11} {"decision":<8} {"M":>4} {"V":>4} {"?":>4}   conditioner right among M+V (95% CI)')
    for (a, dec), c in sorted(by.items()):
        p, lo, hi = wilson(c['M'], c['M'] + c['V'])
        print(f'{a:<11} {dec:<8} {c["M"]:4d} {c["V"]:4d} {c["?"]:4d}   {p * 100:5.1f}%  [{lo * 100:4.1f}, {hi * 100:4.1f}]')
    print('\ncorrect rows: that share is the false-correction rate (lower is better).\n'
          'abstain rows: who was right where the rule refused to decide (no target is right there).')


def analyze(path, clear=0.65):
    """Why edits abstain or are corrected: per attribute, how often the conditioner and the precomputed r34
    disagree on the side of 0.5, how many of the views clearly contradict the conditioner, and which."""
    rows = list(csv.DictReader(open(path)))
    vcols = ['precomputed_r34'] + [k for k in rows[0] if k.startswith('t') and '_' in k]
    by = defaultdict(list)
    for r in rows:
        by[r['attribute']].append(r)
    print(f'views: {vcols}\n')
    for a, rs in by.items():
        n = len(rs)
        side = sum((float(r['conditioner']) > 0.5) != (float(r['precomputed_r34']) > 0.5) for r in rs)
        k_hist = defaultdict(int)
        per_view = defaultdict(int)
        for r in rs:
            c = float(r['conditioner'])
            against = [v for v in vcols if (float(r[v]) < 1 - clear) if c > 0.5] if c > 0.5 else \
                      [v for v in vcols if float(r[v]) > clear]
            k_hist[len(against)] += 1
            if r['decision'] == 'abstain':
                for v in against:
                    per_view[v] += 1
        n_ab = sum(r['decision'] == 'abstain' for r in rs)
        print(f'{a}: conditioner vs precomputed r34 on opposite sides of 0.5: {side}/{n} ({side / n * 100:.1f}%)')
        # per view: a conditioner that reads the reconstruction (--conditioner_input recon) should agree
        # with t*_recon about as well as a photo-reading one agrees with t*_photo
        print('   opposite sides of 0.5, conditioner vs each view: ' + ', '.join(
            f'{v} {sum((float(r["conditioner"]) > 0.5) != (float(r[v]) > 0.5) for r in rs) / n * 100:.1f}%'
            for v in vcols))
        print('   views clearly against the conditioner (count: faces): '
              + ', '.join(f'{k}: {k_hist[k]}' for k in sorted(k_hist)))
        if n_ab:
            print(f'   among the {n_ab} abstained, how often each view was one of those against: '
                  + ', '.join(f'{v} {per_view[v] / n_ab * 100:.0f}%' for v in vcols))
        print()


def main():
    if '--score' in sys.argv:
        i = sys.argv.index('--score')
        return score(sys.argv[i + 1])
    if '--analyze' in sys.argv:
        i = sys.argv.index('--analyze')
        return analyze(sys.argv[i + 1])

    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import Image, ImageDraw
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from common.ops import load_network
    from common.verified_direction import verified_direction
    from evaluation.evaluate_sdflow import apply_run_config, build_parser, load_models
    from models.attribute_estimator import make_attr_classifier
    from models.dataset import SDFlowDataset

    p = build_parser()
    p.add_argument('--num_faces', type=int, default=3000, help='training faces to scan')
    p.add_argument('--main_teacher_only', action='store_true',
                   help='read the views with the run\'s main teacher (r34) only, not its extra teacher: the '
                        'mechanism check (conditioner vs t0_photo / t0_recon) needs only that one; the '
                        'decisions then use fewer views than training did')
    p.add_argument('--margin', type=float, default=0.1, help='--train_consistent_reading')
    p.add_argument('--clear', type=float, default=0.65, help='--verify_clear')
    p.add_argument('--beat', type=float, default=0.0, help='--verify_beat')
    p.add_argument('--views', nargs='+', default=['photo', 'flip', 'recon'], choices=['photo', 'flip', 'recon'])
    p.add_argument('--sheet_faces', type=int, default=40, help='faces per attribute and decision on the sheets')
    p.add_argument('--out_dir', default='./output/audit_verified')
    p.add_argument('--score', default=None)
    p.add_argument('--analyze', default=None, help='a decisions.csv: why edits abstain (no model needed)')
    p.add_argument('--split', default='train', choices=['train', 'test'],
                   help='train (what training sees) or test (the first --num_faces test faces, i.e. the '
                        'development faces for --num_faces <= 500): the conditioner was fitted to the '
                        'training faces, so its misread rate there can understate the one at evaluation')
    args = apply_run_config(p.parse_args())
    cfg = json.load(open(os.path.join(args.checkpoint_dir, 'config.json')))
    os.makedirs(os.path.join(args.out_dir, 'sheets'), exist_ok=True)

    _, conditioner, G, id_criterion, _, _, _ = load_models(args)
    teachers = []
    for bb, w in [(cfg.get('attribute_backbone', 'r34'), cfg.get('attribute_weights')),
                  (cfg.get('extra_teacher_backbone'), None if args.main_teacher_only else cfg.get('extra_teacher_weights'))]:
        if not w or w == 'none':
            continue
        t = make_attr_classifier(bb)
        t.load_state_dict(load_network(w))
        teachers.append(t.cuda().eval())
    print(f'{len(teachers)} teacher(s): {cfg.get("attribute_weights")}, {cfg.get("extra_teacher_weights")}')

    img_size = cfg.get('img_size', 512)
    tf = T.Compose([T.ToTensor(), T.Resize((img_size, img_size)), T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=cfg.get('index_file', './data/ffhq.txt'), image_root=cfg.get('image_root', 'data/FFHQ'),
                       latents_file=cfg.get('latent_file', './data/ffhq_e4e_latents.pth'),
                       preds_file=cfg.get('preds_file', './data/ffhq_e4e_preds.pth'), train=args.split == 'train',
                       transform=tf)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    attrs = list(args.attribute_index)
    resize = cfg.get('train_resize', 'nearest')

    rows, tiles = [], defaultdict(list)          # tiles[(attr, decision)] -> [(row index, photo, recon)]
    seen = 0
    with torch.no_grad():
        for img, latent, pred in tqdm(loader, desc='audit'):
            if seen >= args.num_faces:
                break
            img, latent = img.cuda(), latent.cuda()
            pred = pred.float().reshape(pred.size(0), -1)[:, :40].cuda()
            if pred.min() < 0 or pred.max() > 1:
                pred = torch.sigmoid(pred)
            _, _, attr_cond = conditioner.make_condition(img, latent, id_criterion)
            rec = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
            rec = F.interpolate(rec, (img_size, img_size)) if resize == 'nearest' else \
                F.interpolate(rec, (img_size, img_size), mode=resize)
            ph = F.interpolate(img, (256, 256), mode='bilinear', align_corners=False)
            rc = F.interpolate(rec, (256, 256), mode='bilinear', align_corners=False)
            views = {'photo': ph, 'flip': torch.flip(ph, dims=[3]), 'recon': rc}
            probs = {(k, v): torch.sigmoid(t(views[v])[0][:, :40])
                     for k, t in enumerate(teachers) for v in args.views}
            for li, g in enumerate(attrs):
                reads = [pred[:, g]] + [probs[key][:, g] for key in sorted(probs)]
                c = attr_cond[:, li].float()
                o = verified_direction(c, torch.stack(reads).float(), args.margin, args.clear, args.beat)
                for b in range(img.size(0)):
                    dec = 'correct' if o['correct'][b] else ('abstain' if o['abstain'][b] else 'follow')
                    r = {'face': seen + b, 'attribute': NAMES.get(g, str(g)), 'decision': dec,
                         'conditioner': round(float(c[b]), 3), 'precomputed_r34': round(float(reads[0][b]), 3)}
                    for j, key in enumerate(sorted(probs)):
                        r[f't{key[0]}_{key[1]}'] = round(float(reads[1 + j][b]), 3)
                    rows.append(r)
                    if dec != 'follow' and len(tiles[(r['attribute'], dec)]) < args.sheet_faces:
                        tiles[(r['attribute'], dec)].append((len(rows) - 1, ph[b].cpu(), rc[b].cpu()))
            seen += img.size(0)

    with open(os.path.join(args.out_dir, 'decisions.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # summary
    cnt = defaultdict(lambda: defaultdict(int))
    for r in rows:
        cnt[r['attribute']][r['decision']] += 1
    lines = [f'# Verified-correction audit: {args.checkpoint_dir} step {args.step}, {seen} {args.split} faces',
             f'margin {args.margin}, clear {args.clear}, beat {args.beat}, views precomputed r34 + '
             f'{len(teachers)} teacher(s) x {args.views}', '',
             '| attribute | follow | correct | abstain |', '|---|---|---|---|']
    for a in [NAMES.get(g, str(g)) for g in attrs]:
        n = sum(cnt[a].values())
        lines.append(f'| {a} | ' + ' | '.join(f'{cnt[a][d]} ({cnt[a][d] / n * 100:.1f}%)'
                                              for d in ('follow', 'correct', 'abstain')) + ' |')
    open(os.path.join(args.out_dir, 'summary.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    # sheets + the labelling template
    to_pil = T.ToPILImage()
    lab = []
    for (a, dec), items in tiles.items():
        cols, tw, th = 4, 512, 280
        for page in range(0, len(items), 20):
            chunk = items[page:page + 20]
            sheet = Image.new('RGB', (cols * tw, ((len(chunk) + cols - 1) // cols) * th), 'white')
            draw = ImageDraw.Draw(sheet)
            for k, (ri, phb, rcb) in enumerate(chunk):
                x, y = (k % cols) * tw, (k // cols) * th
                sheet.paste(to_pil((phb + 1) / 2), (x, y))
                sheet.paste(to_pil((rcb + 1) / 2), (x + 256, y))
                r = rows[ri]
                tv = [v for kk, v in r.items() if kk.startswith('t')]
                num = page + k + 1
                draw.text((x + 4, y + 258), f'#{num} face {r["face"]}  model {r["conditioner"]:.2f}  '
                          f'views {min(tv + [r["precomputed_r34"]]):.2f}-{max(tv + [r["precomputed_r34"]]):.2f}',
                          fill='black')
                lab.append({'sheet': f'{a}_{dec}_{page // 20 + 1}.png', 'number': num, 'face': r['face'],
                            'attribute': a, 'decision': dec, 'conditioner': r['conditioner'],
                            'views_min': min(tv + [r['precomputed_r34']]),
                            'views_max': max(tv + [r['precomputed_r34']]), 'human': ''})
            sheet.save(os.path.join(args.out_dir, 'sheets', f'{a}_{dec}_{page // 20 + 1}.png'))
    if lab:
        with open(os.path.join(args.out_dir, 'labels.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(lab[0].keys()))
            w.writeheader()
            w.writerows(lab)
    print(f'\nwrote {args.out_dir}/decisions.csv, summary.md, sheets/ and labels.csv '
          f'(left: photo, right: e4e reconstruction; fill the human column with M / V / ?)')


if __name__ == '__main__':
    main()
