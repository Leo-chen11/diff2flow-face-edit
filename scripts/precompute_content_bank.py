"""Build the reference bank for --content_bank_path (see models/content_cond.py).

For every training-split face that is clearly OLD (Young pred <= --old_thresh)
or clearly YOUNG (Young pred >= --young_thresh), render G(w) -- the StyleGAN
reconstruction, the same domain the edited output lives in -- and store the
VGG skin/hair texture statistics of it, split by gender (Male pred >= 0.5).
Training then gives an aging edit of a male source the texture of a random
real OLD MALE face as its content, and vice versa.

Training split only (train=True): the bank's files are also used to make sure
a source is never its own reference, and eval images must never be used as
references (they are what eval scores).

Nothing here trains or touches a checkpoint.

Usage:
    # 1. LOOK FIRST: a few references per bucket with their skin (red) / hair
    #    (green) masks. Check the old buckets really look old, male buckets
    #    male, and masks cover skin / hair.
    python -m scripts.precompute_content_bank --preview --out_dir ./content_bank_preview

    # 2. build the bank
    python -m scripts.precompute_content_bank --dump --out ./data/content_bank_age.pth
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'stylegan2'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw

from common.face_parser import FaceParser
from models.content_cond import (BUCKET_OLD, BUCKET_YOUNG, RegionTextureStats)
from models.dataset import SDFlowDataset
from models.stylegan2.model import Generator


def build_dataset(args):
    # Images are not needed (G(w) is what gets measured); the transform is only
    # there because SDFlowDataset always loads one.
    tf = T.Compose([T.ToTensor(), T.Resize((64, 64)), T.Normalize(mean=0.5, std=0.5)])
    return SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                         latents_file=args.latent_file, preds_file=args.preds_file,
                         train=True, transform=tf)


def select_rows(dataset, args):
    """Returns [(dataset_index, bucket, male)], capped per (bucket, gender)."""
    g = torch.Generator().manual_seed(args.seed)
    buckets = {(b, m): [] for b in (BUCKET_OLD, BUCKET_YOUNG) for m in (True, False)}
    for idx in range(len(dataset)):
        pred = dataset._lookup_precomputed(dataset.preds, dataset.image_list[idx])
        young, male = float(pred[39]), float(pred[20]) >= 0.5
        if young <= args.old_thresh:
            buckets[(BUCKET_OLD, male)].append(idx)
        elif young >= args.young_thresh:
            buckets[(BUCKET_YOUNG, male)].append(idx)
    rows = []
    for (b, m), idxs in sorted(buckets.items()):
        total = len(idxs)
        if total > args.max_per_bucket:
            perm = torch.randperm(total, generator=g)[:args.max_per_bucket].tolist()
            idxs = [idxs[i] for i in sorted(perm)]
        print(f'  {"old  " if b == BUCKET_OLD else "young"} {"male  " if m else "female"}: '
              f'{total} candidates, using {len(idxs)}')
        rows += [(i, b, m) for i in idxs]
    return rows


@torch.no_grad()
def render(G, latents):
    img = G([latents], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
    return F.interpolate(img, (512, 512), mode='bilinear', align_corners=False)


def build_models(args):
    ckpt = torch.load(args.stygan2_weights, map_location='cpu')
    G = Generator(size=1024, style_dim=512, n_mlp=8)
    G.load_state_dict(ckpt['g_ema'])
    G.cuda().eval()
    parser = FaceParser(weights_path=args.face_parser_weights).cuda().eval()
    tex = RegionTextureStats(res=args.res, min_frac=args.min_frac, erode=args.erode).cuda()
    return G, parser, tex


def run_dump(args):
    dataset = build_dataset(args)
    rows = select_rows(dataset, args)
    if not rows:
        raise SystemExit('no references selected; check --old_thresh / --young_thresh')
    G, parser, tex = build_models(args)
    all_stats, all_valid = [], []
    for s in range(0, len(rows), args.batch):
        chunk = rows[s:s + args.batch]
        lat = torch.stack([dataset._lookup_precomputed(dataset.latents, dataset.image_list[i])
                           for i, _, _ in chunk]).cuda()
        img = render(G, lat)
        with torch.no_grad():
            st, va = tex.stats(img, tex.region_masks(parser, img))
        all_stats.append(st.cpu())
        all_valid.append(va.cpu())
        if (s // args.batch) % 50 == 0:
            print(f'  {s + len(chunk)}/{len(rows)}')
    stats = torch.cat(all_stats)
    valid = torch.cat(all_valid)
    R = valid.shape[1]
    rd = stats.shape[1] // R
    dim_mean = torch.zeros(stats.shape[1])
    dim_std = torch.ones(stats.shape[1])
    for r in range(R):
        sl = slice(r * rd, (r + 1) * rd)
        v = stats[valid[:, r], sl]
        if v.shape[0] >= 2:
            dim_mean[sl] = v.mean(0)
            dim_std[sl] = v.std(0)
    out = {
        'stats': stats,
        'valid': valid,
        'male': torch.tensor([m for _, _, m in rows]),
        'bucket': torch.tensor([b for _, b, _ in rows]),
        'files': [dataset.image_list[i] for i, _, _ in rows],
        'dim_mean': dim_mean,
        'dim_std': dim_std,
        'config': {'res': args.res, 'erode': args.erode, 'min_frac': args.min_frac,
                   'old_thresh': args.old_thresh, 'young_thresh': args.young_thresh,
                   'regions': ['skin', 'hair'], 'vgg_taps': ['relu2_2', 'relu3_3']},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    print(f'saved {len(rows)} references, dim {stats.shape[1]}, to {args.out}')
    print(f'  skin valid {valid[:, 0].float().mean():.1%}, hair valid {valid[:, 1].float().mean():.1%}')


def _to_pil(img, size):
    x = ((img.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x).resize((size, size))


def run_preview(args):
    dataset = build_dataset(args)
    args.max_per_bucket = args.num_faces
    rows = select_rows(dataset, args)
    G, parser, tex = build_models(args)
    os.makedirs(args.out_dir, exist_ok=True)
    names = {BUCKET_OLD: 'old', BUCKET_YOUNG: 'young'}
    for b in (BUCKET_OLD, BUCKET_YOUNG):
        for m in (True, False):
            sel = [i for i, bb, mm in rows if bb == b and mm == m]
            if not sel:
                continue
            lat = torch.stack([dataset._lookup_precomputed(dataset.latents, dataset.image_list[i])
                               for i in sel]).cuda()
            img = render(G, lat)
            masks = tex.region_masks(parser, img)
            T_ = args.tile
            canvas = Image.new('RGB', (T_ * len(sel), T_ * 2))
            for k in range(len(sel)):
                face = F.interpolate(img[k:k + 1], (tex.res, tex.res), mode='bilinear',
                                     align_corners=False)[0]
                over = face.clone()
                over[0] = torch.where(masks['skin'][k, 0] > 0.5, torch.ones_like(over[0]), over[0])
                over[1] = torch.where(masks['hair'][k, 0] > 0.5, torch.ones_like(over[1]), over[1])
                canvas.paste(_to_pil(face, T_), (k * T_, 0))
                canvas.paste(_to_pil(0.5 * face + 0.5 * over, T_), (k * T_, T_))
            ImageDraw.Draw(canvas).text((4, 4), f'{names[b]} {"male" if m else "female"}',
                                        fill=(255, 255, 0))
            path = os.path.join(args.out_dir, f'bank_{names[b]}_{"male" if m else "female"}.png')
            canvas.save(path)
            print(f'  wrote {path}')


def main():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preview', action='store_true')
    mode.add_argument('--dump', action='store_true')
    p.add_argument('--out', default='./data/content_bank_age.pth')
    p.add_argument('--out_dir', default='./content_bank_preview')
    p.add_argument('--num_faces', type=int, default=8, help='--preview: faces per bucket.')
    p.add_argument('--tile', type=int, default=192)
    p.add_argument('--old_thresh', type=float, default=0.2,
                   help='Young pred <= this counts as an OLD reference.')
    p.add_argument('--young_thresh', type=float, default=0.8,
                   help='Young pred >= this counts as a YOUNG reference.')
    p.add_argument('--max_per_bucket', type=int, default=2000,
                   help='Cap per (old/young, male/female) bucket.')
    p.add_argument('--res', type=int, default=256,
                   help='Resolution the texture statistics are measured at. Training reads '
                        'this back from the bank, so it only has to be set here.')
    p.add_argument('--erode', type=int, default=4)
    p.add_argument('--min_frac', type=float, default=0.01)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--index_file', default='./data/ffhq.txt')
    p.add_argument('--image_root', default='data/FFHQ')
    p.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    p.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth')
    p.add_argument('--stygan2_weights', default='./data/stylegan2-ffhq-config-f.pt')
    p.add_argument('--face_parser_weights', default='./data/parsing_bisenet.pth')
    args = p.parse_args()
    if args.preview:
        run_preview(args)
    else:
        run_dump(args)


if __name__ == '__main__':
    main()
