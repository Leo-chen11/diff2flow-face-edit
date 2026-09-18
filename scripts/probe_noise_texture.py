"""Does StyleGAN's per-layer NOISE affect how real the aging texture looks?

Every render in this project passes randomize_noise=False (12 call sites,
no exceptions), so the generator reads the fixed noise buffers registered
once at __init__ as torch.randn. Those buffers are never optimised, never
conditioned on the attribute being edited, and correspond to no particular
face -- they are one arbitrary draw.

That matters because W+ and noise carry different things. W+ decides which
convolution filters fire and how strongly, per layer, uniformly across
space. The irregular, non-repeating fine detail that makes skin read as
real -- pores, creases, blemishes -- is what StyleGAN's noise inputs are
for. An aging edit driven purely through W+ can only reach the first kind,
which is a plausible mechanism for wrinkles that look drawn on rather than
grown.

This does not change any training or eval behaviour. It re-renders the SAME
edited latent under several noise draws and asks whether the result moves.
It works by overwriting the generator's noise buffers between renders --
the latent computation is deterministic and noise only enters at render
time -- so nothing in evaluate_sdflow.py needs to change.

Two outputs:

  numbers  -- high-frequency energy inside the BiSeNet SKIN region (class 1),
              which is a proxy for how much fine texture is present, reported
              for the source reconstruction and for each noise draw of the
              edit, plus the spread across draws. If the spread is tiny
              relative to the source-to-edit change, noise is not the lever
              and the 512-band ControlNet route is the one left.
  montage  -- src | edit@noise0 | edit@noise1 | ... so the question the
              numbers cannot answer (does any draw look MORE convincing)
              stays a visual one.

Usage:
    python -m scripts.probe_noise_texture \
        --checkpoint_dir ./output/SDFlow/substyle3_glasses_v18 --step 120000 \
        --attr 39 --direction rm --edit_scale 1.0 \
        --num_samples 8 --num_noise 4 \
        --out_dir ./noise_probe
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
from torch.utils import data
from PIL import Image, ImageDraw

from evaluation.evaluate_sdflow import (
    ATTR_NAMES, CelebAAttrClassifierJudge, _latest_step, apply_run_config,
    edit_single_attribute, is_clear, load_models, resolve_controlnet_disable_attrs,
)
from common.face_parser import FaceParser
from models.dataset import SDFlowDataset
from scripts.dump_attr_failures import save_montage, to_pil

SKIN_CLASS = [1]        # BiSeNet: 0=background, 1=skin (see face_parser.py)


def _gaussian_blur(x, sigma=2.0):
    ks = int(sigma * 6) | 1                      # odd kernel
    coords = torch.arange(ks, device=x.device, dtype=x.dtype) - ks // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, 1, -1)
    c = x.shape[1]
    x = F.conv2d(x, g.expand(c, 1, 1, ks), padding=(0, ks // 2), groups=c)
    x = F.conv2d(x, g.transpose(-1, -2).expand(c, 1, ks, 1), padding=(ks // 2, 0), groups=c)
    return x


@torch.no_grad()
def skin_hf_energy(img, face_parser, sigma=2.0):
    """Mean squared high-frequency residual inside the skin region.

    img - blur(img) keeps what a Gaussian removes: pores, creases, fine
    shading. Restricted to BiSeNet's skin class so hair -- which carries far
    more high-frequency energy than any wrinkle and would swamp the
    measurement -- is excluded.
    """
    hf = img - _gaussian_blur(img, sigma)
    mask = face_parser.get_region_mask(img, SKIN_CLASS, blur_sigma=0)
    denom = (mask.sum() * img.shape[1]).clamp(min=1.0)
    return ((hf ** 2) * mask).sum() / denom


def reseed_noise(G, seed):
    """Overwrite the generator's fixed noise buffers with a fresh draw.

    randomize_noise=False makes G read these buffers, so replacing them and
    re-rendering the same latent isolates the noise contribution exactly.
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    for i in range(G.num_layers):
        buf = getattr(G.noises, f'noise_{i}')
        buf.copy_(torch.randn(buf.shape, generator=g).to(buf.device))


def snapshot_noise(G):
    return [getattr(G.noises, f'noise_{i}').clone() for i in range(G.num_layers)]


def restore_noise(G, snap):
    for i, t in enumerate(snap):
        getattr(G.noises, f'noise_{i}').copy_(t)


@torch.no_grad()
def main(args):
    prior, conditioner, G, id_criterion, attr_teacher, \
        attribute_index, direction_bank, control_encoder = load_models(args)
    if args.attr not in args.attribute_index:
        raise SystemExit(f'attribute_index {args.attribute_index} has no attr {args.attr}.')
    local_idx = args.attribute_index.index(args.attr)
    attr_name = ATTR_NAMES.get(args.attr, f'attr{args.attr}')

    face_parser = FaceParser(weights_path=args.face_parser_weights).cuda().eval()
    judge = None
    if args.celeba_attr_judge_weights:
        judge = CelebAAttrClassifierJudge(args.celeba_attr_judge_weights, 'cuda')

    print(f'auditing {attr_name} direction={args.direction} scale={args.edit_scale}')
    print(f'{args.num_noise} noise draws per sample, {args.num_samples} samples\n')

    img_transform = T.Compose([
        T.ToTensor(), T.Resize((args.img_size, args.img_size)),
        T.Normalize(mean=0.5, std=0.5),
    ])
    dataset = SDFlowDataset(
        index_file=args.index_file, image_root=args.image_root,
        latents_file=args.latent_file, preds_file=args.preds_file,
        train=False, transform=img_transform,
    )
    loader = data.DataLoader(dataset, shuffle=False, batch_size=1,
                             num_workers=2, drop_last=False)

    base_noise = snapshot_noise(G)
    rows, src_hf_all, spread_all, gap_all = [], [], [], []
    seen = 0
    for img, latent, pred in loader:
        if seen >= args.num_samples:
            break
        img = img.cuda(); latent = latent.cuda()
        _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
        c = attr_cond[0, local_idx].item()
        # Gate on the conditioner -- the reading that actually decides which
        # way edit_single_attribute moves this sample.
        if args.direction == 'rm' and c < 0.5:
            continue
        if args.direction == 'add' and c >= 0.5:
            continue
        if not is_clear(c):
            continue
        seen += 1

        restore_noise(G, base_noise)
        src = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
        src = F.interpolate(src, (args.img_size, args.img_size))
        src_hf = skin_hf_energy(src, face_parser, args.blur_sigma_hf).item()

        panels = [to_pil(F.interpolate(src, (args.tile, args.tile))[0])]
        hfs, scores = [], []
        for k in range(args.num_noise):
            reseed_noise(G, args.noise_seed + k)
            edited = edit_single_attribute(
                prior, conditioner, G, id_criterion, img, latent, attr_cond, id_cond,
                local_idx, args.edit_scale, direction_bank, attr_global_idx=args.attr,
                control_encoder=control_encoder,
                controlnet_max_norm=getattr(args, 'controlnet_max_norm', 0.0),
                controlnet_disable_attrs=getattr(args, 'controlnet_disable_attrs', None),
            )
            hfs.append(skin_hf_energy(edited, face_parser, args.blur_sigma_hf).item())
            if judge is not None:
                scores.append(judge.scores(F.interpolate(edited, (256, 256)))[0, args.attr].item())
            panels.append(to_pil(F.interpolate(edited, (args.tile, args.tile))[0]))

        mean_hf = sum(hfs) / len(hfs)
        spread = max(hfs) - min(hfs)
        gap = mean_hf - src_hf
        src_hf_all.append(src_hf); spread_all.append(spread); gap_all.append(gap)
        sc = ('  judge ' + ' '.join(f'{s:.2f}' for s in scores)) if scores else ''
        print(f'  #{seen:<2} src_hf={src_hf:.5f}  edit_hf={mean_hf:.5f} '
              f'(gap {gap:+.5f})  spread across noise={spread:.5f}{sc}')

        w, h = panels[0].size
        canvas = Image.new('RGB', (w * len(panels), h + 20), (20, 20, 20))
        d = ImageDraw.Draw(canvas)
        d.text((4, 5), f'src hf={src_hf:.4f}', fill=(170, 170, 170))
        for i, p in enumerate(panels):
            canvas.paste(p, (i * w, 20))
            if i > 0:
                d.text((i * w + 4, 5), f'noise{i-1} hf={hfs[i-1]:.4f}', fill=(200, 200, 120))
        rows.append(canvas)

    restore_noise(G, base_noise)

    if not rows:
        raise SystemExit('no samples matched that attribute/direction.')

    n = len(gap_all)
    mean_gap = sum(gap_all) / n
    mean_spread = sum(spread_all) / n
    print()
    print(f'=== over {n} samples ===')
    print(f'  source -> edit change in skin HF energy : {mean_gap:+.5f}')
    print(f'  variation across noise draws            : {mean_spread:.5f}')
    if abs(mean_gap) > 1e-9:
        ratio = mean_spread / abs(mean_gap)
        print(f'  noise spread / edit effect              : {ratio:.2f}x')
        print()
        if ratio >= 0.5:
            print('  READ: noise moves skin texture about as much as the EDIT does.')
            print('  It is a real lever -- worth conditioning on the attribute instead')
            print('  of leaving it at one fixed arbitrary draw.')
        else:
            print('  READ: noise barely moves skin texture next to the edit itself.')
            print('  It is NOT the lever. The remaining untried route is injecting')
            print('  ControlNet at 512, the band this project\'s own control_encoder')
            print('  docstring names as where wrinkles/pores live.')
    print()
    print('  The numbers cannot say whether any draw looks MORE convincing --')
    print('  only that it differs. Open the montage for that.')

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f'{attr_name}_{args.direction}_noise_sweep.png')
    save_montage(rows, path, cols=1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--step', type=int, default=None)
    p.add_argument('--attr', type=int, default=39)
    p.add_argument('--direction', default='rm', choices=['add', 'rm'])
    p.add_argument('--edit_scale', type=float, default=1.0)
    p.add_argument('--num_samples', type=int, default=8)
    p.add_argument('--num_noise', type=int, default=4,
                   help='How many independent noise draws to re-render each edit under.')
    p.add_argument('--noise_seed', type=int, default=1234)
    p.add_argument('--tile', type=int, default=256, help='Montage tile size.')
    p.add_argument('--out_dir', default='./noise_probe')
    p.add_argument('--blur_sigma_hf', type=float, default=2.0,
                   help='Gaussian sigma whose removed detail counts as high frequency.')
    p.add_argument('--face_parser_weights', default='./data/parsing_bisenet.pth')
    p.add_argument('--celeba_attr_judge_weights', default=None,
                   help='Optional: also print the judge score per noise draw.')
    p.add_argument('--controlnet_disable_attrs', nargs='*', type=int, default=None)

    p.add_argument('--index_file',   default='./data/ffhq.txt')
    p.add_argument('--image_root',   default='data/FFHQ')
    p.add_argument('--latent_file',  default='./data/ffhq_e4e_latents.pth')
    p.add_argument('--preds_file',   default='./data/ffhq_e4e_preds.pth')
    p.add_argument('--stygan2_weights', default='./data/stylegan2-ffhq-config-f.pt')
    p.add_argument('--attribute_weights', default='./data/r34_a40_age_256_classifier.pth')
    p.add_argument('--direction_bank_path', default=None)

    p.add_argument('--img_size',         type=int,   default=512)
    p.add_argument('--attribute_index',  nargs='*',  type=int,   default=[15, 20, 39])
    p.add_argument('--flow_modules',     default='512-512-512-512-512')
    p.add_argument('--num_blocks',       type=int,   default=1)
    p.add_argument('--velocity_field',   default='lag_dof')
    p.add_argument('--id_cond_dim',      type=int,   default=32)
    p.add_argument('--id_cond_scale',    type=float, default=0.25)
    p.add_argument('--attr_backbone',    default='resnet50')
    p.add_argument('--conditioner_backbone', default='resnet',
                   choices=['resnet', 'clip', 'resnet_clip'])
    p.add_argument('--clip_model', default='ViT-B/32')
    p.add_argument('--clip_judge_model', default='ViT-L/14')
    p.add_argument('--clip_calibration', default=None)
    p.add_argument('--fused_hidden_dim', type=int, default=256)
    p.add_argument('--lag_gate_hidden_dim', type=int,   default=64)
    p.add_argument('--lag_gate_init_bias',  type=float, default=-0.5)
    p.add_argument('--direction_residual_scale', type=float, default=0.05)
    p.add_argument('--glasses_residual_scale',   type=float, default=0.05)
    p.add_argument('--bypass_glasses_direction_bank',
                   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--guided_delta_max_norm', type=float, default=0.0)
    p.add_argument('--override_residual_scale', type=float, default=None)
    p.add_argument('--age_fine_layer_scale', type=float, default=None)
    p.add_argument('--age_fine_layer_start', type=int, default=10)
    p.add_argument('--force_bank_directions', action='store_true')
    p.add_argument('--disable_controlnet', action='store_true')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--ignore_run_config', action='store_true')

    args = p.parse_args()
    args = apply_run_config(args)
    args = resolve_controlnet_disable_attrs(args)
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
        if args.step is None:
            raise ValueError(f'No checkpoints in {args.checkpoint_dir}/save_models/')
        print(f'Auto-detected latest step: {args.step}')
    main(args)
