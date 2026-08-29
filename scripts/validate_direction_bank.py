"""
Validate a precomputed direction bank BEFORE spending tens of thousands of
training steps on top of it.

scripts/precompute_directions_stratified.py only ever prints internal
diagnostics (cosine similarity between K sub-directions, cross-attribute
cosine similarity) -- it never checks whether the direction it just wrote
actually DOES anything sensible to a real face. The Direction Bank's
magnitude_net is untrained at this point (random init, only learned once
train_sdflow.py starts), so this cannot use AttributeDirectionBank.forward()
-- instead it applies the bank's raw, precomputed geometry directly:

    w' = w + sign * alpha * layer_norms[a].mean(K) * direction_units[a].mean(K)

i.e. exactly the magnitude the high/low groups were observed to differ by
(alpha=1.0), in the ADD or RM direction appropriate for each source sample,
averaged over the K stratified sub-directions into one representative
vector -- the same averaging precompute_directions_stratified.py itself uses
for cross-attribute decorrelation (representative_direction()). This is the
classic InterFaceGAN-style "apply w + alpha*n, check what happens" sanity
check, just reading the vector back out of the saved .pth instead of a
freshly-fit SVM boundary.

Scores with the SAME independent judges evaluate_sdflow.py uses (CLIP
zero-shot by default, optional --celeba_attr_judge_weights), so a number
here is directly comparable in spirit to the eval script's AccCLIP/AccCeleb
columns -- except this whole thing runs in a couple of minutes on ~200
samples, before the flow, conditioner, or any training loss has ever been
touched. A bad --extreme_pct, --K, or missing --decorrelate_cross_attr shows
up here immediately instead of surfacing 80,000 steps later as "the scores
and the images are both bad" with no way to tell whether the fault is the
geometry or the training run built on top of it.

Usage:
  python scripts/validate_direction_bank.py \\
      --direction_bank_path ./data/direction_bank_k4_stratified.pth \\
      --num_samples 200 --alphas 0.5 1.0 1.5
"""
import argparse
import os
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'stylegan2'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils import data
from tqdm import tqdm
import numpy as np

from evaluation.evaluate_sdflow import (
    ATTR_NAMES, CLIPAttributeJudge, CelebAAttrClassifierJudge, GlassesParserJudge,
    IndependentIDJudge, is_clear, strict_success, parse_clip_calibration,
)
from models.dataset import SDFlowDataset
from models.stylegan2.model import Generator


def load_bank(path):
    bank = torch.load(path, map_location='cpu')
    direction_units = bank['direction_units']   # (num_attrs, K, 18, 512), unit vectors
    layer_norms = bank['layer_norms']           # (num_attrs, K, 18)
    attribute_index = [int(a) for a in bank['attribute_index']]
    # Representative (K-averaged) RAW displacement per attribute, same
    # averaging precompute_directions_stratified.py uses for decorrelation.
    raw = layer_norms.unsqueeze(-1) * direction_units          # (num_attrs, K, 18, 512)
    representative = raw.mean(dim=1)                            # (num_attrs, 18, 512)
    return attribute_index, representative


@torch.no_grad()
def main(args):
    device = 'cuda'
    attribute_index, rep_dirs = load_bank(args.direction_bank_path)
    print(f'Bank attribute_index: {attribute_index}')
    for i, a in enumerate(attribute_index):
        print(f'  attr {a} ({ATTR_NAMES.get(a, "?")}): representative norm per layer '
              f'mean={rep_dirs[i].norm(dim=-1).mean():.3f} '
              f'min={rep_dirs[i].norm(dim=-1).min():.3f} '
              f'max={rep_dirs[i].norm(dim=-1).max():.3f}')
    rep_dirs = rep_dirs.to(device)

    ckpt = torch.load(args.stygan2_weights, map_location='cpu')
    G = Generator(size=1024, style_dim=512, n_mlp=8)
    G.load_state_dict(ckpt['g_ema'])
    G.to(device).eval()
    for p in G.parameters():
        p.requires_grad_(False)

    clip_judge = CLIPAttributeJudge(attribute_index, args.clip_judge_model, device,
                                     calibration=parse_clip_calibration(args.clip_calibration))
    glasses_parser = None
    if args.glasses_judge == 'parser' and 15 in attribute_index:
        glasses_parser = GlassesParserJudge(args.face_parser_weights, device,
                                            area_thresh=args.glasses_area_thresh)
        print('[Judge] Eyeglasses scored by BiSeNet parser (matches evaluate_sdflow.py default).')
    celeb_judge = None
    if args.celeba_attr_judge_weights:
        celeb_judge = CelebAAttrClassifierJudge(args.celeba_attr_judge_weights, device)
    indep_id = IndependentIDJudge(device, pretrained=args.id_indep_pretrained)

    img_transform = T.Compose([
        T.ToTensor(), T.Resize((args.img_size, args.img_size)),
        T.Normalize(mean=0.5, std=0.5),
    ])
    dataset = SDFlowDataset(
        index_file=args.index_file, image_root=args.image_root,
        latents_file=args.latent_file, preds_file=args.preds_file,
        train=False, transform=img_transform,
    )
    loader = data.DataLoader(dataset, shuffle=False, batch_size=args.batch, num_workers=4)

    local_idx = {a: i for i, a in enumerate(attribute_index)}
    attr_names = [ATTR_NAMES.get(a, f'attr{a}') for a in attribute_index]

    metrics = {alpha: defaultdict(list) for alpha in args.alphas}
    seen = 0
    for img, latent, _pred in tqdm(loader, desc='validating'):
        if seen >= args.num_samples:
            break
        img = img.cuda()
        latent = latent.cuda()
        B = img.size(0)

        src_256 = F.interpolate(img, (256, 256))
        src_scores = clip_judge.scores(src_256)
        if glasses_parser is not None and 15 in local_idx:
            src_scores[:, local_idx[15]] = glasses_parser.glasses_prob(src_256)
        src_celeb = celeb_judge.scores(src_256)[:, attribute_index] if celeb_judge is not None else None
        src_id = indep_id.extract(src_256)

        for alpha in args.alphas:
            for a_idx, a in enumerate(attribute_index):
                s = src_scores[:, a_idx]
                sign = torch.where(s < 0.5, torch.ones_like(s), -torch.ones_like(s))  # add vs rm
                new_latent = latent + (sign.view(B, 1, 1) * alpha) * rep_dirs[a_idx].unsqueeze(0)
                edited = G([new_latent], input_is_latent=True,
                          randomize_noise=False)[0].clamp(-1, 1)
                edited_256 = F.interpolate(edited, (256, 256))

                edit_scores = clip_judge.scores(edited_256)
                if glasses_parser is not None and a == 15:
                    edit_scores[:, a_idx] = glasses_parser.glasses_prob(edited_256)
                edit_celeb = celeb_judge.scores(edited_256)[:, attribute_index] if celeb_judge is not None else None
                edit_id = indep_id.extract(edited_256)
                id_cos = (src_id * edit_id).sum(dim=1)

                for b in range(B):
                    sb = s[b].item()
                    if not is_clear(sb):
                        continue
                    eb = edit_scores[b, a_idx].item()
                    metrics[alpha][f'acc_clip_{attr_names[a_idx]}'].append(
                        float(strict_success(sb, eb, 0.0)))
                    metrics[alpha][f'id_{attr_names[a_idx]}'].append(id_cos[b].item())
                    if celeb_judge is not None:
                        ceb = edit_celeb[b, a_idx].item()
                        csb = src_celeb[b, a_idx].item()
                        if is_clear(csb):
                            metrics[alpha][f'acc_celeb_{attr_names[a_idx]}'].append(
                                float(strict_success(csb, ceb, 0.0)))
                    # Leakage on every OTHER attribute's CLIP score.
                    for other_idx, other_a in enumerate(attribute_index):
                        if other_idx == a_idx:
                            continue
                        leak = abs(edit_scores[b, other_idx].item() - src_scores[b, other_idx].item())
                        metrics[alpha][f'leak_{attr_names[a_idx]}'].append(leak)

        seen += B

    print(f'\n{seen} samples evaluated (raw direction geometry, no flow, no training)\n')
    for alpha in args.alphas:
        print(f'{"="*70}\nalpha = {alpha}\n{"="*70}')
        header = f'  {"Attribute":<12} {"AccCLIP":>8}'
        if celeb_judge is not None:
            header += f' {"AccCeleb":>9}'
        header += f' {"ID_ind":>8} {"Leak":>8}'
        print(header)
        for name in attr_names:
            m = metrics[alpha]
            acc = np.mean(m[f'acc_clip_{name}']) if m[f'acc_clip_{name}'] else float('nan')
            idv = np.mean(m[f'id_{name}']) if m[f'id_{name}'] else float('nan')
            leak = np.mean(m[f'leak_{name}']) if m[f'leak_{name}'] else float('nan')
            row = f'  {name:<12} {acc*100:7.1f}%'
            if celeb_judge is not None:
                acc_c = np.mean(m[f'acc_celeb_{name}']) if m[f'acc_celeb_{name}'] else float('nan')
                row += f' {acc_c*100:8.1f}%'
            row += f' {idv:8.4f} {leak:8.4f}'
            print(row)
    print('\nRead this as: does AccCLIP move toward 100% and ID_ind stay high as alpha increases? '
          'A direction that never crosses ~70-80% AccCLIP even at alpha=1.5, or whose ID_ind '
          'collapses well before that, has a geometry problem worth fixing (K, --extreme_pct, '
          '--strata_margin, --decorrelate_cross_attr, --decorrelate_age_color) BEFORE training on '
          'top of it -- training cannot fix a direction that points the wrong way.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--direction_bank_path', required=True)
    p.add_argument('--num_samples', type=int, default=200)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--alphas', nargs='*', type=float, default=[0.5, 1.0, 1.5])
    p.add_argument('--img_size', type=int, default=512)

    p.add_argument('--index_file', default='./data/ffhq.txt')
    p.add_argument('--image_root', default='data/FFHQ')
    p.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    p.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth')
    p.add_argument('--stygan2_weights', default='./data/stylegan2-ffhq-config-f.pt')

    p.add_argument('--clip_judge_model', default='ViT-L/14')
    p.add_argument('--clip_calibration', default=None)
    p.add_argument('--glasses_judge', default='parser', choices=['clip', 'parser'])
    p.add_argument('--face_parser_weights', default='./data/parsing_bisenet.pth')
    p.add_argument('--glasses_area_thresh', type=float, default=0.0010)
    p.add_argument('--celeba_attr_judge_weights', default=None)
    p.add_argument('--id_indep_pretrained', default='casia-webface',
                   choices=['casia-webface', 'vggface2'])

    args = p.parse_args()
    main(args)
