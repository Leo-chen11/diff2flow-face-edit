"""
Evaluation script: generates Identity Preservation and Attribute Preservation
curves vs Editing Accuracy (reproduces Fig. 4 in the SDFlow paper).

Usage:
    cd /path/to/SDFlow-main
    python evaluation/eval_curves.py \
        --labeldist_ckpt ./output/SDFlow/default/save_models/labeldist-0017000 \
        --prior_ckpt     ./output/SDFlow/default/save_models/prior-0017000

Requires: pip install facenet-pytorch matplotlib
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'stylegan2'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils import data
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.attribute_estimator import AttributeEstimator
from models.flows.flow import cnf
from models.dataset import SDFlowDataset
from models.stylegan2.model import Generator
from common.ops import load_network


def load_arcface(device):
    from facenet_pytorch import InceptionResnetV1
    model = InceptionResnetV1(pretrained='vggface2').to(device).eval()
    return model


def arcface_features(model, imgs_minus1_1, target_size=160):
    """imgs: [-1,1] tensor [B,3,H,W] → L2-normalized embedding [B,512]"""
    x = F.interpolate(imgs_minus1_1, size=target_size, mode='bilinear', align_corners=False)
    with torch.no_grad():
        emb = model(x)
    return F.normalize(emb, dim=1)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    attribute_index = torch.tensor(args.attribute_index, dtype=torch.long)
    num_attrs = len(args.attribute_index)

    # ── Load models ────────────────────────────────────────────────────────────
    print('Loading AttributeEstimator...')
    labeldist = AttributeEstimator(backbone=args.backbone, attribute_dim=num_attrs).to(device)
    labeldist.load_state_dict(load_network(args.labeldist_ckpt))
    labeldist.eval()

    print('Loading Flow model...')
    prior = cnf(512, args.flow_modules, num_attrs, args.num_blocks).to(device)
    prior.load_state_dict(load_network(args.prior_ckpt))
    prior.eval()

    print('Loading StyleGAN2...')
    ckpt = torch.load(args.stylegan2_weights, map_location='cpu')
    G = Generator(size=1024, style_dim=512, n_mlp=8)
    G.load_state_dict(ckpt['g_ema'])
    G.to(device).eval()

    print('Loading ArcFace (facenet-pytorch)...')
    arcface = load_arcface(device)

    # ── Dataset ────────────────────────────────────────────────────────────────
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((args.img_size, args.img_size)),
        T.Normalize(mean=0.5, std=0.5),
    ])
    test_dataset = SDFlowDataset(
        index_file=args.index_file,
        image_root=args.image_root,
        latents_file=args.latent_file,
        preds_file=args.preds_file,
        train=False,
        transform=transform,
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=args.batch,
        shuffle=False, drop_last=False, num_workers=4,
    )
    print(f'Test set: {len(test_dataset)} images')

    # ── Evaluation ─────────────────────────────────────────────────────────────
    # scales control editing strength: 0.1 = mild, 1.0 = full flip
    scales = [round(s, 1) for s in np.linspace(0.1, 1.0, 10)]

    # per scale: lists of per-sample values
    records = {s: {'edit_acc': [], 'id_sim': [], 'attr_pres': []} for s in scales}

    with torch.no_grad():
        for img, latent, pred in tqdm(test_loader, desc='Evaluating'):
            img    = img.to(device)
            latent = latent.to(device)
            pred   = pred.to(device)
            B = img.shape[0]

            # Current attribute distribution from trained estimator
            label_dist = labeldist(img, latent)          # [B, num_attrs]

            # Encode to latent z via forward flow
            z, _ = prior(latent, label_dist, torch.zeros(B, 18, 1, device=device))

            # Original face from StyleGAN (for identity comparison)
            orig_faces = G([latent], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
            orig_faces = F.interpolate(orig_faces, (args.img_size, args.img_size), mode='bilinear', align_corners=False)
            orig_emb   = arcface_features(arcface, orig_faces)  # [B, 512]

            # Evaluate each attribute independently
            for attr_idx in range(num_attrs):
                orig_attr = label_dist[:, attr_idx]        # [B], continuous [0,1]
                has_attr  = (orig_attr > 0.5)              # [B], bool

                for scale in scales:
                    # Build target label: move attribute toward 0 or 1 by `scale`
                    target_vals = torch.where(
                        has_attr,
                        torch.full_like(orig_attr, 1.0 - scale),  # remove attribute
                        torch.full_like(orig_attr, scale),         # add attribute
                    )
                    new_label_dist = label_dist.clone()
                    new_label_dist[:, attr_idx] = target_vals

                    # Decode to new latent via reverse flow
                    new_latents, _ = prior(
                        z, new_label_dist,
                        torch.zeros(B, 18, 1, device=device),
                        reverse=True,
                    )

                    # Generate edited face
                    new_faces = G([new_latents], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
                    new_faces = F.interpolate(new_faces, args.img_size)

                    # Re-predict attributes on edited face
                    recon_dist = labeldist(new_faces, new_latents)  # [B, num_attrs]

                    # ── Editing Accuracy ──────────────────────────────────────
                    # Did the target attribute flip in the right direction?
                    recon_attr   = recon_dist[:, attr_idx]
                    target_bin   = (target_vals > 0.5).float()
                    recon_bin    = (recon_attr  > 0.5).float()
                    edit_success = (target_bin == recon_bin).float()   # [B]

                    # ── Attribute Preservation ────────────────────────────────
                    # Non-target attributes should stay the same
                    non_target = [j for j in range(num_attrs) if j != attr_idx]
                    if non_target:
                        orig_nt  = (label_dist[:, non_target] > 0.5).float()
                        recon_nt = (recon_dist[:, non_target] > 0.5).float()
                        attr_pres = (orig_nt == recon_nt).float().mean(dim=1)  # [B]
                    else:
                        attr_pres = torch.ones(B, device=device)

                    # ── Identity Preservation ─────────────────────────────────
                    new_emb = arcface_features(arcface, new_faces)     # [B, 512]
                    id_sim  = (orig_emb * new_emb).sum(dim=1)          # [B]

                    records[scale]['edit_acc'].extend(edit_success.cpu().tolist())
                    records[scale]['id_sim'].extend(id_sim.cpu().tolist())
                    records[scale]['attr_pres'].extend(attr_pres.cpu().tolist())

    # ── Aggregate ──────────────────────────────────────────────────────────────
    edit_accs  = [np.mean(records[s]['edit_acc'])  * 100 for s in scales]
    id_sims    = [np.mean(records[s]['id_sim'])          for s in scales]
    attr_press = [np.mean(records[s]['attr_pres']) * 100 for s in scales]

    # ── Print table ───────────────────────────────────────────────────────────
    print('\n=== Evaluation Results ===')
    print(f"{'Scale':>6} | {'Edit Acc (%)':>12} | {'ID Cosine':>10} | {'Attr Pres (%)':>13}")
    print('-' * 52)
    for i, s in enumerate(scales):
        print(f'{s:>6.1f} | {edit_accs[i]:>12.2f} | {id_sims[i]:>10.4f} | {attr_press[i]:>13.2f}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(edit_accs, id_sims, 'r.-', label='Ours', linewidth=2, markersize=8)
    axes[0].set_xlabel('Editing Accuracy (%)')
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_title('Identity Preservation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(edit_accs, attr_press, 'r.-', label='Ours', linewidth=2, markersize=8)
    axes[1].set_xlabel('Editing Accuracy (%)')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Attribute Preservation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'eval_curves.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nCurves saved to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # checkpoints
    parser.add_argument('--labeldist_ckpt', required=True,
                        help='e.g. ./output/SDFlow/default/save_models/labeldist-0017000')
    parser.add_argument('--prior_ckpt', required=True,
                        help='e.g. ./output/SDFlow/default/save_models/prior-0017000')
    parser.add_argument('--stylegan2_weights', default='./data/stylegan2-ffhq-config-f.pt')

    # data
    parser.add_argument('--latent_file',  default='./data/ffhq_e4e_latents.pth')
    parser.add_argument('--preds_file',   default='./data/ffhq_e4e_preds.pth')
    parser.add_argument('--index_file',   default='./data/ffhq.txt')
    parser.add_argument('--image_root',   default='data/FFHQ')

    # model
    parser.add_argument('--attribute_index', nargs='*', default=[15, 20, 39], type=int)
    parser.add_argument('--flow_modules', default='512-512-512-512-512')
    parser.add_argument('--num_blocks',   type=int, default=1)
    parser.add_argument('--backbone',     default='resnet50')

    # eval
    parser.add_argument('--batch',      type=int, default=8)
    parser.add_argument('--img_size',   type=int, default=256)
    parser.add_argument('--output_dir', default='./output/eval')

    args = parser.parse_args()
    main(args)
