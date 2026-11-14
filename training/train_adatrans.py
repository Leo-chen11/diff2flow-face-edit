"""Train the AdaTrans baseline (models/adatrans.py) on this repo's data, as the official
training/train.py and train_realnvp_and_classifier.ipynb do, so that it is compared on the same
generator, inversion, train split, attribute teacher and test faces as every other method.

Two stages:

    density   the RealNVP density over single W+ layer vectors of the TRAIN latents
              (official notebook: AdamW 1e-4, batch 256, 10000 iterations)
    edit      one AdaTrans per attribute (official train.py): c = +1 / -1 (flip the attribute of
              every sample), 10 steps, loss =
                  1.0 * MSE(edited latent, source latent)                           recon
                + 1.0 * BCE(r34(G(edit)), targets) on [the attribute + --keeps]      cls
                + 1.0 * density NLL of every trajectory point                         nll
                + 0.1 * identity loss (edit vs source reconstruction)                 id
                + 0.1 * LPIPS (alex)                                                  lpips
              on 256 latents a step, the image terms on the first --syn_bs of them; AdamW 1e-4,
              gradients accumulated over 8 iterations, 10000 iterations.

Deviations from the official code, all listed in config.json:
  * identity loss: the official one is insightface's iresnet34. This uses the facenet (VGGFace2)
    identity loss the editor of this repo trains with (common/id_loss.py), so both methods are
    trained against the same identity model and neither against the evaluation's ArcFace IR-SE50.
  * --keeps: the official scripts also keep age (-1, its 6 ordinal age heads); this repo's preds
    file has no age heads, so age is left out. Defaults are the official train.sh choices otherwise
    (15: 20 / 20: 15 / 31: 15 20 / 39: 15 20); Bangs (5), which the official scripts do not train,
    keeps 15 20 like Smiling and Young.
  * LPIPS is the pip `lpips` package (alex, v0.1), the same network as the official copy -- and as
    the LPIPS column of this repo's evaluation, so AdaTrans is trained on that metric: say so next
    to its LPIPS numbers.
  * the generator is this repo's StyleGAN2 (the same FFHQ weights in rosinality's format), rendered
    with constant noise and resized to 256 with antialiased bicubic, as the official decoder.

    python training/train_adatrans.py density --out ./output/baselines/adatrans
    python training/train_adatrans.py edit --attr 15 --out ./output/baselines/adatrans
    ... --attr 20 / 39 / 31 / 5

Writes <out>/density.pth and <out>/<attr>/model.pth (+ config.json); scripts/editing_curves.py
--method adatrans --adatrans_dir <out> evaluates them, and also reads the official released
checkpoints laid out as <dir>/<attr>/save_models/model-latest.
"""
import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'stylegan2'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F

OFFICIAL_KEEPS = {15: [20], 20: [15], 31: [15, 20], 39: [15, 20], 5: [15, 20]}


def load_train_split(args):
    """(N, 18, 512) absolute W+ latents and (N, 40) attribute probabilities of the TRAIN split."""
    from models.dataset import SDFlowDataset
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file, train=True)
    files = list(ds.image_list)
    W = torch.stack([torch.as_tensor(ds._lookup_precomputed(ds.latents, f)).float().reshape(18, 512) for f in files])
    P = torch.stack([torch.as_tensor(ds._lookup_precomputed(ds.preds, f)).float().reshape(-1)[:40] for f in files])
    if P.min() < 0 or P.max() > 1:          # logits, not probabilities
        P = torch.sigmoid(P)
    print(f'train split: {len(files)} faces')
    return W, P


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)


def train_density(args, W):
    from models.adatrans import LatentDensity
    dens = LatentDensity(style_dim=512, n_layer=10).cuda()
    opt = torch.optim.AdamW(dens.parameters(), lr=args.lr, weight_decay=0.0)
    path = os.path.join(args.out, 'density.pth')
    t0 = time.time()
    for it in range(1, args.iters + 1):
        idx = torch.randint(0, len(W), (args.batch,))
        dens.train()
        loss, logz, log_det, _ = dens(W[idx].cuda())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 100 == 0:
            print(f'[density] {it}/{args.iters}  nll/dim {loss.item():.4f}  logz {logz.mean().item():.4f}  '
                  f'logdet {log_det.mean().item():.2f}  [{time.time() - t0:.0f}s]')
        if it % args.save_freq == 0 or it == args.iters:
            torch.save(dens.state_dict(), path)
    save_json(os.path.join(args.out, 'density_config.json'), vars(args))
    print(f'saved {path}')


def train_edit(args, W, P):
    import lpips
    from common.id_loss import IDLoss
    from models.adatrans import AdaTrans, LatentDensity
    from models.attribute_estimator import AttributeClassifier
    from models.stylegan2.model import Generator
    from common.ops import load_network

    a = int(args.attr)
    keeps = [int(k) for k in (args.keeps if args.keeps is not None else OFFICIAL_KEEPS.get(a, [15, 20]))]
    keeps = [k for k in keeps if k != a and 0 <= k < 40]
    class_idx = torch.tensor([a] + keeps).long().cuda()
    out_dir = os.path.join(args.out, str(a))
    print(f'AdaTrans for attribute {a}; keeps {keeps}')

    density_path = args.density or os.path.join(args.out, 'density.pth')
    if not os.path.exists(density_path):
        raise SystemExit(f'no {density_path}: run python training/train_adatrans.py density --out {args.out} first')
    dens = LatentDensity(style_dim=512, n_layer=10).cuda().eval()
    dens.load_state_dict(torch.load(density_path, map_location='cpu'))
    for p in dens.parameters():
        p.requires_grad_(False)

    ckpt = torch.load(args.stygan2_weights, map_location='cpu')
    G = Generator(size=1024, style_dim=512, n_mlp=8)
    G.load_state_dict(ckpt['g_ema'])
    G.cuda().eval()
    for p in G.parameters():
        p.requires_grad_(False)

    def render(w):
        img = G([w], input_is_latent=True, randomize_noise=False)[0]
        return F.interpolate(img, size=256, mode='bicubic', align_corners=True, antialias=True)

    clf = AttributeClassifier(backbone='r34')
    clf.load_state_dict(load_network(args.attribute_weights))
    clf.cuda().eval()
    for p in clf.parameters():
        p.requires_grad_(False)
    id_loss_fn = IDLoss(crop=True).cuda().eval()
    for p in id_loss_fn.parameters():
        p.requires_grad_(False)
    lpips_fn = lpips.LPIPS(net='alex').cuda().eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    model = AdaTrans(c_dim=1, max_steps=args.max_steps, n_layers=10, hid_dim=512, n_styles=18, style_dim=512).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    cfg = dict(vars(args), keeps=keeps, density=density_path,
               deviations={'id_loss': 'facenet VGGFace2 (common/id_loss.py), official: insightface iresnet34',
                           'keeps': 'no age (-1): the preds file has no age heads',
                           'lpips': 'pip lpips alex v0.1'})
    save_json(os.path.join(out_dir, 'config.json'), cfg)

    order = torch.randperm(len(W))
    cursor = 0
    t0 = time.time()
    opt.zero_grad()
    for it in range(1, args.iters + 1):
        if cursor + args.batch > len(W):          # whole batches of a reshuffled pass, as drop_last
            order, cursor = torch.randperm(len(W)), 0
        idx = order[cursor:cursor + args.batch]
        cursor += args.batch
        latents, sources = W[idx].cuda(), P[idx].cuda()
        targets = sources.clone()
        targets[:, a] = 1 - (sources[:, a] > 0.5).float()
        c = targets[:, [a]] * 2 - 1                  # official: 0 -> -1, 1 stays
        model.train()
        trajectory, step_size = model(latents, c)
        new_styles = trajectory[-1]
        n = args.syn_bs
        new_images = render(new_styles[:n])
        outs = clf(new_images)[0][:, :40]
        cls_loss = F.binary_cross_entropy_with_logits(outs[:, class_idx], targets[:n, class_idx])
        with torch.no_grad():
            target_images = render(latents[:n])
            e_src = F.normalize(id_loss_fn.extract_features(target_images), dim=1)
        e_new = F.normalize(id_loss_fn.extract_features(new_images), dim=1)
        id_loss = -(e_src * e_new).sum(1).mean()
        lpips_loss = lpips_fn(target_images, new_images).mean()
        nll_loss = dens(trajectory[1:].reshape(-1, 18, 512))[0]
        recon_loss = F.mse_loss(latents, new_styles)
        loss = (args.recon_loss_weight * recon_loss + args.cls_loss_weight * cls_loss + args.nll_loss_weight * nll_loss
                + args.id_loss_weight * id_loss + args.lpips_loss_weight * lpips_loss)
        (loss / args.acc_steps).backward()
        if it % args.acc_steps == 0:
            opt.step()
            opt.zero_grad()
        if it % 50 == 0:
            with torch.no_grad():
                p_edit = torch.sigmoid(outs[:, a])
                ok = torch.where(targets[:n, a] > 0.5, p_edit > 0.5, p_edit < 0.5).float().mean()
            print(f'[edit {a}] {it}/{args.iters}  recon {recon_loss.item():.4f}  cls {cls_loss.item():.4f}  '
                  f'nll {nll_loss.item():.4f}  id {id_loss.item():.4f}  lpips {lpips_loss.item():.4f}  '
                  f'step {step_size.sum(1).mean().item():.3f}  teacher-acc {ok.item():.2f}  [{time.time() - t0:.0f}s]')
        if it % args.save_freq == 0 or it == args.iters:
            torch.save(model.state_dict(), os.path.join(out_dir, 'model.pth'))
    print(f'saved {os.path.join(out_dir, "model.pth")}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('what', choices=['density', 'edit'])
    p.add_argument('--out', default='./output/baselines/adatrans')
    p.add_argument('--attr', type=int, help='edit: the CelebA id to train (one model per attribute)')
    p.add_argument('--keeps', nargs='*', type=int, default=None,
                   help='edit: attributes the classifier loss keeps at the source value '
                        '(default: the official choice for the attribute, without age)')
    p.add_argument('--density', default=None, help='edit: density.pth (default <out>/density.pth)')
    p.add_argument('--index_file', default='./data/ffhq.txt')
    p.add_argument('--image_root', default='data/FFHQ')
    p.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth')
    p.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth')
    p.add_argument('--stygan2_weights', default='./data/stylegan2-ffhq-config-f.pt')
    p.add_argument('--attribute_weights', default='./data/r34_a40_age_256_classifier.pth')
    p.add_argument('--iters', type=int, default=10000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--syn_bs', type=int, default=16, help='edit: latents rendered for the image losses per step')
    p.add_argument('--acc_steps', type=int, default=8)
    p.add_argument('--max_steps', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--recon_loss_weight', type=float, default=1.0)
    p.add_argument('--cls_loss_weight', type=float, default=1.0)
    p.add_argument('--nll_loss_weight', type=float, default=1.0)
    p.add_argument('--id_loss_weight', type=float, default=0.1)
    p.add_argument('--lpips_loss_weight', type=float, default=0.1)
    p.add_argument('--save_freq', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    if args.what == 'edit' and args.attr is None:
        raise SystemExit('edit needs --attr')
    W, P = load_train_split(args)
    if args.what == 'density':
        train_density(args, W)
    else:
        train_edit(args, W, P)


if __name__ == '__main__':
    main()
