"""The original SDFlow recipe (Li et al., github.com/phil329/SDFlow, training/train_sdflow.py),
ported onto this repo's modules, for the "SDFlow retrained on the same attributes" baseline.

Nothing of the LAG-DOF / direction-bank / teacher / DDS machinery of training/train_sdflow.py is
here. What it trains is exactly the released script's:

    labeldist   an attribute estimator (AttributeEstimator, ImageNet-initialised ResNet) that reads
                the face and gives the label distribution the flow is conditioned on
    prior       the conditional CNF (models.flows.flow.cnf, original shared velocity field,
                condition = the label distribution only: no identity vector)

    loss = kd * MSE(labeldist(img), precomputed attribute reading)               # kd_loss
         + nll * negative log-likelihood of the W+ latent under the flow         # log_p2
         + reg * MSE(edited latent, source latent)                               # reg_loss
         + mi  * MSE(labeldist(G(edited latent)), the label distribution asked for)   # mi_loss

all four weights 1, Adam 1e-4, batch 8, 3 epochs, 256 px, seed 0, as the released defaults. Each
step edits ONE attribute PER SAMPLE (drawn per sample), set to a value drawn uniformly from [0, 1]
(models/flows/utils.py modify_one_attribute(mode='uniform') of the released code; this repo's
copy only has the whole-batch 'negative' mode, so the uniform rule is written out below).

Differences from the released script, none of which change what is trained:
  * --labeldist_backbone (default resnet50, as the released script; its released checkpoint was
    built with resnet34 -- scripts/editing_curves.py --method sdflow reads either)
  * --save_freq 5000 (released: 1000), --seed, wandb project / entity / mode as arguments
    (the released script has placeholders), config.json written next to the checkpoints
  * StyleGAN2's parameters get no gradient (the released script computes and discards them; the
    gradient to the edited latent, which is what trains the flow, is the same)
  * --image_root / --index_file default to this repo's data/FFHQ and data/ffhq.txt; the released
    one reads FFHQ pre-resized to 256 px, here T.Resize does it

Checkpoints are written as save_models/{prior,labeldist,optimizer}-<step:07d>, the names the
released editor and scripts/editing_curves.py --method sdflow read.

    python training/train_sdflow_original.py --attribute_index 15 20 39 31 5 --run_name sdflow_orig_5attr

Then, on the held-out faces:
    python -m scripts.editing_curves --checkpoint_dir <final run> --step <its step> \\
        --independent_attr_weights ./data/r50_celebahq_eval.pth --independent_attr_backbone r50 \\
        --method sdflow --sdflow_ckpt_dir output/SDFlow/sdflow_orig_5attr/save_models \\
        --sdflow_step <N> --sdflow_attrs 15 20 39 31 5 \\
        --scales 0 0.25 0.5 1 1.5 2 --sample_offset 500 --num_samples 500
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'stylegan2'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import optim
from torch.utils import data
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from models.flows.utils import standard_normal_logprob


def modify_one_attribute_uniform(attributes):
    """The released modify_one_attribute(mode='uniform'): every sample draws ITS OWN column and
    sets it to a value uniform in [0, 1] (the others keep their value). Returns (idx (B,),
    new_attributes (B, A)); new_attributes carries no gradient."""
    attributes = attributes.to(torch.float32)
    bs, columns = attributes.shape[:2]
    new_attributes = attributes.detach().clone()
    idx = torch.randint(0, columns, (bs,), device=attributes.device)
    new_attributes[torch.arange(bs, device=attributes.device), idx] = \
        torch.rand((bs,), dtype=torch.float32, device=attributes.device)
    return idx, new_attributes


def sdflow_losses(prior, labeldist, G, img, latent, attributes, img_size, weights):
    """One step of the released loss. img / latent / attributes are on the model's device.
    Returns (total, {name: tensor}); names are the released script's."""
    B = latent.size(0)
    zero_pad = torch.zeros(B, 18, 1).to(latent)

    label_dist = labeldist(img, latent)
    kd_loss = F.mse_loss(label_dist, attributes)

    approx21, delta_log_p2 = prior(latent, label_dist, zero_pad)
    # base distribution: standard normal
    approx2 = standard_normal_logprob(approx21).view(B, -1).sum(1, keepdim=True)
    delta_log_p2 = delta_log_p2.view(B, 18, 1).sum(1)
    log_p2 = -(approx2 - delta_log_p2).mean() / (18 * 512)

    _, new_label_dist = modify_one_attribute_uniform(label_dist)
    new_latents, _ = prior(approx21, new_label_dist, zero_pad, reverse=True)

    new_face_tensors = G([new_latents], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
    new_face_tensors = F.interpolate(new_face_tensors, (img_size, img_size))
    recon_label_dist = labeldist(new_face_tensors, new_latents)

    mi_loss = F.mse_loss(recon_label_dist, new_label_dist)
    reg_loss = F.mse_loss(new_latents, latent)

    loss = (weights['kd'] * kd_loss + weights['nll'] * log_p2
            + weights['reg'] * reg_loss + weights['mi'] * mi_loss)
    return loss, {'kd_loss': kd_loss, 'log_p2': log_p2, 'mi_loss': mi_loss, 'reg_loss': reg_loss}


@torch.no_grad()
def preview_grid(prior, G, test_label_dist, origin_latent, attributes, mid_latent, img_size=256):
    """The released preview: the reconstruction, then the face with each attribute flipped."""
    batchsize = origin_latent.shape[0]
    recon = G([origin_latent.squeeze(1)], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
    images = [torchvision.utils.make_grid(recon, nrow=1, normalize=True, value_range=(-1, 1))]
    groups = attributes.shape[-1]
    zero_padding = torch.zeros((batchsize, 18, 1)).to(origin_latent)
    for i in range(groups):
        new_label_dist = test_label_dist.detach().clone()
        new_label_dist[:, i] = 1 - attributes[:, i]
        new_latents, _ = prior(mid_latent, new_label_dist, zero_padding, reverse=True)
        tmp = G([new_latents], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
        images.append(torchvision.utils.make_grid(tmp, nrow=1, normalize=True, value_range=(-1, 1)))
    merge = torch.cat(images, dim=2)
    return to_pil_image(merge.cpu()).resize((img_size * (groups + 1), img_size * batchsize))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The original SDFlow trainer (released recipe)')
    parser.add_argument('--latent_file', default='./data/ffhq_e4e_latents.pth', type=str)
    parser.add_argument('--preds_file', default='./data/ffhq_e4e_preds.pth', type=str)
    parser.add_argument('--index_file', default='./data/ffhq.txt', type=str)
    parser.add_argument('--image_root', default='data/FFHQ', type=str)

    # model structure
    parser.add_argument('--flow_modules', type=str, default='512-512-512-512-512')
    parser.add_argument('--attribute_index', nargs='*', default=[15, 20, 39], type=int,
                        help='CelebA attribute ids; the released default is 15 20 39 (Eyeglasses, Male, Young)')
    parser.add_argument('--labeldist_backbone', default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'],
                        help='the attribute estimator. resnet50 is what the released training script builds; '
                             'its released checkpoint loads as resnet34. Only resnet50 is ImageNet-initialised '
                             'by models/attribute_estimator.py (the others start from scratch).')
    parser.add_argument('--stygan2_weights', default='./data/stylegan2-ffhq-config-f.pt', type=str)

    # naming, saving
    parser.add_argument('--model_name', default='SDFlow', type=str)
    parser.add_argument('--run_name', default='default', type=str)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=5000, help='the released script saves every 1000')
    parser.add_argument('--wandb_project', default='SDFlow', type=str)
    parser.add_argument('--wandb_entity', default=None, type=str)
    parser.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'])

    # training
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)

    # loss weights
    parser.add_argument('--nll_loss_weight', type=float, default=1)
    parser.add_argument('--reg_loss_weight', type=float, default=1)
    parser.add_argument('--kd_loss_weight', type=float, default=1)
    parser.add_argument('--mi_loss_weight', type=float, default=1)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.environ['WANDB_MODE'] = args.wandb_mode

    from common.loggerx import WANDBLoggerX
    from common.version import code_version
    from models.attribute_estimator import AttributeEstimator
    from models.dataset import SDFlowDataset
    from models.flows.flow import cnf
    from models.stylegan2.model import Generator

    args.code_version = code_version()
    print(f'** code version {args.code_version}, seed {args.seed}')
    save_root = os.path.join('./output', args.model_name, args.run_name)
    wandb_kwargs = dict(project=args.wandb_project, name='{}_{}'.format(args.model_name, args.run_name))
    if args.wandb_entity:
        wandb_kwargs['entity'] = args.wandb_entity
    logger = WANDBLoggerX(save_root=save_root, print_freq=args.print_freq, config=args, **wandb_kwargs)
    with open(os.path.join(save_root, 'config.json'), 'w') as _f:
        json.dump(vars(args), _f, indent=2, default=str)

    attribute_index = torch.tensor(args.attribute_index, dtype=int)
    # the released call: default velocity field (original), train_T=True, adjoint, dopri5
    prior = cnf(512, args.flow_modules, len(args.attribute_index), 1).cuda()

    transform = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)), T.Normalize(mean=0.5, std=0.5)])
    train_dataset = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                                  latents_file=args.latent_file, preds_file=args.preds_file,
                                  train=True, transform=transform)
    test_dataset = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                                 latents_file=args.latent_file, preds_file=args.preds_file,
                                 train=False, transform=transform)
    train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                                   num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_batch = args.batch if args.batch <= 8 else 8
    test_loader = data.DataLoader(test_dataset, shuffle=False, batch_size=test_batch,
                                  pin_memory=False, drop_last=True)
    print(f'** dataloader initialized: {len(train_dataset)} train faces, {len(train_loader)} steps per '
          f'epoch x {args.epochs} epochs = {len(train_loader) * args.epochs} steps')

    labeldist = AttributeEstimator(backbone=args.labeldist_backbone,
                                   attribute_dim=len(args.attribute_index)).cuda()
    optimizer = optim.Adam(list(prior.parameters()) + list(labeldist.parameters()), lr=args.lr)
    # names the checkpoints (loggerx.get_varname): prior-<step>, labeldist-<step>, optimizer-<step>
    logger.modules = [prior, labeldist, optimizer]

    ckpt = torch.load(args.stygan2_weights, map_location='cpu')
    G = Generator(size=1024, style_dim=512, n_mlp=8)
    G.load_state_dict(ckpt['g_ema'])
    G.cuda().eval()
    for p in G.parameters():
        p.requires_grad_(False)
    print('** StyleGAN2 model initialization success !')

    test_img, test_latent, test_pred = next(iter(test_loader))
    test_img, test_latent, test_pred = test_img.cuda(), test_latent.cuda(), test_pred.cuda()
    test_attributes = test_pred[:, attribute_index]

    weights = {'kd': args.kd_loss_weight, 'nll': args.nll_loss_weight,
               'reg': args.reg_loss_weight, 'mi': args.mi_loss_weight}
    for epoch in range(args.epochs):
        for i, datas in tqdm(enumerate(train_loader), total=len(train_loader)):
            n_iter = epoch * len(train_loader) + i
            img, latent, pred = datas
            img, latent, pred = img.cuda(), latent.cuda(), pred.cuda()
            attributes = pred[:, attribute_index]

            loss, parts = sdflow_losses(prior, labeldist, G, img, latent, attributes, args.img_size, weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if n_iter % args.save_freq == 0:
                with torch.no_grad():
                    test_label_dist = labeldist(test_img, test_latent)
                    test_mid_latent, _ = prior(test_latent, test_label_dist,
                                               torch.zeros(test_batch, 18, 1).to(test_latent))
                    logger.save_image(preview_grid(prior, G, test_label_dist, test_latent,
                                                   test_attributes, test_mid_latent), n_iter, 'test')
                logger.checkpoints(n_iter)

            logger.msg(parts, n_iter)

    # the last step the loop reached is not a multiple of save_freq in general
    if n_iter % args.save_freq != 0:
        logger.checkpoints(n_iter)
    print(f'** done: checkpoints in {save_root}/save_models (last step {n_iter})')
