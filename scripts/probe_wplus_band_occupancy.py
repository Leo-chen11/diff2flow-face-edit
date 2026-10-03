"""AXIS 2: which W+ SCALE does each attribute's edit actually live at?

This project has repeatedly reasoned about W+ layers by assertion -- "12-17
carry wrinkles/skin texture", "64x64 sits where mid-level structure lives",
"age needs fine layers, gender needs coarse" -- and then set flags
(--age_dds_fine_layer_start, --reg_fine_weight_min_override,
--controlnet_res) on the strength of those assertions. None of them was ever
MEASURED on this checkpoint's own learned directions. This script measures
it.

For each attribute and each direction it computes the W+ delta that the
editing path actually produces (flow reverse pass -> direction bank ->
guided_delta, exactly as evaluation/evaluate_sdflow.py's
edit_single_attribute computes it, minus the render), and reports how that
delta's energy is distributed over the 18 W+ layers and over the StyleGAN2
resolutions those layers drive.

WHY IT MATTERS. The plan this was written for -- deriving a DC-ControlNet
style "element" decomposition automatically instead of hand-writing one per
attribute -- has two halves: WHERE an attribute manifests (spatial, see
scripts/precompute_attr_saliency.py) and AT WHAT SCALE (this script). The
content condition for an attribute is then "the statistics of this
attribute's population, inside the where, at the what-scale". If the
measurement here does NOT separate the attributes -- if Male and Young
occupy the same bands -- then the scale axis carries no information and
that half of the plan is dead, for the cost of one script and no training.

IT ALSO AUDITS A MECHANISM YOU ALREADY HAVE. models/control_encoder.py's
AttributeControlEncoder already keeps a LEARNABLE gain per (slot,
resolution) -- log_gain is shaped (num_slots, len(out_res)) -- so the model
can already decide "Young leans on 256, Male leans on 64" by itself. This
script prints those learned gains next to the measured W+ occupancy. Three
readings:

  they agree      -> the injection is being spent where the W+ edit already
                     concentrates; the scale axis is real and already
                     exploited, so conditioning on it adds little.
  they disagree   -> the injection is fighting the W+ path rather than
                     complementing it. That is a finding on its own and
                     probably worth a flag before any new mechanism.
  gains are flat  -> nothing taught them to differ; an explicit scale prior
                     has something to add.

Nothing here trains, renders, or writes a checkpoint. Read-only.

Usage:
    python -m scripts.probe_wplus_band_occupancy \
        --checkpoint_dir ./output/SDFlow/substyle3_glasses_v24 --step 85000 \
        --num_samples 200 --edit_scale 1.0
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'stylegan2'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torchvision.transforms as T
from torch.utils import data

from evaluation.evaluate_sdflow import (
    ATTR_NAMES, _latest_step, apply_run_config, is_clear, load_models,
    resolve_controlnet_disable_attrs,
)
from models.dataset import SDFlowDataset


def layer_resolution_map(n_latent=18, max_res=1024):
    """W+ index -> the StyleGAN2 resolution that index modulates.

    Read off models/stylegan2/model.py's Generator.forward() rather than
    assumed:

        out  = self.conv1(out, latent[:, 0])          # 4x4
        skip = self.to_rgb1(out, latent[:, 1])        # 4x4
        i = 1
        for conv1, conv2, to_rgb in ...:              # 8, 16, ..., 1024
            out  = conv1(out,  latent[:, i])
            out  = conv2(out,  latent[:, i + 1])
            skip = to_rgb(out, latent[:, i + 2])
            i += 2

    so index 0 drives 4x4, and for index j >= 1 the block at resolution
    2^(3 + (j-1)//2) is the one it modulates. Note to_rgb's index collides
    with the NEXT block's conv1 index (j = i + 2 = (i + 2)), which is why
    the last index runs one block past the top and has to be clamped: it is
    the final to_rgb, which stays at max_res.
    """
    res = []
    for j in range(n_latent):
        r = 4 if j == 0 else min(2 ** (3 + (j - 1) // 2), max_res)
        res.append(r)
    return res


@torch.no_grad()
def edit_delta_wplus(prior, direction_bank, latent, attr_cond, id_cond,
                     attr_local_idx, edit_scale, bypass_bank=False):
    """The W+ displacement an edit produces -- the latent half of
    evaluate_sdflow.edit_single_attribute, copied step for step and stopping
    before the generator call.

    Deliberately NOT a call into edit_single_attribute: that function returns
    a rendered image, and the quantity this script needs (new_latents -
    latent) is not on its way out. Keeping the arithmetic identical is what
    makes the measurement describe the real editing path rather than a
    plausible-looking reimplementation of it, so any change to that function's
    latent path has to be mirrored here.
    """
    B = latent.size(0)
    zero_pad = torch.zeros(B, 18, 1, device=latent.device)

    src_cond = torch.cat([id_cond, attr_cond], dim=1)
    mid_latent, _ = prior(latent, src_cond, zero_pad)

    new_attr_cond = attr_cond.clone()
    src = attr_cond[:, attr_local_idx]
    new_attr_cond[:, attr_local_idx] = src * (1.0 - edit_scale) + (1.0 - src) * edit_scale
    new_cond = torch.cat([id_cond, new_attr_cond], dim=1)

    new_latents_raw, _ = prior(mid_latent, new_cond, zero_pad, reverse=True)

    if direction_bank is not None and not bypass_bank:
        flow_delta = new_latents_raw - latent
        attr_delta = new_attr_cond - attr_cond
        batch_attr_idx = torch.full((B,), attr_local_idx,
                                    device=latent.device, dtype=torch.long)
        guided_delta = direction_bank(flow_delta, attr_delta,
                                      attr_idx=batch_attr_idx, latent=latent)
        return guided_delta
    return new_latents_raw - latent


def learned_gain_table(control_encoder, attribute_index):
    """{(attr_abs, 'add'|'rm'): {res: gain}} from the encoder's own log_gain.

    Slot layout mirrors _PerDirectionSlots.slot_index: attr-major, add then
    rm, when per_direction is on; one slot per attribute otherwise (both
    directions then report the same number, which is the honest reading --
    they share the parameter).
    """
    if control_encoder is None or not hasattr(control_encoder, 'log_gain'):
        return {}
    with torch.no_grad():
        gains = torch.nn.functional.softplus(control_encoder.log_gain).cpu()
    out_res = getattr(control_encoder, 'out_res', None)
    if out_res is None:
        return {}
    per_dir = getattr(control_encoder, 'per_direction', False)
    table = {}
    for local_idx, attr_abs in enumerate(attribute_index):
        for d, is_rm in (('add', 0), ('rm', 1)):
            slot = local_idx * 2 + is_rm if per_dir else local_idx
            if slot >= gains.size(0):
                continue
            table[(attr_abs, d)] = {r: gains[slot, i].item()
                                    for i, r in enumerate(out_res)}
    return table


def main(args):
    prior, conditioner, G, id_criterion, attr_teacher, \
        attribute_index, direction_bank, control_encoder = load_models(args)

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

    n_latent = G.n_latent
    res_of_layer = layer_resolution_map(n_latent, max_res=2 ** G.log_size)

    # {(attr_abs, dir): [sum of per-sample normalized energy over layers]}
    acc = {}
    counts = {}
    for attr_abs in args.attribute_index:
        for d in ('add', 'rm'):
            acc[(attr_abs, d)] = torch.zeros(n_latent, dtype=torch.float64)
            counts[(attr_abs, d)] = 0

    seen = 0
    for img, latent, _pred in loader:
        if seen >= args.num_samples:
            break
        img = img.cuda()
        latent = latent.cuda()
        _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)

        used = False
        for local_idx, attr_abs in enumerate(args.attribute_index):
            c = attr_cond[0, local_idx].item()
            # Same gate every probe in this project uses: skip samples the
            # conditioner itself is unsure about, so the measured direction
            # is not an average of two opposite edits.
            if not is_clear(c):
                continue
            direction = 'rm' if c >= 0.5 else 'add'
            bypass = args.bypass_glasses_direction_bank and attr_abs == 15
            delta = edit_delta_wplus(
                prior, direction_bank, latent, attr_cond, id_cond,
                local_idx, args.edit_scale, bypass_bank=bypass)

            energy = delta[0].pow(2).mean(dim=1).double().cpu()   # (18,)
            total = energy.sum()
            if total <= 0:
                continue
            # Normalize PER SAMPLE before accumulating: without this the
            # average is dominated by whichever faces happen to get a large
            # edit, and the question here is about the SHAPE of the
            # distribution over layers, not its magnitude.
            acc[(attr_abs, direction)] += energy / total
            counts[(attr_abs, direction)] += 1
            used = True
        if used:
            seen += 1

    print(f'\nsamples used: {seen}   edit_scale={args.edit_scale}   '
          f'n_latent={n_latent}')
    print('W+ delta energy share per layer (per-sample normalized, then averaged)\n')

    gain_table = learned_gain_table(control_encoder, args.attribute_index)

    # Per-layer table
    header = f'{"layer":>5} {"res":>6} |'
    cols = []
    for attr_abs in args.attribute_index:
        for d in ('add', 'rm'):
            if counts[(attr_abs, d)] == 0:
                continue
            cols.append((attr_abs, d))
            header += f' {ATTR_NAMES.get(attr_abs, attr_abs)}-{d:<3}'
    print(header)
    print('-' * len(header))
    for j in range(n_latent):
        line = f'{j:>5} {res_of_layer[j]:>6} |'
        for key in cols:
            share = (acc[key][j] / max(counts[key], 1)).item()
            line += f' {share:>10.3f}'
        print(line)

    # Resolution-band rollup -- the number that matters for --controlnet_res
    print('\nrolled up to RESOLUTION bands:\n')
    bands = sorted(set(res_of_layer))
    header = f'{"res":>6} |'
    for key in cols:
        header += f' {ATTR_NAMES.get(key[0], key[0])}-{key[1]:<3}'
    print(header)
    print('-' * len(header))
    for r in bands:
        idxs = [j for j in range(n_latent) if res_of_layer[j] == r]
        line = f'{r:>6} |'
        for key in cols:
            share = sum((acc[key][j] / max(counts[key], 1)).item() for j in idxs)
            line += f' {share:>10.3f}'
        print(line)

    # This project's own group boundaries (reg_loss_global/coarse/fine in
    # training/train_sdflow.py), so the measurement can be read against the
    # flags that already exist rather than only in the abstract.
    print('\nrolled up to THIS PROJECT\'S reg_loss groups:')
    groups = [('global  [0:2]', range(0, 2)),
              ('coarse  [2:4]', range(2, 4)),
              ('fine    [4:] ', range(4, n_latent))]
    header = f'{"group":>14} |'
    for key in cols:
        header += f' {ATTR_NAMES.get(key[0], key[0])}-{key[1]:<3}'
    print(header)
    print('-' * len(header))
    for name, rng in groups:
        line = f'{name:>14} |'
        for key in cols:
            share = sum((acc[key][j] / max(counts[key], 1)).item() for j in rng)
            line += f' {share:>10.3f}'
        print(line)

    if gain_table:
        print('\nControlNet LEARNED gains per (slot, resolution) -- compare '
              'against the resolution table above:')
        res_list = sorted(next(iter(gain_table.values())).keys())
        header = f'{"attr-dir":>16} |' + ''.join(f' {r:>10}' for r in res_list)
        print(header)
        print('-' * len(header))
        for key in cols:
            g = gain_table.get(key)
            if g is None:
                continue
            name = f'{ATTR_NAMES.get(key[0], key[0])}-{key[1]}'
            print(f'{name:>16} |' + ''.join(f' {g[r]:>10.3f}' for r in res_list))
        if not getattr(control_encoder, 'per_direction', False):
            print('  NOTE: per_direction is OFF -- add and rm SHARE one gain '
                  'parameter, so their rows are identical by construction, '
                  'not by measurement.')
    else:
        print('\n(no control_encoder gains to report -- run has no ControlNet '
              'injection, or its checkpoint did not load)')

    print('\nHOW TO READ THIS')
    print('  The scale axis carries information only if the columns DIFFER.')
    print('  Two columns with the same shape mean W+ spends those two edits')
    print('  at the same scales, and conditioning on scale cannot separate')
    print('  them -- whatever the prior assumption was.')
    print('  Compare the learned-gain table against the resolution table:')
    print('  agreement means the injection already follows the W+ edit;')
    print('  disagreement means it is pulling somewhere else, which is a')
    print('  finding in its own right.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--step', type=int, default=None)
    p.add_argument('--num_samples', type=int, default=200)
    p.add_argument('--edit_scale', type=float, default=1.0)

    p.add_argument('--index_file',   default='./data/ffhq.txt')
    p.add_argument('--image_root',   default='data/FFHQ')
    p.add_argument('--latent_file',  default='./data/ffhq_e4e_latents.pth')
    p.add_argument('--preds_file',   default='./data/ffhq_e4e_preds.pth')
    p.add_argument('--stygan2_weights', default='./data/stylegan2-ffhq-config-f.pt')
    p.add_argument('--attribute_weights', default='./data/r34_a40_age_256_classifier.pth')
    p.add_argument('--direction_bank_path', default=None)

    p.add_argument('--img_size',         type=int,   default=512)
    p.add_argument('--attribute_index',  nargs='*',  type=int, default=[15, 20, 39])
    p.add_argument('--flow_modules',     default='512-512-512-512-512')
    p.add_argument('--num_blocks',       type=int,   default=1)
    p.add_argument('--velocity_field',   default='lag_dof')
    p.add_argument('--id_cond_dim',      type=int,   default=32)
    p.add_argument('--id_cond_scale',    type=float, default=0.25)
    p.add_argument('--attr_backbone',    default='resnet50')
    p.add_argument('--conditioner_backbone', default='resnet',
                   choices=['resnet', 'clip', 'resnet_clip'])
    p.add_argument('--clip_model', default='ViT-B/32')
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
    p.add_argument('--controlnet_embed_res', type=int, default=64)
    p.add_argument('--controlnet_channels', type=int, default=512)
    p.add_argument('--controlnet_hidden_dim', type=int, default=256)
    p.add_argument('--controlnet_disable_attrs', nargs='*', type=int, default=None)
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
