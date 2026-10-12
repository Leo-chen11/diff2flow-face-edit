"""How closely does the flow's own edit follow the direction bank, per attribute,
and how face-specific is it? Runs on saved checkpoints, so it works on runs
that finished before per-attribute logging existed.

For each checkpoint step and edited attribute, on test faces:
  cos      cos(flow_delta, dir_delta)        flow edit vs the bank's linear edit
  cos2     cos^2                             share of the flow edit's energy along it
  face_flow  mean pairwise cos of flow_delta between different faces (within
             add edits and within remove edits; the two point opposite ways)
  face_bank  the same for dir_delta          (a shared linear direction -> ~1)
  |dW|     mean flow edit norm
face_flow well below face_bank = the flow edits each face differently.

    python scripts/flow_bank_alignment.py --checkpoint_dir output/SDFlow/<run> \
        --steps 60000 65000 70000 75000 80000
Output: <checkpoint_dir>/flow_bank_alignment.json
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def grouped_pairwise_cos(x, sign):
    """Mean pairwise cos within the add group and within the remove group,
    weighted by pair count: add and remove edits point opposite ways, so
    mixing them would cancel even for a perfectly shared direction."""
    tot, pairs = 0.0, 0
    for g in (sign > 0, sign <= 0):
        n = int(g.sum())
        if n >= 2:
            tot += mean_pairwise_cos(x[g]) * n * (n - 1)
            pairs += n * (n - 1)
    return tot / pairs if pairs else float('nan')


def mean_pairwise_cos(x):
    """x: (N, D). Mean cosine over all pairs i != j."""
    u = F.normalize(x.reshape(x.size(0), -1).float(), dim=1)
    n = u.size(0)
    if n < 2:
        return float('nan')
    s = u.sum(0)
    return float((s @ s - n) / (n * (n - 1)))


@torch.no_grad()
def measure(prior, bank, batches, names, global_idx, scale, target_fn):
    out = {}
    for a, name in enumerate(names):
        flows, dirs, signs = [], [], []
        for latent, id_cond, attr_cond in batches:
            B = latent.size(0)
            zero = torch.zeros(B, latent.size(1), 1, device=latent.device, dtype=latent.dtype)
            src_cond = torch.cat([id_cond, attr_cond], dim=1)
            mid, _ = prior(latent, src_cond, zero)
            new_attr = attr_cond.clone()
            new_attr[:, a] = target_fn(attr_cond[:, a], scale, global_idx[a])
            new_raw, _ = prior(mid, torch.cat([id_cond, new_attr], dim=1), zero, reverse=True)
            flow_delta = new_raw - latent
            idx = torch.full((B,), a, device=latent.device, dtype=torch.long)
            bank._id_cond = id_cond
            bank(flow_delta, new_attr - attr_cond, attr_idx=idx, latent=latent, route_scores=attr_cond)
            signs.append((new_attr[:, a] - attr_cond[:, a]).float().cpu())
            flows.append(flow_delta.float().cpu())
            dirs.append(bank._last_dir_delta.float().cpu())
        fl, di, sg = torch.cat(flows), torch.cat(dirs), torch.cat(signs)
        cos = F.cosine_similarity(fl.reshape(fl.size(0), -1), di.reshape(di.size(0), -1), dim=1)
        out[name] = {
            'cos': float(cos.mean()), 'cos2': float((cos ** 2).mean()),
            'face_flow': grouped_pairwise_cos(fl, sg), 'face_bank': grouped_pairwise_cos(di, sg),
            'flow_norm': float(fl.reshape(fl.size(0), -1).norm(dim=1).mean()),
            'n': int(fl.size(0)),
        }
    return out


def main():
    import torchvision.transforms as T
    from torch.utils import data
    from tqdm import tqdm
    from evaluation.evaluate_sdflow import (
        ATTR_NAMES, _latest_step, apply_run_config, build_parser, edited_attr_value,
        load_models, resolve_controlnet_disable_attrs)
    from models.dataset import SDFlowDataset

    p = build_parser()
    p.add_argument('--steps', nargs='+', type=int, default=None,
                   help='checkpoint steps to measure (default: --step, else the latest)')
    p.add_argument('--align_faces', type=int, default=128)
    p.add_argument('--align_scale', type=float, default=0.9)
    args = p.parse_args()
    if not os.path.isdir(args.checkpoint_dir):
        raise SystemExit(f'checkpoint_dir does not exist on this machine: {args.checkpoint_dir}')
    args = apply_run_config(args)
    args = resolve_controlnet_disable_attrs(args)
    steps = args.steps or [args.step if args.step is not None else _latest_step(args.checkpoint_dir)]

    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)), T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file,
                       train=False, transform=tf)
    loader = data.DataLoader(ds, shuffle=False, batch_size=args.batch, num_workers=4)
    raw, seen = [], 0
    for img, latent, _ in loader:
        if seen >= args.align_faces:
            break
        raw.append((img, latent))
        seen += img.size(0)

    names = [ATTR_NAMES.get(i, f'attr{i}') for i in args.attribute_index]
    results = {}
    for step in steps:
        args.step = step
        prior, conditioner, _G, id_criterion, _t, _ai, bank, _ctrl = load_models(args)
        if bank is None:
            raise SystemExit('this run has no direction bank; nothing to compare against')
        batches = []
        with torch.no_grad():
            for img, latent in tqdm(raw, desc=f'conditions @ {step}'):
                img, latent = img.cuda(), latent.cuda()
                # the conditioner is trained with the flow, so rebuild per checkpoint
                _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
                batches.append((latent, id_cond, attr_cond))
        res = measure(prior, bank, batches, names, list(args.attribute_index), args.align_scale,
                      lambda src, scale, gidx: edited_attr_value(src, scale, gidx))
        results[str(step)] = res
        print(f'\nstep {step}  ({res[names[0]]["n"]} faces, scale {args.align_scale:g})')
        print(f'   {"":<12}{"cos":>7}{"cos^2":>7}{"face_flow":>11}{"face_bank":>11}{"|dW|":>8}')
        for n in names:
            r = res[n]
            print(f'   {n:<12}{r["cos"]:7.3f}{r["cos2"]:7.3f}{r["face_flow"]:11.3f}'
                  f'{r["face_bank"]:11.3f}{r["flow_norm"]:8.2f}')
        del prior, conditioner, _G, id_criterion, _t, bank, _ctrl
        torch.cuda.empty_cache()

    path = args.out_json or os.path.join(args.checkpoint_dir, 'flow_bank_alignment.json')
    with open(path, 'w') as f:
        json.dump({'checkpoint_dir': args.checkpoint_dir, 'scale': args.align_scale,
                   'results': results}, f, indent=2)
    print(f'\nsaved {path}')


if __name__ == '__main__':
    main()
