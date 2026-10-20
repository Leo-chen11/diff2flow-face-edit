"""How much do the two BatchNorm train/eval mismatches cost a checkpoint? No
training; run it before deciding whether a training run that fixes them is
worth it. (The trainer has no flag for either fix yet: the flow side would set
MovingBatchNorm's update_stats=False, the conditioner side would keep its
ResNet's BatchNorm in eval mode while it trains.)

  flow  MovingBatchNorm1d (bn_lag=0) normalizes the forward pass with the
        running stats from BEFORE this batch's update, then updates them, and
        the reverse pass uses the UPDATED stats. Training encodes and decodes in
        train mode, so its round trip w -> z -> w' does not return w.
  cond  the conditioner's ResNet is trained in train mode, i.e. it reads each
        face with the statistics of its batch -- and training's batches are
        built half below / half above the score thresholds on one attribute,
        far from the data distribution. Eval reads with the running stats.

On attribute-balanced batches built like training's (half the faces below
--score_balance_low, half above --score_balance_high on the edited attribute,
--diag_batch faces per batch):

  flow     ||w' - w|| / ||w|| of the round trip under the SAME condition:
             eval          eval mode (what inference does)
             train_update  train mode, stats updating (every run so far)
             train_freeze  train mode, update_stats=False (the flow fix)
           and `offset_vs_edit`: the train_update offset ||w' - w|| over the
           norm of a real raw flow edit (eval, --diag_scale), i.e. how large
           the spurious part of training's flow_delta was next to the edit.
  cond     the conditioner's reading in train mode (batch stats) vs eval mode
           (running stats) on the same batch:
             abs_diff      mean |train - eval| on the edited attribute
             flip          share of faces whose reading falls on the other
                           side of 0.5 -- training edited them the other way
                           from eval (edit_is_rm = conditioner > 0.5)
             agree_eval / agree_train
                           share agreeing with the dataset preds' side of 0.5

Every train-mode pass restores the module's state afterwards, so nothing is
carried between batches and the checkpoint on disk is never touched.

    python scripts/diagnose_bn.py --checkpoint_dir output/SDFlow/<run> --step <N>
Output: <checkpoint_dir>/diagnose_bn_step<N>.json
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def set_flow_bn_update(flow, on):
    for m in flow.modules():
        if hasattr(m, 'update_stats'):
            m.update_stats = on


def snapshot(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


@torch.no_grad()
def round_trip(prior, latent, cond, mode):
    """Per-face ||w' - w|| for w -> z -> w' with the same condition."""
    state = snapshot(prior)
    try:
        if mode == 'eval':
            prior.eval()
        else:
            prior.train()
            set_flow_bn_update(prior, mode == 'train_update')
        zero = torch.zeros(latent.size(0), latent.size(1), 1, device=latent.device, dtype=latent.dtype)
        z, _ = prior(latent, cond, zero)
        w2, _ = prior(z, cond, zero, reverse=True)
        return (w2 - latent).reshape(latent.size(0), -1).norm(dim=1)
    finally:
        prior.load_state_dict(state)
        set_flow_bn_update(prior, True)
        prior.eval()


@torch.no_grad()
def cond_reading(conditioner, img, train):
    state = snapshot(conditioner)
    try:
        conditioner.train(train)
        read = conditioner.attr_estimator(img)
        return read[0] if isinstance(read, tuple) else read
    finally:
        conditioner.load_state_dict(state)
        conditioner.eval()


def balanced_batches(ds, global_idx, n_batches, batch, low, high, seed=0):
    """Index lists, each half low-score / half high-score on global_idx."""
    scores = torch.tensor([float(ds._lookup_precomputed(ds.preds, f)[global_idx])
                           for f in ds.image_list])
    g = torch.Generator().manual_seed(seed)
    lo = torch.nonzero(scores <= low).view(-1)
    hi = torch.nonzero(scores >= high).view(-1)
    lo, hi = lo[torch.randperm(len(lo), generator=g)], hi[torch.randperm(len(hi), generator=g)]
    n_lo, n_hi = batch // 2, batch - batch // 2
    n_batches = min(n_batches, len(lo) // max(n_lo, 1), len(hi) // max(n_hi, 1))
    return [lo[b * n_lo:(b + 1) * n_lo].tolist() + hi[b * n_hi:(b + 1) * n_hi].tolist()
            for b in range(n_batches)]


def main():
    import torchvision.transforms as T
    from evaluation.evaluate_sdflow import (
        ATTR_NAMES, _latest_step, apply_run_config, build_parser, edited_attr_value,
        load_models, resolve_controlnet_disable_attrs)
    from models.dataset import SDFlowDataset

    p = build_parser()
    p.add_argument('--diag_batch', type=int, default=8,
                   help='faces per batch; use the run\'s training --batch (BN statistics depend on it)')
    p.add_argument('--diag_batches', type=int, default=16, help='batches per attribute')
    p.add_argument('--diag_scale', type=float, default=0.9,
                   help='edit strength of the reference flow edit for offset_vs_edit')
    p.add_argument('--score_balance_low', type=float, default=0.35)
    p.add_argument('--score_balance_high', type=float, default=0.65)
    args = p.parse_args()
    if not os.path.isdir(args.checkpoint_dir):
        raise SystemExit(f'checkpoint_dir does not exist on this machine: {args.checkpoint_dir}')
    args = apply_run_config(args)
    args = resolve_controlnet_disable_attrs(args)
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)

    # load_models returns 8 values on this branch (+control_encoder), 7 after the cleanup
    prior, conditioner, _G, id_criterion = load_models(args)[:4]
    prior.eval()
    conditioner.eval()
    if not any(hasattr(m, 'update_stats') for m in prior.modules()):
        raise SystemExit('models/flows/normalization.py has no update_stats: update the code first')

    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)), T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file,
                       train=False, transform=tf)

    results = {'step': args.step, 'batch': args.diag_batch}
    for local_idx, global_idx in enumerate(args.attribute_index):
        name = ATTR_NAMES.get(global_idx, f'attr{global_idx}')
        acc = {k: [] for k in ('eval', 'train_update', 'train_freeze', 'w_norm', 'edit_norm',
                               'abs_diff', 'flip', 'agree_eval', 'agree_train')}
        for idxs in balanced_batches(ds, global_idx, args.diag_batches, args.diag_batch,
                                     args.score_balance_low, args.score_balance_high):
            items = [ds[i] for i in idxs]
            img = torch.stack([it[0] for it in items]).cuda()
            latent = torch.stack([it[1] for it in items]).cuda()
            pred = torch.stack([it[2] for it in items])[:, global_idx].float()

            with torch.no_grad():
                cond, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
            for mode in ('eval', 'train_update', 'train_freeze'):
                acc[mode].append(round_trip(prior, latent, cond, mode).cpu())
            acc['w_norm'].append(latent.reshape(latent.size(0), -1).norm(dim=1).cpu())

            # reference: a real raw flow edit at --diag_scale, eval mode
            with torch.no_grad():
                zero = torch.zeros(latent.size(0), latent.size(1), 1, device=latent.device)
                mid, _ = prior(latent, cond, zero)
                new_attr = attr_cond.clone()
                new_attr[:, local_idx] = edited_attr_value(attr_cond[:, local_idx], args.diag_scale,
                                                           global_idx)
                raw, _ = prior(mid, torch.cat([id_cond, new_attr], 1), zero, reverse=True)
            acc['edit_norm'].append((raw - latent).reshape(latent.size(0), -1).norm(dim=1).cpu())

            r_eval = cond_reading(conditioner, img, False)[:, local_idx].float().cpu()
            r_train = cond_reading(conditioner, img, True)[:, local_idx].float().cpu()
            acc['abs_diff'].append((r_train - r_eval).abs())
            acc['flip'].append(((r_train > 0.5) != (r_eval > 0.5)).float())
            acc['agree_eval'].append(((r_eval > 0.5) == (pred > 0.5)).float())
            acc['agree_train'].append(((r_train > 0.5) == (pred > 0.5)).float())

        if not acc['eval']:
            print(f'{name}: not enough low/high-score test faces for one batch, skipped')
            continue
        c = {k: torch.cat(v) for k, v in acc.items()}
        rel = {m: float((c[m] / c['w_norm']).mean()) for m in ('eval', 'train_update', 'train_freeze')}
        results[name] = {
            'n': int(c['eval'].numel()),
            'flow_roundtrip_rel': rel,
            'offset_vs_edit': float((c['train_update'] / c['edit_norm'].clamp(min=1e-8)).mean()),
            'offset_vs_edit_freeze': float((c['train_freeze'] / c['edit_norm'].clamp(min=1e-8)).mean()),
            'edit_norm': float(c['edit_norm'].mean()),
            'cond': {k: float(c[k].mean()) for k in ('abs_diff', 'flip', 'agree_eval', 'agree_train')},
        }

    print(f'\nstep {args.step}, batch {args.diag_batch}')
    print(f'{"attr":<12} {"n":>4}  {"rt_eval":>9} {"rt_upd":>9} {"rt_frz":>9}  '
          f'{"off/edit":>8} {"frz/edit":>8}  {"c_diff":>6} {"c_flip":>6} {"agr_ev":>6} {"agr_tr":>6}')
    for name, r in results.items():
        if not isinstance(r, dict):
            continue
        f, cd = r['flow_roundtrip_rel'], r['cond']
        print(f'{name:<12} {r["n"]:>4}  {f["eval"]:>9.2e} {f["train_update"]:>9.2e} '
              f'{f["train_freeze"]:>9.2e}  {r["offset_vs_edit"]:>8.3f} {r["offset_vs_edit_freeze"]:>8.3f}  '
              f'{cd["abs_diff"]:>6.3f} {cd["flip"]:>6.1%} {cd["agree_eval"]:>6.1%} {cd["agree_train"]:>6.1%}')

    out = os.path.join(args.checkpoint_dir, f'diagnose_bn_step{args.step}.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
