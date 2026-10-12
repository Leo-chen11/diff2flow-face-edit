"""Does the LAG-DOF layer gate do anything attribute-specific?

LAG-DOF writes the CNF velocity as  g(w, t, c) * v_attr + v_base,  with g a
per-layer gate in [0, 1]^18. The design claim is that the gate learns WHICH
StyleGAN layers each attribute edit should use. This script checks that on a
trained checkpoint, using the flow alone (no StyleGAN, no direction bank, no
ControlNet), in four ways:

  1. Gate profile    mean decode-pass gate per (edited attribute, layer), the
                     spread across attributes / faces / layers, and the share of
                     gates that are saturated (> 0.9 or < 0.1).
  2. Edit response   |gate(edit) - gate(no edit)|: how much the gate moves when
                     the target condition changes (same face, same latent).
  3. Time profile    mean gate per ODE-time bin.
  4. Frozen gate     re-run the same edits with the gate replaced by (a) its own
                     per-layer mean, (b) 1.0, and measure how much the W+ edit
                     changes. This is the functional test: if freezing it does
                     not change the edit, the gate is not doing the work.
The frozen-gate numbers are read against a noise floor: the same edit run twice
with the trained gate (the ODE solver's tolerance alone makes runs differ a bit).
Also reports where the flow's W+ edit itself lands (coarse 0-3 / mid 4-9 /
fine 10-17 layers) and the encode->decode round-trip error, so the edit size
can be trusted against numerical error.

Gate values are read from the solver's function evaluations, tagged encode or
decode, and averaged uniformly over ODE time (adaptive solvers take unequal
steps, so a plain mean over evaluations would weight some times more).

Usage (same checkpoint / data flags as evaluate_sdflow.py; config.json is read):
    python scripts/diagnose_gate.py --checkpoint_dir output/SDFlow/<run> --step 80000 \
        --diag_faces 64 --diag_scale 0.9

Thresholds in the printed verdict are rules of thumb, not tests: read the
numbers. Output: <checkpoint_dir>/gate_diag_s<step>/gate_diag.json (+ .png).
"""
import argparse
import json
import os
import sys
import warnings

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models', 'stylegan2'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from models.flows.odefunc import LAGDOFODEnet

LAYER_GROUPS = {'coarse 0-3': (0, 4), 'mid 4-9': (4, 10), 'fine 10-17': (10, 18)}


class GateTap:
    """Tags and records every layer-gate evaluation of the flow, and can freeze
    the gate to a constant. Patches the gate modules' and the flow's forward on
    the instances; restore() undoes it."""

    def __init__(self, prior):
        self.gates = [m.layer_gate for m in prior.modules()
                      if isinstance(m, LAGDOFODEnet) and m.mode in ('lag', 'lag_dof')]
        if not self.gates:
            raise ValueError('this flow has no active LayerGate (velocity_field must be '
                             "'lag' or 'lag_dof'); there is nothing to diagnose")
        self.prior = prior
        self.mode = 'normal'        # normal | const | open
        self.const = None           # (num_layers,) used when mode == 'const'
        self.on = False
        self.reverse = False
        self.rec = []               # (gate idx, reverse, t, gate (B, L))
        self._gate_fwd = [g.forward for g in self.gates]
        self._prior_fwd = prior.forward
        for i, g in enumerate(self.gates):
            g.forward = self._wrap_gate(i, self._gate_fwd[i])
        prior.forward = self._wrap_prior

    def _wrap_gate(self, idx, orig):
        def fwd(context, y):
            out = orig(context, y)                      # (B, L, 1)
            if self.mode == 'const':
                out = self.const.to(out).view(1, -1, 1).expand_as(out)
            elif self.mode == 'open':
                out = torch.ones_like(out)
            elif self.on:
                self.rec.append((idx, self.reverse, float(context[0, 0].detach()),
                                 out.detach().float().cpu()[..., 0]))
            return out
        return fwd

    def _wrap_prior(self, *args, **kwargs):
        # SequentialFlow.forward(x, context, logpx=None, reverse=False, ...)
        self.reverse = bool(kwargs.get('reverse', args[3] if len(args) > 3 else False))
        return self._prior_fwd(*args, **kwargs)

    def take(self):
        rec, self.rec = self.rec, []
        return rec

    def restore(self):
        for g, f in zip(self.gates, self._gate_fwd):
            g.forward = f
        self.prior.forward = self._prior_fwd


def time_binned(rec, reverse, n_bins):
    """(B, n_bins, L) mean gate per normalised-ODE-time bin (NaN for empty bins)."""
    rs = [r for r in rec if r[1] == reverse]
    if not rs:
        raise RuntimeError('no gate evaluations were recorded for this pass')
    t = np.array([r[2] for r in rs], dtype=np.float64)
    idx = np.minimum((t / max(t.max(), 1e-8) * n_bins).astype(int), n_bins - 1)
    G = torch.stack([r[3] for r in rs])                 # (E, B, L)
    out = torch.full((G.size(1), n_bins, G.size(2)), float('nan'))
    for b in range(n_bins):
        m = torch.from_numpy(idx == b)
        if m.any():
            out[:, b] = G[m].mean(0)
    return out


def _uniform(binned):
    """(B, L) gate averaged uniformly over time bins."""
    return torch.nanmean(binned, dim=1)


def _unit(v):
    return v.reshape(v.size(0), -1)


@torch.no_grad()
def run_diagnostic(prior, batches, names, global_idx, scale, target_fn, n_bins=5, log=print):
    """batches: list of (latent (B,L,512), id_cond (B,id), attr_cond (B,K)) tensors on
    the flow's device. target_fn(src (B,), scale, attr_global_idx) -> target value."""
    tap = GateTap(prior)
    K = len(names)
    try:
        # ── Phase 1: the trained gate, decode pass per edited attribute ─────
        gdec = [[] for _ in range(K)]       # (B, L) uniform-in-time
        bdec = [[] for _ in range(K)]       # (B, n_bins, L)
        delta = [[] for _ in range(K)]      # (B, L, 512) flow's W+ edit
        gno, genc, rt = [], [], []
        tap.mode, tap.on = 'normal', True
        for latent, id_cond, attr_cond in batches:
            zero = torch.zeros(latent.size(0), latent.size(1), 1, device=latent.device,
                               dtype=latent.dtype)
            src_cond = torch.cat([id_cond, attr_cond], dim=1)
            tap.take()
            mid, _ = prior(latent, src_cond, zero)
            genc.append(_uniform(time_binned(tap.take(), False, n_bins)))
            back, _ = prior(mid, src_cond, zero, reverse=True)       # no edit
            gno.append(_uniform(time_binned(tap.take(), True, n_bins)))
            rt.append(back - latent)
            for a in range(K):
                new_attr = attr_cond.clone()
                new_attr[:, a] = target_fn(attr_cond[:, a], scale, global_idx[a])
                new_cond = torch.cat([id_cond, new_attr], dim=1)
                new_raw, _ = prior(mid, new_cond, zero, reverse=True)
                binned = time_binned(tap.take(), True, n_bins)
                bdec[a].append(binned)
                gdec[a].append(_uniform(binned))
                delta[a].append((new_raw - latent).cpu())
        tap.on = False
        gdec = [torch.cat(x).numpy() for x in gdec]
        bdec = [torch.cat(x).numpy() for x in bdec]
        delta = [torch.cat(x) for x in delta]
        gno = torch.cat(gno).numpy()
        genc = torch.cat(genc).numpy()
        rt = torch.cat(rt).cpu()
        n_faces = gno.shape[0]

        # the frozen constant: per-layer mean over every recorded gate value
        const = torch.from_numpy(np.concatenate(gdec + [genc]).mean(0)).float()

        # ── Phase 2: same edits with the gate frozen ────────────────────────
        frozen = {}
        for mode in ('repeat', 'const', 'open'):
            # 'repeat' = trained gate again: the solver-noise floor to compare against
            tap.mode, tap.const = ('normal' if mode == 'repeat' else mode), const
            deltas = [[] for _ in range(K)]
            for latent, id_cond, attr_cond in batches:
                zero = torch.zeros(latent.size(0), latent.size(1), 1, device=latent.device,
                                   dtype=latent.dtype)
                src_cond = torch.cat([id_cond, attr_cond], dim=1)
                mid, _ = prior(latent, src_cond, zero)
                for a in range(K):
                    new_attr = attr_cond.clone()
                    new_attr[:, a] = target_fn(attr_cond[:, a], scale, global_idx[a])
                    new_raw, _ = prior(mid, torch.cat([id_cond, new_attr], dim=1), zero, reverse=True)
                    deltas[a].append((new_raw - latent).cpu())
            frozen[mode] = [torch.cat(x) for x in deltas]
        tap.mode = 'normal'
    finally:
        tap.restore()

    # ── Statistics ──────────────────────────────────────────────────────────
    L = gdec[0].shape[1]
    prof = np.stack([g.mean(0) for g in gdec])                    # (K, L)
    sens = np.stack([np.abs(gdec[a] - gno).mean(0) for a in range(K)])   # (K, L)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)     # a time bin may have no solver step
        time_profile = np.nanmean(np.stack(bdec), axis=(0, 1, 3)).tolist()
        time_spread = [float(np.nanstd(np.nanmean(np.stack(bdec)[:, :, b], axis=(0, 1))))
                       for b in range(bdec[0].shape[1])]
    out = {
        'n_faces': int(n_faces), 'scale': float(scale), 'names': list(names),
        'const_gate': const.tolist(),
        'profile': prof.tolist(),
        'gate_min': float(prof.min()), 'gate_max': float(prof.max()),
        'range': float(prof.max() - prof.min()),
        'spread_attr': float(prof.std(0).mean()),                  # across attributes
        'spread_layer': float(prof.mean(0).std()),                 # across the 18 layers
        'spread_face': float(np.mean([g.std(0).mean() for g in gdec])),
        'edit_response': float(sens.mean()),
        'edit_response_per_attr': sens.mean(1).tolist(),
        'frac_open': float(np.mean([(g > 0.9).mean() for g in gdec])),
        'frac_closed': float(np.mean([(g < 0.1).mean() for g in gdec])),
        'time_profile': time_profile,                              # per bin
        'time_layer_spread': time_spread,
    }
    norm = [_unit(d).norm(dim=1) for d in delta]
    out['delta_norm'] = [float(n.mean()) for n in norm]
    out['roundtrip_rel'] = [float((_unit(rt).norm(dim=1) / n.clamp(min=1e-8)).mean()) for n in norm]
    energy = [(d ** 2).sum(-1).mean(0).numpy() for d in delta]             # (L,) per attr
    out['energy_share'] = {
        names[a]: {gname: float(energy[a][lo:hi].sum() / max(energy[a].sum(), 1e-12))
                   for gname, (lo, hi) in LAYER_GROUPS.items()} for a in range(K)}
    for mode in ('repeat', 'const', 'open'):
        rel, cos = [], []
        for a in range(K):
            diff = _unit(delta[a] - frozen[mode][a]).norm(dim=1)
            rel.append(float((diff / norm[a].clamp(min=1e-8)).mean()))
            cos.append(float(torch.nn.functional.cosine_similarity(
                _unit(delta[a]), _unit(frozen[mode][a]), dim=1).mean()))
        out[f'frozen_{mode}_rel_change'] = rel
        out[f'frozen_{mode}_cos'] = cos
    noise = max(out['frozen_repeat_rel_change'])
    out['noise_floor'] = float(noise)
    effect = max(out['frozen_const_rel_change'])

    out['verdict'] = {
        'gate_varies_with_attribute': ('NO' if out['spread_attr'] < 0.02 and out['edit_response'] < 0.02
                                       else 'YES'),
        'gate_prefers_some_layers': 'NO' if out['spread_layer'] < 0.03 else 'YES',
        # freezing must change the edit by clearly more than re-running it does
        'gate_matters_for_the_edit': (
            'NO' if effect < max(0.05, 3 * noise) else
            'MINOR' if effect < max(0.15, 3 * noise) else 'YES'),
    }
    report(out, prof, sens, log)
    out['_arrays'] = {'bdec': bdec}
    return out


def report(o, prof, sens, log=print):
    names, L = o['names'], prof.shape[1]
    log(f'\nGate diagnostic: {o["n_faces"]} faces, edit scale {o["scale"]:g}, decode pass, '
        f'gate averaged uniformly over ODE time.\n')
    log('1. Mean gate per edited attribute x layer')
    log('   ' + ' ' * 12 + ' '.join(f'{l:>4d}' for l in range(L)))
    for a, n in enumerate(names):
        log(f'   {n:<12}' + ' '.join(f'{v:4.2f}' for v in prof[a]))
    log(f'   range {o["gate_min"]:.3f}..{o["gate_max"]:.3f} (spread {o["range"]:.3f}) | '
        f'std across attributes {o["spread_attr"]:.4f} | across layers {o["spread_layer"]:.4f} | '
        f'across faces {o["spread_face"]:.4f}')
    log(f'   saturated: {o["frac_open"] * 100:.0f}% of gates > 0.9, {o["frac_closed"] * 100:.0f}% < 0.1')
    log('\n2. Does the gate react to the edit? mean |gate(edit) - gate(no edit)|  '
        f'(overall {o["edit_response"]:.4f})')
    log('   ' + '  '.join(f'{n} {v:.4f}' for n, v in zip(names, o['edit_response_per_attr'])))
    log('\n3. Gate by ODE time (decode pass; bin 0 = start of the pass)')
    log('   ' + '  '.join(f'bin{b} {v:.3f}' for b, v in enumerate(o['time_profile'])))
    log('\n4. Where the flow edits W+ (share of squared edit norm) and edit size')
    log(f'   {"":<12}' + ''.join(f'{g:>14}' for g in LAYER_GROUPS) + '   |dW|  round-trip err/|dW|')
    for a, n in enumerate(names):
        s = o['energy_share'][n]
        log(f'   {n:<12}' + ''.join(f'{s[g] * 100:13.0f}%' for g in LAYER_GROUPS)
            + f'  {o["delta_norm"][a]:6.2f}  {o["roundtrip_rel"][a]:8.3f}')
    log('\n5. Frozen gate: relative change of the W+ edit  |dW_trained - dW_frozen| / |dW_trained|')
    log(f'   {"":<12}{"re-run (noise)":>15}{"gate = own mean":>17}{"cos":>7}{"gate = 1":>12}{"cos":>7}')
    for a, n in enumerate(names):
        log(f'   {n:<12}{o["frozen_repeat_rel_change"][a]:15.4f}{o["frozen_const_rel_change"][a]:17.4f}'
            f'{o["frozen_const_cos"][a]:7.3f}{o["frozen_open_rel_change"][a]:12.4f}'
            f'{o["frozen_open_cos"][a]:7.3f}')
    v = o['verdict']
    log('\nVerdict (rules of thumb: attribute spread/response < 0.02, layer spread < 0.03, '
        'frozen change < max(5%, 3x noise floor) / < max(15%, 3x noise floor)):')
    log(f'   gate varies with the edited attribute : {v["gate_varies_with_attribute"]}')
    log(f'   gate prefers some layers              : {v["gate_prefers_some_layers"]}')
    log(f'   freezing the gate changes the edit    : {v["gate_matters_for_the_edit"]}')
    log('   NO / NO / NO     -> the gate is decoration; LAG-DOF\'s attribute- and layer-aware gating '
        'has no support here.')
    log('   matters but NO attribute variation -> it shapes the edit per face / layer / time, identically '
        'for every attribute: not an attribute selector.')


def _nan_to_none(x):
    if isinstance(x, float) and x != x:
        return None
    if isinstance(x, dict):
        return {k: _nan_to_none(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_nan_to_none(v) for v in x]
    return x


def save_figure(o, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    prof = np.array(o['profile'])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    im = ax[0].imshow(prof, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    ax[0].set_yticks(range(len(o['names'])))
    ax[0].set_yticklabels(o['names'])
    ax[0].set_xlabel('W+ layer')
    ax[0].set_title('Mean gate per edited attribute')
    fig.colorbar(im, ax=ax[0], fraction=0.03)
    ax[1].plot(o['time_profile'], marker='o')
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel('ODE time bin (decode pass)')
    ax[1].set_ylabel('mean gate')
    ax[1].set_title('Gate over time')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    return True


def main():
    import torchvision.transforms as T
    from torch.utils import data
    from tqdm import tqdm
    from evaluation.evaluate_sdflow import (
        ATTR_NAMES, _latest_step, apply_run_config, build_parser, edited_attr_value,
        load_models, resolve_controlnet_disable_attrs)
    from models.dataset import SDFlowDataset

    p = build_parser()
    p.add_argument('--diag_faces', type=int, default=64, help='test faces to use')
    p.add_argument('--diag_scale', type=float, default=0.9, help='edit strength')
    p.add_argument('--diag_bins', type=int, default=5, help='ODE-time bins')
    p.add_argument('--out_dir', default=None)
    args = p.parse_args()
    if not os.path.isdir(args.checkpoint_dir):
        raise SystemExit(f'checkpoint_dir does not exist on this machine: {args.checkpoint_dir}\n'
                         f'(a /mnt/c/... path only exists inside WSL)')
    args = apply_run_config(args)
    args = resolve_controlnet_disable_attrs(args)
    if args.step is None:
        args.step = _latest_step(args.checkpoint_dir)
    out_dir = args.out_dir or os.path.join(args.checkpoint_dir, f'gate_diag_s{args.step}')
    os.makedirs(out_dir, exist_ok=True)

    prior, conditioner, _G, id_criterion, _t, attribute_index, _bank, _ctrl = load_models(args)
    tf = T.Compose([T.ToTensor(), T.Resize((args.img_size, args.img_size)), T.Normalize(mean=0.5, std=0.5)])
    ds = SDFlowDataset(index_file=args.index_file, image_root=args.image_root,
                       latents_file=args.latent_file, preds_file=args.preds_file,
                       train=False, transform=tf)
    loader = data.DataLoader(ds, shuffle=False, batch_size=args.batch, num_workers=4)
    batches, seen = [], 0
    with torch.no_grad():
        for img, latent, _ in tqdm(loader, desc='conditions'):
            if seen >= args.diag_faces:
                break
            img, latent = img.cuda(), latent.cuda()
            _, id_cond, attr_cond = conditioner.make_condition(img, latent, id_criterion)
            batches.append((latent, id_cond, attr_cond))
            seen += img.size(0)

    names = [ATTR_NAMES.get(i, f'attr{i}') for i in args.attribute_index]
    res = run_diagnostic(
        prior, batches, names, list(args.attribute_index), args.diag_scale,
        lambda src, scale, gidx: edited_attr_value(src, scale, gidx), n_bins=args.diag_bins)
    res.pop('_arrays', None)
    res['checkpoint_dir'], res['step'] = args.checkpoint_dir, args.step
    path = os.path.join(out_dir, 'gate_diag.json')
    with open(path, 'w') as f:
        json.dump(_nan_to_none(res), f, indent=2)
    print(f'\nsaved {path}')
    if save_figure(res, os.path.join(out_dir, 'gate_diag.png')):
        print(f'saved {os.path.join(out_dir, "gate_diag.png")}')


if __name__ == '__main__':
    main()
