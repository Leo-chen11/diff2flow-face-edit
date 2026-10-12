"""Eval-time ablation: replace the learned LAG-DOF LayerGate with a fixed table.

scripts/diagnose_gate.py showed the gate is the same for every edited attribute,
so the question left is whether its per-face / per-step adaptivity matters for
the final metrics, or whether a constant table would do. The tables come from
that script's gate_diag.json:

    mean      per-layer constant (18,)              -- drops time and face variation
    schedule  per (ODE-time bin, layer) (n_bins,18) -- keeps the time schedule,
                                                       drops face/attribute variation
    open      gate = 1 everywhere                   -- removes the gate entirely
"""
import json

import torch

from models.flows.odefunc import LAGDOFODEnet

GATE_FREEZE_MODES = ('mean', 'schedule', 'open')


def _gates(prior):
    return [m.layer_gate for m in prior.modules()
            if isinstance(m, LAGDOFODEnet) and m.mode in ('lag', 'lag_dof')]


def freeze_layer_gates(prior, mode, profile_json=None):
    """Patch every active LayerGate of `prior` in place. Returns the number patched."""
    if mode not in GATE_FREEZE_MODES:
        raise ValueError(f'gate freeze mode must be one of {GATE_FREEZE_MODES}, got {mode!r}')
    gates = _gates(prior)
    if not gates:
        raise ValueError('this flow has no active LayerGate (velocity_field must be '
                         "'lag' or 'lag_dof'); there is nothing to freeze")
    prof = None
    if mode != 'open':
        if profile_json is None:
            raise ValueError(f'--gate_freeze {mode} needs the gate_diag.json written by '
                             'scripts/diagnose_gate.py')
        with open(profile_json) as f:
            diag = json.load(f)
        if mode == 'mean':
            prof = torch.tensor(diag['const_gate'], dtype=torch.float32)          # (L,)
        else:
            if 'time_layer_profile' not in diag:
                raise ValueError(f'{profile_json} has no time_layer_profile: re-run '
                                 'scripts/diagnose_gate.py with the current code')
            rows = [[float('nan') if v is None else v for v in r]
                    for r in diag['time_layer_profile']]
            prof = torch.tensor(rows, dtype=torch.float32)                        # (n_bins, L)
            const = torch.tensor(diag['const_gate'], dtype=torch.float32)
            prof = torch.where(torch.isnan(prof), const.expand_as(prof), prof)    # empty bins
            t_max = float(diag['t_max'])

    def make(orig):
        def fwd(context, y):
            out = orig(context, y)                                                # (B, L, 1)
            if mode == 'open':
                return torch.ones_like(out)
            p = prof.to(out)
            if mode == 'mean':
                return p.view(1, -1, 1).expand_as(out)
            n_bins = p.size(0)
            t = context[:, 0].float()
            idx = (t / max(t_max, 1e-8) * n_bins).long().clamp(0, n_bins - 1)
            return p[idx].unsqueeze(-1)
        return fwd

    for g in gates:
        g.forward = make(g.forward)
    return len(gates)
