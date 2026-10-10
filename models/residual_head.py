"""A small per-attribute network that replaces the flow's residual at edit time.

scripts/analyze_residual.py showed the direction bank's residual lives in a
few directions per attribute (top 4 components keep the full model's accuracy
at matched identity), and that one fixed direction scaled by the requested
attribute change already recovers most of it. This head predicts the
residual's coefficients on those directions as

    c(face) = attr_delta * (base + net(face))

base is the fixed-direction solution (a per-attribute constant); net, zero-
initialised, adds the face-dependent part from what the bank already sees --
the source latent, the edit sign, the conditioner's attribute scores and
identity code. An edit then needs no ODE solve for the residual:

    residual(face) = sum_j  c_j(face) * PC_j

Trained by scripts/distill_residual_head.py on the flow's own coefficients;
loaded by evaluate_sdflow.py --residual_head.
"""
import torch
import torch.nn as nn


def head_features(latent, attr_delta_col, route_scores, id_cond):
    """(n, 512 + 1 + A + id_dim): layer-mean latent, edit sign, conditioner
    attribute scores, identity code. The edit size enters multiplicatively in
    ResidualHead.coeffs, not here."""
    parts = [latent.mean(dim=1), torch.sign(attr_delta_col).view(-1, 1)]
    if route_scores is not None:
        parts.append(route_scores)
    if id_cond is not None:
        parts.append(id_cond)
    return torch.cat([p.float() for p in parts], dim=1)


class ResidualHead(nn.Module):
    def __init__(self, in_dim, basis, hidden=256):
        super().__init__()
        k = basis.shape[0]
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, k))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.base = nn.Parameter(torch.zeros(k))
        self.register_buffer('basis', basis.float().clone())          # (k, L*D) orthonormal rows
        self.register_buffer('x_mean', torch.zeros(in_dim))
        self.register_buffer('x_std', torch.ones(in_dim))
        self.in_dim = int(in_dim)

    def coeffs(self, x, attr_delta_col):
        return attr_delta_col.float().view(-1, 1) * (self.base + self.net((x - self.x_mean) / self.x_std))

    def forward(self, latent, attr_delta_col, route_scores=None, id_cond=None):
        """Residual term (n, L*D) for these samples."""
        x = head_features(latent, attr_delta_col, route_scores, id_cond)
        if x.shape[1] != self.in_dim:
            raise ValueError(f'residual head expects {self.in_dim} input features, got {x.shape[1]} '
                             f'(trained with a different attribute set or identity code?)')
        return self.coeffs(x, attr_delta_col) @ self.basis
