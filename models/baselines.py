"""Re-implementations of two published W+ editing baselines, trained on this
repo's e4e latents and r34 labels so that every method is compared on the same
generator, inversion, test faces and judges (scripts/train_baselines.py trains
them, scripts/editing_curves.py --method lt / styleflow evaluates them).

Latent Transformer (Yao et al., ICCV 2021, github.com/InterDigitalInc/latent-transformer):
    a latent attribute classifier LCNet on the flattened W+, and per attribute a
    transformer T(w, a) = w + a * F(w), F one equalized-lr linear layer per W+ layer.

StyleFlow (Abdal et al., TOG 2021): a continuous normalizing flow on W+ conditioned
    on the attribute values, trained by maximum likelihood only; an edit encodes w
    under its attributes and decodes under the changed ones. It is the same CNF as
    the released SDFlow (models/flows/flow.cnf with the original velocity field)
    without SDFlow's attribute estimator and disentanglement losses.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.flows.flow import cnf


class LCNet(nn.Module):
    """Latent classifier of the Latent Transformer: MLP on the flattened W+ (18 x 512)."""

    def __init__(self, fmaps=(9216, 2048, 512, 40)):
        super().__init__()
        self.fcs = nn.ModuleList(nn.Linear(fmaps[i], fmaps[i + 1]) for i in range(len(fmaps) - 1))

    def forward(self, w):
        x = w.flatten(1)
        for layer in self.fcs[:-1]:
            x = F.leaky_relu(layer(x), 0.2)
        return self.fcs[-1](x)


class _DenseEqualized(nn.Module):
    """The Latent Transformer's Dense_layer: xavier init, He-scaled at run time."""

    def __init__(self, size=512):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(size, size))
        self.bias = nn.Parameter(torch.zeros(size))
        nn.init.xavier_uniform_(self.weight)
        self.coef = 1.0 / np.sqrt(size)

    def forward(self, x):
        return F.linear(x, self.weight * self.coef, self.bias)


class LatentTransformer(nn.Module):
    """T(w, a) = w + a * F(w) with F one linear map per W+ layer (the paper's F_mapping)."""

    def __init__(self, layers=18, size=512):
        super().__init__()
        self.dense = nn.ModuleList(_DenseEqualized(size) for _ in range(layers))

    def forward(self, w, coeff):
        """w: (B, 18, 512); coeff: (B,) signed strength."""
        delta = torch.stack([d(w[:, i]) for i, d in enumerate(self.dense)], dim=1)
        return w + coeff.view(-1, 1, 1) * delta

    @staticmethod
    def train_coeff(p):
        """The paper's get_coeff: the step that takes the latent classifier's
        probability p to the other side (1 - p to add, -p to remove)."""
        return torch.where(p > 0.5, -p, 1 - p)


def styleflow_cnf(n_attrs):
    return cnf(512, '512-512-512-512-512', n_attrs, 1)


def styleflow_edit(flow, w, cond, idx, direction, strength):
    """Encode w under cond, move attribute idx toward its target end (1 to add,
    0 to remove) by strength (1 = reach it, >1 = past it), decode.
    w: (B, 18, 512); cond: (B, A); direction: (B,) +1 add / -1 remove."""
    z = flow(w, cond, torch.zeros(w.size(0), w.size(1), 1).to(w))[0]
    new = cond.clone()
    target = (direction > 0).to(cond)
    new[:, idx] = cond[:, idx] + strength * (target - cond[:, idx])
    return flow(z, new, reverse=True)
