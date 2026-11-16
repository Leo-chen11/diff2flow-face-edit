"""PluGeN (Wolczyk et al., "PluGeN: Multi-Label Conditional Generation From Pre-Trained Models",
AAAI 2022; journal version TPAMI 2024) as a baseline: an independent implementation of the method's
StyleGAN experiment, with the hyper-parameters of the official code (github.com/gmum/plugen,
StyleGAN/train.py and NICE.py, CC BY-NC-SA 4.0 -- academic use only; no released weights).

The method: an invertible NICE flow (additive couplings, so volume preserving) maps a W latent to a
code z whose first K coordinates are the K attributes -- trained so that z[:K] ~ N(label, sigma)
and z[K:] ~ N(0, 1), sigma decaying over training. An edit encodes the latent, writes the wanted
attribute values into z[:K] and decodes.

The official flow works on ONE W vector (it reads row 0 of a W+ and copies its output to all 18
rows). On e4e W+ latents (18 different rows) the flow reads the MEAN of the rows and the change it
makes to that vector is added to every row -- for a W latent this is exactly the official edit.
(models/gradctrl.py lifts its W-space method the same way.)
"""
import torch
from torch import nn


class AdditiveCoupling(nn.Module):
    """x_b <- x_b + m(x_a): half of the coordinates (even or odd ones) shift the other half."""

    def __init__(self, dim, n_layers, parity, hidden):
        super().__init__()
        mask = torch.zeros(dim)
        mask[parity::2] = 1.0
        self.register_buffer('mask', mask.view(1, -1))
        layers = [nn.Linear(dim, hidden), nn.LeakyReLU(0.1)]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.LeakyReLU(0.1)]
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def shift(self, x):
        return self.net(x * self.mask) * (1 - self.mask)

    def forward(self, x):
        return x + self.shift(x)

    def inverse(self, z):
        return z - self.shift(z)


class PluGeNFlow(nn.Module):
    """The official StyleGAN setting: 4 additive couplings, each a 4-layer MLP of width 512."""

    def __init__(self, dim=512, n_layers=4, n_couplings=4, hidden=512):
        super().__init__()
        self.couplings = nn.ModuleList(AdditiveCoupling(dim, n_layers, i % 2, hidden) for i in range(n_couplings))

    def forward(self, w):
        for c in self.couplings:
            w = c(w)
        return w

    def inverse(self, z):
        for c in reversed(self.couplings):
            z = c.inverse(z)
        return z


def to_code(p):
    """Attribute probability in [0, 1] -> the flow's label scale [-1, 1]."""
    return 2.0 * p - 1.0


@torch.no_grad()
def plugen_edit(model, latent, k, direction, strength, labels=None):
    """Edit W+ latents (B, 18, 512): attribute coordinate k of the code is set to
    direction * strength (+1 / -1 are the two ends of the label range the flow was trained on;
    strength > 1 goes past them).

    labels: (B, K) source attribute values on the [-1, 1] scale. Given (the official evaluation):
    ALL K attribute coordinates are written -- the other attributes with their labels, the edited
    one with its target. None: the other coordinates keep the values the flow encoded.
    """
    v = latent.mean(1)
    z = model(v)
    if labels is not None:
        z[:, :labels.size(1)] = labels.to(z)
    z[:, k] = direction.to(z).view(-1) * float(strength)
    return latent + (model.inverse(z) - v)[:, None, :]
