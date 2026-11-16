"""GradCtrl (Chen et al., "Exploring Gradient-based Multi-directional Controls in GANs", ECCV 2022),
ported from the official code github.com/zikuncshelly/GradCtrl (MIT License, Copyright (c) 2022
Zikun Chen): classifiers.py (the latent classifier, same parameter names, so the released
model_ffhq/model.pth loads strictly) and manipulate.py (the edit).

The method: a small MLP classifier per attribute on a W latent. An edit walks along the gradient of
the target attribute's probability, with the channels most salient for the attributes to keep
(the top-k of |gradient| of each of them) zeroed, normalised, step 4/9, the walk applied to a range
of W+ layers only; the official FFHQ demo takes 9 steps (a total length of 4).

On e4e W+ latents (18 different rows, the official code edits W latents repeated 18 times):
the classifier reads the MEAN of the 18 rows, and each step's delta is added both to that vector
(what the classifier reads next) and to every row in the attribute's layer range -- for a W latent
this is exactly the official update.
"""
import math

import torch
from torch import nn

# The official FFHQ settings (manipulate.py EXCLUDE_DICT_FFHQ), with its attribute names mapped to
# CelebA ids: gender 20, smile 31, eyeglasses 15, age 39 (Young).
#   attr: (attributes kept, top channels of each to exclude, (first layer, last layer + 1))
OFFICIAL_FFHQ = {
    20: ([31, 15, 39], [250, 100, 100], (2, 10)),
    31: ([20, 15, 39], [200, 50, 100], (2, 8)),
    15: ([20, 31, 39], [200, 100, 250], (1, 4)),
    39: ([20, 15], [100, 300], (4, 10)),
}
# Bangs is not in the official release; this setting is ours (keep the four official attributes,
# 100 channels each, the layers of Smiling). Report it as such, or tune it on the development faces.
OURS_EXTRA = {
    5: ([20, 31, 15, 39], [100, 100, 100, 100], (2, 8)),
}
RELEASED_FFHQ_ATTRS = {'gender': 20, 'smile': 31, 'eyeglasses': 15, 'age': 39, 'pose': None}
STEP_SIZE = 4.0 / 9.0


class BinaryClassifier(nn.Module):
    def __init__(self, latent_size=512, hidden_size=None):
        super().__init__()
        if hidden_size is None:
            self.classifier = nn.Sequential(nn.Linear(latent_size, 1))
        else:
            self.classifier = nn.Sequential(nn.Linear(latent_size, hidden_size),
                                            nn.LeakyReLU(0.1), nn.Linear(hidden_size, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.classifier(x)
        return self.sigmoid(out), out


class MultiLabelClassifier(nn.Module):
    """The official multi-branch classifier (binary branches only): one BinaryClassifier per
    attribute behind an optional shared trunk (none in the released FFHQ model)."""

    def __init__(self, n_attributes, n_shared_layers=0, hidden_size=16, latent_size=512):
        super().__init__()
        if n_shared_layers > 0:
            shared = []
            for _ in range(n_shared_layers):
                shared += [nn.Linear(latent_size, latent_size), nn.LeakyReLU(0.1)]
        else:
            shared = [nn.Identity()]
        self.shared = nn.Sequential(*shared)
        self.n_branches = n_attributes
        self.classifiers = nn.ModuleDict({str(i): BinaryClassifier(latent_size, hidden_size)
                                          for i in range(n_attributes)})

    def logits(self, x):
        """x (B, 512) -> (B, n_branches) logits."""
        h = self.shared(x)
        return torch.cat([self.classifiers[str(i)](h)[1] for i in range(self.n_branches)], dim=1)

    def probs(self, x):
        return torch.sigmoid(self.logits(x))


def gradctrl_edit(model, latent, branch, direction, steps, exclude=(), top=(), layers=(0, 18),
                  step_size=STEP_SIZE, sign=1.0):
    """Edit W+ latents (B, 18, 512) along branch `branch` of `model`.

    direction: (B,) +1 to add the attribute, -1 to remove it (times `sign`, -1 for a branch whose
    probability rises as the attribute goes away). steps: number of steps (the official demo 9);
    a fractional part is a shortened last step. exclude / top: branch indices of the attributes to
    keep and how many of their most salient channels to zero. layers: the W+ rows edited.
    """
    lo, hi = layers
    n_full = int(math.floor(steps + 1e-9))
    frac = float(steps) - n_full
    mults = [1.0] * n_full + ([frac] if frac > 1e-6 else [])
    d = (direction.to(latent).view(-1) * float(sign)).view(-1, 1)
    out = latent.detach().clone()
    v = latent.detach().mean(1)
    with torch.enable_grad():
        for m in mults:
            x = v.detach().requires_grad_(True)
            p = model.probs(x)
            g = torch.autograd.grad(p[:, branch].sum(), x, retain_graph=len(exclude) > 0)[0]
            if len(exclude):
                mask = torch.ones_like(g)
                for e, k in zip(exclude, top):
                    sal = torch.autograd.grad(p[:, e].sum(), x, retain_graph=True)[0].abs()
                    mask.scatter_(1, sal.topk(int(k), dim=1).indices, 0.0)
                g = g * mask
            g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
            delta = (g * d * step_size * m).detach()
            v = v + delta
            out[:, lo:hi] = out[:, lo:hi] + delta[:, None, :]
    return out


def edit_settings(attr, attrs, overrides=None):
    """(exclude branch indices, top channels, layers) of `attr` for a classifier over `attrs`:
    the official FFHQ setting (Bangs: ours), restricted to kept attributes the classifier has."""
    table = dict(OFFICIAL_FFHQ)
    table.update(OURS_EXTRA)
    if overrides:
        table.update(overrides)
    if attr not in table:
        raise KeyError(f'no GradCtrl setting for attribute {attr}')
    keep, top, layers = table[attr]
    pairs = [(attrs.index(k), t) for k, t in zip(keep, top) if k in attrs and k != attr]
    return [i for i, _ in pairs], [t for _, t in pairs], tuple(layers)
