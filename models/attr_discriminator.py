"""Attribute-conditional discriminator (--attr_disc_weight in training/train_sdflow.py).

The training classifiers (teachers) only ask "is the attribute there"; an edit can
answer yes with patterns no real face carries. This discriminator asks "does the
attribute look like it does on real faces": conditioned on (attribute, state), it
separates e4e RECONSTRUCTIONS of training faces that really are in that state from
our edits into that state. Both sides are StyleGAN2 outputs, so it cannot win by
spotting "generated vs photographed" and has to look at the attribute itself.

Projection discriminator (Miyato & Koyama, ICLR 2018) with spectral norm:
    D(x, c) = w . h(x) + <embed(c), h(x)>,   c = 2 * attribute_slot + state
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN


class _DBlock(nn.Module):
    def __init__(self, cin, cout, down=True):
        super().__init__()
        self.c1 = SN(nn.Conv2d(cin, cout, 3, padding=1))
        self.c2 = SN(nn.Conv2d(cout, cout, 3, padding=1))
        self.skip = SN(nn.Conv2d(cin, cout, 1))
        self.down = down

    def forward(self, x):
        h = self.c2(F.leaky_relu(self.c1(F.leaky_relu(x, 0.2)), 0.2))
        s = self.skip(x)
        if self.down:
            h, s = F.avg_pool2d(h, 2), F.avg_pool2d(s, 2)
        return h + s


class AttrConditionalDiscriminator(nn.Module):
    """Input (B, 3, 256, 256) in [-1, 1] (other sizes are resized to 256), condition
    index (B,) in [0, 2 * num_attrs). Returns (B,) logits."""

    def __init__(self, num_attrs, size=256, ch=(32, 64, 128, 256, 256, 256), embed_sn=True):
        super().__init__()
        self.size = size
        blocks, cin = [], 3
        for c in ch:                         # 256 -> 4 with six halvings
            blocks.append(_DBlock(cin, c))
            cin = c
        self.blocks = nn.Sequential(*blocks)
        self.linear = SN(nn.Linear(cin, 1))
        emb = nn.Embedding(2 * num_attrs, cin)
        self.embed = SN(emb) if embed_sn else emb

    def forward(self, x, cond):
        if x.shape[-1] != self.size:
            x = F.interpolate(x, (self.size, self.size), mode='bilinear', align_corners=False)
        h = F.leaky_relu(self.blocks(x), 0.2).sum(dim=(2, 3))
        return self.linear(h).squeeze(1) + (self.embed(cond) * h).sum(1)
