import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeLayerMask(nn.Module):
    """Post-hoc attribute-layer mask baseline for ablations."""

    def __init__(self, num_attrs, num_layers=18, init_logit=-0.75):
        super(AttributeLayerMask, self).__init__()
        self.num_attrs = int(num_attrs)
        self.num_layers = int(num_layers)
        self.base_logits = nn.Parameter(torch.full((self.num_attrs, self.num_layers), init_logit))
        self.src_scale = nn.Parameter(torch.zeros(self.num_attrs, self.num_layers))
        self.tgt_scale = nn.Parameter(torch.zeros(self.num_attrs, self.num_layers))
        self.delta_scale = nn.Parameter(torch.zeros(self.num_attrs, self.num_layers))

    def forward(self, attr_idx, src_attr, tgt_attr):
        attr_idx = attr_idx.long().view(-1)
        src_attr = src_attr.view(-1, 1)
        tgt_attr = tgt_attr.view(-1, 1)
        logits = self.base_logits[attr_idx]
        logits = logits + self.src_scale[attr_idx] * src_attr
        logits = logits + self.tgt_scale[attr_idx] * tgt_attr
        logits = logits + self.delta_scale[attr_idx] * (tgt_attr - src_attr)
        return torch.sigmoid(logits)

    def mean_mask(self):
        return torch.sigmoid(self.base_logits)

    def sparsity_loss(self):
        return 1.0 - self.mean_mask().mean()

    def target_mean_loss(self, target_mean):
        return (self.mean_mask().mean() - float(target_mean)).pow(2)

    def smoothness_loss(self):
        mask = self.mean_mask()
        return (mask[:, 1:] - mask[:, :-1]).abs().mean()

    def affinity_prior_loss(self, attribute_index):
        priors = torch.full_like(self.mean_mask(), 0.25)
        for local_idx, attr in enumerate(attribute_index):
            attr = int(attr)
            if attr == 15:  # Eyeglasses: middle/fine StyleGAN layers.
                priors[local_idx, 4:14] = 0.75
                priors[local_idx, 14:] = 0.55
            elif attr == 20:  # Gender: requires coarse structural change (jaw, cheekbones).
                priors[local_idx, 0:12] = 0.65
            elif attr == 39:  # Young/Age: requires coarse geometry + fine skin texture.
                priors[local_idx, 0:17] = 0.65
            else:
                priors[local_idx, 4:14] = 0.55
        return F.mse_loss(self.mean_mask(), priors)
