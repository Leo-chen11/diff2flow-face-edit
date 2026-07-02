import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentEditDenoiser(nn.Module):
    """Post-hoc conditional delta refiner with FiLM conditioning.

    Applies a one-step correction to a flow-predicted W+ delta using
    attribute and direction information to make the refinement conditional.
    """

    def __init__(self, condition_dim, num_attrs, latent_dim=512, num_layers=18):
        super(LatentEditDenoiser, self).__init__()
        self.condition_dim = int(condition_dim)
        self.num_attrs = int(num_attrs)
        self.latent_dim = int(latent_dim)
        self.num_layers = int(num_layers)

        # Context: [condition, one_hot_attr, src_attr, tgt_attr]
        ctx_dim = condition_dim + num_attrs + 2
        self.cond_encoder = nn.Sequential(
            nn.Linear(ctx_dim, latent_dim),
            nn.Tanh(),
        )
        # FiLM modulation: scale and shift applied after the main net
        self.gamma = nn.Linear(latent_dim, latent_dim)
        self.beta = nn.Linear(latent_dim, latent_dim)
        # Main per-layer net (operates on last dim of [B, 18, 512])
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, delta, condition, attr_idx, src_attr, tgt_attr):
        # delta: [B, 18, 512]
        # condition: [B, condition_dim]
        # attr_idx: [B] long
        # src_attr, tgt_attr: [B] float
        attr_oh = F.one_hot(attr_idx.long(), self.num_attrs).float().to(delta)
        ctx = torch.cat([
            condition.to(delta),
            attr_oh,
            src_attr.view(-1, 1).to(delta),
            tgt_attr.view(-1, 1).to(delta),
        ], dim=-1)                                     # [B, ctx_dim]
        h = self.cond_encoder(ctx)                     # [B, latent_dim]
        gamma = 1.0 + self.gamma(h).unsqueeze(1)       # [B, 1, latent_dim]
        beta = self.beta(h).unsqueeze(1)               # [B, 1, latent_dim]
        return self.net(delta) * gamma + beta          # [B, 18, latent_dim]

    @torch.no_grad()
    def refine(self, delta, condition, attr_idx, src_attr, tgt_attr):
        """Apply one-step post-hoc conditional correction to flow delta."""
        return delta + self.forward(delta, condition, attr_idx, src_attr, tgt_attr)
