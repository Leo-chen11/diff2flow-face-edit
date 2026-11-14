"""AdaTrans (Huang, Ma, Zhang, Shan, "Adaptive Nonlinear Latent Transformation for Conditional
Face Editing", ICCV 2023, arXiv 2307.07790, github.com/Hzzone/AdaTrans), reimplemented for the
baseline comparison. Self-contained (torch only).

    AdaTrans      the editor: from a W+ latent and a condition c (+1 add / -1 remove for a binary
                  attribute), max_steps steps; at each step an LSTM over the trajectory so far, then
                  condition-modulated layers, give a unit direction (one 512-d vector for every W+
                  layer) and a step size in (0, 1). Returns the whole trajectory; editing strength =
                  the number of steps taken.
    LatentDensity the RealNVP density on single W+ layer vectors that keeps the trajectory in the
                  latent distribution (training only).

The module structure and parameter names follow the official code, so its released checkpoints
(save_models/model-latest for the editor) load with strict=True and can be evaluated in this repo's
protocol next to a retrained one (scripts/editing_curves.py --method adatrans). The official
repository has no license file; this is an independent implementation of the published
architecture, written against it, not a copy of it.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── editor ────────────────────────────────────────────────────────────────

def mlp(in_c, hid_c, out_c, n_layer=0):
    layers = [nn.Linear(in_c, hid_c), nn.BatchNorm1d(hid_c), nn.ReLU(True)]
    for _ in range(n_layer):
        layers += [nn.Linear(hid_c, hid_c), nn.BatchNorm1d(hid_c), nn.ReLU(True)]
    layers.append(nn.Linear(hid_c, out_c))
    return nn.Sequential(*layers)


class Affine(nn.Module):
    """x * exp(scale(c)) + bias(c)."""

    def __init__(self, dim_out, dim_c):
        super().__init__()
        self.scale = mlp(dim_c, dim_out, dim_out)
        self.bias = mlp(dim_c, dim_out, dim_out)

    def forward(self, x, c):
        return x * self.scale(c).exp() + self.bias(c)


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c):
        super().__init__()
        self.layer = mlp(dim_in, dim_in, dim_out, 0)
        self.affine = Affine(dim_out, dim_c)

    def forward(self, x, c):
        return self.affine(self.layer(x), c)


class Func(nn.Module):
    """One step: (direction (B, 1, style_dim), step size (B, 1)) from the trajectory so far."""

    def __init__(self, style_dim, hid_c, c_dim, n_styles, n_layer):
        super().__init__()
        self.main = nn.ModuleList([ConcatSquashLinear(hid_c, hid_c, c_dim) for _ in range(n_layer)])
        self.input_layer = mlp(style_dim * n_styles, hid_c, hid_c)
        self.output_layer = mlp(hid_c, hid_c, style_dim)
        self.step_layer = mlp(hid_c, hid_c, 1)
        self.rnn_layers = 2
        self.rnn = nn.LSTM(hid_c, hid_c, self.rnn_layers)
        self.n_styles = n_styles
        self.style_dim = style_dim

    def forward(self, trajectory, c):
        traj = torch.stack(trajectory).flatten(-2)                       # (T, B, n_styles * style_dim)
        T, B = traj.shape[:2]
        y = self.input_layer(traj.reshape(T * B, -1)).view(T, B, -1)
        h0 = torch.zeros(self.rnn_layers, B, y.size(2)).to(y)
        y = self.rnn(y, (h0, torch.zeros_like(h0)))[0]
        y0 = y[-1]
        for layer in self.main:
            y0 = layer(y0, c)
        step_size = torch.sigmoid(self.step_layer(y0))
        dy = F.normalize(self.output_layer(y0), dim=-1).unsqueeze(1)    # the same vector for every layer
        return dy, step_size


class AdaTrans(nn.Module):
    def __init__(self, style_dim=512, hid_dim=512, c_dim=1, n_styles=18, max_steps=10, n_layers=10):
        super().__init__()
        self.n_layers = n_layers
        self.style_dim = style_dim
        self.func = Func(style_dim, hid_dim, c_dim, n_styles, n_layers)
        self.c_dim = c_dim
        self.max_steps = max_steps

    def forward(self, x, target):
        """x: (B, n_styles, style_dim) absolute W+ latents; target: (B, c_dim), +1 / -1 for binary
        attributes. Returns (trajectory (max_steps + 1, B, n_styles, style_dim), step sizes (B, max_steps))."""
        target = target.float()
        y0 = x
        trajectory, step_sizes = [y0], []
        for _ in range(self.max_steps):
            dy, step_size = self.func(trajectory, target)
            y0 = y0 + dy * step_size[:, :, None]
            step_sizes.append(step_size)
            trajectory.append(y0)
        return torch.stack(trajectory), torch.cat(step_sizes, dim=1)


@torch.no_grad()
def adatrans_edit(model, latent, direction, steps):
    """The latent after `steps` steps toward +1 (add) / -1 (remove) (direction: (B,)). A fractional
    `steps` (scripts/comparison_table.py interpolates the strength between measured ones) takes
    the last step only partly: y_n + f * (y_{n+1} - y_n), i.e. f of that step's size."""
    steps = float(steps)
    n = int(math.floor(steps + 1e-9))
    f = steps - n
    model.max_steps = n + (1 if f > 1e-6 else 0)
    if model.max_steps == 0:
        return latent.clone()
    traj = model(latent, direction.view(-1, 1).to(latent))[0]
    return traj[n] if f <= 1e-6 else traj[n] + f * (traj[n + 1] - traj[n])


# ── density model (RealNVP on single W+ layer vectors) ────────────────────

class _WeightNorm(nn.Module):
    """w = v * g / (||v|| + eps), with the norm taken over dim 0 (as the official flow code)."""

    def __init__(self, module, eps=1.0e-5):
        super().__init__()
        self.module = module
        self.eps = eps
        w = module.weight
        g = torch.norm(w, dim=0)
        v = w / (g.expand_as(w) + eps)
        del module._parameters['weight']
        module.register_parameter('weight_g', nn.Parameter(g.detach()))
        module.register_parameter('weight_v', nn.Parameter(v.detach()))

    def forward(self, x):
        m = self.module
        w = m.weight_v * (m.weight_g / (torch.norm(m.weight_v, dim=0) + self.eps)).expand_as(m.weight_v)
        return F.linear(x, w, m.bias)


class _ResBlockLinear(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm1d(ch), nn.ReLU(inplace=True), _WeightNorm(nn.Linear(ch, ch)),
                                 nn.BatchNorm1d(ch), nn.ReLU(inplace=True), _WeightNorm(nn.Linear(ch, ch)))
        self.bridge = nn.Sequential()

    def forward(self, x):
        return self.bridge(x) + self.net(x)


class _MLP(nn.Module):
    def __init__(self, in_ch, out_ch, base=32, n_blocks=2):
        super().__init__()
        self.in_block = nn.Sequential(_WeightNorm(nn.Linear(in_ch, base)))
        self.mid_block = nn.Sequential(*[_ResBlockLinear(base) for _ in range(n_blocks)])
        self.out_block = nn.Sequential(nn.BatchNorm1d(base), nn.ReLU(inplace=True), _WeightNorm(nn.Linear(base, out_ch)))

    def forward(self, x):
        return self.out_block(self.mid_block(self.in_block(x)))


class _FlowBatchNorm(nn.Module):
    """Batch normalisation as an invertible layer (no affine: log_gamma / beta are zero buffers)."""

    def __init__(self, dim, momentum=0.1, eps=1.0e-5):
        super().__init__()
        self.momentum, self.eps = momentum, eps
        for name, val in (('log_gamma', 0.), ('beta', 0.), ('running_mean', 0.), ('running_var', 1.),
                          ('batch_mean', 0.), ('batch_var', 1.)):
            self.register_buffer(name, torch.full((1, dim), val))

    def forward(self, x, log_det):
        if self.training:
            mean = x.mean(0, keepdim=True)
            var = (x - mean).pow(2).mean(0, keepdim=True) + self.eps
            self.batch_mean.data.copy_(mean)
            self.batch_var.data.copy_(var)
            self.running_mean.mul_(1 - self.momentum).add_(mean.detach() * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(var.detach() * self.momentum)
            mean, var = self.batch_mean, self.batch_var
        else:
            mean, var = self.running_mean, self.running_var
        x = (x - mean) / torch.sqrt(var) * torch.exp(self.log_gamma) + self.beta
        return x, log_det + torch.sum(self.log_gamma - 0.5 * torch.log(var))

    def backward(self, x, log_det):
        mean, var = (self.batch_mean, self.batch_var) if self.training else (self.running_mean, self.running_var)
        x = (x - self.beta) / torch.exp(self.log_gamma) * torch.sqrt(var) + mean
        return x, log_det + torch.sum(-self.log_gamma + 0.5 * torch.log(var))


class _AffineCoupling(nn.Module):
    """Real NVP affine coupling on interleaved halves (even / odd coordinates, swapped when odd)."""

    def __init__(self, dim, odd):
        super().__init__()
        self.odd = odd
        self.register_parameter('s_log_scale', nn.Parameter(torch.randn(1) * 0.01))
        self.register_parameter('s_bias', nn.Parameter(torch.randn(1) * 0.01))
        in_ch = dim // 2 if not odd else (dim + 1) // 2
        self.out_chs = dim - in_ch
        self.net = _MLP(in_ch, self.out_chs * 2)

    def _split(self, z):
        z = z.view(z.size(0), z.size(1) // 2, 2)
        z0, z1 = z[:, :, 0], z[:, :, 1]
        return (z1, z0) if self.odd else (z0, z1)

    def _merge(self, z0, z1):
        if self.odd:
            z0, z1 = z1, z0
        return torch.stack([z0, z1], dim=-1).view(z0.size(0), -1).contiguous()

    def _st(self, z1):
        params = self.net(z1)
        return torch.tanh(params[:, self.out_chs:]) * self.s_log_scale + self.s_bias, params[:, :self.out_chs]

    def forward(self, z, log_det):
        z0, z1 = self._split(z)
        s, t = self._st(z1)
        return self._merge(z0 * torch.exp(s) + t, z1), log_det + s.sum(1)

    def backward(self, y, log_det):
        y0, y1 = self._split(y)
        s, t = self._st(y1)
        return self._merge(torch.exp(-s) * (y0 - t), y1), log_det - s.sum(1)


class _Compose(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, z, log_det):
        for layer in self.layers:
            z, log_det = layer(z, log_det)
        return z, log_det

    def backward(self, z, log_det):
        for layer in reversed(self.layers):
            z, log_det = layer.backward(z, log_det)
        return z, log_det


class _RealNVP(nn.Module):
    def __init__(self, dim, n_layers):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(_FlowBatchNorm(dim))
            layers.append(_AffineCoupling(dim, odd=i % 2 != 0))
        self.net = _Compose(layers)

    def forward(self, z):
        return self.net(z, torch.zeros(z.size(0)).to(z))

    def backward(self, z):
        return self.net.backward(z, torch.zeros(z.size(0)).to(z))


class LatentDensity(nn.Module):
    """The official FLOW module: a RealNVP over single W+ layer vectors (B * n_styles of them)."""

    def __init__(self, style_dim=512, n_layer=10):
        super().__init__()
        self.flow = _RealNVP(style_dim, n_layer)

    def forward(self, inputs):
        """inputs: (B, n_styles, style_dim) or (N, style_dim). Returns (loss = negative log-likelihood
        per dimension, logz, log|det J|, z), as the official code."""
        dim = inputs.size(-1)
        latents = inputs.reshape(-1, dim)
        z, log_det = self.flow(latents)
        logz_sum = (-0.5 * math.log(2 * math.pi) - 0.5 * z.pow(2)).sum(dim=1)
        return -torch.mean(logz_sum + log_det) / dim, logz_sum / dim, log_det, z

    def backward(self, inputs):
        bs, n_styles, dim = inputs.size()
        return self.flow.backward(inputs.reshape(bs * n_styles, -1))[0].view(-1, n_styles, dim)
