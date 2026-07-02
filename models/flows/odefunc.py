import copy
import torch
import torch.nn as nn
from . import diffeq_layers

__all__ = ["ODEnet", "ODEfunc"]


def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx.mul(e)

    cnt = 0
    while not e_dzdx_e.requires_grad and cnt < 10:
        # print("RequiresGrad:f=%s, y(rgrad)=%s, e_dzdx:%s, e:%s, e_dzdx_e:%s cnt=%d"
        #       % (f.requires_grad, y.requires_grad, e_dzdx.requires_grad,
        #          e.requires_grad, e_dzdx_e.requires_grad, cnt))
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        e_dzdx_e = e_dzdx * e
        cnt += 1

    approx_tr_dzdx = e_dzdx_e.sum(dim=-1)
    assert approx_tr_dzdx.requires_grad, \
        "(failed to add node to graph) f=%s %s, y(rgrad)=%s, e_dzdx:%s, e:%s, e_dzdx_e:%s cnt:%s" \
        % (
        f.size(), f.requires_grad, y.requires_grad, e_dzdx.requires_grad, e.requires_grad, e_dzdx_e.requires_grad, cnt)
    return approx_tr_dzdx


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Lambda(nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x ** 2),
    "identity": Lambda(lambda x: x),
}


class ODEnet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    """

    def __init__(self, hidden_dims, input_shape, context_dim, layer_type="concat", nonlinearity="softplus"):
        super(ODEnet, self).__init__()
        base_layer = {
            "reslinear":diffeq_layers.ResLinear,
            "ignore": diffeq_layers.IgnoreLinear,
            "squash": diffeq_layers.SquashLinear,
            "scale": diffeq_layers.ScaleLinear,
            "concat": diffeq_layers.ConcatLinear,
            "concat_v2": diffeq_layers.ConcatLinear_v2,
            "concatsquash": diffeq_layers.ConcatSquashLinear,
            "concatscale": diffeq_layers.ConcatScaleLinear,
            "concatnonlinear": diffeq_layers.ConcatNonLinear,
        }[layer_type]

        # build models and add them
        layers = []
        activation_fns = []
        hidden_shape = input_shape

        for dim_out in (hidden_dims + (input_shape[0],)):

            layer_kwargs = {}
            layer = base_layer(hidden_shape[0], dim_out, context_dim, **layer_kwargs)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, context, y):
        dx = y
        for l, layer in enumerate(self.layers):
            dx = layer(context, dx)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        return dx


class LayerGate(nn.Module):
    """Layer-wise attribute gate g_a with shape [B, num_layers, 1]."""

    def __init__(self, context_dim, latent_dim, num_layers=18, hidden_dim=64, init_bias=-1.5):
        super(LayerGate, self).__init__()
        self.num_layers = num_layers
        self.context_proj = nn.Sequential(
            nn.Linear(1 + context_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.layer_emb = nn.Parameter(torch.randn(num_layers, hidden_dim) * 0.02)
        self.to_gate = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.to_gate.bias, init_bias)

    def forward(self, context, y):
        if y.dim() != 3:
            raise ValueError("LayerGate expects W+ latents with shape [B, L, C].")
        if y.size(1) != self.num_layers:
            raise ValueError(
                f"LayerGate was built for {self.num_layers} layers, got {y.size(1)}."
            )
        context_feat = self.context_proj(context).unsqueeze(1)
        latent_feat = self.latent_proj(y)
        layer_feat = self.layer_emb.unsqueeze(0).to(dtype=y.dtype)
        logits = self.to_gate(torch.tanh(context_feat + latent_feat + layer_feat))
        return torch.sigmoid(logits)


class LAGDOFODEnet(nn.Module):
    """Layer-wise Attribute-Gated Disentangled Orthogonal Flow velocity field.

    The CNF state and solver are unchanged. Only the velocity field is decomposed:
        dy/dt = g_a(y, t, c) * v_attr(y, t, c) + v_base(y, t, c).
    """

    def __init__(self, hidden_dims, input_shape, context_dim, layer_type="concat",
                 nonlinearity="softplus", num_layers=18, gate_hidden_dim=64,
                 gate_init_bias=-1.5, mode='lag_dof', attr_context_dim=0):
        super(LAGDOFODEnet, self).__init__()
        if mode not in {'lag', 'dof', 'lag_dof'}:
            raise ValueError(f'Unknown LAG-DOF mode: {mode}')
        self.mode = mode
        self.attr_context_dim = int(attr_context_dim)
        self.attr_velocity = ODEnet(
            hidden_dims=hidden_dims,
            input_shape=input_shape,
            context_dim=context_dim,
            layer_type=layer_type,
            nonlinearity=nonlinearity,
        )
        self.base_velocity = ODEnet(
            hidden_dims=hidden_dims,
            input_shape=input_shape,
            context_dim=context_dim,
            layer_type=layer_type,
            nonlinearity=nonlinearity,
        )
        self.layer_gate = LayerGate(
            context_dim=context_dim,
            latent_dim=input_shape[0],
            num_layers=num_layers,
            hidden_dim=gate_hidden_dim,
            init_bias=gate_init_bias,
        )
        self._last_v_attr = None
        self._last_v_base = None
        self._last_gate = None

    def forward(self, context, y):
        v_attr = self.attr_velocity(context, y)
        if self.attr_context_dim > 0:
            base_context = context.clone()
            base_context[:, -self.attr_context_dim:] = 0.0
        else:
            base_context = context
        v_base = self.base_velocity(base_context, y)
        gate = self.layer_gate(context, y)
        self._last_v_attr = v_attr
        self._last_v_base = v_base
        self._last_gate = gate
        if self.mode == 'lag':
            return gate * v_attr
        if self.mode == 'dof':
            return v_attr + v_base
        return gate * v_attr + v_base

    def regularization_losses(self, eps=1e-8):
        if self._last_v_attr is None:
            zero = next(self.parameters()).new_tensor(0.0)
            return {
                "orth": zero,
                "orth_layer": zero,
                "gate_smooth": zero,
            }

        v_attr = self._last_v_attr
        v_base = self._last_v_base
        gate = self._last_gate

        attr_flat = v_attr.reshape(v_attr.size(0), -1)
        base_flat = v_base.reshape(v_base.size(0), -1)
        zero = gate.new_tensor(0.0)
        if self.mode in {'dof', 'lag_dof'}:
            orth = torch.abs(nn.functional.cosine_similarity(attr_flat, base_flat, dim=1, eps=eps)).mean()
            orth_layer = torch.abs(nn.functional.cosine_similarity(v_attr, v_base, dim=-1, eps=eps)).mean()
        else:
            orth = zero
            orth_layer = zero
        if self.mode in {'lag', 'lag_dof'}:
            gate_smooth = (gate[:, 1:, :] - gate[:, :-1, :]).abs().mean()
        else:
            gate_smooth = zero
        return {
            "orth": orth,
            "orth_layer": orth_layer,
            "gate_smooth": gate_smooth,
        }


class ODEfunc(nn.Module):
    def __init__(self, diffeq):
        super(ODEfunc, self).__init__()
        self.diffeq = diffeq
        self.divergence_fn = divergence_approx
        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def forward(self, t, states):
        y = states[0]
        t = torch.ones(y.size(0), 1).to(y) * t.clone().detach().requires_grad_(True).type_as(y)
        self._num_evals += 1
        for state in states:
            state.requires_grad_(True)

        # Sample and fix the noise.
        if self._e is None:
            self._e = torch.randn_like(y, requires_grad=True).to(y)

        with torch.set_grad_enabled(True):
            if len(states) == 3:  # conditional CNF
                c = states[2]
                tc = torch.cat([t, c.view(y.size(0), -1)], dim=1)

                dy = self.diffeq(tc, y)
                divergence = self.divergence_fn(dy, y, e=self._e).unsqueeze(-1)

                return dy, -divergence, torch.zeros_like(c).requires_grad_(True)
            elif len(states) == 2:  # unconditional CNF
                dy = self.diffeq(t, y)
                divergence = self.divergence_fn(dy, y, e=self._e).view(-1, 1)
                return dy, -divergence
            else:
                assert 0, "`len(states)` should be 2 or 3"
