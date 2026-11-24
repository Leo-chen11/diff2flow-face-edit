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


def divergence_and_vjp(f, y, e):
    """divergence_approx, also returning e^T df/dy (the vector-Jacobian product it is built
    from), whose squared norm is the Hutchinson estimate of the Jacobian's Frobenius norm."""
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    cnt = 0
    while not (e_dzdx * e).requires_grad and cnt < 10:
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        cnt += 1
    approx_tr_dzdx = (e_dzdx * e).sum(dim=-1)
    assert approx_tr_dzdx.requires_grad, "(failed to add node to graph)"
    return approx_tr_dzdx, e_dzdx


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

    def __init__(self, hidden_dims, input_shape, context_dim, layer_type="concat", nonlinearity="softplus",
                 layer_embed_layers=0):
        super(ODEnet, self).__init__()
        if layer_embed_layers and layer_type != "concatsquash":
            raise ValueError('layer_embed_layers needs layer_type concatsquash')
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
        # Layer-aware velocity (--velocity_layer_embed). Without it the net applies the same weights and
        # the same per-sample hyper-network output (gate, bias of every ConcatSquash layer, functions of
        # (t, condition)) to all 18 W+ rows, so it tells row 3 (pose, face shape) from row 12 (colour)
        # only by the row's own values -- and e4e rows are close to each other. Here the W+ row index
        # enters the hyper-network exactly like t does: every layer's gate logit and bias get a learned
        # per-row offset,
        #     h_l = W x_l * sigmoid(G [t, c] + g_row) + (B [t, c] + b_row),
        # i.e. a one-hot row index concatenated to the hyper-network's input. Because the offset sits
        # inside the sigmoid, a unit can open only where the condition AND the row call for it (an
        # attribute x layer interaction), and on the output layer it scales / shifts the velocity of each
        # row directly. No gate that can collapse, no extra loss: g_row = b_row = 0 at init, so a model
        # resumed from a checkpoint without them computes exactly what it did (bit-identical), and if
        # the rows do not help, the offsets stay near 0 and the model stays the old one.
        self.row_gate = self.row_bias = None
        if layer_embed_layers:
            dims = list(hidden_dims) + [input_shape[0]]
            self.row_gate = nn.ParameterList([nn.Parameter(torch.zeros(layer_embed_layers, d)) for d in dims])
            self.row_bias = nn.ParameterList([nn.Parameter(torch.zeros(layer_embed_layers, d)) for d in dims])

    def forward(self, context, y):
        dx = y
        rows = (self.row_bias is not None and dx.dim() == 3 and dx.size(1) == self.row_bias[0].size(0))
        for l, layer in enumerate(self.layers):
            if rows:
                gate = torch.sigmoid(layer._hyper_gate(context).unsqueeze(1) + self.row_gate[l].to(dx.dtype))
                bias = layer._hyper_bias(context).unsqueeze(1) + self.row_bias[l].to(dx.dtype)
                dx = layer._layer(dx) * gate + bias
            else:
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
                 gate_init_bias=-1.5, mode='lag_dof', attr_context_dim=0, layer_embed=False):
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
            layer_embed_layers=num_layers if layer_embed else 0,
        )
        self.base_velocity = ODEnet(
            hidden_dims=hidden_dims,
            input_shape=input_shape,
            context_dim=context_dim,
            layer_type=layer_type,
            nonlinearity=nonlinearity,
            layer_embed_layers=num_layers if layer_embed else 0,
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
        # the part of the velocity that sees the attributes, as it enters the sum: v_attr (dof) or
        # gate * v_attr (lag, lag_dof). The base velocity is computed with the attributes zeroed, so it is
        # the same under the source and the target condition; with this part at 0 an edit returns the
        # source unchanged. ODEfunc's transport_target 'attr' regularises this part only.
        self._last_v_edit = None
        self._last_gate = None

    def forward(self, context, y):
        v_attr = self.attr_velocity(context, y)
        if self.attr_context_dim > 0:
            base_context = context.clone()
            base_context[:, -self.attr_context_dim:] = 0.0
        else:
            base_context = context
        v_base = self.base_velocity(base_context, y)
        # dof sums the two velocities and never uses the gate: skip computing it (same output)
        gate = None if self.mode == 'dof' else self.layer_gate(context, y)
        self._last_v_attr = v_attr
        self._last_v_base = v_base
        self._last_gate = gate
        self._last_v_edit = v_attr if self.mode == 'dof' else gate * v_attr
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
                "gate_sparse": zero,
            }

        v_attr = self._last_v_attr
        v_base = self._last_v_base
        gate = self._last_gate

        attr_flat = v_attr.reshape(v_attr.size(0), -1)
        base_flat = v_base.reshape(v_base.size(0), -1)
        zero = v_attr.new_tensor(0.0)
        if self.mode in {'dof', 'lag_dof'}:
            orth = torch.abs(nn.functional.cosine_similarity(attr_flat, base_flat, dim=1, eps=eps)).mean()
            orth_layer = torch.abs(nn.functional.cosine_similarity(v_attr, v_base, dim=-1, eps=eps)).mean()
        else:
            orth = zero
            orth_layer = zero
        if self.mode in {'lag', 'lag_dof'}:
            gate_smooth = (gate[:, 1:, :] - gate[:, :-1, :]).abs().mean()
            # L_sparse: mean |g_a|. gate_smooth alone only pulls adjacent
            # layers toward each other -- it has no gradient pushing any
            # layer toward 0, so with only that term the gate has no reason
            # to ever close a layer. Diagnostic evidence (scripts/inspect_gate.py)
            # on a trained checkpoint confirmed exactly this: every layer
            # stayed in 0.79-0.99 regardless of attribute or ODE time,
            # i.e. the gate degenerated into an always-open, attribute-
            # agnostic constant instead of the selective layer gate the
            # design intends. This term gives sparsity an actual gradient.
            gate_sparse = gate.abs().mean()
        else:
            gate_smooth = zero
            gate_sparse = zero
        return {
            "orth": orth,
            "orth_layer": orth_layer,
            "gate_smooth": gate_smooth,
            "gate_sparse": gate_sparse,
        }


class ODEfunc(nn.Module):
    def __init__(self, diffeq):
        super(ODEfunc, self).__init__()
        self.diffeq = diffeq
        self.divergence_fn = divergence_approx
        self.register_buffer("_num_evals", torch.tensor(0.))
        # Inference fast path (flow.configure_inference(skip_divergence=True)): an edit only needs the
        # trajectory of the latent, never log p. The divergence term costs a Hutchinson vector-Jacobian
        # product -- a backward pass through the velocity net with create_graph -- at EVERY evaluation,
        # and its random estimate also enters the adaptive solver's error control. With this flag the
        # log-p state gets derivative 0 (it stays where it started), the velocity runs without autograd,
        # and the solver controls the step on the latent alone. The latent's ODE is the same, so the
        # edit agrees with the full solve up to the solver tolerance. Never set during training (the
        # NLL needs the divergence); a plain attribute, not saved in checkpoints.
        self.skip_divergence = False
        # Which velocity the transport regulariser (CNF.transport_reg) measures: 'full' = the whole
        # field (RNODE as published), 'attr' = only the attribute part of a decomposed field
        # (LAGDOFODEnet._last_v_edit). The log-density always uses the whole field's divergence.
        self.transport_target = 'full'

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def forward(self, t, states):
        y = states[0]
        if self.skip_divergence and len(states) == 3:
            self._num_evals += 1
            with torch.no_grad():
                tt = torch.ones(y.size(0), 1).to(y) * t.detach().type_as(y)
                c = states[2]
                dy = self.diffeq(torch.cat([tt, c.view(y.size(0), -1)], dim=1), y)
            return dy, torch.zeros_like(states[1]), torch.zeros_like(c)
        t = torch.ones(y.size(0), 1).to(y) * t.clone().detach().requires_grad_(True).type_as(y)
        self._num_evals += 1
        for state in states:
            state.requires_grad_(True)

        # Sample and fix the noise.
        if self._e is None:
            self._e = torch.randn_like(y, requires_grad=True).to(y)

        with torch.set_grad_enabled(True):
            if len(states) == 5:  # conditional CNF + the transport regularizer states (CNF.transport_reg)
                # RNODE (Finlay et al. 2020, "How to train your neural ODE"): integrate, along the
                # same trajectory, the kinetic energy ||v||^2 and the Hutchinson estimate
                # ||e^T dv/dy||^2 of the Jacobian's squared Frobenius norm. Penalising them makes
                # the flow transport along straighter, smoother paths (closer to optimal
                # transport), which is what lets an adaptive solver take fewer steps.
                # transport_target 'attr': the same two integrals of the ATTRIBUTE part of a decomposed
                # field only -- the part an edit comes from -- while the base part, which models the
                # density of W+ codes, is left free. The divergence for log p stays the whole field's.
                c = states[2]
                tc = torch.cat([t, c.view(y.size(0), -1)], dim=1)
                dy = self.diffeq(tc, y)
                if self.transport_target == 'attr':
                    v = getattr(self.diffeq, '_last_v_edit', None)
                    if v is None:
                        raise RuntimeError("transport_target 'attr' needs a decomposed velocity field "
                                           "(velocity_field dof, lag or lag_dof)")
                    div = self.divergence_fn(dy, y, e=self._e)
                    vjp = torch.autograd.grad(v, y, self._e, create_graph=True)[0]
                else:
                    v = dy
                    div, vjp = divergence_and_vjp(dy, y, self._e)
                kinetic = v.pow(2).sum(dim=-1, keepdim=True)
                jacobian = vjp.pow(2).sum(dim=-1, keepdim=True)
                return dy, -div.unsqueeze(-1), torch.zeros_like(c).requires_grad_(True), kinetic, jacobian
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
