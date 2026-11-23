from .odefunc import LAGDOFODEnet, ODEfunc, ODEnet
from .normalization import MovingBatchNorm1d
from .cnf import CNF, SequentialFlow


def build_model(input_dim, hidden_dims, context_dim, num_blocks, conditional,
                layer_type='concatsquash', nonlinearity='relu',
                velocity_field='original', num_layers=18, gate_hidden_dim=64,
                gate_init_bias=-1.5, use_adjoint=True, train_T=True,
                solver='dopri5', atol=1e-5, rtol=1e-5,
                attr_context_dim=0, layer_embed=False):
    def build_cnf():
        # assert layer_type in {'ignore','squash','scale','concat','concat_v2','concatsquash','concatscale'}
        if velocity_field in {'lag', 'dof', 'lag_dof'}:
            diffeq = LAGDOFODEnet(
                hidden_dims=hidden_dims,
                input_shape=(input_dim,),
                context_dim=context_dim,
                layer_type=layer_type,
                nonlinearity=nonlinearity,
                num_layers=num_layers,
                gate_hidden_dim=gate_hidden_dim,
                gate_init_bias=gate_init_bias,
                mode=velocity_field,
                attr_context_dim=attr_context_dim,
                layer_embed=layer_embed,
            )
        elif velocity_field == 'original':
            diffeq = ODEnet(
                hidden_dims=hidden_dims,
                input_shape=(input_dim,),
                context_dim=context_dim,
                layer_type=layer_type,
                nonlinearity=nonlinearity,
                layer_embed_layers=num_layers if layer_embed else 0,
            )
        else:
            raise ValueError(f'Unknown velocity_field: {velocity_field}')
        odefunc = ODEfunc(
            diffeq=diffeq,
        )
        cnf = CNF(
            odefunc=odefunc,
            T=1.0,
            train_T=train_T,
            conditional=conditional,
            solver=solver,
            use_adjoint=use_adjoint,
            atol=atol,
            rtol=rtol,
        )
        
        # original is dopri5

        return cnf

    chain = [build_cnf() for _ in range(num_blocks)]
    bn_layers = [MovingBatchNorm1d(input_dim, bn_lag=0, sync=False)
                     for _ in range(num_blocks)]
    bn_chain = [MovingBatchNorm1d(input_dim, bn_lag=0, sync=False)]
    for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
    chain = bn_chain
    model = SequentialFlow(chain)

    return model


def cnf(input_dim, dims, zdim, num_blocks, velocity_field='original',
        num_layers=18, gate_hidden_dim=64, gate_init_bias=-1.5,
        use_adjoint=True, train_T=True, solver='dopri5',
        atol=1e-5, rtol=1e-5, attr_context_dim=0, layer_embed=False):
    dims = tuple(map(int, dims.split("-")))
    # input_dim, hidden_dims, context_dim, num_blocks, conditional,layer_type='concatsquash'
    model = build_model(input_dim=input_dim, hidden_dims=dims,
                        context_dim=zdim, num_blocks=num_blocks, conditional=True,
                        layer_type='concatsquash', nonlinearity='tanh',
                        velocity_field=velocity_field, num_layers=num_layers,
                        gate_hidden_dim=gate_hidden_dim,
                        gate_init_bias=gate_init_bias,
                        use_adjoint=use_adjoint,
                        train_T=train_T,
                        solver=solver,
                        atol=atol,
                        rtol=rtol,
                        attr_context_dim=attr_context_dim,
                        layer_embed=layer_embed)
    return model


def state_has_layer_embed(state_dict):
    """Whether a saved flow was trained with --velocity_layer_embed (its per-W+-row biases are in it):
    the eval and the editor build the flow to match the checkpoint, so they cannot silently drop them."""
    return any('.row_bias.' in k for k in state_dict)




FIXED_STEP_SOLVERS = ('euler', 'midpoint', 'rk4', 'heun2', 'heun3', 'explicit_adams', 'implicit_adams', 'fixed_adams')


def cnf_blocks(flow):
    return [m for m in flow.modules() if isinstance(m, CNF)]


def configure_inference(flow, solver=None, tol=None, steps=None, skip_divergence=None):
    """How a flow solves its ODE when it is NOT training (the evaluation and the editor; training
    always uses its own solver, tolerances and the divergence). Returns a short description.

    solver: a torchdiffeq method name (None keeps the trained one, dopri5). Adaptive solvers
        (dopri5, bosh3, adaptive_heun, ...) take tol as both atol and rtol; fixed-step solvers
        (euler, midpoint, rk4, ...) take `steps` steps over each block's integration interval [0, T].
    skip_divergence: integrate the latent only, not log p (models/flows/odefunc.py ODEfunc).
    The function counts it calls in each block's odefunc._num_evals (flow_nfe)."""
    blocks = cnf_blocks(flow)
    for b in blocks:
        if solver is not None:
            b.test_solver = solver
        if tol is not None:
            b.test_atol = b.test_rtol = float(tol)
        if b.test_solver in FIXED_STEP_SOLVERS:
            if not steps:
                raise ValueError(f'solver {b.test_solver} is fixed-step: give the number of steps')
            b.test_options = {'step_size': float(b.T) / int(steps)}
        else:
            b.test_options = None
        if skip_divergence is not None:
            b.odefunc.skip_divergence = bool(skip_divergence)
    b = blocks[0] if blocks else None
    if b is None:
        return 'no CNF block'
    how = (f'{b.test_solver}, {steps} steps' if b.test_solver in FIXED_STEP_SOLVERS
           else f'{b.test_solver}, tol {b.test_atol:g}')
    return how + (', latent only (no log-det)' if b.odefunc.skip_divergence else ', with log-det')


def flow_nfe(flow):
    """Velocity evaluations of the last solve, summed over the flow's CNF blocks."""
    return int(sum(b.num_evals() for b in cnf_blocks(flow)))
