from .odefunc import LAGDOFODEnet, ODEfunc, ODEnet
from .normalization import MovingBatchNorm1d
from .cnf import CNF, SequentialFlow


def build_model(input_dim, hidden_dims, context_dim, num_blocks, conditional,
                layer_type='concatsquash', nonlinearity='relu',
                velocity_field='original', num_layers=18, gate_hidden_dim=64,
                gate_init_bias=-1.5, use_adjoint=True, train_T=True,
                solver='dopri5', atol=1e-5, rtol=1e-5,
                attr_context_dim=0):
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
            )
        elif velocity_field == 'original':
            diffeq = ODEnet(
                hidden_dims=hidden_dims,
                input_shape=(input_dim,),
                context_dim=context_dim,
                layer_type=layer_type,
                nonlinearity=nonlinearity,
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
        atol=1e-5, rtol=1e-5, attr_context_dim=0):
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
                        attr_context_dim=attr_context_dim)
    return model


