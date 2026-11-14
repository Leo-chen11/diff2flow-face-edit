import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_normal
import torchdiffeq

# torchdiffeq 0.0.1 (requirements.txt) takes one tolerance per state tensor. Later
# versions reuse the training tolerances for the adjoint pass, whose augmented state
# has more tensors, and reject the list there; they get the same value as a scalar.
_PER_STATE_TOL = not hasattr(torchdiffeq, '__version__')

__all__ = ["CNF", "SequentialFlow"]


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows."""

    def __init__(self, layer_list):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layer_list)

    def forward(self, x, context, logpx=None, reverse=False, inds=None, integration_times=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                # print(x.shape)
                x = self.chain[i](x, context, logpx, integration_times, reverse)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, context, logpx, integration_times, reverse)
            return x, logpx


class CNF(nn.Module):
    def __init__(self, odefunc, conditional=True, T=1.0, train_T=False, regularization_fns=None,
                 solver='dopri5', atol=1e-5, rtol=1e-5, use_adjoint=True):
        super(CNF, self).__init__()
        self.train_T = train_T
        self.T = T
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
            # print("Training T :", self.T)

        if regularization_fns is not None and len(regularization_fns) > 0:
            raise NotImplementedError("Regularization not supported")
        self.use_adjoint = use_adjoint
        self.odefunc = odefunc
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.conditional = conditional
        # Transport regularizer (train_sdflow.py --transport_kinetic_weight / _jacobian_weight):
        # when True, a conditional solve also integrates the kinetic energy and the Jacobian
        # Frobenius estimate along its trajectory (ODEfunc, 5-state branch) and leaves them, per
        # sample and W+ layer, in self._last_transport. A plain attribute, not state: checkpoints
        # are unchanged, and a model loaded for evaluation solves exactly as before.
        self.transport_reg = False
        self._last_transport = None

    def forward(self, x, context=None, logpx=None, integration_times=None, reverse=False):
        
        if logpx is None:
            _logpx = torch.zeros(*x.shape[:-1], 1).to(x)
        else:
            _logpx = logpx

        track = self.conditional and self.transport_reg
        if self.conditional:
            assert context is not None
            states = (x, _logpx, context)
            if track:
                states = states + (torch.zeros_like(_logpx), torch.zeros_like(_logpx))
        else:
            states = (x, _logpx)
        atol = [self.atol] * len(states) if _PER_STATE_TOL else self.atol
        rtol = [self.rtol] * len(states) if _PER_STATE_TOL else self.rtol
            

        if integration_times is None:
            if self.train_T:
                integration_times = torch.stack(
                    [torch.tensor(0.0).to(x), self.sqrt_end_time * self.sqrt_end_time]
                ).to(x)
                # print("integration times:", integration_times)
            else:
                integration_times = torch.tensor([0., self.T], requires_grad=False).to(x)

        
        integration_times,_ = torch.sort(integration_times,dim=0,descending=reverse)

        
        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()
        odeint = odeint_adjoint if self.use_adjoint else odeint_normal
        if self.training:
            # print('states :',states[0].shape,states[1].shape)
            # print('integration_times : ',integration_times.shape,integration_times)
            
            state_t = odeint(
                self.odefunc,
                states,
                integration_times.to(x),
                atol=atol,
                rtol=rtol,
                method=self.solver,
            )
        else:
            state_t = odeint(
                self.odefunc,
                states,
                integration_times.to(x),
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver,
            )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]
        # integrands are >= 0; a reverse solve runs from T to 0, so its integrals come out negative
        self._last_transport = (state_t[3].abs(), state_t[4].abs()) if track else None

        if logpx is not None:
            return z_t, logpz_t
        else:
            return z_t

    def num_evals(self):
        return self.odefunc._num_evals.item()


