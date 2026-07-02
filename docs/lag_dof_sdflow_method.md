# LAG-DOF-SDFlow

## Core Idea

LAG-DOF keeps the original SDFlow conditional CNF and changes only the internal ODE velocity field. The original field

```text
dw/dt = v_theta(w, t, c)
```

is replaced by

```text
dw/dt = g_a(w, t, c) * v_attr(w, t, c) + v_base(w, t, c)
```

where `w` is a W+ latent with shape `[B, 18, 512]`, `v_attr` models the editable semantic direction, `v_base` models identity-preserving transport, and `g_a` is a learned layer-wise gate with shape `[B, 18, 1]`.

The flow remains reversible because the ODE solver, integration direction, and CNF log-density computation are unchanged. LAG-DOF only changes the differentiable velocity function used by the solver.

## Architecture

### AttrVelocityNet

`AttrVelocityNet` is implemented as an `ODEnet` with the same ConcatSquashLinear blocks as SDFlow. It receives the time-augmented condition `[t, c]` and the current W+ state `w`, and returns `v_attr` with shape `[B, 18, 512]`.

### BaseVelocityNet

`BaseVelocityNet` mirrors `AttrVelocityNet` but has independent weights. It returns `v_base`, which handles condition-preserving transport and should not collapse to the same direction as `v_attr`.

### LayerGate

`LayerGate` predicts `g_a` using:

- a projected time/condition embedding,
- a learned StyleGAN layer embedding for the 18 W+ layers,
- a per-layer latent projection from `w[:, l, :]`.

The gate output is sigmoid-normalized. Its initial bias is negative, so training starts with conservative edits and opens layers only when useful.

### LAG-DOF Context

The usual SDFlow condition is

```text
c_base = [id_condition, attr_condition]
```

For LAG-DOF, the flow context is extended to

```text
c_lag = [c_base, one_hot(a), src_attr_a, tgt_attr_a, tgt_attr_a - src_attr_a]
```

This makes the velocity field explicitly aware of the edited attribute and edit direction.

## Losses

Keep the original SDFlow likelihood and latent losses:

```text
L_nll = negative CNF log likelihood
L_reg = W+ displacement regularization
L_kd  = conditioner attribute distillation
```

Use target/leakage semantic supervision:

```text
L_target  = ||A_a(G(w_edit)) - c_t,a||^2 + margin loss
L_leakage = mean_{j != a} ||A_j(G(w_edit)) - A_j(G(w_src))||^2
```

Add LAG-DOF regularizers:

```text
L_orth       = mean |cos(flatten(v_attr), flatten(v_base))|
L_orth_layer = mean_l |cos(v_attr[:, l, :], v_base[:, l, :])|
L_sparse     = mean |g_a|
L_smooth     = mean_l |g_a[:, l+1] - g_a[:, l]|
```

The full objective is:

```text
L = L_nll
  + lambda_reg L_reg
  + lambda_kd L_kd
  + lambda_target L_target
  + lambda_leak L_leakage
  + lambda_orth L_orth
  + lambda_orth_layer L_orth_layer
  + lambda_sparse L_sparse
  + lambda_smooth L_smooth
  + lambda_id L_id
```

In this codebase, `L_target` corresponds to `changed_loss`, `L_leakage` corresponds to `preserve_loss`, and `L_id` is the ArcFace feature loss.

## Recommended Hyperparameters

Good starting values:

```text
lambda_orth       = 0.05
lambda_orth_layer = 0.05
lambda_sparse     = 0.01
lambda_smooth     = 0.02
lambda_id          = 0.15
gate_hidden_dim    = 64
gate_init_bias     = -1.5
learning_rate      = 1e-4
```

If edits are too weak, reduce `lambda_sparse` to `0.003` or set `gate_init_bias=-1.0`. If identity drift remains high, increase `lambda_orth_layer` to `0.1` and `lambda_id` to `0.2`.

## Paper Method Section Draft

We propose Layer-wise Attribute-Gated Disentangled Orthogonal Flow (LAG-DOF), a velocity-field redesign for SDFlow that preserves the original reversible conditional CNF while improving edit locality in StyleGAN W+ space. Given a source latent code `w_src`, SDFlow maps it to a base latent `z` by integrating a conditional ODE under the source condition, then reverses the flow under a target attribute condition. Instead of using a single shared velocity field for all attributes and StyleGAN layers, LAG-DOF decomposes the velocity into an attribute-editing component and a base identity-preserving component:

```text
dw/dt = g_a(w,t,c) * v_attr(w,t,c) + v_base(w,t,c).
```

The gate `g_a` is predicted per W+ layer and conditioned on the edited attribute, source attribute score, target score, continuous ODE time, and current latent state. This design lets the trajectory itself become attribute-aware and layer-aware, instead of applying a mask to the final displacement after the flow has already produced an edited latent.

To prevent the attribute and base velocities from learning redundant directions, we add global and layer-wise orthogonality penalties between `v_attr` and `v_base`. We further regularize the gate with sparsity and adjacent-layer smoothness losses, encouraging compact, interpretable layer usage while avoiding noisy layer patterns. The resulting model keeps SDFlow's invertible ODE formulation and likelihood training, but learns to restrict semantic edits to the layers required by each attribute.

## Ablations

Recommended comparisons:

| Model | Description |
| --- | --- |
| Original SDFlow | Shared velocity `v_theta(w,t,c)` with no post-hoc correction. |
| SDFlow + Manual Layer Mask | Apply fixed hand-designed W+ masks to `Delta w` after editing. |
| SDFlow + Post-hoc Delta Refiner | Generate raw edit first, then clean/refine `Delta w`. |
| SDFlow + LAG only | Use `g_a * v_theta` or disable orthogonal loss and base/attr split. |
| SDFlow + DOF only | Use `v_attr + v_base` with orthogonal loss but no layer gate. |
| Full LAG-DOF-SDFlow | Use gate, decomposed velocities, orthogonal loss, and gate regularization. |

The implemented training switch is:

```bash
python training/train_sdflow.py --velocity_field original
python training/train_sdflow.py --velocity_field lag
python training/train_sdflow.py --velocity_field dof
python training/train_sdflow.py --velocity_field lag_dof
```

Report target-attribute success, non-target leakage, ArcFace identity similarity, W+ displacement magnitude by layer group, and learned gate visualizations per attribute.
