

## Direction Bank Modes (`--bank_mode`)

With `--direction_bank_path`, `--bank_mode` decides who actually produces the edit. The mode and the anneal schedule are saved in the run's `config.json` (not in the checkpoint, so `state_dict` keys are unchanged); `evaluate_sdflow.py` (and every script built on its `load_models`) and `models/editor.py` restore them, and for `anneal` use the mix the schedule had at the evaluated step. Runs without `--bank_mode` behave as `replace`.

| Mode | Final W+ delta | Role of the bank |
| --- | --- | --- |
| `replace` (default) | `bank_dir x MLP magnitude + rs * flow_residual` | Produces the edit; the flow only adds a small orthogonal residual. |
| `anneal` | `mix * replace + (1 - mix) * flow_delta` | Curriculum: `mix` decays from `--bank_mix_start` to `--bank_mix_end` over `--bank_anneal_steps`. A soft prior loss fades in as `(1 - mix) * --bank_prior_weight`. |
| `prior` | `flow_delta` | Training target only: cosine hinge `relu(margin - cos(flow_delta, bank_pred))`, controlled by `--bank_prior_weight` and `--bank_prior_margin`. |
| `flow_magnitude` | `flow_delta` projected onto the edited attribute's bank axes + `rs * residual` | Fixes the axes; the flow sets per-sample, per-layer magnitudes. Components along other attributes' axes are dropped. |

Training logs `dir_bank_flow_share` (fraction of the edit coming from the flow), `dir_bank_flow_bank_cos`, `dir_bank_mix`, and `loss_bank_prior`.

Re-evaluate an existing checkpoint under another rule without retraining (an explicit `--bank_mode` overrides config.json; the result JSON gets a `_bank<mode>` suffix):

```bash
python evaluation/evaluate_sdflow.py --checkpoint_dir <run> --step <N> --bank_mode prior            # flow alone
python evaluation/evaluate_sdflow.py --checkpoint_dir <run> --step <N> --bank_mode flow_magnitude
python evaluation/evaluate_sdflow.py --checkpoint_dir <run> --step <N> --bank_mode anneal --bank_mix 0.5
```

Train each mode:

```bash
python training/train_sdflow.py --direction_bank_path <bank.pt> --bank_mode anneal \
    --resume_dir <best_run> --resume_step <N> --resume_direction_bank \
    --bank_mix_start 1.0 --bank_mix_end 0.0 --bank_anneal_steps 20000
python training/train_sdflow.py --direction_bank_path <bank.pt> --bank_mode prior --bank_prior_weight 0.1 --bank_prior_margin 0.5
python training/train_sdflow.py --direction_bank_path <bank.pt> --bank_mode flow_magnitude
```
