

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

## Consensus Metric (`evaluate_sdflow.py --consensus`)

Every judge splits add/rm from its own reading of the source, and the model edits in the direction its conditioner reads. When those readings disagree, a correct edit is scored as a failure (e.g. short-haired women that CelebA calls male are put in Male-rm, while the conditioner reads them as female and edits toward male).

`--consensus` additionally scores each edit on the subset of sources that CLIP (the BiSeNet parser for Eyeglasses) and the CelebA classifier both read clearly (outside 0.35-0.65) and the same way. The subset depends only on the source image and the two external judges, never on the model, so every checkpoint is scored on the same faces. It needs both judges (`./data/celeba_attr_resnet18.pth`).

Reported per attribute under `"consensus"` and pooled under `"consensus_overall"` (existing fields are unchanged; the output name gets a `_cons` suffix):

| Field | Meaning |
| --- | --- |
| `cover` | Share of sources kept. The kept faces are the less ambiguous ones, so always report it next to the full-set numbers. |
| `acc_clip`, `acc_celeb`, `acc_both` | Strict success by CLIP, by CelebA, and by both. |
| `acc_both_add`, `acc_both_rm` | `acc_both` split by the consensus direction. |
| `dir_agree` | How often the model was asked to edit the consensus way (its conditioner's reading under `--edit_direction cond`). |
| `acc_both_dir_agree`, `acc_both_dir_disagree` | `acc_both` split by that: failures in the second are caused by the model's reading of the source, not by its edit. |
| `id_indep`, `id_arc`, `lpips` | Identity and perceptual distance on the same subset. |

```bash
python evaluation/evaluate_sdflow.py --checkpoint_dir <run> --step <N> --consensus
```
