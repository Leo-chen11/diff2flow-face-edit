"""Read the LEARNED per-attribute [global, coarse, fine] reg-loss weights
straight out of a saved checkpoint, no model load required.

Why this matters here: LearnableRegLossWeights's own docstring records that
in an earlier run, age(39)'s fine weight collapsed 0.5 -> 5e-6 by 46k steps,
"freeing the fine W+ layers and producing texture/color artifacts" -- and
--reg_weight_min (default 0.1) exists specifically to floor that collapse.

That leaves an open question this project's own comments don't answer: with
the floor in place, did age's fine weight settle near the floor (fine
layers still heavily discouraged from moving -> a structural reason the
skin-smoothing probe found what it found, since real wrinkle synthesis
needs SUBSTANTIAL fine-layer movement that this weight directly penalizes),
or well above it (meaning this weight is not the bottleneck and the next
place to look is elsewhere)? This script answers that directly from the
checkpoint instead of guessing.

Usage:
    python -m scripts.inspect_reg_fine_weight \
        --checkpoint_dir ./output/SDFlow/substyle3_glasses_v18 --step 120000 \
        --attribute_index 15 20 39
"""
import argparse
import os

import torch


def _inverse_softplus(x):
    # Mirrors training/train_sdflow.py's own helper exactly, so the decoded
    # weights match what training actually computed.
    x = x.clamp(min=1e-6)
    return torch.log(torch.expm1(x))


def main(args):
    # Matches load_module_checkpoint's own naming exactly:
    # os.path.join(save_dir, '{prefix}-{step, zero-padded to 7}')
    path = os.path.join(args.checkpoint_dir, 'save_models',
                        'reg_loss_weights-{}'.format(str(args.step).zfill(7)))
    if not os.path.exists(path):
        raise SystemExit(f'not found: {path}\n'
                         f'List what is actually there with:\n'
                         f'  ls {os.path.join(args.checkpoint_dir, "save_models")} | grep reg_loss_weights')

    state = torch.load(path, map_location='cpu')   # this IS the state_dict, per load_module_checkpoint
    raw = state['log_weights_raw']            # (n_edit_attrs, 3): [global, coarse, fine]
    weights = torch.nn.functional.softplus(raw).clamp(min=args.reg_weight_min)

    names = ['global', 'coarse', 'fine']
    print(f'loaded {path}')
    print(f'floor (--reg_weight_min) = {args.reg_weight_min}\n')
    for i, attr in enumerate(args.attribute_index):
        row = weights[i].tolist()
        print(f'attr {attr}:')
        for name, w in zip(names, row):
            at_floor = '  <- AT FLOOR (fine layers maximally discouraged from moving)' \
                if name == 'fine' and abs(w - args.reg_weight_min) < 1e-3 else ''
            print(f'    {name:<7} = {w:.4f}{at_floor}')
    print()
    print('READ: if attr 39\'s fine weight sits near the floor, it is being held')
    print('there against the pull toward zero the docstring documents -- i.e. still')
    print('actively discouraging fine-layer movement, just not able to collapse')
    print('all the way. That directly opposes what wrinkle synthesis in the fine')
    print('W+ layers needs, and would explain the skin-smoothing this project\'s')
    print('own noise probe found without needing a new loss to fix it -- lowering')
    print('the floor for age specifically, or dropping this regularizer\'s reach')
    print('into the layers --age_dds_fine_layer_start unlocked, would be the direct')
    print('fix. If it sits well above the floor, this weight is not the bottleneck.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--attribute_index', nargs='*', type=int, default=[15, 20, 39])
    p.add_argument('--reg_weight_min', type=float, default=0.1,
                   help='Must match the --reg_weight_min the run was actually trained '
                        'with (v18 config.json: 0.3, not the argparse default 0.1) -- '
                        'the floor is applied at read time here, same as in training.')
    main(p.parse_args())
