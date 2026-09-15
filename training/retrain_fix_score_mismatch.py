"""
Fine-tune an existing SDFlow checkpoint with the training-time flags that fix
the score/visual mismatches reported for this run, instead of the eval-time
band-aids in evaluate_sdflow.py (--glasses_judge parser, --controlnet_disable_attrs,
--age_fine_layer_scale, --composite_face_region). Those eval flags only change
how the SAME generator is scored/post-processed; they cannot make the eyeglasses
structure more complete or make aging actually visible. Every flag below already
exists in training/train_sdflow.py -- this script just assembles the specific
combination whose docstrings describe exactly these three problems, and resumes
from your checkpoint instead of retraining from scratch.

What each problem needs and why (see train_sdflow.py for the full docstrings):

  1. Eyeglasses structure not fully/cleanly rendered
     --local_region_loss_weight 0.5
         Forces the edit budget into the eye/brow/glasses BiSeNet region; outside
         it, the edited image must pixel-match the source reconstruction. Directly
         attacks what the code calls the "~55-60% real eyeglasses accuracy
         ceiling" instead of leaning on ControlNet's feature-map injection hack.
     --clip_prompt_glasses_weight 3.0
         Forces real, CLIP-visible glasses instead of a decision-boundary trick
         the frozen r34 teacher alone rewards (the documented ~91% teacher vs
         ~44% CLIP gap on eyeglasses-add).
     --dds_face_mask
         Keeps the diffusion teacher's gradient inside the allowed local region
         for eyeglasses too, instead of asking it to denoise the whole image.

  2. Aging (Young) not visibly convincing, color-cast artifact
     --clip_prompt_mode directional
         The default 'absolute' mode was found to let the model reach a high
         CLIP "looks old" score via a red/orange color shift instead of real
         structural aging -- this is the actual source of the color-cast
         artifact evaluate_sdflow.py's --age_fine_layer_scale only masks at
         eval time. 'directional' rewards movement along the CLIP pos/neg axis
         from the image's OWN source instead, closing off the color-shift
         shortcut.
     --age_dds_fine_layer_start 12
         The default --dds_fine_layer_start 7 blocks the diffusion teacher's
         gradient from fine W+ layers (7-17) -- exactly where wrinkle/skin
         texture/gray hair live -- so the diffusion teacher currently CANNOT
         teach real aging texture. 12 is the documented middle-ground (vs. 18
         = fully unblocked) that lets the teacher reach most texture layers
         without the full-unblock risk.
     --age_diffusion_weight 0.1 --age_diffusion_interval 8
         Age currently gets the same tiny 0.01 weight as glasses/gender and
         LESS frequent diffusion supervision (16 vs 8 steps) despite being
         the hardest attribute -- both documented as backwards.

  3. General leakage / ID drift (helps ID score across all attributes)
     --residual_max_norm 10.0
         Clips per-sample Direction Bank residual norm, the documented
         suggested value to prevent residual explosion from large DDS
         gradients.
     --controlnet_reg_weight 0.01 (up from default 0.001)
         Stronger L2 penalty on control_skips magnitude. Eval found ControlNet
         injection helps eyeglasses but gives gender/age no measured benefit
         while adding a sparkle artifact; raising the reg weight lets training
         itself learn to shrink the injection toward zero for attributes that
         don't need it, instead of the eval-time --controlnet_disable_attrs
         hard override. Only meaningful if the checkpoint was trained with
         --use_controlnet_injection in the first place -- see --skip_controlnet_tweak.

Usage:
  python training/retrain_fix_score_mismatch.py \\
      --checkpoint_dir ./output/SDFlow/v16 --resume_step 51000 \\
      --run_name v16_fix_score_mismatch \\
      --attribute_index 15 20 39

Anything after -- is forwarded to train_sdflow.py verbatim and OVERRIDES the
recommended flags above (last occurrence wins in argparse), so you can dial
individual pieces back without editing this file, e.g. to skip the ControlNet
regularization change:

  python training/retrain_fix_score_mismatch.py --checkpoint_dir ./output/SDFlow/v16 \\
      --resume_step 51000 --run_name v16_test -- --controlnet_reg_weight 0.001

This only ASSEMBLES and prints/runs the train_sdflow.py command -- it does not
duplicate or reimplement any training logic, so it can't drift from the real
training code path.
"""
import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, 'training', 'train_sdflow.py')


def _latest_step(checkpoint_dir, module_name='prior'):
    """Same auto-detect logic as evaluate_sdflow.py's _latest_step, duplicated
    here (instead of imported) so this launcher stays a lightweight subprocess
    wrapper that doesn't need torch/CLIP/etc. just to list a directory."""
    d = os.path.join(checkpoint_dir, 'save_models')
    if not os.path.isdir(d):
        return None
    steps = []
    for f in os.listdir(d):
        if f.startswith(f'{module_name}-'):
            try:
                steps.append(int(f.split('-')[1]))
            except ValueError:
                pass
    return max(steps) if steps else None


def build_recommended_flags(args):
    flags = [
        '--resume_dir', args.checkpoint_dir,
        '--run_name', args.run_name,
        '--model_name', args.model_name,
        '--attribute_index', *[str(i) for i in args.attribute_index],
        '--resume_optimizer' if args.resume_optimizer else '--no-resume_optimizer',
        '--resume_direction_bank' if args.resume_direction_bank else '--no-resume_direction_bank',

        # -- problem 1: eyeglasses structure --
        '--local_region_loss_weight', '0.5',
        '--use_clip_prompt_loss',
        '--clip_prompt_glasses_weight', '3.0',

        # -- problem 2: aging visibility / color-cast --
        '--clip_prompt_mode', 'directional',
        '--age_dds_fine_layer_start', '12',
        '--age_diffusion_weight', '0.1',
        '--age_diffusion_interval', '8',

        # -- problem 3: leakage / ID drift --
        '--residual_max_norm', '10.0',
    ]
    if args.resume_step is not None:
        flags += ['--resume_step', str(args.resume_step)]
    if not args.skip_dds_face_mask:
        flags += ['--dds_face_mask']
    if not args.skip_diffusion_guidance:
        flags += ['--use_diffusion_guidance']
    if not args.skip_controlnet_tweak:
        flags += ['--controlnet_reg_weight', '0.01']
    return flags


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint_dir', required=True,
                    help='Existing run dir to resume from (passed as --resume_dir).')
    p.add_argument('--resume_step', type=int, default=None,
                    help='Checkpoint step to resume from (default: latest).')
    p.add_argument('--run_name', required=True,
                    help='New run name -- writes to ./output/<model_name>/<run_name>, '
                         'never overwrites --checkpoint_dir.')
    p.add_argument('--model_name', default='SDFlow')
    p.add_argument('--attribute_index', nargs='*', type=int, default=[15, 20, 39])
    p.add_argument('--resume_optimizer', action='store_true',
                    help='Load optimizer state too. Usually left off for this kind of '
                         'targeted fine-tune (fresh optimizer state adapts faster to the '
                         'new loss weights).')
    p.add_argument('--resume_direction_bank', action='store_true',
                    help='Load the direction_bank checkpoint state. Leave off (default) so '
                         'the bank re-initializes from --direction_bank_path with the new '
                         'residual_max_norm/controlnet settings instead of inheriting state '
                         'tuned for the old loss weights.')
    p.add_argument('--skip_dds_face_mask', action='store_true',
                    help='Do not add --dds_face_mask. Only skip if you already validated '
                         'unmasked DDS is fine for your checkpoint.')
    p.add_argument('--skip_diffusion_guidance', action='store_true',
                    help="Do not add --use_diffusion_guidance. Skip only if this checkpoint "
                         "never used diffusion guidance and you don't want to introduce it now "
                         "-- --age_dds_fine_layer_start / --age_diffusion_weight / "
                         "--age_diffusion_interval do nothing without it.")
    p.add_argument('--skip_controlnet_tweak', action='store_true',
                    help='Do not add --controlnet_reg_weight 0.01. Skip if this checkpoint was '
                         'not trained with --use_controlnet_injection (the flag would just be '
                         'silently unused) or you want to keep the original reg weight.')
    p.add_argument('--dry_run', action='store_true',
                    help='Print the assembled train_sdflow.py command and exit without running it.')
    args, passthrough = p.parse_known_args()

    if passthrough and passthrough[0] == '--':
        passthrough = passthrough[1:]

    # train_sdflow.py raises if --resume_dir is set without --resume_step, so
    # auto-detect the latest step here (same logic evaluate_sdflow.py/
    # render_preview.py already use) instead of forcing the caller to look it up.
    if args.resume_step is None:
        args.resume_step = _latest_step(args.checkpoint_dir)
        if args.resume_step is None:
            raise SystemExit(f'No checkpoints found under {args.checkpoint_dir}/save_models/ -- '
                              f'pass --resume_step explicitly if they are laid out differently.')
        print(f'Auto-detected latest step: {args.resume_step}')

    cmd = [sys.executable, TRAIN_SCRIPT] + build_recommended_flags(args) + passthrough

    print('Resuming from:', args.checkpoint_dir,
          f'(step {args.resume_step})' if args.resume_step is not None else '(latest step)')
    print('New run will be written to:',
          os.path.join(PROJECT_ROOT, 'output', args.model_name, args.run_name))
    print('\nCommand:')
    print(' \\\n    '.join(cmd))

    if args.dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == '__main__':
    main()
