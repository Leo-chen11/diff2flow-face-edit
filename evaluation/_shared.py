"""Helpers that several evaluation/ scripts had each kept their own copy of.

Nothing here is new behaviour -- each function is the byte-identical body
that was previously duplicated across the scripts listed in its docstring.
They live here so a fix to one no longer has to be remembered in three
places; that was a real hazard, because the copies had already started to
drift in whitespace and quote style even while staying semantically equal.

Deliberately NOT collected here, because the same-named copies are NOT
equivalent and merging them would change behaviour:

  - write_csv       eval_strength_sweep.py writes nothing for an empty row
                    list; compare_strength_sweeps.py writes an empty file,
                    and unions keys across all rows instead of trusting
                    rows[0] for the header.
  - reduce_tensor   common/ops.py skips the divide when world_size is None;
                    models/flows/utils.py falls back to
                    dist.get_world_size() and always divides.
  - _latest_step    evaluate_sdflow.py and retrain_fix_score_mismatch.py
                    differ in how they filter checkpoint filenames.
  - to_pil          dump_multi_attr_edit.py resizes, dump_attr_failures.py
                    does not.
"""
import argparse
import os

import torch.nn.functional as F


def attr_name_map(indices, names):
    """Previously duplicated in eval_strength_sweep.py,
    probe_direction_identity_contamination.py, eval_adversarial_robustness.py."""
    if names and len(names) != len(indices):
        raise ValueError('--attribute_names must have the same length as --attribute_index.')
    if names:
        return {int(idx): name for idx, name in zip(indices, names)}
    return {int(idx): f'attr_{idx}' for idx in indices}


def generate_faces(generator, latents, img_size):
    """Previously duplicated in eval_strength_sweep.py,
    eval_adversarial_robustness.py."""
    faces = generator([latents], input_is_latent=True, randomize_noise=False)[0].clamp(-1, 1)
    if faces.size(-1) != img_size:
        faces = F.interpolate(faces, (img_size, img_size), mode='bilinear', align_corners=False)
    return faces


def parse_named_path(value):
    """argparse type for NAME=path pairs. Previously duplicated in
    plot_strength_curves.py, compare_strength_sweeps.py."""
    if '=' not in value:
        raise argparse.ArgumentTypeError('Expected NAME=path/to/metrics_summary.csv')
    name, path = value.split('=', 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise argparse.ArgumentTypeError('Run name cannot be empty.')
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f'File does not exist: {path}')
    return name, path
