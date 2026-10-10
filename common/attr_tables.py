"""Attribute tables shared by training, evaluation and scripts.

One copy of each table: training/train_sdflow.py, evaluation/evaluate_sdflow.py
and the scripts import from here. Light on purpose (no torch models), so a
script can read a table without pulling in the training or eval stack.
"""

# Official CelebA list_attr_celeba.txt column order (0-indexed). Index 15 =
# Eyeglasses, 20 = Male, 24 = No_Beard, 31 = Smiling, 33 = Wavy_Hair, 39 =
# Young -- matches this project's attribute indices directly.
CELEBA_ALL_ATTRS = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
    'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
    'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young',
]

# Attributes that may change WITH an edit of the key attribute (CelebA global
# indices): they are part of what the edit means or are hidden by it. Used by
# --preserve_boundaries (eval) and the "penal" side-effect columns of
# scripts/compare_runs_matched_id.py. Chosen from the --leak40 side-effect
# report: e.g. Male edits move facial hair / makeup / earrings, Smiling moves
# cheekbones and an open mouth, Young moves gray hair, eye bags and jowls,
# glasses occlude the eye region, bangs hide the hairline. Everything else
# (Big_Nose, Chubby, hair colour, Straight_Hair, ...) should stay put.
PRESERVE40_ALLOW = {
    15: [1, 3, 12, 23],                              # Eyeglasses: brows, eye bags, narrow eyes (occluded)
    20: [0, 1, 12, 16, 18, 22, 24, 30, 34, 36],      # Male: stubble, brows, goatee, makeup, mustache,
                                                     #   no-beard, sideburns, earrings, lipstick
    39: [3, 4, 13, 14, 17, 28],                      # Young: eye bags, bald, chubby, double chin,
                                                     #   gray hair, receding hairline
    31: [19, 21, 23],                                # Smiling: cheekbones, mouth open, narrow eyes
    5: [28, 32, 33],                                 # Bangs: receding hairline, straight / wavy hair
}

# Training soft-target ends, keyed by ABSOLUTE CelebA attribute index:
# (target when removing, target when adding). Also what evaluate_sdflow.py
# --edit_target train aims at.
SOFT_TARGET_TABLE = {
    15: (0.10, 0.90),   # eyeglasses needs a stronger local-edit signal
    20: (0.20, 0.80),   # gender should move without forcing a full identity flip
    39: (0.20, 0.80),   # age is the most identity-sensitive edit; conservative
}
DEFAULT_SOFT_TARGET = (0.20, 0.80)

# BiSeNet skin class: the region a --controlnet_region_cond checkpoint reads
# for age (39) edits, in training and in eval.
AGE_TEXTURE_REGION_CLASS = [1]


def bank_num_k(bank, default=1):
    """Directions per attribute (K) of a direction-bank file: its 'num_k' (or the
    older 'K'), else the slot axis of a 4-D direction_units (A, K, L, D)."""
    for key in ('num_k', 'K'):
        if bank.get(key) is not None:
            return int(bank[key])
    du = bank.get('direction_units')
    if du is not None and getattr(du, 'ndim', 0) == 4:
        return int(du.shape[1])
    return int(default)
