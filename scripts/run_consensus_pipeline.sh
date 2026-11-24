#!/usr/bin/env bash
# The whole held-out comparison on the consensus faces, one step after the other:
#   1. scripts/run_heldout_curves.sh   every method's curve (topped up to span R50 40-100%), the
#                                      coverage check and the 7-point tables / figures
#   2. scripts/comparison_table.py     every method at R50 70 / 85 / 95%: accuracy (R50, r18), ArcFace,
#                                      SSIM, LPIPS, FID / KID, attribute preservation, CLIP
# Step 2 starts as soon as step 1 ends; a method whose curve is missing (it failed in step 1) is left
# out of step 2 and listed, instead of stopping the table.
#
#   nohup bash scripts/run_consensus_pipeline.sh > consensus_pipeline.log 2>&1 &
# Environment: RUN STEP (the model), SDF SDF_STEP (the retrained SDFlow), CR (default 0.1),
#   TARGETS (default "70 85 95"), METHODS (as scripts/run_heldout_curves.sh), CMP_OUT (default
#   output/comparison_consensus)
set -u
cd "$(dirname "$0")/.."
export RUN=${RUN:-output/multi_v1_min_2teach_seed1_bn_ctrl_wsd_from50k_long}
export STEP=${STEP:-130000}
export SDF=${SDF:-output/sdflow_original_5attr/save_models}
export SDF_STEP=${SDF_STEP:-17549}
export CR=${CR:-0.1}
export FACE_SET=consensus
export METHODS=${METHODS:-ours ours_nocr sdflow5 sdflow5_cr styleflow styleflow_cr adatrans gradctrl plugen deltaedit lt interfacegan}
TARGETS=${TARGETS:-70 85 95}
CMP_OUT=${CMP_OUT:-output/comparison_consensus}
C=output/curves_consensus

echo "##### 1/2 curves  $(date '+%F %T')"
bash scripts/run_heldout_curves.sh

echo "##### 2/2 comparison table at R50 $TARGETS%  $(date '+%F %T')"
SPECS=(--method_spec "No-edit:noedit")
add() {   # add <name> <kind> <csv>
    if [ -f "$3" ]; then SPECS+=(--method_spec "$1:$2:$3"); else echo "  - $1: no $3 (left out of the table)"; fi
}
add "Ours"          model+cr      "$C/ours_cr${CR}_off500.csv"
add "Ours (no CR)"  model         "$C/ours_nocr_off500.csv"
add "SDFlow"        sdflow        "$C/sdflow5_off500.csv"
add "SDFlow+CR"     sdflow+cr     "$C/sdflow5_cr${CR}_off500.csv"
add "StyleFlow"     styleflow     "$C/styleflow_off500.csv"
add "StyleFlow+CR"  styleflow+cr  "$C/styleflow_cr${CR}_off500.csv"
add "AdaTrans"      adatrans      "$C/adatrans_off500.csv"
add "GradCtrl"      gradctrl      "$C/gradctrl_off500.csv"
add "PluGeN"        plugen        "$C/plugen_off500.csv"
add "DeltaEdit"     deltaedit     "$C/deltaedit_off500.csv"
add "LT"            lt            "$C/lt_off500.csv"
add "InterFaceGAN"  interfacegan  "$C/interfacegan_off500.csv"

python -u -m scripts.comparison_table --checkpoint_dir "$RUN" --step "$STEP" \
    --independent_attr_weights ./data/r50_celebahq_eval.pth --independent_attr_backbone r50 \
    --face_set consensus --sample_offset 500 --num_samples 500 --target_acc $TARGETS \
    --consistent_reading "$CR" \
    --sdflow_ckpt_dir "$SDF" --sdflow_step "$SDF_STEP" --sdflow_attrs 15 20 39 31 5 --sdflow_img_size 256 \
    --plugen_others encoded --out_dir "$CMP_OUT" "${SPECS[@]}" \
    || echo "!! comparison table failed (see above)"

echo "##### done  $(date '+%F %T')"
echo "  curves check:   $C/coverage_check.md"
echo "  7-point tables: $C/ladder/ladder_r50_{all,common}.md (.png)"
echo "  full table:     $CMP_OUT/ (one .md per target accuracy, comparison.json)"
