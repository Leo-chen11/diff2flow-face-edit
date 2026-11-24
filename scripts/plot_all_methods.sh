#!/usr/bin/env bash
# Every method's held-out strength curves (scripts/run_heldout_curves.sh) on one figure per attribute
# (and "All"): Identity (ArcFace) and a preservation panel against editing accuracy, via
# evaluation/plot_strength_curves.py. Methods whose CSV is not there yet are skipped and listed.
#
#   bash scripts/plot_all_methods.sh
# Environment:
#   CURVES   folder of the <method>_off500.csv files (default output/curves; the consensus-face curves of
#            FACE_SET=consensus bash scripts/run_heldout_curves.sh: CURVES=output/curves_consensus)
#   OUT      output folder (default output/figures/all_methods, or .../all_methods_consensus)
#   OURS     your model's CSV (default: $CURVES/ours_cr0.1_off500.csv, else $CURVES/final_cr0.1_off500.csv)
# Writes, each as .png and .pdf plus a table (.csv) of every curve read at 40 60 80 90 95%:
#   $OUT/r50/        accuracy by the R50 judge,         right panel: other attributes preserved
#   $OUT/r18/        accuracy by the CelebA r18 judge,  right panel: other attributes preserved
#   $OUT/r50_region/ accuracy by R50,                   right panel: pixels outside the edited region
set -u
cd "$(dirname "$0")/.."
CURVES=${CURVES:-output/curves}
case $(basename "$CURVES") in
    curves_consensus) OUT=${OUT:-output/figures/all_methods_consensus} ;;
    *)                OUT=${OUT:-output/figures/all_methods} ;;
esac
if [ -z "${OURS:-}" ]; then
    OURS=$CURVES/ours_cr0.1_off500.csv
    [ -f "$OURS" ] || OURS=$CURVES/final_cr0.1_off500.csv
fi

# name in the legend = CSV
METHODS=(
    "Ours=$OURS"
    "Ours (no CR)=$CURVES/ours_nocr_off500.csv"
    "SDFlow=$CURVES/sdflow5_off500.csv"
    "SDFlow+CR=$CURVES/sdflow5_cr0.1_off500.csv"
    "StyleFlow+CR=$CURVES/styleflow_cr0.1_off500.csv"
    "AdaTrans=$CURVES/adatrans_off500.csv"
    "DeltaEdit=$CURVES/deltaedit_off500.csv"
    "GradCtrl=$CURVES/gradctrl_off500.csv"
    "PluGeN=$CURVES/plugen_off500.csv"
    "StyleFlow=$CURVES/styleflow_off500.csv"
    "LT=$CURVES/lt_off500.csv"
    "InterFaceGAN=$CURVES/interfacegan_off500.csv"
)

RUNS=()
for m in "${METHODS[@]}"; do
    if [ -f "${m#*=}" ]; then
        RUNS+=(--run "$m")
        echo "  + ${m%%=*}  ${m#*=}"
    else
        echo "  - ${m%%=*}  (no ${m#*=} yet: skipped)"
    fi
done
[ ${#RUNS[@]} -gt 0 ] || { echo "no curve CSV found in $CURVES"; exit 1; }

READ="--read_at 40 60 80 90 95"
python evaluation/plot_strength_curves.py "${RUNS[@]}" $READ --output_dir "$OUT/r50" \
    && python evaluation/plot_strength_curves.py "${RUNS[@]}" $READ --x_metric celeb_success --output_dir "$OUT/r18" \
    && python evaluation/plot_strength_curves.py "${RUNS[@]}" $READ --right_panel region --ylim_bg 0 100 \
           --output_dir "$OUT/r50_region" \
    && echo "figures: $OUT/{r50,r18,r50_region}/curves_<attribute>.png (.pdf beside each)"
# every method read at R50 / r18 40 50 ... 100%, with the reason for each missing point
for j in r50 r18; do
    for pool in all common; do
        python -m scripts.accuracy_ladder --curves "$CURVES" --judge $j --pool $pool --out_dir "$OUT/ladder" > /dev/null \
            && echo "ladder: $OUT/ladder/ladder_${j}_${pool}.{md,png}"
    done
done
