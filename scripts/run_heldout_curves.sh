#!/usr/bin/env bash
# Strength curves of every method on the held-out faces (--sample_offset 500 --num_samples 500),
# the input of scripts/comparison_table.py. One method after another; a method whose CSV is
# already there is skipped, so a stopped run is resumed by running this again.
#
#   bash scripts/run_heldout_curves.sh                 (defaults below)
#   RUN=output/<run> STEP=130000 METHODS="ours adatrans" bash scripts/run_heldout_curves.sh
# Environment:
#   RUN      the final run (default output/multi_v1_min_2teach_seed1_bn_ctrl_wsd_from50k_long)
#   STEP     default 130000
#   SDF      retrained 5-attribute SDFlow ckpt dir (default output/sdflow_original_5attr/save_models,
#            where scripts/pack_eval_bundle.sh puts it)      SDF_STEP default 17549
#   METHODS  any of: ours adatrans gradctrl plugen deltaedit sdflow5 lt styleflow interfacegan (default all);
#            interfacegan_lr = the older
#            InterFaceGAN-style logistic-regression boundaries (not the official fit);
#            ours_nocr = ours without the reading rule; sdflow5_cr / styleflow_cr = the two conditional
#            baselines WITH the same reading rule (CR margin) -- the three methods that edit from a
#            reading of the source, so the only ones the rule acts on
#   CR       the reading rule's margin for ours / sdflow5_cr / styleflow_cr (default 0.1)
#   FACE_SET direction (default: the faces R50 reads clearly, every curve so far) or consensus (the faces
#            R50 AND the CelebA r18 read clearly and the same way: the request does not depend on which
#            judge makes it; the protocol for any table with the reading rule in it)
#   OUT      default output/curves (FACE_SET=consensus: output/curves_consensus)
# CSVs: $OUT/<method>_off500.csv, logs beside them. The strength grids reach below 40% R50 accuracy
# and up to each method's ceiling, so scripts/accuracy_ladder.py can read every method at 40-100%.
#
#   FACE_SET=consensus METHODS="ours ours_nocr sdflow5 sdflow5_cr styleflow styleflow_cr adatrans gradctrl \
#       plugen deltaedit lt interfacegan" bash scripts/run_heldout_curves.sh
set -u
cd "$(dirname "$0")/.."
RUN=${RUN:-output/multi_v1_min_2teach_seed1_bn_ctrl_wsd_from50k_long}
STEP=${STEP:-130000}
SDF=${SDF:-output/sdflow_original_5attr/save_models}
SDF_STEP=${SDF_STEP:-17549}
FACE_SET=${FACE_SET:-direction}
CR=${CR:-0.1}
if [ "$FACE_SET" = consensus ]; then OUT=${OUT:-output/curves_consensus}; else OUT=${OUT:-output/curves}; fi
METHODS=${METHODS:-ours adatrans gradctrl plugen deltaedit sdflow5 lt styleflow interfacegan}
J="--independent_attr_weights ./data/r50_celebahq_eval.pth --independent_attr_backbone r50"
H="--sample_offset 500 --num_samples 500 --face_set $FACE_SET"
mkdir -p "$OUT"

run() {   # <csv name> <editing_curves args...>
    name=$1; shift
    out=$OUT/${name}_off500.csv
    [ -f "$out" ] && { echo "skip $name ($out exists)"; return; }
    echo "== $name  $(date '+%F %T')"
    python -u -m scripts.editing_curves --checkpoint_dir "$RUN" --step "$STEP" $J $H --out_csv "$out" "$@" \
        > "$OUT/${name}_off500.log" 2>&1 || echo "!! $name failed, see $OUT/${name}_off500.log"
}

for m in $METHODS; do
    case $m in
        ours)         run ours_cr$CR --method model --consistent_reading "$CR" \
                          --scales 0.1 0.2 0.3 0.4 0.5 0.6 0.8 0.95 1.0 1.25 1.5 2.0 ;;
        ours_nocr)    run ours_nocr --method model --scales 0.1 0.2 0.3 0.4 0.5 0.6 0.8 0.95 1.0 1.25 1.5 2.0 ;;
        adatrans)     run adatrans --method adatrans --adatrans_dir ./output/baselines/adatrans \
                          --scales 0.5 1 2 3 4 5 6 8 10 12 14 16 20 ;;
        gradctrl)     run gradctrl --method gradctrl --gradctrl_dir ./output/baselines/gradctrl \
                          --scales 1 2 3 4 5 6 7 8 9 11 13 15 18 22 26 32 ;;
        plugen)       run plugen --method plugen --plugen_dir ./output/baselines/plugen --plugen_others encoded \
                          --scales 0 0.1 0.25 0.5 0.75 1 1.5 2 3 4 5 6 8 ;;
        deltaedit)    run deltaedit --method deltaedit --deltaedit_dir ./data/deltaedit \
                          --scales 0.1 0.25 0.5 0.75 1 1.5 2 3 4 6 8 12 ;;
        sdflow5)      run sdflow5 --method sdflow --sdflow_ckpt_dir "$SDF" --sdflow_step "$SDF_STEP" \
                          --sdflow_attrs 15 20 39 31 5 --sdflow_img_size 256 \
                          --scales -1 -0.75 -0.5 -0.25 0 0.25 0.5 1 1.5 2 3 ;;
        sdflow5_cr)   run sdflow5_cr$CR --method sdflow --sdflow_ckpt_dir "$SDF" --sdflow_step "$SDF_STEP" \
                          --sdflow_attrs 15 20 39 31 5 --sdflow_img_size 256 --consistent_reading "$CR" \
                          --scales -1 -0.75 -0.5 -0.25 0 0.25 0.5 1 1.5 2 3 ;;
        lt)           run lt --method lt --scales 0.1 0.25 0.5 0.75 1 1.5 2 3 4 ;;
        styleflow)    run styleflow --method styleflow --scales 0.1 0.25 0.5 0.75 1 1.5 2 3 4 5 ;;
        styleflow_cr) run styleflow_cr$CR --method styleflow --consistent_reading "$CR" \
                          --scales 0.1 0.25 0.5 0.75 1 1.5 2 3 4 5 ;;
        interfacegan) run interfacegan --method interfacegan \
                          --interfacegan_boundaries ./data/interfacegan_boundaries.pth \
                          --scales 0.25 0.5 1 1.5 2 2.5 3 4 5 6 8 10 ;;
        interfacegan_lr) run interfacegan_lr --method boundary --scales 1 2 3 4 5 6 ;;
        *)            echo "unknown method $m" ;;
    esac
done
echo "all done $(date '+%F %T')"
