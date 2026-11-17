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
#   METHODS  any of: ours adatrans gradctrl plugen deltaedit sdflow5 lt styleflow interfacegan (default all)
#   OUT      default output/curves
# CSVs: $OUT/<method>_off500.csv, logs beside them.
set -u
cd "$(dirname "$0")/.."
RUN=${RUN:-output/multi_v1_min_2teach_seed1_bn_ctrl_wsd_from50k_long}
STEP=${STEP:-130000}
SDF=${SDF:-output/sdflow_original_5attr/save_models}
SDF_STEP=${SDF_STEP:-17549}
OUT=${OUT:-output/curves}
METHODS=${METHODS:-ours adatrans gradctrl plugen deltaedit sdflow5 lt styleflow interfacegan}
J="--independent_attr_weights ./data/r50_celebahq_eval.pth --independent_attr_backbone r50"
H="--sample_offset 500 --num_samples 500"
mkdir -p "$OUT"

run() {   # <csv name> <editing_curves args...>
    name=$1; shift
    out=$OUT/${name}_off500.csv
    [ -f "$out" ] && { echo "skip $name ($out exists)"; return; }
    echo "== $name  $(date '+%F %T')"
    python -m scripts.editing_curves --checkpoint_dir "$RUN" --step "$STEP" $J $H --out_csv "$out" "$@" \
        > "$OUT/${name}_off500.log" 2>&1 || echo "!! $name failed, see $OUT/${name}_off500.log"
}

for m in $METHODS; do
    case $m in
        ours)         run ours_cr0.1 --method model --consistent_reading 0.1 \
                          --scales 0.2 0.4 0.6 0.8 1.0 1.25 1.5 ;;
        adatrans)     run adatrans --method adatrans --adatrans_dir ./output/baselines/adatrans \
                          --scales 1 2 3 4 5 6 8 10 12 14 16 ;;
        gradctrl)     run gradctrl --method gradctrl --gradctrl_dir ./output/baselines/gradctrl \
                          --scales 1 2 3 4 5 6 7 8 9 11 13 15 18 22 26 ;;
        plugen)       run plugen --method plugen --plugen_dir ./output/baselines/plugen \
                          --scales 0.25 0.5 0.75 1 1.5 2 3 ;;
        deltaedit)    run deltaedit --method deltaedit --deltaedit_dir ./data/deltaedit \
                          --scales 0.25 0.5 0.75 1 1.5 2 3 4 ;;
        sdflow5)      run sdflow5 --method sdflow --sdflow_ckpt_dir "$SDF" --sdflow_step "$SDF_STEP" \
                          --sdflow_attrs 15 20 39 31 5 --sdflow_img_size 256 \
                          --scales -1 -0.75 -0.5 -0.25 0 0.25 0.5 1 1.5 2 ;;
        lt)           run lt --method lt --scales 0.25 0.5 0.75 1 1.5 2 ;;
        styleflow)    run styleflow --method styleflow --scales 0.25 0.5 0.75 1 1.5 2 3 ;;
        interfacegan) run interfacegan --method boundary --scales 1 2 3 4 5 6 ;;
        *)            echo "unknown method $m" ;;
    esac
done
echo "all done $(date '+%F %T')"
