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
# CSVs: $OUT/<method>_off500.csv, logs beside them. After each method, up to ROUNDS (default 3)
# coverage rounds add strengths until its curve spans R50 40-100% (scripts/ladder_check.py); a CSV that
# already exists is not re-run, only topped up. At the end: $OUT/coverage_check.md (PASS / FAIL per
# method) and the 7-point tables and figures in $OUT/ladder/.
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

ROUNDS=${ROUNDS:-3}     # coverage rounds per method (scripts/ladder_check.py suggest); 0 = off

run() {   # <csv name> <smallest valid strength> <editing_curves args...>
    name=$1; min=$2; shift 2
    out=$OUT/${name}_off500.csv
    log=$OUT/${name}_off500.log
    if [ -f "$out" ]; then
        echo "-- $name: $out exists (only topped up below)"
    else
        echo "== $name  $(date '+%F %T')"
        python -u -m scripts.editing_curves --checkpoint_dir "$RUN" --step "$STEP" $J $H --out_csv "$out" "$@" \
            > "$log" 2>&1 || { echo "!! $name failed, see $log"; return; }
    fi
    # make the curve span R50 40-100%: smaller strengths where its weakest already edits past 40%,
    # larger ones where it is still rising at its strongest (scripts/ladder_check.py); the later
    # --scales replaces the method's grid for that extra run only
    for r in $(seq 1 "$ROUNDS"); do
        extra=$(python -m scripts.ladder_check suggest "$out" --min_strength "$min")
        [ -z "$extra" ] && break
        echo "   $name coverage round $r: adding strengths $extra  $(date '+%T')"
        tmp=$OUT/${name}_extra_off500.csv
        python -u -m scripts.editing_curves --checkpoint_dir "$RUN" --step "$STEP" $J $H "$@" \
            --scales $extra --out_csv "$tmp" >> "$log" 2>&1 || { echo "!! $name extra strengths failed, see $log"; break; }
        python -m scripts.ladder_check merge "$out" "$tmp" --out "$out" && rm -f "$tmp" "${tmp%.csv}.json"
    done
}

for m in $METHODS; do
    case $m in
        ours)         run ours_cr$CR 0 --method model --consistent_reading "$CR" \
                          --scales 0.1 0.2 0.3 0.4 0.5 0.6 0.8 0.95 1.0 1.25 1.5 2.0 ;;
        ours_nocr)    run ours_nocr 0 --method model --scales 0.1 0.2 0.3 0.4 0.5 0.6 0.8 0.95 1.0 1.25 1.5 2.0 ;;
        adatrans)     run adatrans 0 --method adatrans --adatrans_dir ./output/baselines/adatrans \
                          --scales 0.5 1 2 3 4 5 6 8 10 12 14 16 20 ;;
        gradctrl)     run gradctrl 0 --method gradctrl --gradctrl_dir ./output/baselines/gradctrl \
                          --scales 1 2 3 4 5 6 7 8 9 11 13 15 18 22 26 32 ;;
        plugen)       run plugen 0 --method plugen --plugen_dir ./output/baselines/plugen --plugen_others encoded \
                          --scales 0 0.1 0.25 0.5 0.75 1 1.5 2 3 4 5 6 8 ;;
        deltaedit)    run deltaedit 0 --method deltaedit --deltaedit_dir ./data/deltaedit \
                          --scales 0.1 0.25 0.5 0.75 1 1.5 2 3 4 6 8 12 ;;
        sdflow5)      run sdflow5 -5 --method sdflow --sdflow_ckpt_dir "$SDF" --sdflow_step "$SDF_STEP" \
                          --sdflow_attrs 15 20 39 31 5 --sdflow_img_size 256 \
                          --scales -1 -0.75 -0.5 -0.25 0 0.25 0.5 1 1.5 2 3 ;;
        sdflow5_cr)   run sdflow5_cr$CR -5 --method sdflow --sdflow_ckpt_dir "$SDF" --sdflow_step "$SDF_STEP" \
                          --sdflow_attrs 15 20 39 31 5 --sdflow_img_size 256 --consistent_reading "$CR" \
                          --scales -1 -0.75 -0.5 -0.25 0 0.25 0.5 1 1.5 2 3 ;;
        lt)           run lt 0 --method lt --scales 0.1 0.25 0.5 0.75 1 1.5 2 3 4 ;;
        styleflow)    run styleflow 0 --method styleflow --scales 0.1 0.25 0.5 0.75 1 1.5 2 3 4 5 ;;
        styleflow_cr) run styleflow_cr$CR 0 --method styleflow --consistent_reading "$CR" \
                          --scales 0.1 0.25 0.5 0.75 1 1.5 2 3 4 5 ;;
        interfacegan) run interfacegan 0 --method interfacegan \
                          --interfacegan_boundaries ./data/interfacegan_boundaries.pth \
                          --scales 0.25 0.5 1 1.5 2 2.5 3 4 5 6 8 10 ;;
        interfacegan_lr) run interfacegan_lr 0 --method boundary --scales 1 2 3 4 5 6 ;;
        *)            echo "unknown method $m" ;;
    esac
done
echo "all done $(date '+%F %T')"
# the standard: every method read at R50 40 50 60 70 80 90 100%, or the missing points explained
python -m scripts.ladder_check report --curves "$OUT" || echo "(coverage check: FAIL -- run this script again to top up, see $OUT/coverage_check.md)"
for pool in all common; do python -m scripts.accuracy_ladder --curves "$OUT" --pool $pool > /dev/null; done
echo "tables and figures: $OUT/ladder/ladder_r50_{all,common}.{md,png}"
