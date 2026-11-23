#!/usr/bin/env bash
# Edit speed of every method, and speed against quality for the ways of solving our flow's ODE.
#   1. scripts/speed_benchmark.py: ms per face, NFE, peak memory for every method (all solver
#      variants for the flow methods), batch 1 and 8
#   2. scripts/editing_curves.py once per solver variant (our model, CR 0.1) on the DEVELOPMENT faces:
#      choosing a solver is a design decision, so it is made off the held-out faces
#   3. scripts/solver_tradeoff.py: one table of speed-up against identity / accuracy at matched accuracy
# A step whose output exists is skipped, so a stopped study resumes by running this again. Run nothing
# else on the GPU meanwhile (it changes the times).
#
#   RUN=<final run dir> STEP=130000 bash scripts/run_speed_study.sh
# Environment:
#   RUN, STEP        the model (default: the final long run at 130000)
#   VARIANTS         solver variants (default below; see scripts/speed_benchmark.py)
#   METHODS          methods timed in step 1 (default: all; missing ones are skipped)
#   OFFSET / NUM     faces of the quality curves (default 0 / 500: the development faces)
#   CR               the reading rule's margin for the curves (default 0.1; CR= for none)
#   OUT              default output/speed_study
#   SKIP_CURVES=1    step 1 only
set -u
cd "$(dirname "$0")/.."
RUN=${RUN:-output/multi_v1_min_2teach_seed1_bn_ctrl_wsd_from50k_long}
STEP=${STEP:-130000}
VARIANTS=${VARIANTS:-trained fast dopri5@1e-3+fast dopri5@1e-4+fast rk4x8+fast rk4x4+fast midpointx4+fast}
METHODS=${METHODS:-ours sdflow styleflow adatrans gradctrl plugen deltaedit lt interfacegan}
OFFSET=${OFFSET:-0}
NUM=${NUM:-500}
CR=${CR-0.1}
OUT=${OUT:-output/speed_study}
J="--independent_attr_weights ./data/r50_celebahq_eval.pth --independent_attr_backbone r50"
mkdir -p "$OUT/curves"

echo "== 1. speed  $(date '+%F %T')"
if [ -f "$OUT/speed.json" ]; then echo "skip ($OUT/speed.json exists)"; else
    python -u -m scripts.speed_benchmark --checkpoint_dir "$RUN" --step "$STEP" $J \
        --methods $METHODS --ode $VARIANTS --out_dir "$OUT" > "$OUT/speed.log" 2>&1 \
        || { echo "!! speed benchmark failed, see $OUT/speed.log"; exit 1; }
    tail -n 25 "$OUT/speed.log"
fi
[ "${SKIP_CURVES:-0}" = 1 ] && exit 0

echo "== 2. quality per solver variant (faces $OFFSET..$((OFFSET + NUM)))"
for v in $VARIANTS; do
    csv="$OUT/curves/$v.csv"
    [ -f "$csv" ] && { echo "skip $v"; continue; }
    flags=$(python -m scripts.solver_tradeoff --flags "$v")
    echo "   $v: $flags  $(date '+%T')"
    python -u -m scripts.editing_curves --checkpoint_dir "$RUN" --step "$STEP" $J --method model \
        --scales 0.2 0.4 0.6 0.8 1.0 1.25 1.5 --sample_offset "$OFFSET" --num_samples "$NUM" \
        ${CR:+--consistent_reading $CR} $flags --out_csv "$csv" > "$OUT/curves/$v.log" 2>&1 \
        || echo "!! $v failed, see $OUT/curves/$v.log"
done

echo "== 3. trade-off"
python -m scripts.solver_tradeoff --speed "$OUT/speed.json" --curves_dir "$OUT/curves" --batch 1
python -m scripts.solver_tradeoff --speed "$OUT/speed.json" --curves_dir "$OUT/curves" --batch 8 --out "$OUT/tradeoff_b8.md"
echo "done $(date '+%F %T'): $OUT/speed.md, $OUT/tradeoff.md (.png), $OUT/tradeoff_b8.md"
