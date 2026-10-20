#!/usr/bin/env bash
# Step 1 (scripts/diagnose_bn.py), then step 2 (evaluate_sdflow.py with the
# old nearest resize and with --eval_resize area) on one checkpoint, back to
# back. A failed step 1 does not stop step 2. Logs: <checkpoint_dir>/diag_logs/
#
#   bash scripts/diag_then_eval.sh <checkpoint_dir> [step] [train_batch]
#
# step defaults to the latest checkpoint, train_batch (the run's training
# --batch, which the BatchNorm statistics depend on) to 8.
set -uo pipefail

RUN=${1:?usage: bash scripts/diag_then_eval.sh <checkpoint_dir> [step] [train_batch]}
RUN=$(cd "$RUN" && pwd) || exit 1     # absolute, before moving to the repo root
cd "$(dirname "$0")/.."
STEP=${2:-}
BATCH=${3:-8}
STEP_ARG=()
[ -n "$STEP" ] && STEP_ARG=(--step "$STEP")
LOG="$RUN/diag_logs"
mkdir -p "$LOG"

echo "== step 1: diagnose_bn =="
python scripts/diagnose_bn.py --checkpoint_dir "$RUN" ${STEP_ARG[@]+"${STEP_ARG[@]}"} \
    --diag_batch "$BATCH" 2>&1 | tee "$LOG/1_diagnose_bn.log" \
    || echo "!! diagnose_bn failed (see $LOG/1_diagnose_bn.log); continuing with step 2"

echo "== step 2a: eval, nearest resize (as every earlier eval) =="
python evaluation/evaluate_sdflow.py --checkpoint_dir "$RUN" ${STEP_ARG[@]+"${STEP_ARG[@]}"} \
    --consensus 2>&1 | tee "$LOG/2a_eval_nearest.log"
S2A=$?

echo "== step 2b: eval, area resize =="
python evaluation/evaluate_sdflow.py --checkpoint_dir "$RUN" ${STEP_ARG[@]+"${STEP_ARG[@]}"} \
    --consensus --eval_resize area 2>&1 | tee "$LOG/2b_eval_area.log"
S2B=$?

echo "== done: logs in $LOG (2a exit $S2A, 2b exit $S2B) =="
