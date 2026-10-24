#!/usr/bin/env bash
# BatchNorm-fix ablation from one checkpoint: resume it for MAX_STEPS steps as
#   ctrl    the parent run's exact flags
#   flowbn  + --flow_bn_stats freeze                       (3+ GPUs only)
#   both    + --flow_bn_stats freeze --conditioner_bn freeze
#   both_area  both + --train_resize area                  (only via ARMS)
#   maleid  + --id_hinge_threshold_override 39:0.68,20:0.72  (only via ARMS)
#           a lower ID hinge floor for Young and Male, whose real edits cost
#           identity; needs a parent trained with --id_loss_hinge
#   accum   + --grad_accum_steps 5 --ema_decay 0.995         (only via ARMS)
#           one optimizer step per 5 iterations, i.e. per round of all five
#           attributes; max_steps still counts iterations, so this arm sees
#           the same data with 1/5 the updates (decay 0.995 per update is
#           ~0.999 per iteration, the default)
# then evaluate each arm's last step (--consensus --eval_resize area, plus the
# R50 judge when ./data/r50_celebahq_eval.pth exists, R50=<path> to change it;
# the ConvNeXt judge likewise from ./data/convnext_base_a40_512_classifier.pth, CNX=<path>).
# Arms are dealt round-robin to the GPUs; arms sharing a GPU run one after
# the other. Everything runs in the background; logs in logs/bn_ablation/.
#
#   bash scripts/launch_bn_ablation.sh <run_dir> <step> "<gpu ids>" [max_steps]
#   bash scripts/launch_bn_ablation.sh output/SDFlow/myrun 80000 "0 1 2"
#
# Prints every command before launching it; pass DRY_RUN=1 to only print.
# TAG=<suffix> appends _<suffix> to every run name and log directory, so a
# repeat does not overwrite an earlier one; SEED=<n> overrides the parent
# run's --seed (a repeat with the same seed mostly reproduces the same run).
# ARMS="<arm> ..." picks the arms instead of the GPU-count default.
# EVAL_ALSO="<run name> ..." evaluates those existing runs under
# output/<model>/ at the same final step and with the same eval flags, after
# the first GPU's queue (e.g. an earlier arm, re-scored with the R50 judge).
set -uo pipefail

RUN=${1:?usage: bash scripts/launch_bn_ablation.sh <run_dir> <step> "<gpu ids>" [max_steps]}
STEP=${2:?step missing}
read -r -a GPUS <<< "${3:?gpu ids missing, e.g. \"0 1\"}"
MAX_STEPS=${4:-20000}
RUN=$(cd "$RUN" && pwd) || exit 1
cd "$(dirname "$0")/.."

CFG="$RUN/config.json"
[ -f "$CFG" ] || { echo "no config.json in $RUN"; exit 1; }
PRIOR=$(printf '%s/save_models/prior-%07d' "$RUN" "$STEP")
[ -f "$PRIOR" ] || { echo "no checkpoint $PRIOR"; exit 1; }
MODEL=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('model_name', 'SDFlow'))" "$CFG")
BASE=$(basename "$RUN")
[ "$BASE" = best ] && BASE=$(basename "$(dirname "$RUN")")
END=$((STEP + MAX_STEPS))
TAG=${TAG:+_$TAG}
LOG=logs/bn_ablation$TAG
mkdir -p "$LOG"
R50=${R50:-./data/r50_celebahq_eval.pth}
CNX=${CNX:-./data/convnext_base_a40_512_classifier.pth}

if [ -n "${ARMS:-}" ]; then read -r -a ARMS <<< "$ARMS"
elif [ "${#GPUS[@]}" -ge 3 ]; then ARMS=(ctrl flowbn both); else ARMS=(ctrl both); fi

arm_flags() {
    case $1 in
        ctrl)   ;;
        flowbn) echo "--set flow_bn_stats=freeze" ;;
        both)   echo "--set flow_bn_stats=freeze --set conditioner_bn=freeze" ;;
        both_area) echo "--set flow_bn_stats=freeze --set conditioner_bn=freeze --set train_resize=area" ;;
        maleid) echo "--set id_hinge_threshold_override=39:0.68,20:0.72" ;;
        accum)  echo "--set grad_accum_steps=5 --set ema_decay=0.995" ;;
        *)      echo "unknown arm $1" >&2; exit 1 ;;
    esac
}

for arm in "${ARMS[@]}"; do arm_flags "$arm" > /dev/null; done   # an unknown arm stops here
for arm in "${ARMS[@]}"; do
    if [ "$arm" = maleid ] && [ "$(python -c "import json,sys; print(bool(json.load(open(sys.argv[1])).get('id_loss_hinge')))" "$CFG")" != True ]; then
        echo "arm maleid: $RUN was not trained with --id_loss_hinge, the override would do nothing"; exit 1
    fi
done

eval_for() {
    local cmd="python evaluation/evaluate_sdflow.py --checkpoint_dir output/$MODEL/$1 --step $END --consensus --eval_resize area"
    # the R50 judge too, when its weights are here (evaluate_sdflow loads none by default)
    [ -f "$R50" ] && cmd+=" --independent_attr_weights $R50 --independent_attr_backbone r50"
    [ -f "$CNX" ] && cmd+=" --cnx_judge_weights $CNX"
    echo "$cmd"
}

declare -A QUEUE
for i in "${!ARMS[@]}"; do
    arm=${ARMS[$i]}
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    name="${BASE}_bn_${arm}$TAG"
    # shellcheck disable=SC2046
    train=$(python -m scripts.rebuild_train_cmd --config "$CFG" \
        --set "run_name=$name" --set "resume_dir=$RUN" --set "resume_step=$STEP" \
        --set "max_steps=$MAX_STEPS" ${SEED:+--set seed=$SEED} $(arm_flags "$arm") \
        2>>"$LOG/rebuild.log") \
        || { echo "rebuild_train_cmd failed for $arm, see $LOG/rebuild.log"; exit 1; }
    train=$(echo "$train" | tr -d '\\\n' | tr -s ' ')
    eval_cmd=$(eval_for "$name")
    echo "== $arm on GPU $gpu =="
    echo "   $train"
    echo "   $eval_cmd"
    QUEUE[$gpu]+="CUDA_VISIBLE_DEVICES=$gpu $train > $LOG/${arm}_train.log 2>&1 && CUDA_VISIBLE_DEVICES=$gpu $eval_cmd > $LOG/${arm}_eval.log 2>&1 || echo '!! $arm failed, see $LOG/${arm}_*.log' >> $LOG/status.txt; "
done

for extra in ${EVAL_ALSO:-}; do
    [ -d "output/$MODEL/$extra" ] || { echo "EVAL_ALSO: no output/$MODEL/$extra"; exit 1; }
    gpu=${GPUS[0]}
    cmd=$(eval_for "$extra")
    echo "== re-evaluate $extra on GPU $gpu =="
    echo "   $cmd"
    QUEUE[$gpu]+="CUDA_VISIBLE_DEVICES=$gpu $cmd > $LOG/eval_${extra}.log 2>&1 || echo '!! eval $extra failed' >> $LOG/status.txt; "
done

[ "${DRY_RUN:-0}" = 1 ] && { echo "(DRY_RUN: nothing launched)"; exit 0; }
for gpu in "${!QUEUE[@]}"; do
    nohup bash -c "${QUEUE[$gpu]} echo 'GPU $gpu queue done' >> $LOG/status.txt" > /dev/null 2>&1 &
done
echo "launched. progress: tail -f $LOG/*_train.log   finished arms: cat $LOG/status.txt"
