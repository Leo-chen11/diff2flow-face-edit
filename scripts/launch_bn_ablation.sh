#!/usr/bin/env bash
# BatchNorm-fix ablation from one checkpoint: resume it for MAX_STEPS steps as
#   ctrl    the parent run's exact flags
#   flowbn  + --flow_bn_stats freeze                       (3+ GPUs only)
#   both    + --flow_bn_stats freeze --conditioner_bn freeze
# then evaluate each arm's last step (--consensus --eval_resize area, plus the
# R50 judge when ./data/r50_celebahq_eval.pth exists; R50=<path> to change it).
# Arms are dealt round-robin to the GPUs; arms sharing a GPU run one after
# the other. Everything runs in the background; logs in logs/bn_ablation/.
#
#   bash scripts/launch_bn_ablation.sh <run_dir> <step> "<gpu ids>" [max_steps]
#   bash scripts/launch_bn_ablation.sh output/SDFlow/myrun 80000 "0 1 2"
#
# Prints every command before launching it; pass DRY_RUN=1 to only print.
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
LOG=logs/bn_ablation
mkdir -p "$LOG"
R50=${R50:-./data/r50_celebahq_eval.pth}

if [ "${#GPUS[@]}" -ge 3 ]; then ARMS=(ctrl flowbn both); else ARMS=(ctrl both); fi

arm_flags() {
    case $1 in
        ctrl)   ;;
        flowbn) echo "--set flow_bn_stats=freeze" ;;
        both)   echo "--set flow_bn_stats=freeze --set conditioner_bn=freeze" ;;
    esac
}

declare -A QUEUE
for i in "${!ARMS[@]}"; do
    arm=${ARMS[$i]}
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    name="${BASE}_bn_${arm}"
    # shellcheck disable=SC2046
    train=$(python -m scripts.rebuild_train_cmd --config "$CFG" \
        --set "run_name=$name" --set "resume_dir=$RUN" --set "resume_step=$STEP" \
        --set "max_steps=$MAX_STEPS" $(arm_flags "$arm") 2>>"$LOG/rebuild.log") \
        || { echo "rebuild_train_cmd failed for $arm, see $LOG/rebuild.log"; exit 1; }
    train=$(echo "$train" | tr -d '\\\n' | tr -s ' ')
    eval_cmd="python evaluation/evaluate_sdflow.py --checkpoint_dir output/$MODEL/$name --step $END --consensus --eval_resize area"
    # the R50 judge too, when its weights are here (evaluate_sdflow loads none by default)
    [ -f "$R50" ] && eval_cmd+=" --independent_attr_weights $R50 --independent_attr_backbone r50"
    echo "== $arm on GPU $gpu =="
    echo "   $train"
    echo "   $eval_cmd"
    QUEUE[$gpu]+="CUDA_VISIBLE_DEVICES=$gpu $train > $LOG/${arm}_train.log 2>&1 && CUDA_VISIBLE_DEVICES=$gpu $eval_cmd > $LOG/${arm}_eval.log 2>&1 || echo '!! $arm failed, see $LOG/${arm}_*.log' >> $LOG/status.txt; "
done

[ "${DRY_RUN:-0}" = 1 ] && { echo "(DRY_RUN: nothing launched)"; exit 0; }
for gpu in "${!QUEUE[@]}"; do
    nohup bash -c "${QUEUE[$gpu]} echo 'GPU $gpu queue done' >> $LOG/status.txt" > /dev/null 2>&1 &
done
echo "launched. progress: tail -f $LOG/*_train.log   finished arms: cat $LOG/status.txt"
