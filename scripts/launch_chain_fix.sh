#!/usr/bin/env bash
# Retrain a two-stage run chain from scratch with the BatchNorm and resize
# fixes on, and score it against the original chain at both stage ends.
#
#   stage 1  <stage1 run>'s exact flags, from step 0
#            + --flow_bn_stats dataset --conditioner_bn freeze
#              --conditioner_bn_freeze_after 2000 --train_resize area
#   stage 2  <stage2 run>'s exact flags, resumed from the NEW stage 1
#            + --flow_bn_stats freeze --conditioner_bn freeze --train_resize area
#
# One GPU, in order: stage 1 -> eval new and original at stage 1's last step
# -> stage 2 -> eval new and original at stage 2's last step. Evals use
# --consensus --eval_resize area, plus the R50 judge when
# ./data/r50_celebahq_eval.pth exists (R50=<path> to change it). New runs are
# named <run>_fix (TAG=<suffix> to change it); logs in logs/chain_fix<TAG>/.
#
#   bash scripts/launch_chain_fix.sh <stage1 run dir> <stage2 run dir> <gpu id>
#   DRY_RUN=1 ... prints the commands only.
#
# The two stages' changes are S1_SETS / S2_SETS (key=value, space-separated; defaults: the fixes above),
# so the same chain retrains with any other change in BOTH stages, e.g. the velocity-field ablation
# (the dof checkpoints do not load into the original field, so it cannot resume from stage 1):
#   TAG=orig S1_SETS="velocity_field=original" S2_SETS="velocity_field=original" \
#       bash scripts/launch_chain_fix.sh <stage1 run dir> <stage2 run dir> <gpu id>
# After stage 2 the new run is also scored by scripts/editing_curves.py on the held-out faces
# (--sample_offset 500), the protocol of the paper's tables.
set -uo pipefail

S1=${1:?usage: bash scripts/launch_chain_fix.sh <stage1 run dir> <stage2 run dir> <gpu id>}
S2=${2:?stage2 run dir missing}
GPU=${3:?gpu id missing}
S1=$(cd "$S1" && pwd) || exit 1
S2=$(cd "$S2" && pwd) || exit 1
cd "$(dirname "$0")/.."

TAG=_${TAG:-fix}
LOG=logs/chain$TAG
mkdir -p "$LOG"
R50=${R50:-./data/r50_celebahq_eval.pth}

cfg() { python -c "import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print('' if v is None else v)" "$1/config.json" "$2"; }
for d in "$S1" "$S2"; do [ -f "$d/config.json" ] || { echo "no config.json in $d"; exit 1; }; done
[ -z "$(cfg "$S1" resume_dir)" ] || { echo "stage 1 ($S1) was itself resumed; this script starts it from step 0"; exit 1; }
[ "$(basename "$(cfg "$S2" resume_dir)")" = "$(basename "$S1")" ] \
    || { echo "stage 2 ($S2) did not resume from $S1 (its resume_dir: $(cfg "$S2" resume_dir))"; exit 1; }
MODEL=$(cfg "$S1" model_name); MODEL=${MODEL:-SDFlow}
END1=$(cfg "$S1" max_steps)
[ -n "$END1" ] || { echo "stage 1 has no max_steps; cannot tell where it ended"; exit 1; }
RES2=$(cfg "$S2" resume_step)
[ "$RES2" = "$END1" ] || { echo "stage 2 resumed at $RES2, stage 1 ended at $END1"; exit 1; }
END2=$((RES2 + $(cfg "$S2" max_steps)))
N1=$(basename "$S1")$TAG
N2=$(basename "$S2")$TAG

build() {   # <config dir> --set ... -> one-line train command
    local c=$1; shift
    python -m scripts.rebuild_train_cmd --config "$c/config.json" "$@" 2>>"$LOG/rebuild.log" \
        | tr -d '\\\n' | tr -s ' '
}
# The flags each stage changes (key=value, space-separated). Default: the BatchNorm and resize fixes.
# Any other change to retrain the chain with (e.g. velocity_field=original in both stages) goes here.
S1_SETS=${S1_SETS-"flow_bn_stats=dataset conditioner_bn=freeze conditioner_bn_freeze_after=2000 train_resize=area"}
S2_SETS=${S2_SETS-"flow_bn_stats=freeze conditioner_bn=freeze conditioner_bn_freeze_after=0 train_resize=area"}
sets() { for kv in $1; do printf -- '--set\n%s\n' "$kv"; done; }
mapfile -t A1 < <(sets "$S1_SETS")
mapfile -t A2 < <(sets "$S2_SETS")
T1=$(build "$S1" --set "run_name=$N1" ${A1[@]+"${A1[@]}"}) || exit 1
T2=$(build "$S2" --set "run_name=$N2" --set "resume_dir=output/$MODEL/$N1" --set "resume_step=$END1" \
    ${A2[@]+"${A2[@]}"}) || exit 1
[ -n "$T1" ] && [ -n "$T2" ] || { echo "rebuild_train_cmd failed, see $LOG/rebuild.log"; exit 1; }

ev() {   # <run dir> <step>
    local cmd="python evaluation/evaluate_sdflow.py --checkpoint_dir $1 --step $2 --consensus --eval_resize area"
    [ -f "$R50" ] && cmd+=" --independent_attr_weights $R50 --independent_attr_backbone r50"
    echo "$cmd"
}

STEPS=(
    "stage1_train|$T1"
    "stage1_eval_new|$(ev "output/$MODEL/$N1" "$END1")"
    "stage1_eval_orig|$(ev "$S1" "$END1")"
    "stage2_train|$T2"
    "stage2_eval_new|$(ev "output/$MODEL/$N2" "$END2")"
    "stage2_eval_orig|$(ev "$S2" "$END2")"
    "stage2_curves_new|python -m scripts.editing_curves --checkpoint_dir output/$MODEL/$N2 --step $END2 --independent_attr_weights $R50 --independent_attr_backbone r50 --method model --scales 0.2 0.4 0.6 0.8 1.0 1.25 1.5 --sample_offset 500 --num_samples 500"
)
Q=""
for s in "${STEPS[@]}"; do
    name=${s%%|*}; cmd=${s#*|}
    echo "== $name =="
    echo "   $cmd"
    # a failed training stops everything after it; a failed eval does not
    case $name in
        *_train) Q+="CUDA_VISIBLE_DEVICES=$GPU $cmd > $LOG/$name.log 2>&1 && echo '$name done' >> $LOG/status.txt || { echo '!! $name failed, see $LOG/$name.log' >> $LOG/status.txt; exit 1; }; " ;;
        *)       Q+="CUDA_VISIBLE_DEVICES=$GPU $cmd > $LOG/$name.log 2>&1 && echo '$name done' >> $LOG/status.txt || echo '!! $name failed, see $LOG/$name.log' >> $LOG/status.txt; " ;;
    esac
done

[ "${DRY_RUN:-0}" = 1 ] && { echo "(DRY_RUN: nothing launched)"; exit 0; }
nohup bash -c "$Q echo 'all done' >> $LOG/status.txt" > /dev/null 2>&1 &
echo "launched. progress: tail -f $LOG/stage1_train.log   status: cat $LOG/status.txt"
