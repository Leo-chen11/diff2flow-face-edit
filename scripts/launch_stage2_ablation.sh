#!/usr/bin/env bash
# Stage-2 ablations of the final model. Each arm re-runs STAGE 2 -- resumed from the same stage-1
# checkpoint, with exactly the final run's flags (its config.json) -- with ONE thing removed, then
# scores it on the held-out faces with scripts/editing_curves.py (the protocol of the paper's tables).
#
#   ctrl      the final run's own flags: the control, or with SEED=<n> a seed repeat of stage 2
#             (stage 1 is not repeated)
#   nodds     --no-use_diffusion_guidance        (frozen Stable Diffusion DDS: Young and Male)
#   noclip    --no-use_clip_prompt_loss          (CLIP directional prompt loss)
#   noloc     --local_region_loss_weight 0 and --no-local_region_head (both locality losses)
#   nohead    --no-local_region_head only        (the head locality, added on top of the eyeglasses one)
#   oneteach  r34 alone: no ConvNeXt second teacher, teacher_combine mean
#   idcont    continuous ID loss (1 - cos) instead of the hinge
#
# The term is removed from the resume step on: the first stage's steps still had it. Stage 2 is where
# the edit moves from the direction bank to the flow, so it is the phase these terms shape the flow in;
# a full-chain ablation (stage 1 too) is scripts/launch_chain_fix.sh with S1_SETS / S2_SETS.
# velocity_field (dof -> original) is NOT an arm here: the checkpoints do not load into the other
# architecture, so it needs the full chain.
#
#   bash scripts/launch_stage2_ablation.sh <final stage-2 run dir> "<gpu ids>"
#   bash scripts/launch_stage2_ablation.sh output/SDFlow/multi_v1_min_2teach_seed1_bn_ctrl_wsd_from50k_long "0 1"
#
# Arms are dealt round-robin to the GPUs; arms sharing a GPU run one after the other. Everything runs
# in the background; logs in logs/stage2_ablation<TAG>/, runs named <run>_abl_<arm><TAG>.
# Environment:
#   ARMS="<arm> ..."     pick the arms (default: all but ctrl)
#   TAG=<suffix>         appended to every run name and the log directory
#   SEED=<n>             overrides the run's --seed
#   RESUME_DIR=<dir>     the stage-1 run dir to resume from, when the config.json's own resume_dir
#                        (an absolute path of the machine it was trained on) does not exist here
#   R50=<path>           the R50 judge (default ./data/r50_celebahq_eval.pth)
#   SCALES="..."         edit strengths of the curves (default 0.2 0.4 0.6 0.8 1.0 1.25 1.5)
#   OFFSET / NUM         held-out faces: skip the first OFFSET (default 500), score NUM (default 500)
#   DRY_RUN=1            print the commands only
set -uo pipefail

RUN=${1:?usage: bash scripts/launch_stage2_ablation.sh <final stage-2 run dir> "<gpu ids>"}
read -r -a GPUS <<< "${2:?gpu ids missing, e.g. \"0 1\"}"
RUN=$(cd "$RUN" && pwd) || exit 1
cd "$(dirname "$0")/.."

CFG="$RUN/config.json"
[ -f "$CFG" ] || { echo "no config.json in $RUN"; exit 1; }
cfg() { python -c "import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print('' if v is None else v)" "$CFG" "$1"; }
[ -n "$(cfg resume_dir)" ] && [ -n "$(cfg resume_step)" ] \
    || { echo "$RUN is not a stage-2 run (its config has no resume_dir / resume_step)"; exit 1; }
[ -n "$(cfg max_steps)" ] || { echo "$RUN has no max_steps"; exit 1; }
MODEL=$(cfg model_name); MODEL=${MODEL:-SDFlow}
BASE=$(basename "$RUN")
END=$(( $(cfg resume_step) + $(cfg max_steps) ))
[ -f "$RUN/save_models/prior-$(printf '%07d' "$END")" ] \
    || echo "note: $RUN has no checkpoint at step $END (the control is still scored from the arms' own runs)"
TAG=${TAG:+_$TAG}
LOG=logs/stage2_ablation$TAG
mkdir -p "$LOG"
R50=${R50:-./data/r50_celebahq_eval.pth}
SCALES=${SCALES:-0.2 0.4 0.6 0.8 1.0 1.25 1.5}
OFFSET=${OFFSET:-500}
NUM=${NUM:-500}

if [ -n "${ARMS:-}" ]; then read -r -a ARMS <<< "$ARMS"; else ARMS=(nodds noclip noloc nohead oneteach idcont); fi

arm_sets() {   # the --set key=value pairs of an arm
    case $1 in
        ctrl)     ;;
        nodds)    echo "use_diffusion_guidance=false" ;;
        noclip)   echo "use_clip_prompt_loss=false" ;;
        noloc)    echo "local_region_loss_weight=0 local_region_head=false" ;;
        nohead)   echo "local_region_head=false" ;;
        oneteach) echo "extra_teacher_weights=none teacher_combine=mean" ;;
        idcont)   echo "id_loss_hinge=false" ;;
        *)        echo "unknown arm $1" >&2; exit 1 ;;
    esac
}
for arm in "${ARMS[@]}"; do arm_sets "$arm" > /dev/null; done   # an unknown arm stops here

declare -A QUEUE
for i in "${!ARMS[@]}"; do
    arm=${ARMS[$i]}
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    name="${BASE}_abl_${arm}$TAG"
    sets=(--set "run_name=$name")
    [ -n "${SEED:-}" ] && sets+=(--set "seed=$SEED")
    [ -n "${RESUME_DIR:-}" ] && sets+=(--set "resume_dir=$RESUME_DIR")
    for kv in $(arm_sets "$arm"); do sets+=(--set "$kv"); done
    train=$(python -m scripts.rebuild_train_cmd --config "$CFG" "${sets[@]}" 2>>"$LOG/rebuild.log") \
        || { echo "rebuild_train_cmd failed for $arm, see $LOG/rebuild.log"; exit 1; }
    train=$(echo "$train" | tr -d '\\\n' | tr -s ' ')
    curves="python -m scripts.editing_curves --checkpoint_dir output/$MODEL/$name --step $END"
    curves+=" --independent_attr_weights $R50 --independent_attr_backbone r50 --method model"
    curves+=" --scales $SCALES --sample_offset $OFFSET --num_samples $NUM"
    echo "== $arm on GPU $gpu =="
    echo "   $train"
    echo "   $curves"
    QUEUE[$gpu]+="CUDA_VISIBLE_DEVICES=$gpu $train > $LOG/${arm}_train.log 2>&1 && CUDA_VISIBLE_DEVICES=$gpu $curves > $LOG/${arm}_curves.log 2>&1 || echo '!! $arm failed, see $LOG/${arm}_*.log' >> $LOG/status.txt; "
done

[ "${DRY_RUN:-0}" = 1 ] && { echo "(DRY_RUN: nothing launched)"; exit 0; }
for gpu in "${!QUEUE[@]}"; do
    nohup bash -c "${QUEUE[$gpu]} echo 'GPU $gpu queue done' >> $LOG/status.txt" > /dev/null 2>&1 &
done
echo "launched. progress: tail -f $LOG/*_train.log   finished arms: cat $LOG/status.txt"
echo "curves: output/$MODEL/<run>_abl_<arm>/curves_model_s${END}_off${OFFSET}.csv"
