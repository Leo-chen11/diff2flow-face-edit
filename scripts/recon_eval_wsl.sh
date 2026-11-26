#!/usr/bin/env bash
# The evaluations the recon ablation still needs (docs/improvement_round2.md, section 4: the three
# conditions), on a machine that reaches the runs through another copy of the project (WSL -> the
# Windows desktop folder):
#   copy   config.json + the step's checkpoints + existing curves of nodds_b8, nodds_s2_b8 and
#          recon_nodds_b8 into output/abl/ (nothing existing is overwritten), and the ConvNeXt teacher
#          the audit loads, if it is not in data/ yet
#   run    the no-CR held-out curve of each (skipped where it exists), then the mechanism audit of
#          recon_nodds_b8 and nodds_b8 on the development faces, then what to paste
#
#   bash scripts/recon_eval_wsl.sh copy
#   nohup bash scripts/recon_eval_wsl.sh run > recon_eval.log 2>&1 &
# Environment: WIN (the other copy), P (the run prefix), STEP (default 100000), RUNS
set -u
cd "$(dirname "$0")/.."
WIN=${WIN:-/mnt/c/Users/User/Desktop/diff2flow-face-edit-claude-generated-image-score-mismatch-ysovdw}
P=${P:-multi_v1_min_2teach_seed1_bn_ctrl_wsd_from50k}
STEP=${STEP:-100000}
RUNS=${RUNS:-nodds_b8 nodds_s2_b8 recon_nodds_b8}
A=output/abl
S=$(printf '%07d' "$STEP")
CNX=data/convnext_base_a40_512_classifier.pth

cpn() {   # cpn <files...> <dir>: copy each file the dir does not have yet; fails if a source is missing
    local dst=${!#} n=$(($# - 1)) f rc=0
    mkdir -p "$dst"
    for f in "${@:1:$n}"; do
        [ -e "$f" ] || { rc=1; continue; }
        [ -e "$dst/$(basename "$f")" ] || cp "$f" "$dst/" || rc=1
    done
    return $rc
}

copy() {
    mkdir -p "$A"
    ok=1
    for r in $RUNS; do
        s="$WIN/output/${P}_abl_$r"; d="$A/${P}_abl_$r"
        if [ ! -d "$s" ]; then echo "ERROR $r: no folder $s"; ok=0; continue; fi
        mkdir -p "$d/save_models"
        cpn "$s/config.json" "$d"
        if ! cpn "$s"/save_models/*-"$S" "$d/save_models"; then
            echo "ERROR $r: no step $STEP; its last steps:"; ls "$s/save_models" | sed 's/.*-//' | sort -u | tail -3; ok=0
        fi
        cpn "$s"/curves_* "$d" || true
        echo "$r: $(ls "$d/save_models" | wc -l) checkpoint files, $(ls "$d" | grep -c '^curves_.*csv$') curves"
    done
    if [ ! -f "$CNX" ]; then
        c=$(find "$WIN" -name "$(basename "$CNX")" 2>/dev/null | head -n 1)
        if [ -n "$c" ] && cpn "$c" data; then echo "copied the ConvNeXt teacher from $c"
        else echo "note: no $CNX here or in $WIN -- the audit reads with r34 only (enough for the mechanism check)"; fi
    fi
    echo; echo "paths in the recon config (they should be relative, ./data/...):"
    grep -o '"[a-z_]*\(weights\|file\|root\)": *"[^"]*"' "$A/${P}_abl_recon_nodds_b8/config.json" 2>/dev/null
    echo; [ $ok = 1 ] && echo "copy OK -> nohup bash scripts/recon_eval_wsl.sh run > recon_eval.log 2>&1 &" \
                      || echo "copy INCOMPLETE (see ERROR above)"
}

run() {
    for r in $RUNS; do
        d="$A/${P}_abl_$r"; f="$d/curves_model_s${STEP}_off500_crnone.csv"
        if [ -f "$f" ]; then echo "== $r: no-CR curve exists"; continue; fi
        echo "== $r: no-CR curve  $(date '+%F %T')"
        python -u -m scripts.editing_curves --checkpoint_dir "$d" --step "$STEP" \
            --independent_attr_weights ./data/r50_celebahq_eval.pth --independent_attr_backbone r50 \
            --method model --scales 0.2 0.4 0.6 0.8 1.0 1.25 1.5 --sample_offset 500 --num_samples 500 \
            --out_csv "$f" > "${f%.csv}.log" 2>&1 || echo "ERROR $r curve, see ${f%.csv}.log"
    done
    for r in recon_nodds_b8 nodds_b8; do
        o=output/audit_mech/${r}_test
        if [ -f "$o/decisions.csv" ]; then echo "== audit $r exists"; continue; fi
        echo "== audit $r (development faces)  $(date '+%F %T')"
        extra=(); [ -f "$CNX" ] || extra=(--main_teacher_only)
        python -u -m scripts.audit_verified_direction --checkpoint_dir "$A/${P}_abl_$r" --step "$STEP" \
            ${extra[@]+"${extra[@]}"} --split test --num_faces 500 --out_dir "$o" > "output/audit_mech_${r}_test.log" 2>&1 \
            || echo "ERROR audit $r, see output/audit_mech_${r}_test.log"
    done
    echo; echo "########## paste everything below"
    for r in $RUNS; do
        echo "#### curve $r (no CR)"
        grep '^All,' "$A/${P}_abl_$r/curves_model_s${STEP}_off500_crnone.csv" 2>/dev/null || echo "(missing)"
    done
    for r in recon_nodds_b8 nodds_b8; do
        echo "#### audit $r"
        [ -f "output/audit_mech/${r}_test/decisions.csv" ] && \
            python -m scripts.audit_verified_direction --analyze "output/audit_mech/${r}_test/decisions.csv" \
            | grep -v 'views clearly\|among the' || echo "(missing)"
    done
    echo "all done $(date '+%F %T')"
}

case "${1:-}" in
    copy) copy ;;
    run) run ;;
    *) echo "usage: bash scripts/recon_eval_wsl.sh copy|run"; exit 1 ;;
esac
