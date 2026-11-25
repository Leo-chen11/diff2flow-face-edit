#!/usr/bin/env bash
# Get the baselines that have no held-out curve yet ready, and (START=1) run their curves.
#   SDFlow (retrained, 5 attributes), GradCtrl, PluGeN, DeltaEdit, InterFaceGAN
# Their files are looked for in the other copy of the project (WIN, default the Windows desktop folder
# seen from WSL) and copied here where the evaluation expects them; nothing existing is overwritten.
# InterFaceGAN's boundaries are fitted here (CPU, minutes) when no copy is found. Then a status table says
# which methods can run now; START=1 runs exactly those (no CR anywhere: the main comparison).
#
#   bash scripts/setup_remaining_baselines.sh              # prepare and report
#   START=1 bash scripts/setup_remaining_baselines.sh      # ... and run the curves of the ready ones
# Environment: WIN (other copy of the project), RUN (the final model dir), FACE_SET (default consensus)
set -u
cd "$(dirname "$0")/.."
WIN=${WIN:-/mnt/c/Users/User/Desktop/diff2flow-face-edit-claude-generated-image-score-mismatch-ysovdw}
RUN=${RUN:-output/multi_v1_min_2teach_seed1_bn_ctrl_wsd_from50k_long}
FACE_SET=${FACE_SET:-consensus}
SDF_DIR=output/sdflow_original_5attr/save_models
SDF_STEP=${SDF_STEP:-17549}

cp_new() {   # cp_new <src> <dst dir>: copy if the source exists and the target does not
    [ -e "$1" ] || return 1
    mkdir -p "$2"; [ -e "$2/$(basename "$1")" ] || cp "$1" "$2/"
}
find_one() { find "$WIN" -name "$1" 2>/dev/null | head -n 1; }

echo "== looking in $WIN"
[ -d "$WIN" ] || echo "   (this folder does not exist: set WIN=<other copy of the project>)"
for f in config.json model.pth; do
    cp_new "$WIN/output/baselines/gradctrl/$f" output/baselines/gradctrl
    cp_new "$WIN/output/baselines/plugen/$f" output/baselines/plugen
done
for f in prior labeldist; do
    src=$(find_one "$f-$(printf '%07d' "$SDF_STEP")")
    [ -n "$src" ] && cp_new "$src" "$SDF_DIR"
done
for f in net_face.pth fs3.npy; do
    src=$(find_one "$f"); [ -n "$src" ] && cp_new "$src" data/deltaedit
done
if [ ! -f data/interfacegan_boundaries.pth ]; then
    src=$(find_one interfacegan_boundaries.pth)
    if [ -n "$src" ]; then cp "$src" data/
    elif python -c "import sklearn" 2>/dev/null; then
        echo "== fitting the InterFaceGAN boundaries (official train_boundary, CPU)"
        python -m scripts.fit_interfacegan --out ./data/interfacegan_boundaries.pth > logs_fit_interfacegan.txt 2>&1 \
            && tail -n 7 logs_fit_interfacegan.txt || echo "!! fit failed, see logs_fit_interfacegan.txt"
    else
        echo "!! scikit-learn missing: pip install scikit-learn, then run this again"
    fi
fi

echo; echo "== status"
READY=""
check() {   # check <method> <files...>
    m=$1; shift; miss=""
    for f in "$@"; do [ -e "$f" ] || miss="$miss $f"; done
    if [ -z "$miss" ]; then echo "  ready    $m"; READY="$READY $m"; else echo "  MISSING  $m:$miss"; fi
}
check sdflow5 "$SDF_DIR/prior-$(printf '%07d' "$SDF_STEP")" "$SDF_DIR/labeldist-$(printf '%07d' "$SDF_STEP")"
check gradctrl output/baselines/gradctrl/config.json output/baselines/gradctrl/model.pth
check plugen output/baselines/plugen/config.json output/baselines/plugen/model.pth
check deltaedit data/deltaedit/net_face.pth data/deltaedit/fs3.npy
check interfacegan data/interfacegan_boundaries.pth
READY=$(echo $READY)
for m in sdflow5 gradctrl plugen deltaedit interfacegan; do   # those that already have a curve need no run
    [ -f "output/curves_$FACE_SET/${m}_off500.csv" ] && READY=$(echo " $READY " | sed "s/ $m / /" | xargs)
done
echo; echo "== curves still to run, files ready: ${READY:-none}"
if [ "${START:-0}" = 1 ] && [ -n "$READY" ]; then
    RUN="$RUN" FACE_SET="$FACE_SET" METHODS="$READY" nohup bash scripts/run_heldout_curves.sh > heldout_topup.log 2>&1 &
    echo "started in the background (log: heldout_topup.log): grep '==\\|round\\|!!' heldout_topup.log"
elif [ -n "$READY" ]; then
    echo "run them:  START=1 bash scripts/setup_remaining_baselines.sh"
fi
