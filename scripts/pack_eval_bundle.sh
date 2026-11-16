#!/usr/bin/env bash
# Pack what the held-out evaluation (scripts/editing_curves.py, scripts/comparison_table.py,
# scripts/render_edits.py) needs into ONE tar, to evaluate on another machine. Run it in the repo
# root of the machine that has the files; on the other machine, in its repo root:  tar -xf <tar>
#
# Inside the tar everything sits where the eval scripts look by default:
#   data/...                                   judges, generator, latents, boundaries, bank
#   data/FFHQ/<file>                           only the first N_TEST test-split images
#   output/<run name>/config.json + save_models/*-<STEP>
#   output/sdflow_original_5attr/save_models/{prior,labeldist}-<SDF_STEP>
#   output/baselines/{adatrans,lt,styleflow}/...
# Symlinks are followed (a run linked from the Desktop is packed as its files).
#
#   RUN=output/<run> STEP=130000 SDF=<dir with prior-0017549> bash scripts/pack_eval_bundle.sh
# Environment:
#   RUN        the run dir (required)            STEP      its step (default 130000)
#   SDF        retrained SDFlow ckpt dir         SDF_STEP  default 17549   (skipped when unset)
#   ADA LT STYLEFLOW   baseline dirs (default output/baselines/{adatrans,lt,styleflow})
#   N_TEST     test images to pack (default 1000: covers --sample_offset 500 --num_samples 500)
#   OUT        output tar (default eval_bundle.tar)
# Anything missing is reported and skipped; the tar holds the rest.
set -uo pipefail
cd "$(dirname "$0")/.."

RUN=${RUN:?set RUN=<run dir>}
STEP=${STEP:-130000}
SDF=${SDF:-}
SDF_STEP=${SDF_STEP:-17549}
ADA=${ADA:-output/baselines/adatrans}
LT=${LT:-output/baselines/lt}
STYLEFLOW=${STYLEFLOW:-output/baselines/styleflow}
N_TEST=${N_TEST:-1000}
OUT=${OUT:-eval_bundle.tar}
S=$(printf '%07d' "$STEP")

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
MISSING=()
add() {   # add <source> <path inside the tar>
    if [ -e "$1" ]; then
        mkdir -p "$STAGE/$(dirname "$2")"
        ln -s "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")" "$STAGE/$2"
    else
        MISSING+=("$1")
    fi
}

for f in ffhq.txt ffhq_e4e_latents.pth ffhq_e4e_preds.pth stylegan2-ffhq-config-f.pt \
         r34_a40_age_256_classifier.pth r50_celebahq_eval.pth celeba_attr_resnet18.pth \
         model_ir_se50.pth parsing_bisenet.pth direction_bank_multi.pth attr_boundaries_wplus.pth; do
    add "data/$f" "data/$f"
done

NAME=$(basename "$(cd "$RUN" && pwd)")
add "$RUN/config.json" "output/$NAME/config.json"
found=0
for f in "$RUN"/save_models/*-"$S"; do
    [ -e "$f" ] || continue
    add "$f" "output/$NAME/save_models/$(basename "$f")"; found=1
done
[ $found = 1 ] || MISSING+=("$RUN/save_models/*-$S")

if [ -n "$SDF" ]; then
    T=$(printf '%07d' "$SDF_STEP")
    for m in prior labeldist; do
        add "$SDF/$m-$T" "output/sdflow_original_5attr/save_models/$m-$T"
    done
fi
add "$ADA" output/baselines/adatrans
add "$LT" output/baselines/lt
add "$STYLEFLOW" output/baselines/styleflow

if [ -f data/ffhq.txt ]; then
    python - "$N_TEST" > "$STAGE/test_files.txt" << 'EOF'
import sys
import pandas as pd
df = pd.read_csv('data/ffhq.txt', index_col=None)
df = df[df['split'].values.astype(bool)]           # train=False rows, as models/dataset.py
for p in df['path'].values[:int(sys.argv[1])]:
    print(p)
EOF
    while IFS= read -r p; do add "data/FFHQ/$p" "data/FFHQ/$p"; done < "$STAGE/test_files.txt"
    rm "$STAGE/test_files.txt"
fi

echo "packing into $OUT ..."
tar -chf "$OUT" -C "$STAGE" . || { echo "tar failed"; exit 1; }
echo "done: $OUT ($(du -h "$OUT" | cut -f1)); run dir inside: output/$NAME"
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "NOT packed (missing here):"
    printf '  %s\n' "${MISSING[@]}" | head -30
    [ ${#MISSING[@]} -gt 30 ] && echo "  ... ${#MISSING[@]} in all"
fi
