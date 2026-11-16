#!/bin/bash
#
# AntiStyler vs SaliSAM-FastSAM (and Saliuitl) on SaliSAM's UCYHPC_inria_yolo11n_patch_vanish
# (288 native-resolution INRIA test frames, every person patched), with SaliSAM's YOLO11n COCO
# victim and SaliSAM's evaluators. From any directory:
#
#     bash salisam_comparisons/run_antistyler_inria_yolo11.sh
#     LIM=5 OUT=/tmp/smoke bash salisam_comparisons/run_antistyler_inria_yolo11.sh   # smoke test
#     STAGES="evaluate compare" bash salisam_comparisons/run_antistyler_inria_yolo11.sh
#
# STAGES (all by default, in this order):
#   defend        antistyler_defend.py on the attacked frames, for each of MODES
#   defend_clean  the same on the clean frames (--clean; for LPR / D(clean) AP)
#   evaluate      evaluate_antistyler.py with SaliSAM's venv: SaliSAM's evaluate_coco /
#                 saliuitl_metrics / attack_success on each AntiStyler run, plus a rescore of
#                 $SALISAM_RUN (and its _clean twin) with identical settings
#   compare       compare.py: one table -> $OUT/comparison.csv, with the Saliuitl
#                 comparison's run when $SALIUITL_RUN has been scored
#
# MODES: antistyled (the official notebook's output; method AntiStyler) and masked_input
# (the paper's reading: the mask zero-filled on the native input; AntiStyler-masked).
#
# Overrides: SALISAM_ROOT ATTACKED VARIANT MODES LIM OUT SALISAM_RUN SALIUITL_RUN
# PY (this repo's venv; runs the defense) EVAL_PY (SaliSAM's venv; runs the evaluators)
# CONFIG; EXTRA flags go last to antistyler_defend.py.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
SALISAM_ROOT=${SALISAM_ROOT:-$(realpath ../SaliSAM)}
export SALISAM_ROOT
EVAL_PY=${EVAL_PY:-$SALISAM_ROOT/.venv/bin/python}
ATTACKED=${ATTACKED:-$SALISAM_ROOT/data/UCYHPC_inria_yolo11n_patch_vanish}
VARIANT=${VARIANT:-all_instance}
MODES=${MODES:-"antistyled masked_input"}
CONFIG=${CONFIG:-salisam_comparisons/antistyler_inria.json}
LIM=${LIM:-}
OUT=${OUT:-salisam_comparisons/outputs}
SUFFIX=$(basename "$ATTACKED")_${VARIANT}_yolo11
SALISAM_RUN=${SALISAM_RUN:-$SALISAM_ROOT/outputs/SaliSAM_FastSAM_$SUFFIX}
SALIUITL_RUN=${SALIUITL_RUN:-$(realpath -m ../Saliuitl/salisam_comparisons/outputs/Saliuitl_$SUFFIX)}
STAGES=${STAGES:-"defend defend_clean evaluate compare"}
EXTRA=${EXTRA:-}
LOGS=$OUT/logs
mkdir -p "$LOGS"

has() { [[ " $STAGES " == *" $1 "* ]]; }
method() { [ "$1" = antistyled ] && echo AntiStyler || echo AntiStyler-masked; }
run_dir() { echo "$OUT/$(method "$1")_$SUFFIX"; }

defend() {   # $1: mode, $2: output dir, rest: extra flags
    local mode=$1 out=$2; shift 2
    # shellcheck disable=SC2086
    "$PY" salisam_comparisons/antistyler_defend.py \
        --imgdir "$ATTACKED/clean" --patch_imgdir "$ATTACKED/$VARIANT" \
        --config "$CONFIG" --defense_output "$mode" --victim_args "$SALISAM_RUN/args.json" \
        --output_dir "$out" ${LIM:+--lim "$LIM"} "$@" $EXTRA 2>&1 | tee "$LOGS/$(basename "$out").log"
}

for mode in $MODES; do
    RUN=$(run_dir "$mode")
    if has defend; then
        echo "== defend ($mode) -> $RUN"
        defend "$mode" "$RUN"
    fi
    if has defend_clean; then
        echo "== defend_clean ($mode) -> ${RUN}_clean"
        defend "$mode" "${RUN}_clean" --clean
    fi
done

if has evaluate; then
    for mode in $MODES; do
        RUN=$(run_dir "$mode")
        echo "== evaluate $RUN"
        "$EVAL_PY" salisam_comparisons/evaluate_antistyler.py --run "$RUN" --visualize 2>&1 \
            | tee "$LOGS/evaluate_$(basename "$RUN").log"
    done
    if [ -d "$SALISAM_RUN/recovered" ]; then
        echo "== evaluate (rescore) $SALISAM_RUN"
        "$EVAL_PY" salisam_comparisons/evaluate_antistyler.py --run "$SALISAM_RUN" \
            --out_root "$OUT/salisam_rescored" 2>&1 | tee "$LOGS/rescore_$(basename "$SALISAM_RUN").log"
    else
        echo "NOTE: no SaliSAM run at $SALISAM_RUN; set SALISAM_RUN to rescore one"
    fi
fi

if has compare; then
    echo "== compare"
    RUNS=()
    RESCORED=$OUT/salisam_rescored/$(basename "$SALISAM_RUN")
    [ -d "$RESCORED" ] && RUNS+=("SaliSAM-FastSAM=$RESCORED")
    [ -d "$SALIUITL_RUN/coco_eval_rec" ] && RUNS+=("Saliuitl=$SALIUITL_RUN")
    for mode in $MODES; do RUNS+=("$(method "$mode")=$(run_dir "$mode")"); done
    "$PY" salisam_comparisons/compare.py "${RUNS[@]}" --csv "$OUT/comparison.csv"
fi
