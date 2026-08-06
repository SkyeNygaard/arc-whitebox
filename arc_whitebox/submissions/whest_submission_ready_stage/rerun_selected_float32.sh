#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 7 ]; then
 echo "usage: $0 MLP_KPROP_REPO RESULTS_JSON MODEL_NPZ MOMENTS_DIR WEIGHTS_DIR READINESS_OUTPUT_DIR FLOAT32_OUTPUT" >&2;exit 2
fi
REPO="$(cd "$1" && pwd)";RESULTS="$2";MODEL="$3";MOMENTS="$4";WEIGHTS="$5";READY="$6";OUT="$7"
HERE="$(cd "$(dirname "$0")" && pwd)";mkdir -p "$OUT";runpy(){ uv run --project "$REPO" python "$@"; }
CONFIG=$(runpy - "$READY/selected_plan.json" <<'PY'
import json,sys;print(json.load(open(sys.argv[1]))['selected'][0]['config_text'])
PY
)
CAL=$(runpy - "$READY/selected_plan.json" <<'PY'
import json,sys;print(json.load(open(sys.argv[1]))['selected'][0].get('calibration') or 'NONE')
PY
)
ARGS=();if [ "$CAL" != NONE ];then ARGS=(--calibration "$CAL");fi
runpy "$HERE/eval_factorized_k3_hybrid_v2.py" --results-json "$RESULTS" --model "$MODEL" \
 --moments-dir "$MOMENTS" --weights-dir "$WEIGHTS" --split test --configs "$CONFIG" \
 "${ARGS[@]}" --device cpu --dtype float32 --output "$OUT/test15_float32.json"
runpy "$HERE/summarize_factorized_k3_v2.py" "$OUT/test15_float32.json" | tee "$OUT/test15_float32_summary.txt"
