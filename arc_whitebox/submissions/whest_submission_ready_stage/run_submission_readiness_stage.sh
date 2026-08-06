#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 6 ]; then
  echo "usage: $0 MLP_KPROP_REPO RESULTS_JSON MODEL_NPZ MOMENTS_DIR WEIGHTS_DIR OUTPUT_DIR [DEVICE] [DTYPE]" >&2
  exit 2
fi
REPO="$(cd "$1" && pwd)"
RESULTS="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
MODEL="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
MOMENTS="$(cd "$4" && pwd)"
WEIGHTS="$(cd "$5" && pwd)"
OUT="$6"
DEVICE="${7:-cpu}"
DTYPE="${8:-float64}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"

runpy() { uv run --project "$REPO" python "$@"; }

runpy "$HERE/api_compat_check.py" --output "$OUT/api_compat.json"
runpy "$HERE/check_numeric_inputs.py" --results-json "$RESULTS" --weights-dir "$WEIGHTS" > "$OUT/numeric_inputs.jsonl"

# The original 15 test MLPs remain untouched. Seven validation MLPs fit feature
# calibration; the other eight select the hybrid configuration.
runpy "$HERE/calibrate_factorized_k3_features.py" \
  --results-json "$RESULTS" --moments-dir "$MOMENTS" --weights-dir "$WEIGHTS" \
  --start-mlp 0 --max-mlps 7 --pairs-per-layer 4096 \
  --device "$DEVICE" --dtype "$DTYPE" --output "$OUT/calibration_layerwise.json"
runpy "$HERE/make_calibration_variants.py" "$OUT/calibration_layerwise.json" \
  --global-output "$OUT/calibration_global.json"

CONFIG_GRID="0.35,0,0;0.5,0,0;0.65,0,0;0.35,0,0.5;0.5,0,0.5;0.65,0,0.5;0.35,0,1;0.5,0,1;0.65,0,1;0.35,0.5,0;0.5,0.5,0;0.65,0.5,0;0.35,0.5,0.5;0.5,0.5,0.5;0.65,0.5,0.5;0.35,0.5,1;0.5,0.5,1;0.65,0.5,1;0.35,1,0;0.5,1,0;0.65,1,0;0.35,1,0.5;0.5,1,0.5;0.65,1,0.5;0.35,1,1;0.5,1,1;0.65,1,1"

# Broad 3-MLP search for each feature-calibration variant.
for MODE in none global layerwise; do
  CAL_ARGS=()
  if [ "$MODE" = global ]; then CAL_ARGS=(--calibration "$OUT/calibration_global.json"); fi
  if [ "$MODE" = layerwise ]; then CAL_ARGS=(--calibration "$OUT/calibration_layerwise.json"); fi
  runpy "$HERE/eval_factorized_k3_hybrid_v2.py" \
    --results-json "$RESULTS" --model "$MODEL" --moments-dir "$MOMENTS" --weights-dir "$WEIGHTS" \
    --split valid --start-mlp 7 --max-mlps 3 --configs "$CONFIG_GRID" \
    ${CAL_ARGS[@]+"${CAL_ARGS[@]}"} --device "$DEVICE" --dtype "$DTYPE" --output "$OUT/smoke_${MODE}.json"
done

# Carry the top four configurations from every calibration mode into the full
# eight-MLP tuning subset. This avoids selecting calibration from only 3 MLPs.
for MODE in none global layerwise; do
  runpy "$HERE/select_hybrid_configs.py" "$OUT/smoke_${MODE}.json" \
    --top-k 4 --min-win 0.34 --max-guard 0.75 \
    --output "$OUT/smoke_shortlist_${MODE}.json"
  SHORT_CONFIGS=$(runpy - "$OUT/smoke_shortlist_${MODE}.json" <<'PYCODE'
import json,sys
r=json.load(open(sys.argv[1]))
print(';'.join(x['config_text'] for x in r['selected']))
PYCODE
)
  CAL_ARGS=()
  if [ "$MODE" = global ]; then CAL_ARGS=(--calibration "$OUT/calibration_global.json"); fi
  if [ "$MODE" = layerwise ]; then CAL_ARGS=(--calibration "$OUT/calibration_layerwise.json"); fi
  runpy "$HERE/eval_factorized_k3_hybrid_v2.py" \
    --results-json "$RESULTS" --model "$MODEL" --moments-dir "$MOMENTS" --weights-dir "$WEIGHTS" \
    --split valid --start-mlp 7 --max-mlps 8 --configs "$SHORT_CONFIGS" \
    ${CAL_ARGS[@]+"${CAL_ARGS[@]}"} --device "$DEVICE" --dtype "$DTYPE" --output "$OUT/tuning8_${MODE}.json"
  runpy "$HERE/summarize_factorized_k3_v2.py" "$OUT/tuning8_${MODE}.json" \
    | tee "$OUT/tuning8_${MODE}_summary.txt"
done

# Select exactly one plan using validation only.
runpy "$HERE/select_hybrid_configs.py" \
  "$OUT/tuning8_none.json" "$OUT/tuning8_global.json" "$OUT/tuning8_layerwise.json" \
  --top-k 1 --min-win 0.625 --max-guard 0.75 --output "$OUT/selected_plan.json"
FINAL_CONFIG=$(runpy - "$OUT/selected_plan.json" <<'PYCODE'
import json,sys
print(json.load(open(sys.argv[1]))['selected'][0]['config_text'])
PYCODE
)
FINAL_CAL=$(runpy - "$OUT/selected_plan.json" <<'PYCODE'
import json,sys
print(json.load(open(sys.argv[1]))['selected'][0].get('calibration') or 'NONE')
PYCODE
)
FINAL_CAL_ARGS=()
if [ "$FINAL_CAL" != NONE ]; then FINAL_CAL_ARGS=(--calibration "$FINAL_CAL"); fi

# One and only one evaluation on the untouched 15-MLP test split.
runpy "$HERE/eval_factorized_k3_hybrid_v2.py" \
  --results-json "$RESULTS" --model "$MODEL" --moments-dir "$MOMENTS" --weights-dir "$WEIGHTS" \
  --split test --configs "$FINAL_CONFIG" ${FINAL_CAL_ARGS[@]+"${FINAL_CAL_ARGS[@]}"} \
  --device "$DEVICE" --dtype "$DTYPE" --output "$OUT/test15.json"
runpy "$HERE/summarize_factorized_k3_v2.py" "$OUT/test15.json" | tee "$OUT/test15_summary.txt"

echo "Complete. Final held-out result: $OUT/test15.json"
