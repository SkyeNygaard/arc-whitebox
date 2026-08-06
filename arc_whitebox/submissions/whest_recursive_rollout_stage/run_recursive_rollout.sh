#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 5 ]]; then
  echo "usage: $0 RESULTS_JSON MODEL_NPZ MOMENTS_DIR WEIGHTS_DIR OUTPUT_PREFIX [FEATURE_MODE]" >&2
  exit 2
fi
RESULTS=$1
MODEL=$2
MOMENTS=$3
WEIGHTS=$4
PREFIX=$5
MODE=${6:-oracle_x1}
HERE=$(cd "$(dirname "$0")" && pwd)
RAW="${PREFIX}_raw.json"
CLIP="${PREFIX}_psdclip.json"

# Primary test: no PSD repair. Eigenvalues are diagnosed but not modified.
python "$HERE/eval_recursive_rollout_x1.py" \
  --results-json "$RESULTS" \
  --model "$MODEL" \
  --moments-dir "$MOMENTS" \
  --weights-dir "$WEIGHTS" \
  --output "$RAW" \
  --feature-mode "$MODE" \
  --psd-mode diagnose \
  --alpha-grid 0,0.2,0.35,0.5,0.65,0.8,1.0 \
  --fit-metric final_mean_mse \
  --validation-limit 0
python "$HERE/summarize_recursive_rollout.py" "$RAW"

ALPHA=$(python - "$RAW" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['best_alpha'])
PY
)

# Upper-bound/stability test: same held-out alpha with diagonal-preserving PSD repair.
python "$HERE/eval_recursive_rollout_x1.py" \
  --results-json "$RESULTS" \
  --model "$MODEL" \
  --moments-dir "$MOMENTS" \
  --weights-dir "$WEIGHTS" \
  --output "$CLIP" \
  --feature-mode "$MODE" \
  --psd-mode clip \
  --fixed-alpha "$ALPHA" \
  --validation-limit 0
python "$HERE/summarize_recursive_rollout.py" "$CLIP"
