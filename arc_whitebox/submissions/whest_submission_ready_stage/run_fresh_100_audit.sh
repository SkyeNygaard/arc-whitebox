#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 7 ]; then
 echo "usage: $0 MLP_KPROP_REPO RESULTS_JSON MODEL_NPZ READINESS_OUTPUT MOMENTS_DIR WEIGHTS_DIR AUDIT_OUTPUT [INDICES]" >&2;exit 2
fi
REPO="$(cd "$1" && pwd)";RESULTS="$2";MODEL="$3";READY="$4";MOMENTS="$5";WEIGHTS="$6";OUT="$7";INDICES="${8:-100-199}"
HERE="$(cd "$(dirname "$0")" && pwd)";mkdir -p "$OUT" "$MOMENTS" "$WEIGHTS";runpy(){ uv run --project "$REPO" python "$@"; }
# Downloading needs huggingface_hub and pyarrow, which need not be dependencies of
# ARC's research environment. Resolve them into a throwaway env rather than assuming
# the caller's bare `python` has them (on macOS that is often the 3.9 system Python).
# Version specifiers must stay quoted: unquoted ">=" would be a shell redirect.
DL=(uv run --with 'huggingface_hub>=1.0' --with 'numpy>=2.0' --with 'pyarrow>=15' python)
"${DL[@]}" "$HERE/download_higher_moments.py" "$INDICES" --output "$MOMENTS"
"${DL[@]}" "$HERE/download_official_weights.py" --indices "$INDICES" --output "$WEIGHTS"
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
 --moments-dir "$MOMENTS" --weights-dir "$WEIGHTS" --indices "$INDICES" --configs "$CONFIG" \
 ${ARGS[@]+"${ARGS[@]}"} --device cpu --dtype float64 --output "$OUT/fresh100.json"
runpy "$HERE/summarize_factorized_k3_v2.py" "$OUT/fresh100.json" | tee "$OUT/fresh100_summary.txt"
runpy "$HERE/submission_gate.py" "$OUT/fresh100.json" --output "$OUT/fresh100_gate.json"
