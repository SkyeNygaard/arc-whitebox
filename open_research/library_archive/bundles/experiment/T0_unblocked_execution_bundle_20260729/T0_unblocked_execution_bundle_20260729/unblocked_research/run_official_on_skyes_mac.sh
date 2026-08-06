#!/usr/bin/env bash
set -euo pipefail
# Usage: bash run_official_on_skyes_mac.sh /path/to/arc_whitebox /path/to/T0_official_grader_instrumentation_20260729
PROJECT_ROOT="${1:-arc_whitebox}"
T0_ROOT="${2:-$(cd "$(dirname "$0")" && pwd)/T0_official_grader_instrumentation_20260729}"
WH="${WHEST:-$PROJECT_ROOT/.venv/bin/whest}"
PY="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
DATASET="${WHEST_DATASET:-$PROJECT_ROOT/data/official_phase1_mini}"
OUT="${T0_OUTPUT:-$PROJECT_ROOT/results/t0_official_$(date +%Y%m%d_%H%M%S)}"
THREADS="${WHEST_THREADS:-4}"
[[ -x "$WH" ]] || { echo "Missing whest executable: $WH" >&2; exit 2; }
[[ -x "$PY" ]] || { echo "Missing Python executable: $PY" >&2; exit 2; }
[[ -e "$DATASET" ]] || { echo "Missing local dataset: $DATASET" >&2; echo "Set WHEST_DATASET=hf://aicrowd/arc-whestbench-public-2026@v1-phase1 to download it." >&2; exit 2; }
[[ -d "$T0_ROOT/packages" ]] || { echo "Missing T0 package root: $T0_ROOT/packages" >&2; exit 2; }
mkdir -p "$OUT"
export OPENBLAS_NUM_THREADS="$THREADS" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
arms=(production_baseline A42 A43 A43_delta64 A43_basis096 A43_basis064 A43_basis032)
sha256sum "$T0_ROOT"/packages/*.tar.gz > "$OUT/package_hashes.sha256"
printf '%s\n' "$(date -u +%FT%TZ)" > "$OUT/started_at_utc.txt"
"$PY" - <<PY > "$OUT/environment.json"
import json,platform,sys
try:
 import numpy
except Exception: numpy=None
try:
 import flopscope
except Exception: flopscope=None
try:
 import whestbench
except Exception: whestbench=None
print(json.dumps({
 'python':sys.version,'platform':platform.platform(),
 'numpy':getattr(numpy,'__version__',None),
 'flopscope':getattr(flopscope,'__version__',None),
 'whestbench':getattr(whestbench,'__version__',None),
 'dataset':r'''$DATASET''','threads':int('$THREADS')},indent=2))
PY
for arm in "${arms[@]}"; do
  pkg="$T0_ROOT/packages/$arm"
  run="$OUT/$arm"; mkdir -p "$run"
  echo "===== $arm =====" | tee "$run/header.txt"
  cmd=("$WH" run --estimator "$pkg/estimator.py" --dataset "$DATASET" --split mini --runner subprocess --max-threads "$THREADS" --profile)
  printf '%q ' "${cmd[@]}" > "$run/command.sh"; echo >> "$run/command.sh"
  set +e
  "${cmd[@]}" --format json > "$run/stdout.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]] && grep -qiE 'unrecognized arguments.*format|no such option.*format' "$run/stdout.log"; then
    "${cmd[@]}" > "$run/stdout.log" 2>&1; rc=$?
  fi
  set -e
  echo "$rc" > "$run/returncode.txt"
  cat "$run/stdout.log"
  [[ $rc -eq 0 ]] || { echo "$arm failed; see $run/stdout.log" >&2; exit "$rc"; }
done
"$PY" "$(cd "$(dirname "$0")" && pwd)/aggregate_official_results.py" --input "$OUT" --output "$OUT" 2>/dev/null || true
printf '%s\n' "$(date -u +%FT%TZ)" > "$OUT/finished_at_utc.txt"
echo "Completed: $OUT"
