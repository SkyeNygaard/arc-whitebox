#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
python "$HERE/code/t16_dense_sanity.py"
python "$HERE/code/finite_width_direct_sanity.py"
python "$HERE/code/verify_reopened_outputs.py"
python "$HERE/code/verify_agent5_outputs.py"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python "$HERE/code/group_invariant_phase_audit.py"
sha256sum -c "$HERE/SHA256SUMS.txt"
