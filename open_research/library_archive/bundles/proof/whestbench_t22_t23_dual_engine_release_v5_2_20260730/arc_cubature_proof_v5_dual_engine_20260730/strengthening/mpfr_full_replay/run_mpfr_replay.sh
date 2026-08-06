#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROOF="$(cd "$HERE/../.." && pwd)"
RUN="${1:-$(mktemp -d /tmp/whestbench-mpfr-replay-XXXXXX)}"
mkdir -p "$RUN/gcc" "$RUN/clang"
LIB="$(ldconfig -p | awk '/libmpfr\.so\.6/{print $NF; exit}')"
[[ -n "$LIB" && -f "$LIB" ]] || { echo 'libmpfr.so.6 not found' >&2; exit 2; }
compile_and_run() {
  local cc="$1" dst="$2"
  "$cc" -O3 -Wall -Wextra -Wno-unused-function "$HERE/mpfr_full_replay.c" "$LIB" -lgmp -o "$dst/mpfr_full_replay"
  "$cc" -O3 -Wall -Wextra -Wno-unused-function "$HERE/mpfr_global_minorant.c" "$LIB" -lgmp -o "$dst/mpfr_global_minorant"
  "$cc" -O3 -Wall -Wextra -Wno-unused-function "$HERE/mpfr_kernel_mean_512.c" "$LIB" -lgmp -o "$dst/mpfr_kernel_mean_512"
  "$cc" -O3 -Wall -Wextra -Wno-unused-function "$HERE/mpfr_final_theorem.c" "$LIB" -lgmp -o "$dst/mpfr_final_theorem"
  "$dst/mpfr_full_replay" "$HERE/hpp.tsv" "$HERE/certified.tsv" verify "$dst/verify_certified.json" "$dst/verify_certified.log"
  "$dst/mpfr_full_replay" "$HERE/hpp.tsv" "$HERE/mesh.tsv" regen "$dst/regenerate_mesh.json" "$dst/regenerate_mesh.log"
  "$dst/mpfr_global_minorant" "$HERE/h.tsv" "$HERE/hp.tsv" "$HERE/hpp.tsv" "$HERE/global.tsv" "$dst/global_minorant.json" "$dst/global_minorant.log"
  "$dst/mpfr_kernel_mean_512" "$dst/kernel_mean_512.json"
  "$dst/mpfr_final_theorem" "$HERE/gegenbauer_coeff.tsv" "$dst/kernel_mean_512.json" "$dst/final_theorem.json"
  python "$HERE/verify_exact_coverage.py" --mesh "$HERE/mesh.tsv" --certified "$HERE/certified.tsv" --global-boxes "$HERE/global.tsv" --out "$dst/exact_coverage.json"
  python "$HERE/verify_mpfr_results.py" --run-dir "$dst" --proof-dir "$PROOF" --out "$dst/verification_summary.json"
}
compile_and_run gcc "$RUN/gcc"
if command -v clang >/dev/null 2>&1; then
  compile_and_run clang "$RUN/clang"
  for n in verify_certified.json regenerate_mesh.json global_minorant.json kernel_mean_512.json final_theorem.json exact_coverage.json verification_summary.json; do
    cmp "$RUN/gcc/$n" "$RUN/clang/$n"
  done
  echo 'GCC and Clang outputs are byte-identical.'
fi
echo "MPFR replay passed: $RUN"
