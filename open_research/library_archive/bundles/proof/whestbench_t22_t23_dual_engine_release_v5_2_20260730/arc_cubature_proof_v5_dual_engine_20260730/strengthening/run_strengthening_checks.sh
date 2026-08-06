#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
S="$ROOT/strengthening"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
python "$S/code/independent_kerdock_construction.py" --asset "$S/assets/kerdock_mub5_seed3.npz" --out "$TMP/kerdock.json"
python "$S/code/verify_sign_logic_derived.py" --certificate "$ROOT/results/FORMAL_CERTIFICATE_D256_L32.json" --out "$TMP/sign.json"
PYTHONPATH="$S/pydecimal_shim" python "$S/code/independent_pydecimal_scalar_replay.py" --proof-dir "$ROOT" --out "$TMP/scalar.json"
python "$S/code/run_pydecimal_component_replay.py" --proof-dir "$ROOT" --out "$TMP/components.json"
for pair in \
  "INDEPENDENT_KERDOCK_CONSTRUCTION.json kerdock.json" \
  "DERIVED_SIGN_LOGIC_AUDIT.json sign.json" \
  "INDEPENDENT_PYDECIMAL_SCALAR_REPLAY.json scalar.json"; do
  read -r archived regenerated <<<"$pair"
  cmp "$S/results/$archived" "$TMP/$regenerated"
done
python "$ROOT/verify_full_artifact_manifest.py"
echo 'All strengthening checks passed.'
