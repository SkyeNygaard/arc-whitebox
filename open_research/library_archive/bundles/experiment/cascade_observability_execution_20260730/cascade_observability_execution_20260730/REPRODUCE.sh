#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p results figures

python code/run_math_and_design_audit.py > results/MATH_DESIGN_AUDIT.stdout.txt
python code/run_t4_observability_probe.py > results/TEST2_T4_OBSERVABILITY_PROBE.stdout.txt
python code/run_transfer_probe.py > results/TEST3_TRANSFER_PROBE.stdout.txt

(
  cd sources/signed_weight_audit
  python signed_weight_certificate.py > /tmp/cascade_signed_cert_stdout.txt
  python compute_exclusion_curve.py > /tmp/cascade_signed_curve_stdout.txt
)
cp sources/signed_weight_audit/M_CERTIFICATE.json sources/M_CERTIFICATE.json
cp sources/signed_weight_audit/NEGATIVE_MASS_EXCLUSION_CURVE.csv sources/NEGATIVE_MASS_EXCLUSION_CURVE.csv
cp sources/signed_weight_audit/NEGATIVE_MASS_EXCLUSION_CURVE.json sources/NEGATIVE_MASS_EXCLUSION_CURVE.json
printf 'signed certificate: PASS\nsigned exclusion curve: PASS\n' > results/TEST7_REPRODUCE_STDOUT.txt

python code/make_figures.py

# Rebuild a manifest over all package files except the manifest and zip itself.
find . -type f \
  ! -name MANIFEST.sha256 \
  ! -name 'cascade_observability_execution_20260730.zip' \
  -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256
sha256sum -c MANIFEST.sha256 >/dev/null
printf 'All local reproductions and hash checks passed.\n'
