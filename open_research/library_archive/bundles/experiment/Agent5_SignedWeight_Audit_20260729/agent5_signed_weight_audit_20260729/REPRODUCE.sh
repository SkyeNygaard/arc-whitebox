#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python signed_weight_certificate.py > /dev/null
python compute_exclusion_curve.py > /dev/null
python kerdock_signed_stress.py > /dev/null
# The low-dimensional global searches are deterministic but slower.
python lowdim_signed_node_search.py > /dev/null
rm -rf __pycache__ vendor_proof/__pycache__
sha256sum $(find . -maxdepth 2 -type f ! -name SHA256SUMS.txt ! -path '*/__pycache__/*' ! -name '*.pyc' | sort) > SHA256SUMS.txt
printf 'Agent 5 artifacts reproduced.\n'
