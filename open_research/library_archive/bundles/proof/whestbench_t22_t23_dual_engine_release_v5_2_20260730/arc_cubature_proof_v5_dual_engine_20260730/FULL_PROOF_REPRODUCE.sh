#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

rm -f results/formal_base_certificate_d256_L32.json
rm -f results/formal_gpp_chunk_*.json
rm -f results/FORMAL_CERTIFICATE_D256_L32.json
rm -f results/FORMAL_SIGN_LOGIC_AUDIT.json
rm -f results/FORMAL_INTERVAL_AUDIT.json
rm -f results/KERDOCK_MULTIPLICITY_PROOF.json
rm -f results/FORMAL_KERNEL_MEAN_D256_L32.json
rm -f results/FORMAL_DELSARTE_BOUND_D256_L32.json
rm -f results/FORMAL_NEAR_OPTIMALITY_THEOREM_D256_L32.json
rm -f results/THEOREM_PACKAGE_VERIFICATION.json

python formal_base_certificate.py
ranges=(
  '0 100' '100 100' '200 100' '300 100' '400 100' '500 50'
  '550 1' '551 9' '560 5' '565 5' '570 5' '575 10' '585 15'
  '600 5' '605 1' '606 1' '607 6' '613 14'
  '627 100' '727 100' '827 100' '927 100' '1027 52'
)
for spec in "${ranges[@]}"; do
  read -r start count <<<"$spec"
  python formal_sign_chunk.py --start "$start" --count "$count"
done
python verify_theorem_package.py
python verify_manifest.py
