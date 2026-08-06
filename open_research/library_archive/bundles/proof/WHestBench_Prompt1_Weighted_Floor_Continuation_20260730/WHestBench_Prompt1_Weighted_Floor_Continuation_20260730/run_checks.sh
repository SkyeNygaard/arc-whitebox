#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python certify_degree123.py
python crosscheck_degree123_closed_projection.py
python verify_degree123_dual_chunk.py 0 30
python verify_degree123_dual_chunk.py 31 60
python verify_degree123_dual_chunk.py 61 75
python verify_degree123_dual_chunk.py 76 90
python verify_degree123_dual_chunk.py 91 105
python verify_degree123_dual_chunk.py 106 123
python aggregate_degree123_dual.py
python - <<'PY'
import json
p=json.load(open('SIGNED_RANK_DEGREE123_RECHECK.json'))
d=json.load(open('DEGREE123_ENTRYWISE_DUAL_RECHECK.json'))
assert p['status']=='PASS'
assert p['fraction_kerdock_lower_rigorous'].startswith('0.909436093131522')
assert d['status']=='PASS_EXACT_RATIONAL_VERIFICATION'
assert d['checked_positive_entries']==7381
assert d['degree123_family_upper_fraction_of_kerdock_rigorous'].startswith('0.944932965937273')
assert d['target_1_05_impossible_in_declared_class_rigorous']
print('ALL CERTIFICATE CHECKS PASS')
PY
