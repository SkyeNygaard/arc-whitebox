#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
python code/verify_oracle_proof_completion.py
python code/verify_weighted_rank_floor.py
python - <<'PY'
import json
from pathlib import Path
root=Path('.')
for p in sorted((root/'results').glob('*.json')):
    json.load(open(p))
required=[
'T42_POSITIVE_DEFINITE_AUXILIARY_RESIDUAL.md',
'T43_ARBITRARY_SIGNED_NODE_RANK_FLOOR.md',
'T44_PHASE_INFORMATION_BOUNDS.md',
'T45_SYMMETRY_DEFECT_ALIGNMENT_BOUND.md',
'T46_GAUGE_INVARIANT_COEFFICIENT_OBSTRUCTION.md',
'T47_WEIGHTED_HARMONIC_RANK_FLOOR.md',
'whestbench_canonical_research_ledger_20260730_reconciled_v18_oracle_proof_completion.xlsx',
]
missing=[x for x in required if not (root/x).exists()]
if missing: raise SystemExit(f'missing: {missing}')
frozen=json.load(open(root/'results/weighted_rank_floor_degree15_frozen.json'))
recomputed=json.load(open(root/'results/weighted_rank_floor_degree15_recomputed.json'))
pairs=[('binding_degree','binding_degree'),('floor_lower','floor_lower'),('fraction_kerdock','fraction_of_kerdock_mse'),('improvement_cap','maximum_improvement_factor')]
for a,b in pairs:
    if str(frozen[a])!=str(recomputed[b]):
        raise SystemExit(f'T47 mismatch {a}/{b}: {frozen[a]} != {recomputed[b]}')
print('PACKAGE PASS')
PY
