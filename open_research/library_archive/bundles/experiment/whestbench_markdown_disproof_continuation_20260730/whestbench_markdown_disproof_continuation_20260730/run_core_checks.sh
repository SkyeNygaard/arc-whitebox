#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/code"
python verify_weighted_rank_l23_independent.py
PYTHONPATH=. python verify_weighted_rank_l23_sympy.py
python - <<'PYCHECK'
import json
from decimal import Decimal
from pathlib import Path
r=Path.cwd().parent/'results'
a=json.load(open(r/'INDEPENDENT_WEIGHTED_RANK_L23_RECOMPUTED.json'))
b=json.load(open(r/'SYMPY_WEIGHTED_RANK_L23_RECOMPUTED.json'))
assert a['binding_degree']==b['binding_degree']==8
pairs=[('floor_lower','floor_lower'),('fraction_kerdock','fraction_of_kerdock_mse'),('improvement_cap','maximum_improvement_factor')]
for ka,kb in pairs:
    x=Decimal(a[ka]); y=Decimal(b[kb])
    assert abs(x-y)/abs(x) < Decimal('1e-68'), (ka,x,y)
print('PASS: dual-stack degree-23 certificate agrees beyond 68 relative decimal digits')
for name in ['INDEPENDENT_WEIGHTED_RANK_L23_RECOMPUTED.json','SYMPY_WEIGHTED_RANK_L23_RECOMPUTED.json']:
    (r/name).unlink(missing_ok=True)
PYCHECK
