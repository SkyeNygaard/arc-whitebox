#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from decimal import Decimal
from pathlib import Path

STALE_KEYS={
 'additive_suboptimality','multiplicative_ratio_kerdock_over_optimum',
 'relative_excess','relative_excess_percent','optimum_mse_lower_bound'
}
REQUIRED={
 'actual_additive_suboptimality','actual_multiplicative_ratio_kerdock_over_infimum',
 'actual_relative_excess','actual_relative_excess_percent',
 'certificate_expression_B_minus_A0_interval','certified_optimum_mse_lower_bound'
}

def validate(path:Path):
 d=json.loads(path.read_text()); errors=[]
 missing=sorted(REQUIRED-set(d));
 if missing: errors.append(f'missing canonical keys: {missing}')
 present=sorted(STALE_KEYS & set(d));
 if present: errors.append(f'stale two-sided keys present: {present}')
 def eq(pathkeys,val):
  z=d
  try:
   for k in pathkeys:z=z[k]
   if Decimal(str(z))!=Decimal(val): errors.append(f'{".".join(pathkeys)} must equal {val}, got {z}')
  except KeyError: pass
 eq(['actual_additive_suboptimality','lower'],'0')
 eq(['actual_multiplicative_ratio_kerdock_over_infimum','lower'],'1')
 eq(['actual_relative_excess','lower'],'0')
 eq(['actual_relative_excess_percent','lower'],'0')
 conclusion=d.get('human_readable_conclusion','').lower()
 if 'infimum' not in conclusion: errors.append('human-readable conclusion should say infimum')
 return {'file':str(path),'passed':not errors,'errors':errors}

if __name__=='__main__':
 rows=[validate(Path(p)) for p in sys.argv[1:]]
 print(json.dumps(rows,indent=2))
 if any(not r['passed'] for r in rows): raise SystemExit(1)
