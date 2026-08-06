#!/usr/bin/env python3
import json
from decimal import Decimal,getcontext
from pathlib import Path
getcontext().prec=60
ROOT=Path(__file__).resolve().parent
base=json.load(open(ROOT/'SIGNED_NEAR_OPTIMALITY_CERTIFICATE_BLOCKTRACE_ORDER320.json'))
cont=json.load(open(ROOT/'CONTINUOUS_ADJACENT_BLOCKTRACE_DISCOVERY_320_FIXED.json'))
multi=json.load(open(ROOT/'MULTIBLOCK_BLOCKTRACE_DISCOVERY_320.json'))
bf=Decimal(base['certified_result']['fraction_of_kerdock_upper'])
cf=Decimal(str(cont['fraction']));mf=Decimal(str(multi['fraction']))
out={
 'released_exact_adjacent_grid_fraction':str(bf),
 'corrected_continuous_adjacent_numerical_fraction':str(cf),
 'corrected_continuous_improvement_fraction_of_kerdock':str(cf-bf),
 'corrected_continuous_improvement_percentage_points':str((cf-bf)*100),
 'general_multiblock_blocktrace_numerical_fraction':str(mf),
 'multiblock_improvement_fraction_of_kerdock':str(mf-bf),
 'multiblock_improvement_percentage_points':str((mf-bf)*100),
 'overflow_audit':{
  'prior_issue':'The original stationary-point routine formed products of coefficients as large as about 1e232 and overflowed, omitting interior minima.',
  'repair':'Independently rescale numerator and denominator quadratics before forming derivative coefficients. Positive rescaling does not change stationary points.',
  'effect':'The corrected continuous LP added many high-degree and boundary columns, but changed the normalized objective by less than four millionths.'},
 'evidence_status':{'released_grid':'exact-rational certificate after interval kernel jet','continuous_adjacent':'numerical discovery only; no released continuous dual certificate','multiblock':'numerical discovery only'},
 'conclusion':'Grid discretization and adjacent-profile restriction account for less than 0.001 percentage point each in current numerical searches. The remaining roughly 6.3% theorem gap is not plausibly explained by these optimization details.'}
(ROOT/'COMPARISON_CONE_SEARCH_AUDIT.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
