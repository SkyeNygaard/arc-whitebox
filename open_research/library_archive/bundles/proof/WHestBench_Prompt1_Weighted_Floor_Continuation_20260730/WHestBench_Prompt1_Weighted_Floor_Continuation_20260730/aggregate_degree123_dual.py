from fractions import Fraction
from decimal import Decimal,getcontext
import json,glob
from pathlib import Path
getcontext().prec=180
OUT=Path(__file__).resolve().parent
cert=json.load(open(OUT/'SIGNED_RANK_DEGREE123_CERTIFICATE.json'))
cand=json.load(open(OUT/'DEGREE123_ENTRYWISE_DUAL_CANDIDATE.json'))
parts=[json.load(open(p)) for p in sorted(glob.glob(str(OUT/'DEGREE123_DUAL_CHUNK_*.json')))]
# Ensure exact coverage and all checks.
ranges=sorted((p['lo'],p['hi']) for p in parts)
assert ranges==[(0,30),(31,60),(61,75),(76,90),(91,105),(106,123)],ranges
assert all(p['positive'] for p in parts)
assert sum(p['checked'] for p in parts)==7381
global_min=min(parts,key=lambda p:Decimal(p['minimum_margin_decimal']))
y=[Fraction(s)*Fraction(1001,1000) for s in cand['dual_weights']]
U=sum(y,Fraction(0));P=Fraction(cert['floor_exact']);upper=P*U
formal=json.load(open(OUT/'KERDOCK_MSE_CERTIFIED_INTERVAL.json'))['kerdock_mse_interval'];Klower=Fraction(formal['lower']);frac=upper/Klower
out={'status':'PASS_EXACT_RATIONAL_VERIFICATION','degree_cutoff':123,
'kernel_upper_source':'direct-C MPFR order-511 jet, exact untruncated closed monomial projections, and rigorous positive-tail bound',
'dual_weight_inflation':'1001/1000','dual_weights':[str(Decimal(v.numerator)/Decimal(v.denominator)) for v in y],
'fixed_selection':cand['fixed_selection'],'reference_floor_exact':cert['floor_exact'],
'checked_positive_entries':7381,'verification_chunks':ranges,
'minimum_margin_decimal':global_min['minimum_margin_decimal'],'minimum_margin_pair':global_min['minimum_margin_pair'],
'minimum_margin_exact_sha256':global_min.get('minimum_exact_margin_sha256','available in chunk record'),
'dual_objective_upper_factor_over_D123':str(Decimal(U.numerator)/Decimal(U.denominator)),
'degree123_family_upper_floor':str(Decimal(upper.numerator)/Decimal(upper.denominator)),
'degree123_family_upper_fraction_of_kerdock_rigorous':str(Decimal(frac.numerator)/Decimal(frac.denominator)),
'target_1_05_floor_20_over_21':str(Decimal(20)/Decimal(21)),
'target_1_05_impossible_in_declared_class_rigorous':frac<Fraction(20,21),
'theorem_class':'finite or countable sums of squared rotation-invariant harmonic comparison kernels supported on degrees 0..123; equivalently completely-positive mixtures of nonnegative harmonic weight vectors',
'logical_role':'upper bound on the best lower floor certifiable by the declared comparison class, not an upper bound on cubature risk'}
json.dump(out,open(OUT/'DEGREE123_ENTRYWISE_DUAL_RECHECK.json','w'),indent=2)
json.dump(out,open(OUT/'DEGREE123_ENTRYWISE_DUAL_RECHECK.json','w'),indent=2)
print(json.dumps({k:out[k] for k in ['status','checked_positive_entries','minimum_margin_decimal','minimum_margin_pair','dual_objective_upper_factor_over_D123','degree123_family_upper_fraction_of_kerdock_rigorous','target_1_05_impossible_in_declared_class_rigorous']},indent=2))
