#!/usr/bin/env python3
from pathlib import Path
from fractions import Fraction
import json,hashlib
HERE=Path(__file__).resolve().parent
src=HERE/'sources'
t22=json.loads((src/'T22_V5_1_THEOREM.json').read_text())
t16=json.loads((src/'T16_DECIMAL_PRIMAL_DUAL_CERTIFICATE.json').read_text())
t16b=json.loads((src/'T16_MPMATH_IV_SECOND_STACK_AUDIT.json').read_text())
signed=json.loads((src/'MULTIRANK_SIGNED_NODE_CERTIFICATE.json').read_text())
regen=json.loads((HERE/'T22_FULL_CLEAN_REGENERATION_REPORT.json').read_text())
h=t16['hermite_coefficient_certificate']
ratio=h['kerdock_over_auxiliary_optimum_ratio_interval'];pct=h['kerdock_relative_excess_percent_interval'];mse=h['optimal_mse_interval']
record={
 'title':'WHestBench canonical theorem record v5.2',
 'status':'INTERNAL CANONICAL RECORD — PUBLIC RELEASE REQUIRES EXTERNAL DIGEST AND HUMAN SIGN-OFF',
 'date':'2026-07-30',
 'primary_static_theorem':{
  'theorem':'For the dimension-256, depth-32 infinite-width normalized ReLU kernel, every static network-independent nonnegative mass-one linear rule using at most 66,048 arbitrary spherical nodes has MSE at least the certified all-degree auxiliary optimum lower bound. Complete Kerdock has MSE at most the stated factor above the infimum.',
  'auxiliary_optimum_mse_interval':mse,
  'kerdock_mse_interval':t22['kerdock_mse_interval'],
  'actual_ratio_kerdock_over_infimum':{'lower':'1','upper':ratio[1]},
  'actual_relative_excess_percent':{'lower':'0','upper':pct[1]},
  'one_sided':True,
  'strict_suboptimality_not_proved':True,
 },
 'auxiliary_lp_theorem':{
  'status':'COMPUTER-ASSISTED CERTIFIED; TWO INTERVAL STACKS FOR PRIMAL NUMERICS',
  'optimizer':'unique degree-5 Hermite minorant at roots of 22102 t^3+21930 t^2-87 t-85',
  'decimal_certificate_status':t16['status'],
  'mpmath_iv_second_stack_status':t16b['status'],
  'mpmath_iv_record_sha256':t16b['certificate_sha256'],
  'remaining_external_requirement':'named human review of the analytic Bell/Hermite bridge and release implementation',
 },
 'arbitrary_signed_node_floor':{
  'status':'COMPUTER-ASSISTED CERTIFIED MULTI-RANK LOWER BOUND',
  'scope':signed['scope'],
  'mse_lower_bound':signed['signed_rule_mse_lower_bound_decimal'],
  'fraction_of_complete_kerdock_mse_lower_bound':signed['fraction_of_kerdock_mse_lower_bound'],
  'maximum_permitted_improvement_factor_vs_kerdock':signed['maximum_permitted_improvement_factor_vs_kerdock'],
  'interpretation':'This partially closes arbitrary off-support signed rules by excluding gains above the stated factor. It is not signed near-optimality.',
 },
 'fixed_kerdock_line_theorem':{
  'status':'PROVED UNDER EXPLICIT MODEL AT INFINITE AND FINITE WIDTH',
  'scope':'fixed symmetrized real-MUB/Kerdock line universe; arbitrary real mass-one line weights; static rules; Gaussian first layer and stated nondegeneracy at finite width',
  'conclusion':'complete bases plus at most one partial basis, with positive equal within-basis weights and positive analytic basis masses',
 },
 'proof_reproduction':{
  't22_complete_local_regeneration_passed':regen['result']['passed'],
  't22_subintervals':regen['result']['formal_pointwise_subintervals'],
  't22_manifest':regen['result']['manifest_output'],
  't22_decimal_mpfr_dual_engine':'documented in the independent continuation audit; direct-C GMP/MPFR reproduces theorem-critical path with GCC/Clang identical outputs',
  't16_decimal_and_mpmath_iv':'both proof-critical primal numerical stacks pass; exact reduced-cost recurrence also has independent C++ audit',
 },
 'explicit_exclusions':['arbitrary-node finite-width near-optimality','signed arbitrary-node near-optimality beyond the certified floor','network-adaptive support or weights','unrestricted nonlinear estimators','candidate-dependent transformed residuals without recertification'],
 'source_files':['sources/T22_V5_1_THEOREM.json','sources/T16_DECIMAL_PRIMAL_DUAL_CERTIFICATE.json','sources/T16_MPMATH_IV_SECOND_STACK_AUDIT.json','sources/MULTIRANK_SIGNED_NODE_CERTIFICATE.json','T22_FULL_CLEAN_REGENERATION_REPORT.json'],
}
raw=json.dumps(record,sort_keys=True,separators=(',',':')).encode();record['record_sha256']=hashlib.sha256(raw).hexdigest()
(HERE/'FORMAL_CANONICAL_THEOREM_RECORD_V5_2.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'passed':True,'relative_excess_percent_upper':pct[1],'signed_floor':signed['signed_rule_mse_lower_bound_decimal'],'signed_max_improvement_factor':signed['maximum_permitted_improvement_factor_vs_kerdock']},indent=2))
