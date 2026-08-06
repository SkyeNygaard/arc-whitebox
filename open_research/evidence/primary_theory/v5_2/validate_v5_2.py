#!/usr/bin/env python3
from pathlib import Path
from fractions import Fraction
import json,hashlib
H=Path(__file__).resolve().parent
r=json.loads((H/'FORMAL_CANONICAL_THEOREM_RECORD_V5_2.json').read_text())
t16=json.loads((H/'sources/T16_DECIMAL_PRIMAL_DUAL_CERTIFICATE.json').read_text())
t16b=json.loads((H/'sources/T16_MPMATH_IV_SECOND_STACK_AUDIT.json').read_text())
s=json.loads((H/'sources/MULTIRANK_SIGNED_NODE_CERTIFICATE.json').read_text())
regen=json.loads((H/'T22_FULL_CLEAN_REGENERATION_REPORT.json').read_text())
assert r['primary_static_theorem']['one_sided']
assert r['primary_static_theorem']['actual_ratio_kerdock_over_infimum']['lower']=='1'
assert r['primary_static_theorem']['actual_relative_excess_percent']['lower']=='0'
assert r['primary_static_theorem']['actual_relative_excess_percent']['upper']==t16['hermite_coefficient_certificate']['kerdock_relative_excess_percent_interval'][1]
assert t16b['status']=='PASSED'
assert regen['result']['passed'] and regen['result']['formal_pointwise_subintervals']==1421 and regen['result']['manifest_output']=='manifest verified: 59 files'
obj=Fraction(s['signed_rule_mse_lower_bound'])
assert obj>0 and all(x['passes'] for x in s['constraint_audit'])
kmse_hi=Fraction('2.4336603575430052276094665026697645914811206370055599695108464279151347033914533e-7')
assert kmse_hi/obj < Fraction(31,10) # strict certified improvement ceiling below 3.1x
copy=dict(r);stored=copy.pop('record_sha256');raw=json.dumps(copy,sort_keys=True,separators=(',',':')).encode();assert hashlib.sha256(raw).hexdigest()==stored
print(json.dumps({'passed':True,'t22_clean_regeneration':True,'t16_second_interval_stack':True,'signed_floor_exact':str(obj),'signed_max_improvement_below_3_1x':True},indent=2))
