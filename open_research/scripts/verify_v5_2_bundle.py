#!/usr/bin/env python3
"""Cross-check the recovered v5.2 theorem artifacts included in this release.

This is a consistency validator, not an independent interval reconstruction.
"""
from pathlib import Path
from decimal import Decimal
import hashlib, json

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / 'evidence' / 'primary_theory' / 'v5_2'

record = json.loads((V / 'FORMAL_CANONICAL_THEOREM_RECORD_V5_2.json').read_text())
t16 = json.loads((V / 'T16_PRIMAL_DUAL_CERTIFICATE.json').read_text())
t16b = json.loads((V / 'T16_MPMATH_IV_SECOND_STACK_AUDIT.json').read_text())
endpoint = json.loads((V / 'T16_ENDPOINT_CERTIFICATE.json').read_text())

assert record['primary_static_theorem']['one_sided'] is True
assert record['primary_static_theorem']['strict_suboptimality_not_proved'] is True
assert record['auxiliary_lp_theorem']['mpmath_iv_second_stack_status'] == 'PASSED'
assert t16b['status'] == 'PASSED'
assert t16b['krawczyk']['all_nonconstant_positive'] is True
assert Decimal(t16b['outer_log_derivative']['interval']['upper']) < 3
assert Decimal(t16b['kappa6_plus_3B62']['minimum_H_lower']) > 0

# The v5.2 public constant is the upper endpoint of the T16 interval.
expected = t16['hermite_coefficient_certificate']['kerdock_relative_excess_percent_interval'][1]
actual = record['primary_static_theorem']['actual_relative_excess_percent']['upper']
assert actual == expected, (actual, expected)

# Explicit endpoint separation closes the equality-only-at-contacts wording.
assert endpoint['status'] == 'PASS'
plus = [Decimal(x) for x in endpoint['K32_minus_h_at_plus_one']]
minus = [Decimal(x) for x in endpoint['K32_minus_h_at_minus_one']]
assert plus[0] > Decimal('0.017') and plus[1] >= plus[0]
assert minus[0] > Decimal('2e-7') and minus[1] >= minus[0]

# Verify the record's self-hash.
copy = dict(record)
stored = copy.pop('record_sha256')
raw = json.dumps(copy, sort_keys=True, separators=(',', ':')).encode()
assert hashlib.sha256(raw).hexdigest() == stored

regen = (V / 'T22_FULL_CLEAN_REGENERATION_REPORT.md').read_text()
assert '**Status:** Passed locally.' in regen
assert '1,421' in regen and 'manifest verified: 59 files' in regen

print(json.dumps({
    'verified': True,
    'kind': 'bundle consistency check, not independent interval reconstruction',
    'nonnegative_relative_excess_percent_upper': actual,
    't16_second_stack': 'PASSED',
    't16_endpoint_separation': 'PASSED',
    't22_clean_regeneration_report_present': True,
}, indent=2))
