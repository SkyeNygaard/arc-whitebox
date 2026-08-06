#!/usr/bin/env python3
"""Replay a conservative inertia-strengthened signed-rule certificate.

This script deliberately does *not* claim the slightly stronger reoptimized
T70 constant, because that reoptimized rational witness was not recovered.
Instead it starts from the released exact-rational degree-280 comparison
witness and applies the positive-index strengthening to the same frozen
coefficient allocation.  The resulting lower bound is marginally weaker than
T70, but it is fully reconstructible from the files in this directory.

Trust boundary: the K_32 Maclaurin lower endpoints in
K32_MACLAURIN_INTERVALS_ORDER320.json are inherited directed intervals.  The
script exactly verifies every operation downstream of those endpoints.
"""
from __future__ import annotations
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path
import json, math

getcontext().prec = 180
ROOT = Path(__file__).resolve().parent
D, N = 256, 66048


def fdec(s: str) -> Fraction:
    return Fraction(Decimal(s))


def harmonic_dim(l: int) -> int:
    if l == 0:
        return 1
    if l == 1:
        return D
    return math.comb(D + l - 1, l) - math.comb(D + l - 3, l - 2)


def decimal_string(q: Fraction, digits: int = 70) -> str:
    return format(Decimal(q.numerator) / Decimal(q.denominator), f'.{digits}E')


def floor_for_negative_count(cert: dict, q: int) -> Fraction:
    """Lower bound with at least q negative-weight support entries.

    Duplicate locations are first consolidated and zero weights removed. A
    resulting rule with at most N support entries and at least q negative
    weights has at most
    N-q positive weights.  For each frozen comparison profile, the released
    witness uses y_j/B_N units of coefficient capacity, where
    B_N=T_j^2/N-S_{2,j}.  Keeping that exact allocation and replacing B_N by
    B_{N-q}=T_j^2/(N-q)-S_{2,j} is a valid stronger objective.
    """
    assert 1 <= q < N
    total = Fraction(0)
    for row in cert['components']:
        s = int(row['s'])
        r = Fraction(row['r'])
        y = fdec(row['y'])
        ds, dt = harmonic_dim(s), harmonic_dim(s + 1)
        T = Fraction(ds) + r * dt
        S2 = Fraction(ds) + r * r * dt
        base = T * T / N - S2
        strengthened = T * T / (N - q) - S2
        assert base > 0 and strengthened > 0
        total += y * strengthened / base
    return total


def main() -> None:
    cert = json.loads((ROOT / 'SIGNED_NEAR_OPTIMALITY_CERTIFICATE_BLOCKTRACE_ORDER320.json').read_text())
    old_report = json.loads((ROOT / 'SIGNED_NEAR_OPTIMALITY_VERIFICATION_BLOCKTRACE_ORDER320.json').read_text())
    assert old_report['verified'] is True
    assert cert['scope']['dimension'] == D
    assert cert['scope']['maximum_nodes'] == N

    kerdock_upper = fdec(cert['certified_result']['kerdock_mse_upper_bound'])
    old_floor = fdec(cert['certified_result']['mse_lower_bound'])
    q1 = floor_for_negative_count(cert, 1)
    assert q1 > old_floor
    # Universal mass-one signed class is a case split: q=0 is covered by the
    # stronger nonnegative theorem, while every genuinely signed consolidated
    # rule has q>=1 and is covered by q1.
    nonnegative_floor = fdec('2.4330928587565937917467205177357824616906898195111938487488890771313270267856063E-7')
    assert nonnegative_floor > q1

    frontier = []
    for q in (1, 1072, 4160, 8192):
        floor = floor_for_negative_count(cert, q)
        frontier.append({
            'minimum_negative_weight_support_entries': q,
            'mse_lower_bound': decimal_string(floor),
            'fraction_of_kerdock_upper': decimal_string(floor / kerdock_upper),
            'maximum_kerdock_over_rule_factor': decimal_string(kerdock_upper / floor),
        })

    ratio = q1 / kerdock_upper
    gain = kerdock_upper / q1
    risk_reduction = 1 - ratio

    # Frozen headline thresholds.
    assert gain < Fraction(1067168, 1000000)  # < 1.067168
    assert frontier[1]['maximum_kerdock_over_rule_factor'].startswith('1.0498337')
    assert floor_for_negative_count(cert, 4160) > kerdock_upper

    report = {
        'verified': True,
        'status': 'exact-rational downstream replay from inherited kernel intervals',
        'scope': cert['scope'],
        'source_witness': 'SIGNED_NEAR_OPTIMALITY_CERTIFICATE_BLOCKTRACE_ORDER320.json',
        'source_witness_original_floor': decimal_string(old_floor),
        'universal_case_split': {
            'nonnegative_rules': 'covered by the stronger nonnegative mass-one theorem',
            'genuinely_signed_consolidated_rules': 'have at least one negative-weight support entry and are covered by q=1',
        },
        'audited_inertia_strengthened_floor': decimal_string(q1),
        'audited_floor_fraction_of_kerdock_upper': decimal_string(ratio),
        'audited_maximum_kerdock_over_rule_factor': decimal_string(gain),
        'audited_maximum_risk_reduction_fraction': decimal_string(risk_reduction),
        'frontier': frontier,
        'not_claimed': [
            'the unrecovered reoptimized T70 witness and its 0.9370605225569535 constant',
            'an arbitrary-total-mass corollary',
            'independent reconstruction of the inherited K_32 interval endpoints',
            'finite-width, adaptive, nonlinear, or network-dependent estimators',
        ],
    }
    out = ROOT / 'INERTIA_STRENGTHENED_FROZEN_WITNESS_VERIFICATION.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
