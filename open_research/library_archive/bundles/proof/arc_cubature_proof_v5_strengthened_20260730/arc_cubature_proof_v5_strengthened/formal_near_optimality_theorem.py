from __future__ import annotations

"""Assemble the one-sided computer-assisted near-optimality theorem.

The proof supplies an upper bound on Kerdock's ratio to the true optimum.  It
does not supply a positive lower bound on Kerdock's suboptimality; Kerdock could
in principle be exactly optimal.  The machine-readable output reflects this
one-sided logic explicitly.
"""

import json
from decimal import Decimal
from pathlib import Path

from formal_interval_certificate import Directed, I


def read_interval(obj: dict) -> I:
    return I(Decimal(obj['lower']), Decimal(obj['upper']))


def main(prec: int = 80) -> None:
    base = Path(__file__).resolve().parent
    res = base / 'results'
    pointwise = json.loads((res/'FORMAL_CERTIFICATE_D256_L32.json').read_text())
    energy_data = json.loads((res/'FORMAL_DELSARTE_BOUND_D256_L32.json').read_text())
    mean_data = json.loads((res/'FORMAL_KERNEL_MEAN_D256_L32.json').read_text())
    assert pointwise['passed'] and energy_data['passed'] and mean_data['passed']
    assert pointwise['dimension'] == energy_data['dimension'] == mean_data['dimension'] == 256
    assert pointwise['depth'] == energy_data['depth'] == mean_data['depth'] == 32

    dr = Directed(prec)
    kerdock_energy = read_interval(energy_data['kerdock_energy'])
    universal_energy = read_interval(energy_data['universal_energy_lower_bound'])
    A0 = read_interval(mean_data['A0_certified'])

    # Actual Kerdock MSE lies in this interval.
    kerdock_mse = dr.sub(kerdock_energy, A0)

    # C = B-A0 is the certificate expression.  The actual optimum satisfies
    # M_opt >= C >= C.lo.  C.hi is not an upper bound on M_opt.
    certificate_expression = dr.sub(universal_energy, A0)
    optimum_lower = certificate_expression.lo
    if optimum_lower <= 0:
        raise AssertionError(certificate_expression)

    # One-sided ratio: actual M_K / actual M_opt <= M_K.upper / optimum_lower.
    ratio_upper_interval = dr.div(I(kerdock_mse.hi, kerdock_mse.hi),
                                  I(optimum_lower, optimum_lower))
    ratio_upper = ratio_upper_interval.hi
    excess_upper = dr.sub(I(ratio_upper, ratio_upper), dr.integer(1)).hi
    percent_upper = dr.mul(I(excess_upper, excess_upper), dr.integer(100)).hi

    # The additive A0 term cancels exactly.  This is a one-sided upper bound on
    # actual Kerdock suboptimality, not an enclosure of the unknown true gap.
    energy_gap = read_interval(energy_data['kerdock_minus_universal_bound'])
    additive_upper = energy_gap.hi
    assert additive_upper >= 0

    theorem = (
        'For K_32 in dimension 256, every network-independent linear cubature rule '
        'supported on at most 66,048 points, with nonnegative weights summing to one, '
        'has ensemble MSE at least the certified scalar lower bound. The uniform '
        '66,048-point antipodal real-MUB (Kerdock) rule has MSE at most the stated '
        'multiplicative factor above the infimum over that class.'
    )

    out = {
        'title': 'Computer-assisted near-optimality certificate for Kerdock/MUB cubature',
        'dimension': 256,
        'depth': 32,
        'node_budget': 66048,
        'precision': prec,
        'theorem': theorem,
        'kernel_definition': (
            'K_0(t)=t and K_{l+1}(t)=kappa(K_l(t)), where '
            'kappa(t)=(sqrt(1-t^2)+(pi-acos(t))t)/pi.'
        ),
        'randomized_rule_corollary': (
            'The same bound holds after averaging over a randomized rule when the '
            'rule randomness, nodes, and weights are independent of the realized random field.'
        ),
        'scope': {
            'included': [
                'deterministic nodes and weights fixed independently of the random field',
                'randomized rules whose randomness is independent of the random field',
                'nonnegative weights summing to one',
                'support size at most 66,048',
                'the infinite-width depth-32 normalized ReLU kernel in dimension 256',
            ],
            'excluded': [
                'nodes or weights selected from the realized network or observed activations',
                'pilot sampling followed by adaptation',
                'nonlinear estimators',
                'network-dependent analytic-plus-residual estimators',
                'signed cubature weights',
                'finite-width network objectives',
            ],
        },
        'pointwise_certificate': {
            'global_upper_bound_for_h_minus_K': pointwise['global_upper_bound'],
            'certified_subintervals': pointwise['mesh_certified_subintervals'],
            'coverage_exact': pointwise['coverage_exact'],
            'no_gaps_or_overlaps': pointwise['no_gaps_or_overlaps'],
        },
        'kernel_mean_A0_interval': {
            'lower': str(A0.lo), 'upper': str(A0.hi), 'width': str(A0.hi-A0.lo),
        },
        'kerdock_energy_interval': {
            'lower': str(kerdock_energy.lo), 'upper': str(kerdock_energy.hi),
        },
        'universal_energy_lower_bound_certificate_interval': {
            'lower': str(universal_energy.lo), 'upper': str(universal_energy.hi),
        },
        'kerdock_mse_interval': {
            'lower': str(kerdock_mse.lo), 'upper': str(kerdock_mse.hi),
        },
        'certificate_expression_B_minus_A0_interval': {
            'lower': str(certificate_expression.lo),
            'upper': str(certificate_expression.hi),
            'interpretation': 'The actual optimum MSE is at least the lower endpoint only.',
        },
        'certified_optimum_mse_lower_bound': str(optimum_lower),
        'actual_additive_suboptimality': {
            'lower': '0',
            'upper': str(additive_upper),
            'interpretation': 'One-sided enclosure: Kerdock is feasible, and the Delsarte certificate gives the upper endpoint.',
        },
        'actual_multiplicative_ratio_kerdock_over_infimum': {
            'lower': '1',
            'upper': str(ratio_upper),
        },
        'actual_relative_excess': {
            'lower': '0',
            'upper': str(excess_upper),
        },
        'actual_relative_excess_percent': {
            'lower': '0',
            'upper': str(percent_upper),
        },
        'human_readable_conclusion': (
            f'Kerdock is certified to be at most {percent_upper}% above the infimum within the stated class.'
        ),
        'source_artifacts': [
            'FORMAL_CERTIFICATE_D256_L32.json',
            'FORMAL_DELSARTE_BOUND_D256_L32.json',
            'FORMAL_KERNEL_MEAN_D256_L32.json',
        ],
        'passed': True,
    }
    (res/'FORMAL_NEAR_OPTIMALITY_THEOREM_D256_L32.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
