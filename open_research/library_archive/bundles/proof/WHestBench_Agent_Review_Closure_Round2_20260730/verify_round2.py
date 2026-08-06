#!/usr/bin/env python3
"""Independent arithmetic checks for WHestBench closure round 2.

This script checks only algebraic consequences. It is not a formal proof of the
Gaussian-Hermite expansion, whose proof is given in the accompanying theorem
memo.
"""
from __future__ import annotations

import json
import math
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path
from typing import Iterable


def alpha_audit() -> dict[str, str]:
    getcontext().prec = 60
    spherical_mean = Decimal("0.9747299895417149")
    row_average = Decimal("0.9747302329077503")
    alpha = spherical_mean / row_average
    mass_one_risk = row_average - spherical_mean
    absolute_reduction = (row_average - spherical_mean) ** 2 / row_average
    relative_reduction = absolute_reduction / mass_one_risk
    assert abs((Decimal(1) - alpha) - relative_reduction) < Decimal("1e-50")
    return {
        "spherical_kernel_mean": str(spherical_mean),
        "complete_support_row_average": str(row_average),
        "free_mass_alpha": str(alpha),
        "mass_one_risk": str(mass_one_risk),
        "absolute_risk_reduction": str(absolute_reduction),
        "relative_risk_reduction": str(relative_reduction),
    }


def partitions(total: int, max_part: int, min_part: int = 1) -> Iterable[tuple[int, ...]]:
    """Nondecreasing integer partitions with each part <= max_part."""
    if total == 0:
        yield ()
        return
    for first in range(min_part, min(max_part, total) + 1):
        for rest in partitions(total - first, max_part, first):
            yield (first,) + rest


def h(r: int, a: Fraction, b: Fraction) -> Fraction:
    if r == 0:
        return Fraction(0)
    den = a + b * r
    assert den > 0
    return Fraction(r, 1) / den


def canonical_partition(total: int, d: int) -> tuple[int, ...]:
    q, rem = divmod(total, d)
    parts = [d] * q
    if rem:
        parts.append(rem)
    return tuple(sorted(parts))


def support_extremality_checks() -> dict[str, object]:
    # Multiple exact rational sign regimes satisfying a>0, b<0, a+bd>0.
    regimes = [
        (4, Fraction(7, 5), Fraction(-1, 10)),
        (5, Fraction(2, 1), Fraction(-1, 5)),
        (7, Fraction(3, 2), Fraction(-1, 10)),
    ]
    checked = 0
    for d, a, b in regimes:
        assert a > 0 and b < 0 and a + b * d > 0
        # Strictly increasing discrete increments = strict discrete convexity.
        inc = [h(r + 1, a, b) - h(r, a, b) for r in range(d)]
        assert all(y > x for x, y in zip(inc, inc[1:]))
        for p in range(1, 3 * d + 1):
            scored = [(sum(h(r, a, b) for r in part), part)
                      for part in partitions(p, d)]
            best_value = max(v for v, _ in scored)
            best = [part for v, part in scored if v == best_value]
            expected = canonical_partition(p, d)
            assert best == [expected], (d, p, best, expected)
            checked += len(scored)
    return {"exact_partition_cases_scored": checked, "regimes": len(regimes)}


def coefficient_sign_checks() -> dict[str, object]:
    # coeff[r] represents a_{2r}. Repeat exact identities for several finite
    # nonnegative sequences. coeff[0] cancels from all association differences.
    sequences = [
        [Fraction(7, 10), Fraction(3, 25), Fraction(3, 100), Fraction(1, 100)],
        [Fraction(1, 3), Fraction(1, 7), Fraction(0), Fraction(2, 19), Fraction(1, 23)],
        [Fraction(5, 8), Fraction(0), Fraction(1, 17)],
    ]
    d = 256
    for coeff in sequences:
        A = sum(coeff)
        O = coeff[0]
        C = sum(c * Fraction(1, d**r) for r, c in enumerate(coeff))
        a = A - O
        b = O - C
        margin = a + d * b
        expected = sum(coeff[r] * (1 - Fraction(d, d**r))
                       for r in range(2, len(coeff)))
        assert margin == expected
        assert a > 0 and b < 0
        if any(coeff[r] > 0 for r in range(2, len(coeff))):
            assert margin > 0
    return {"coefficient_sequences_checked": len(sequences), "dimension": d}


def nonlinear_relu_counterexample() -> dict[str, float]:
    # In d=4 choose a non-isotropic coefficient vector. Antipodal basis values
    # recover |a_i| and hence ||a|| exactly, while equal-weight averaging uses l1.
    d = 4
    a = [1.0, 2.0, 0.5, -1.5]
    norm = math.sqrt(sum(x * x for x in a))
    l1 = sum(abs(x) for x in a)
    c_d = math.gamma(d / 2) / (2 * math.sqrt(math.pi) * math.gamma((d + 1) / 2))
    true_integral = c_d * norm
    nonlinear_from_basis = c_d * math.sqrt(sum(abs(x) ** 2 for x in a))
    equal_weight_linear = l1 / (2 * d)
    assert math.isclose(nonlinear_from_basis, true_integral, rel_tol=1e-15)
    assert not math.isclose(equal_weight_linear, true_integral, rel_tol=1e-6)
    return {
        "dimension": d,
        "true_integral": true_integral,
        "nonlinear_estimator": nonlinear_from_basis,
        "equal_weight_linear_estimator": equal_weight_linear,
    }


def main() -> None:
    result = {
        "alpha_audit": alpha_audit(),
        "mub_support_extremality": support_extremality_checks(),
        "finite_width_coefficient_signs": coefficient_sign_checks(),
        "nonlinear_relu_counterexample": nonlinear_relu_counterexample(),
        "status": "PASS",
    }
    out = Path(__file__).with_name("ROUND2_VERIFICATION.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
