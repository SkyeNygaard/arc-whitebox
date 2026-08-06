#!/usr/bin/env python3
"""Arithmetic checks for the stronger-claims proof memo."""

from __future__ import annotations

import math


def h(r: int, a: float, b: float) -> float:
    return 0.0 if r == 0 else r / (a + b * r)


def check_degree4_counterexample() -> None:
    lam = 1.0
    # P4(t)=(16t^4-12t^2+1)/5 on S^3.
    A = 1.0 + lam
    O = 1.0 + lam / 5.0
    C = 1.0 - lam / 5.0
    a, b = A - O, O - C
    complete = h(4, a, b)
    balanced = 2.0 * h(2, a, b)
    assert a > 0 and b > 0
    assert math.isclose(complete, 5.0 / 3.0, rel_tol=1e-14)
    assert math.isclose(balanced, 5.0 / 2.0, rel_tol=1e-14)
    assert balanced > complete  # Larger H means lower optimized risk.


def check_t22_arithmetic() -> None:
    delta = 0.0002336550102949
    exact_improvement = delta / (1.0 + delta)
    assert math.isclose(exact_improvement, 0.00023360042838440096, rel_tol=1e-14)
    rk = 2.433660357543006e-7
    lb = 2.433091853440941e-7
    gap = rk - lb
    assert math.isclose(gap, 5.685041020648603e-11, rel_tol=1e-14)
    assert math.isclose(gap / rk, exact_improvement, rel_tol=2e-12)


def check_relu_discrete_convexity() -> None:
    d = 256
    a = 0.011988581160655598
    b = -0.000009468153657654632
    assert a > 0 and b < 0 and a + b * d > 0

    # Increasing discrete increments imply that moving one line from a smaller
    # partial basis to a larger partial basis strictly increases H.
    increments = [h(r + 1, a, b) - h(r, a, b) for r in range(d)]
    for left, right in zip(increments, increments[1:]):
        assert right > left

    for x in range(1, d + 1):
        for y in range(x, d):
            before = h(x, a, b) + h(y, a, b)
            after = h(x - 1, a, b) + h(y + 1, a, b)
            assert after > before


def main() -> None:
    check_degree4_counterexample()
    check_t22_arithmetic()
    check_relu_discrete_convexity()
    print("PASS: stronger-claims arithmetic and convexity checks")


if __name__ == "__main__":
    main()
