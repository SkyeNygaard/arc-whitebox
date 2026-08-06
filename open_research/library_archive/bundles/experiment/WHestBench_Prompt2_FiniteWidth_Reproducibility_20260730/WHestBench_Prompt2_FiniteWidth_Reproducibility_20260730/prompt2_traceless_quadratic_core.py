#!/usr/bin/env python3
"""Exact interval machinery for F(G)=bb^T/||b|| and its traceless part.

Let G~N(0,I_m), b=sqrt(2) ReLU(G), and R=||b||.  The matrix-valued
feature F=bb^T/R has Frobenius kernel R R' q^2.  Its traceless part
F0=F-(R/m)I has kernel R R' (q^2-1/m).

Complete multivariate-Hermite energies are evaluated by finite sums over
integer partitions, sign patterns, and Gamma ratios.  No numerical quadrature
or floating-point arithmetic is used.
"""
from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction
from functools import lru_cache

import prompt2_full_hermite_core as c

M = c.M
I = c.I


@lru_cache(None)
def inverse_radial_factor(active_dim: int, total_power: int) -> I:
    """Base radial factor for r^{-1} times angular degree J.

    The outer sqrt(2) in F_ab is applied after the sign sum.  For k active
    coordinates and total angular degree J, the remaining radial factor is
    2^(J/2) / E[chi_{k+J-1}].
    """
    if active_dim <= 0 or total_power < 2:
        return I.point(0)
    factor = I.point(2 ** (total_power // 2))
    if total_power % 2:
        factor *= c.SQRT2
    return factor / c.chi_mean(active_dim + total_power - 1)


@lru_cache(None)
def shifted_sphere_poly(q: int, numerator_power: int) -> tuple[I, ...]:
    """Angular polynomial for x^p He_q(x) on a positive sphere coordinate."""
    hp = c.hermite_prob(q)
    out = [I.point(0) for _ in range(len(hp) + numerator_power)]
    for j, coefficient in enumerate(hp):
        if coefficient:
            out[j + numerator_power] = (
                c.sphere_gamma_factor(j + numerator_power) * coefficient
            )
    return tuple(out)


def _conv_many(polynomials: list[tuple[I, ...]]) -> list[I]:
    out = [I.point(1)]
    for polynomial in polynomials:
        out = c.poly_conv(out, polynomial)
    return out


@lru_cache(None)
def feature_coefficient(
    lam: tuple[int, ...],
    distinguished: tuple[tuple[int, int], ...],
) -> I:
    """Return E[F_ab(G) He_alpha(G)] for one symmetry category.

    ``lam`` is the partition of the nonzero entries of alpha.  Each
    distinguished item is ``(q,p)``: q=0 denotes a coordinate outside the
    support of alpha, q>0 consumes one selected He_q coordinate, and p is the
    numerator power (2 for a diagonal matrix entry, 1 for each endpoint of an
    off-diagonal entry).  Distinguished coordinates are distinct.
    """
    counts = Counter(lam)
    extra_unselected = 0
    forced_polys: list[tuple[I, ...]] = []
    for q, numerator_power in distinguished:
        if q == 0:
            extra_unselected += 1
        else:
            if counts[q] <= 0:
                return I.point(0)
            counts[q] -= 1
            if counts[q] == 0:
                del counts[q]
        forced_polys.append(shifted_sphere_poly(q, numerator_power))

    remaining_unselected = M - len(lam) - extra_unselected
    if remaining_unselected < 0:
        return I.point(0)

    qs = sorted(counts)
    total = I.point(0)

    def recurse(
        position: int,
        active_polys: list[tuple[I, ...]],
        active_count: int,
        multiplicity: int,
        inactive_factor: I,
    ) -> None:
        nonlocal total
        if position == len(qs):
            polynomial = _conv_many(forced_polys + active_polys)
            radial = I.point(0)
            forced_count = len(forced_polys)
            for total_power, coefficient in enumerate(polynomial):
                if coefficient.lo == 0 and coefficient.hi == 0:
                    continue
                assert total_power >= 2
                sign_sum = I.point(0)
                for k in range(remaining_unselected + 1):
                    active_dim = k + active_count + forced_count
                    if active_dim:
                        sign_sum += (
                            inverse_radial_factor(active_dim, total_power)
                            * math.comb(remaining_unselected, k)
                        )
                radial += coefficient * sign_sum
            total += radial * inactive_factor * multiplicity
            return

        q = qs[position]
        count = counts[q]
        for active in range(count + 1):
            recurse(
                position + 1,
                active_polys + [shifted_sphere_poly(q, 0)] * active,
                active_count + active,
                multiplicity * math.comb(count, active),
                inactive_factor
                * c.negative_half_hermite_mean(q).pow_int(count - active),
            )

    recurse(0, [], 0, 1, I.point(1))
    return total * c.SQRT2 / (2**M)


def matrix_partition_contribution(lam: tuple[int, ...]) -> I:
    """Hermite-energy contribution of all alpha with nonzero partition lam."""
    counts = Counter(lam)
    support = len(lam)
    unselected = M - support
    category = I.point(0)

    # Diagonal matrix entries.
    if unselected:
        z = feature_coefficient(lam, ((0, 2),))
        category += z.square() * unselected
    for q, count in counts.items():
        z = feature_coefficient(lam, ((q, 2),))
        category += z.square() * count

    # Ordered off-diagonal matrix entries; Frobenius norm counts both orders.
    if unselected >= 2:
        z = feature_coefficient(lam, ((0, 1), (0, 1)))
        category += z.square() * unselected * (unselected - 1)
    if unselected:
        for q, count in counts.items():
            z = feature_coefficient(lam, ((q, 1), (0, 1)))
            category += z.square() * (2 * count * unselected)

    qs = sorted(counts)
    for q in qs:
        count = counts[q]
        if count >= 2:
            z = feature_coefficient(lam, ((q, 1), (q, 1)))
            category += z.square() * count * (count - 1)
    for i, q in enumerate(qs):
        for r in qs[i + 1 :]:
            z = feature_coefficient(lam, ((q, 1), (r, 1)))
            category += z.square() * (2 * counts[q] * counts[r])

    return category * Fraction(
        c.multiindex_count(lam), c.alpha_factorial(lam)
    )


def matrix_feature_energy(n: int) -> I:
    total = I.point(0)
    for lam in c.partitions(n):
        total += matrix_partition_contribution(lam)
    return total


def traceless_partition_contribution(lam: tuple[int, ...]) -> I:
    matrix = matrix_partition_contribution(lam)
    scalar = c.A_partition(lam)
    trace_energy = (
        scalar.square()
        * Fraction(c.multiindex_count(lam), c.alpha_factorial(lam))
        / M
    )
    result = matrix - trace_energy
    assert result.lo >= 0
    return result


def traceless_energy(n: int) -> I:
    total = I.point(0)
    for lam in c.partitions(n):
        total += traceless_partition_contribution(lam)
    return total


if __name__ == "__main__":
    import json
    import sys

    degrees = [int(x) for x in sys.argv[1:]] or list(range(0, 9))
    output = {}
    for degree in degrees:
        matrix = matrix_feature_energy(degree)
        traceless = matrix - c.hermite_energy(degree) / M
        output[degree] = {
            "matrix": c.decimal_bounds(matrix, 40),
            "traceless": c.decimal_bounds(traceless, 40),
        }
        print(degree, output[degree], flush=True)
    print(json.dumps(output, indent=2))
