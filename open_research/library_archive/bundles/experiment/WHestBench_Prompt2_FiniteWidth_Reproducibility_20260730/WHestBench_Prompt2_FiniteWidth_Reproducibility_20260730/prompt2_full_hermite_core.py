#!/usr/bin/env python3
"""Independent exact-rational interval verifier for the strengthened Prompt-2 theorem.

No mpmath, scipy, sympy, gamma implementation, or floating-point arithmetic is
used in the certificate path.  The only transcendental constant is pi, enclosed
by exact rational Machin-series bounds.  Square roots are enclosed using integer
square roots at a fixed decimal scale.  All subsequent interval arithmetic is
exact over fractions.Fraction.

The proof captures the complete Hermite chaos of the exact first-layer
conditional-mean component through degree 16, including every triangular monomial-to-Gegenbauer
contribution, rather than only one mode or only the top-degree contribution.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

M = 256
D = 256
DEPTH = 32
NODES = 66_048
SQRT_DIGITS = 130

@dataclass(frozen=True)
class I:
    lo: Fraction
    hi: Fraction

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError((self.lo, self.hi))

    @staticmethod
    def point(x: int | Fraction) -> "I":
        q = Fraction(x)
        return I(q, q)

    def __add__(self, other: "I | int | Fraction") -> "I":
        o = other if isinstance(other, I) else I.point(other)
        return I(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __neg__(self) -> "I":
        return I(-self.hi, -self.lo)

    def __sub__(self, other: "I | int | Fraction") -> "I":
        o = other if isinstance(other, I) else I.point(other)
        return self + (-o)

    def __rsub__(self, other: "I | int | Fraction") -> "I":
        return I.point(other) - self

    def __mul__(self, other: "I | int | Fraction") -> "I":
        o = other if isinstance(other, I) else I.point(other)
        vals = (
            self.lo * o.lo,
            self.lo * o.hi,
            self.hi * o.lo,
            self.hi * o.hi,
        )
        return I(min(vals), max(vals))

    __rmul__ = __mul__

    def reciprocal(self) -> "I":
        if self.lo <= 0 <= self.hi:
            raise ZeroDivisionError(self)
        vals = (Fraction(1, 1) / self.lo, Fraction(1, 1) / self.hi)
        return I(min(vals), max(vals))

    def __truediv__(self, other: "I | int | Fraction") -> "I":
        o = other if isinstance(other, I) else I.point(other)
        return self * o.reciprocal()

    def __rtruediv__(self, other: "I | int | Fraction") -> "I":
        return I.point(other) / self

    def square(self) -> "I":
        if self.lo <= 0 <= self.hi:
            return I(Fraction(0), max(self.lo * self.lo, self.hi * self.hi))
        vals = (self.lo * self.lo, self.hi * self.hi)
        return I(min(vals), max(vals))

    def pow_int(self, n: int) -> "I":
        if n < 0:
            return self.pow_int(-n).reciprocal()
        result = I.point(1)
        base = self
        k = n
        while k:
            if k & 1:
                result = result * base
            base = base * base
            k >>= 1
        return result


def atan_inv_bounds(q: int, last_index: int) -> I:
    """Exact alternating-series enclosure for atan(1/q)."""
    x = Fraction(1, q)
    x2 = x * x
    term_power = x
    s = Fraction(0)
    for j in range(last_index + 1):
        term = term_power / (2 * j + 1)
        s = s + term if j % 2 == 0 else s - term
        term_power *= x2
    j = last_index + 1
    next_term = term_power / (2 * j + 1)
    s_next = s + next_term if j % 2 == 0 else s - next_term
    return I(min(s, s_next), max(s, s_next))


# Machin: pi = 16 atan(1/5) - 4 atan(1/239).
A5 = atan_inv_bounds(5, 110)
A239 = atan_inv_bounds(239, 35)
PI = I(16 * A5.lo - 4 * A239.hi, 16 * A5.hi - 4 * A239.lo)


def sqrt_fraction_bounds(x: Fraction, digits: int = SQRT_DIGITS) -> I:
    if x < 0:
        raise ValueError(x)
    if x == 0:
        return I.point(0)
    scale = 10**digits
    scaled_num = x.numerator * scale * scale
    floor_ratio = scaled_num // x.denominator
    k = math.isqrt(floor_ratio)
    lo = Fraction(k, scale)
    exact = k * k * x.denominator == scaled_num
    hi = lo if exact else Fraction(k + 1, scale)
    assert lo * lo <= x <= hi * hi
    return I(lo, hi)


def sqrt_interval(x: I) -> I:
    low = sqrt_fraction_bounds(x.lo)
    high = sqrt_fraction_bounds(x.hi)
    return I(low.lo, high.hi)


SQRT2 = sqrt_fraction_bounds(Fraction(2))
SQRT_PI = sqrt_interval(PI)
MU1 = SQRT2 / SQRT_PI               # sqrt(2/pi)
MU2 = SQRT_PI / SQRT2               # sqrt(pi/2)


@lru_cache(None)
def chi_mean(k: int) -> I:
    if k == 0:
        return I.point(0)
    if k == 1:
        return MU1
    if k == 2:
        return MU2
    return chi_mean(k - 2) * Fraction(k - 1, k - 2)


def hermite_prob(n: int) -> list[int]:
    if n == 0:
        return [1]
    h0 = [1]
    h1 = [0, 1]
    if n == 1:
        return h1
    for k in range(1, n):
        xh1 = [0] + h1
        h0s = [k * v for v in h0] + [0] * (len(xh1) - len(h0))
        h2 = [xh1[i] - h0s[i] for i in range(len(xh1))]
        h0, h1 = h1, h2
    return h1


def odd_double_factorial(n: int) -> int:
    if n <= 0:
        return 1
    out = 1
    for j in range(1, n + 1, 2):
        out *= j
    return out


@lru_cache(None)
def half_normal_moment(j: int) -> I:
    if j % 2 == 0:
        return I.point(odd_double_factorial(j - 1))
    r = (j - 1) // 2
    return MU1 * (2**r * math.factorial(r))


@lru_cache(None)
def sphere_gamma_factor(j: int) -> I:
    """Gamma((j+1)/2)/sqrt(pi), avoiding a gamma implementation."""
    if j % 2 == 0:
        r = j // 2
        return I.point(Fraction(math.factorial(2 * r), 4**r * math.factorial(r)))
    r = (j - 1) // 2
    return I.point(math.factorial(r)) / SQRT_PI


@lru_cache(None)
def negative_half_hermite_mean(n: int) -> I:
    s = I.point(0)
    for j, c in enumerate(hermite_prob(n)):
        if c:
            s += half_normal_moment(j) * (c * ((-1) ** j))
    return s


@lru_cache(None)
def radial_factor(k: int, total_power: int) -> I:
    """2^((J+1)/2) Gamma((k+J+1)/2)/Gamma((k+J)/2)."""
    if k <= 0:
        return I.point(0)
    factor: I = chi_mean(k + total_power) * (2 ** (total_power // 2))
    if total_power % 2:
        factor = factor * SQRT2
    return factor


@lru_cache(None)
def sphere_poly(q: int) -> tuple[I, ...]:
    return tuple(sphere_gamma_factor(j) * c for j, c in enumerate(hermite_prob(q)))


def poly_conv(a: tuple[I, ...] | list[I], b: tuple[I, ...] | list[I]) -> list[I]:
    out = [I.point(0) for _ in range(len(a) + len(b) - 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = out[i + j] + x * y
    return out


@lru_cache(None)
def active_poly(counts: tuple[tuple[int, int], ...]) -> tuple[I, ...]:
    p: list[I] = [I.point(1)]
    for q, multiplicity in counts:
        base = sphere_poly(q)
        for _ in range(multiplicity):
            p = poly_conv(p, base)
    return tuple(p)


@lru_cache(None)
def radial_binomial_sum(selected_count: int, active_count: int, total_power: int) -> I:
    remaining = M - selected_count
    s = I.point(0)
    for k in range(remaining + 1):
        if k + active_count:
            s += radial_factor(k + active_count, total_power) * math.comb(remaining, k)
    return s


def partitions(n: int, max_part: int | None = None):
    if n == 0:
        yield ()
        return
    if max_part is None or max_part > n:
        max_part = n
    for first in range(max_part, 0, -1):
        for rest in partitions(n - first, first):
            yield (first,) + rest


def A_partition(lam: tuple[int, ...]) -> I:
    s = len(lam)
    counts = Counter(lam)
    qs = sorted(counts)
    total = I.point(0)

    def recurse(pos: int, active_counts: list[tuple[int, int]], active_n: int,
                multiplicity: int, inactive_factor: I) -> None:
        nonlocal total
        if pos == len(qs):
            p = active_poly(tuple(active_counts))
            radial = I.point(0)
            for j, coeff in enumerate(p):
                radial += coeff * radial_binomial_sum(s, active_n, j)
            total += radial * inactive_factor * multiplicity
            return
        q = qs[pos]
        cq = counts[q]
        for r in range(cq + 1):
            new_counts = active_counts + ([(q, r)] if r else [])
            recurse(
                pos + 1,
                new_counts,
                active_n + r,
                multiplicity * math.comb(cq, r),
                inactive_factor * negative_half_hermite_mean(q).pow_int(cq - r),
            )

    recurse(0, [], 0, 1, I.point(1))
    return total * SQRT2 / (2**M)


def multiindex_count(lam: tuple[int, ...]) -> int:
    s = len(lam)
    out = 1
    for j in range(s):
        out *= M - j
    for c in Counter(lam).values():
        out //= math.factorial(c)
    return out


def alpha_factorial(lam: tuple[int, ...]) -> int:
    out = 1
    for q in lam:
        out *= math.factorial(q)
    return out


def hermite_energy(n: int) -> I:
    total = I.point(0)
    for lam in partitions(n):
        a = A_partition(lam)
        total += a.square() * Fraction(multiindex_count(lam), alpha_factorial(lam))
    return total


def poly_add(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    n = max(len(a), len(b))
    out = [Fraction(0) for _ in range(n)]
    for i, x in enumerate(a):
        out[i] += x
    for i, x in enumerate(b):
        out[i] += x
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def poly_scale(a: list[Fraction], c: Fraction) -> list[Fraction]:
    return [c * x for x in a]


def poly_shift(a: list[Fraction]) -> list[Fraction]:
    return [Fraction(0)] + a


def poly_mul(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    out = [Fraction(0) for _ in range(len(a) + len(b) - 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def gegenbauer_normalized(max_degree: int, lam: int) -> list[list[Fraction]]:
    raw = [[Fraction(1)], [Fraction(0), Fraction(2 * lam)]]
    for n in range(1, max_degree):
        lhs = poly_scale(poly_shift(raw[n]), Fraction(2 * (n + lam), n + 1))
        rhs = poly_scale(raw[n - 1], Fraction(n + 2 * lam - 1, n + 1))
        raw.append(poly_add(lhs, poly_scale(rhs, Fraction(-1))))
    out = []
    for p in raw[: max_degree + 1]:
        out.append(poly_scale(p, Fraction(1, 1) / sum(p, Fraction(0))))
    return out


def expand_in_basis(poly: list[Fraction], basis: list[list[Fraction]]) -> list[Fraction]:
    rem = poly[:]
    degree = len(poly) - 1
    coeff = [Fraction(0) for _ in range(degree + 1)]
    for ell in range(degree, -1, -1):
        c = rem[ell] / basis[ell][ell]
        coeff[ell] = c
        for j, x in enumerate(basis[ell]):
            rem[j] -= c * x
    assert all(x == 0 for x in rem)
    return coeff


def harmonic_dim(ell: int, d: int = D) -> int:
    ans = math.comb(d + ell - 1, ell)
    if ell >= 2:
        ans -= math.comb(d + ell - 3, ell - 2)
    return ans


def rank_floor(weights: list[Fraction]) -> Fraction:
    eigenvalues = sorted(
        zip(weights, [harmonic_dim(j) for j in range(len(weights))]),
        key=lambda x: x[0],
        reverse=True,
    )
    remaining = NODES
    tail_sum = Fraction(0)
    tail_sq = Fraction(0)
    for value, multiplicity in eigenvalues:
        kept = min(remaining, multiplicity)
        tail = multiplicity - kept
        tail_sum += tail * value
        tail_sq += tail * value * value
        remaining -= kept
    return tail_sq + tail_sum * tail_sum / NODES


def decimal_bounds(x: I, digits: int = 40) -> list[str]:
    scale = 10**digits
    lo_int = x.lo.numerator * scale // x.lo.denominator
    hi_int = -((-x.hi.numerator * scale) // x.hi.denominator)

    def render(v: int) -> str:
        sign = "-" if v < 0 else ""
        v = abs(v)
        s = str(v).rjust(digits + 1, "0")
        return f"{sign}{s[:-digits]}.{s[-digits:]}"

    return [render(lo_int), render(hi_int)]


def frac_record(x: Fraction, digits: int = 40) -> dict[str, str]:
    return {
        "exact": f"{x.numerator}/{x.denominator}",
        "bounds": decimal_bounds(I.point(x), digits),
    }




def energy_worker(n: int):
    """Separate-process worker for one exact Hermite-chaos energy."""
    return n, hermite_energy(n)


def round_interval_outward(x: I, digits: int = 75) -> I:
    scale = 10**digits
    lo_i = x.lo.numerator * scale // x.lo.denominator
    hi_i = -((-x.hi.numerator * scale) // x.hi.denominator)
    return I(Fraction(lo_i, scale), Fraction(hi_i, scale))


def rounded_energy_worker(n: int):
    """Compute one energy exactly, then return a compact rigorous enclosure."""
    return n, round_interval_outward(hermite_energy(n), 75)
