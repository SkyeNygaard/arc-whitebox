from __future__ import annotations

"""Rigorous spherical mean of the depth-L normalized ReLU kernel.

The normalized ReLU dual activation has a Maclaurin series with nonnegative
coefficients summing to one. Composition preserves both properties. We compute
rigorous intervals for the first M Taylor coefficients of K_L at zero using
interval jet composition. The sphere's odd moments vanish, and the omitted even
terms are bounded by the first omitted even moment because their coefficient
mass is at most one.
"""

import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from formal_interval_certificate import Directed, I, pi_bounds, kappa_interval, kappa_prime_interval


def izero() -> I:
    z = Decimal(0)
    return I(z, z)


def iadd_to(dr: Directed, dst: I, src: I) -> I:
    return dr.add(dst, src)


def poly_mul(a: list[I], b: list[I], M: int, dr: Directed) -> list[I]:
    out = [izero() for _ in range(M + 1)]
    for i, ai in enumerate(a):
        if i > M:
            break
        # Exact zero shortcut is important for speed.
        if ai.lo == 0 and ai.hi == 0:
            continue
        top = min(M - i, len(b) - 1)
        for j in range(top + 1):
            bj = b[j]
            if bj.lo == 0 and bj.hi == 0:
                continue
            out[i + j] = dr.add(out[i + j], dr.mul(ai, bj))
    return out


def invsqrt_series_at(z0: I, M: int, dr: Directed) -> list[I]:
    """Taylor coefficients of (1-(z0+u)^2)^(-1/2) in u, through M.

    Uses a*y' = alpha*y*a' for y=a^alpha, alpha=-1/2.
    """
    one = dr.integer(1)
    two = dr.integer(2)
    a0 = dr.sub(one, dr.mul(z0, z0))
    if a0.lo <= 0:
        raise ValueError(f'nonpositive a0 {a0}')
    a1 = dr.neg(dr.mul(two, z0))
    a2 = dr.integer(-1)
    aa = [a0, a1, a2]
    y = [izero() for _ in range(M + 1)]
    y[0] = dr.div(one, dr.sqrt(a0))
    alpha = Fraction(-1, 2)
    for n in range(1, M + 1):
        s = izero()
        for k in range(1, min(n, 2) + 1):
            factor = (alpha + 1) * k - n
            term = dr.mul(dr.frac_interval(Fraction(factor)), dr.mul(aa[k], y[n - k]))
            s = dr.add(s, term)
        y[n] = dr.div(s, dr.mul(dr.integer(n), a0))
    return y


def kappa_taylor_at(z0: I, M: int, dr: Directed, pi: I) -> list[I]:
    """Taylor coefficients b_j of kappa(z0+u)=sum b_j u^j."""
    b = [izero() for _ in range(M + 1)]
    b[0] = kappa_interval(z0, dr, pi)
    if M >= 1:
        b[1] = kappa_prime_interval(z0, dr, pi)
    if M >= 2:
        q = invsqrt_series_at(z0, M - 2, dr)
        for j in range(2, M + 1):
            denom = dr.mul(pi, dr.integer(j * (j - 1)))
            b[j] = dr.div(q[j - 2], denom)
    return b


def compose_kappa(g: list[I], M: int, dr: Directed, pi: I) -> list[I]:
    z0 = g[0]
    b = kappa_taylor_at(z0, M, dr, pi)
    u = list(g)
    u[0] = izero()  # exact by definition: g(t)-g(0)
    out = [izero() for _ in range(M + 1)]
    power = [izero() for _ in range(M + 1)]
    power[0] = dr.integer(1)
    for j in range(M + 1):
        bj = b[j]
        if not (bj.lo == 0 and bj.hi == 0):
            for k in range(M + 1):
                if power[k].lo == 0 and power[k].hi == 0:
                    continue
                out[k] = dr.add(out[k], dr.mul(bj, power[k]))
        if j != M:
            power = poly_mul(power, u, M, dr)
    return out


def deep_kernel_coefficients(depth: int, M: int, dr: Directed, pi: I) -> list[I]:
    g = [izero() for _ in range(M + 1)]
    g[1] = dr.integer(1)
    for layer in range(depth):
        g = compose_kappa(g, M, dr, pi)
        # All true coefficients are nonnegative. Numerical intervals should
        # respect this; a negative lower endpoint is harmless, but a negative
        # upper endpoint would signal a broken enclosure.
        if any(c.hi < 0 for c in g):
            raise AssertionError(f'negative coefficient interval at layer {layer+1}')
    return g


def sphere_even_moment(d: int, k: int) -> Fraction:
    """E[T^(2k)] for inner product T of a fixed unit vector and U~Unif(S^(d-1))."""
    out = Fraction(1)
    for j in range(k):
        out *= Fraction(2 * j + 1, d + 2 * j)
    return out


def main(prec: int = 120, M: int = 30, d: int = 256, depth: int = 32):
    if M % 2:
        raise ValueError('M should be even')
    base = Path(__file__).resolve().parent
    dr = Directed(prec)
    pl, ph = pi_bounds(prec + 40)
    pi = I(dr.Dlo(pl), dr.Dhi(ph))
    coeff = deep_kernel_coefficients(depth, M, dr, pi)

    A0 = izero()
    terms = []
    for k in range(M // 2 + 1):
        degree = 2 * k
        moment = sphere_even_moment(d, k)
        term = dr.mul(coeff[degree], dr.frac_interval(moment))
        A0 = dr.add(A0, term)
        terms.append({
            'degree': degree,
            'coefficient': {'lower': str(coeff[degree].lo), 'upper': str(coeff[degree].hi)},
            'moment_exact': str(moment),
            'term': {'lower': str(term.lo), 'upper': str(term.hi)},
        })

    # K_L(t)=sum c_k t^k, c_k>=0, sum c_k=K_L(1)=1. Odd moments vanish.
    # For every omitted even k >= M+2, E[T^k] <= E[T^(M+2)].
    tail_moment = sphere_even_moment(d, M // 2 + 1)
    tail = dr.frac_interval(tail_moment)
    A0_with_tail = I(A0.lo, dr.add(A0, tail).hi)

    out = {
        'dimension': d,
        'depth': depth,
        'precision': prec,
        'maximum_computed_degree': M,
        'method': 'interval Taylor jets plus nonnegative-series moment tail bound',
        'lemmas_used': [
            'The normalized ReLU dual activation has nonnegative Maclaurin coefficients summing to one.',
            'Composition of power series with nonnegative coefficients preserves nonnegativity and unit coefficient sum.',
            'Odd spherical moments vanish.',
            'Even spherical moments decrease with degree.',
        ],
        'A0_partial': {'lower': str(A0.lo), 'upper': str(A0.hi)},
        'first_omitted_even_moment_exact': str(tail_moment),
        'first_omitted_even_moment_decimal': {
            'lower': str(tail.lo), 'upper': str(tail.hi)
        },
        'A0_certified': {'lower': str(A0_with_tail.lo), 'upper': str(A0_with_tail.hi)},
        'A0_width': str(A0_with_tail.hi - A0_with_tail.lo),
        'coefficient_intervals': [
            {'degree': i, 'lower': str(c.lo), 'upper': str(c.hi)} for i, c in enumerate(coeff)
        ],
        'even_terms': terms,
        'passed': True,
    }
    path = base / 'results' / 'FORMAL_KERNEL_MEAN_D256_L32.json'
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k:v for k,v in out.items() if k not in ('coefficient_intervals','even_terms')}, indent=2))


if __name__ == '__main__':
    main()
