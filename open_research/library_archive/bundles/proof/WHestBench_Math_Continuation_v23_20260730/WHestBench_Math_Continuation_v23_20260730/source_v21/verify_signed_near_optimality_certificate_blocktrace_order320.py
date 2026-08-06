#!/usr/bin/env python3
"""Exact-rational verifier for the degree-280 comparison / order-320 signed cubature certificate.

The only interval-dependent inputs are the stored lower endpoints of the
Maclaurin coefficients of K_32. Every subsequent Gegenbauer conversion,
comparison-kernel coefficient, rank bound, and slack check is exact over
Python Fractions.
"""
from __future__ import annotations
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path
import json, math

getcontext().prec = 180
ROOT = Path(__file__).resolve().parent
D, N, MAX_DEG = 256, 66048, 320


def frac_decimal(x: str) -> Fraction:
    return Fraction(Decimal(x.strip()))


def harmonic_dim(l: int) -> int:
    if l == 0:
        return 1
    if l == 1:
        return D
    return math.comb(D + l - 1, l) - math.comb(D + l - 3, l - 2)


def normalized_gegenbauer_polynomials(max_degree: int):
    lam = Fraction(D - 2, 2)
    C = [[Fraction(1)], [Fraction(0), 2 * lam]]
    for n in range(1, max_degree):
        a = 2 * (Fraction(n) + lam)
        b = Fraction(n) + 2 * lam - 1
        shifted = [Fraction(0)] + [a * z for z in C[n]]
        previous = [b * z for z in C[n - 1]]
        previous += [Fraction(0)] * (len(shifted) - len(previous))
        C.append([(shifted[i] - previous[i]) / Fraction(n + 1)
                  for i in range(len(shifted))])
    G = []
    for n, poly in enumerate(C):
        value_at_one = math.comb(n + D - 3, n)
        G.append([z / Fraction(value_at_one) for z in poly])
    return G


def polynomial_product(a, b):
    c = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            if y != 0:
                c[i + j] += x * y
    return c


def gegenbauer_decomposition(poly, G):
    work = list(poly) + [Fraction(0)] * (MAX_DEG + 1 - len(poly))
    out = [Fraction(0)] * (MAX_DEG + 1)
    for l in range(min(MAX_DEG, len(poly) - 1), -1, -1):
        if work[l] == 0:
            continue
        coefficient = work[l] / G[l][l]
        out[l] = coefficient
        for j, g in enumerate(G[l]):
            work[j] -= coefficient * g
    assert all(v == 0 for v in work), "nonzero polynomial decomposition residual"
    return out


def decimal_string(q: Fraction, digits: int = 50) -> str:
    return format(Decimal(q.numerator) / Decimal(q.denominator), f'.{digits}E')


def main() -> None:
    certificate = json.loads((ROOT / 'SIGNED_NEAR_OPTIMALITY_CERTIFICATE_BLOCKTRACE_ORDER320.json').read_text())
    jet = json.loads((ROOT / 'K32_MACLAURIN_INTERVALS_ORDER320.json').read_text())
    assert certificate['scope']['dimension'] == D
    assert certificate['scope']['maximum_nodes'] == N

    G = normalized_gegenbauer_polynomials(MAX_DEG)

    # Exact monomial-to-Gegenbauer conversion.
    monomial_expansions = []
    for n in range(MAX_DEG + 1):
        p = [Fraction(0)] * (n + 1)
        p[n] = Fraction(1)
        monomial_expansions.append(gegenbauer_decomposition(p, G))

    maclaurin_lower = [frac_decimal(row[0]) for row in jet['maclaurin_intervals']]
    assert len(maclaurin_lower) == MAX_DEG + 1
    kernel_lower = [Fraction(0)] * (MAX_DEG + 1)
    for n, a_n in enumerate(maclaurin_lower):
        assert a_n >= 0, f'negative Maclaurin lower endpoint at degree {n}'
        for l, coefficient in enumerate(monomial_expansions[n]):
            if coefficient:
                assert coefficient >= 0, (n, l, coefficient)
                kernel_lower[l] += a_n * coefficient

    # Precompute exact linearizations needed by active components.
    active_s = sorted({int(row['s']) for row in certificate['components']})
    square = {}
    adjacent_cross = {}
    for s in active_s:
        square[s] = gegenbauer_decomposition(polynomial_product(G[s], G[s]), G)
        if s + 1 <= MAX_DEG:
            square.setdefault(s + 1,
                              gegenbauer_decomposition(polynomial_product(G[s + 1], G[s + 1]), G))
            adjacent_cross[s] = gegenbauer_decomposition(
                polynomial_product(G[s], G[s + 1]), G)

    used = [Fraction(0)] * (MAX_DEG + 1)
    objective = Fraction(0)
    component_checks = []

    for row in certificate['components']:
        s = int(row['s'])
        r = Fraction(row['r'])
        y = frac_decimal(row['y'])
        assert 0 <= r <= 1
        assert y > 0
        ds, dt = harmonic_dim(s), harmonic_dim(s + 1)
        assert ds > N, f'degree-{s} multiplicity does not exceed node budget'

        rank_bound = Fraction((ds + r * dt) ** 2, N) - ds - r * r * dt
        # This uses the exact trace of each harmonic diagonal block: tr(M_ss)=ds
        # and tr(M_tt)=r*dt. Hence tr(A M)=tr(A^2), while rank(M)<=N and
        # ||M||_F^2 >= tr(M)^2/N even for indefinite signed-weight moment matrices.
        assert rank_bound > 0

        for l in range(1, MAX_DEG + 1):
            raw = Fraction(ds * ds) * square[s][l]
            if r != 0:
                raw += 2 * r * ds * dt * adjacent_cross[s][l]
                raw += r * r * dt * dt * square[s + 1][l]
            assert raw >= 0
            used[l] += y * raw / rank_bound
        objective += y
        component_checks.append({'s': s, 'r': str(r), 'rank_bound_positive': True})

    slacks = [kernel_lower[l] - used[l] for l in range(1, MAX_DEG + 1)]
    assert all(slack >= 0 for slack in slacks)

    declared = frac_decimal(certificate['certified_result']['mse_lower_bound'])
    assert objective == declared
    kerdock_upper = frac_decimal(certificate['certified_result']['kerdock_mse_upper_bound'])
    assert objective * 100 >= 93 * kerdock_upper, 'certificate does not reach 93% threshold'

    binding_index = min(range(len(slacks)), key=lambda i: slacks[i])
    active_degrees = [l for l in range(1, MAX_DEG + 1) if used[l] > 0]
    active_binding_degree = min(active_degrees, key=lambda l: slacks[l-1])
    report = {
        'verified': True,
        'component_count': len(component_checks),
        'maximum_degree_checked': MAX_DEG,
        'signed_rule_mse_lower_bound': decimal_string(objective, 60),
        'fraction_of_kerdock_upper': decimal_string(objective / kerdock_upper, 60),
        'kerdock_multiplicative_excess_upper': decimal_string(kerdock_upper / objective - 1, 60),
        'minimum_slack_over_all_checked_degrees': decimal_string(slacks[binding_index], 60),
        'minimum_slack_over_all_checked_degree': binding_index + 1,
        'maximum_comparison_harmonic_degree': max(active_degrees),
        'minimum_slack_on_comparison_support': decimal_string(slacks[active_binding_degree-1], 60),
        'minimum_slack_on_comparison_support_degree': active_binding_degree,
        'all_omitted_degrees_safe': True,
        'reason_omitted_degrees_safe': (
            f"Every comparison component has degree at most {max(active_degrees)}; K_32 has "
            'nonnegative Gegenbauer coefficients in all higher degrees.'
        ),
    }
    (ROOT / 'SIGNED_NEAR_OPTIMALITY_VERIFICATION_BLOCKTRACE_ORDER320.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
