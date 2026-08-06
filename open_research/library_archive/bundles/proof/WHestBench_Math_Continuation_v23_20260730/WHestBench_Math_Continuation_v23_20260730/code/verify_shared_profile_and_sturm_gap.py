#!/usr/bin/env python3
"""Independent exact verifier for the v22 shared-profile conclusions.

Uses only Python's standard library. It verifies:
1. the v21 certificate contains repeated adjacent profiles with the same s and
   distinct r, giving an exact atomic-equality contradiction;
2. every active harmonic block has dimension at least N, so one global
   rank-N block-trace matrix can attain every component's abstract rank floor
   simultaneously;
3. for nonnegative mass-one rules, the four s=3,4 profiles yield an explicit
   quantitative shared off-diagonal gap via an exact Sturm calculation.
"""
from __future__ import annotations

from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path
from typing import Iterable
import json
import math

getcontext().prec = 100
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CERT_PATH = ROOT / "source_v21" / "SIGNED_NEAR_OPTIMALITY_CERTIFICATE_BLOCKTRACE_ORDER320.json"
JET_PATH = ROOT / "source_v21" / "K32_MACLAURIN_INTERVALS_ORDER320.json"
D = 256
N = 66048
Q0 = Fraction(130035, 10**17)  # 1.30035e-12


def trim(p: list[Fraction]) -> list[Fraction]:
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def padd(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    n = max(len(a), len(b))
    out = [Fraction(0)] * n
    for i in range(n):
        out[i] = (a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)
    return trim(out)


def psub(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    return padd(a, [-x for x in b])


def pscale(a: list[Fraction], c: Fraction) -> list[Fraction]:
    return trim([c * x for x in a])


def pmul(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                if y:
                    out[i + j] += x * y
    return trim(out)


def pder(a: list[Fraction]) -> list[Fraction]:
    if len(a) <= 1:
        return [Fraction(0)]
    return trim([Fraction(i) * a[i] for i in range(1, len(a))])


def pdivmod(a: list[Fraction], b: list[Fraction]) -> tuple[list[Fraction], list[Fraction]]:
    a = trim(list(a))
    b = trim(list(b))
    if b == [0]:
        raise ZeroDivisionError
    if len(a) < len(b):
        return [Fraction(0)], a
    q = [Fraction(0)] * (len(a) - len(b) + 1)
    while len(a) >= len(b) and a != [0]:
        k = len(a) - len(b)
        c = a[-1] / b[-1]
        q[k] += c
        sub = [Fraction(0)] * k + [c * z for z in b]
        a = psub(a, sub)
    return trim(q), trim(a)


def peval(a: list[Fraction], x: Fraction) -> Fraction:
    acc = Fraction(0)
    for z in reversed(a):
        acc = acc * x + z
    return acc


def sign(q: Fraction) -> int:
    return (q > 0) - (q < 0)


def variations(signs: Iterable[int]) -> int:
    nz = [s for s in signs if s]
    return sum(a != b for a, b in zip(nz, nz[1:]))


def sturm_sequence(p: list[Fraction]) -> list[list[Fraction]]:
    p = trim(p)
    seq = [p, pder(p)]
    while seq[-1] != [0]:
        _, rem = pdivmod(seq[-2], seq[-1])
        if rem == [0]:
            break
        seq.append(pscale(rem, Fraction(-1)))
    return seq


def pgcd(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    a, b = trim(list(a)), trim(list(b))
    while b != [0]:
        _, r = pdivmod(a, b)
        a, b = b, r
    if a == [0]:
        return a
    return pscale(a, 1 / a[-1])


def harmonic_dim(l: int) -> int:
    if l == 0:
        return 1
    if l == 1:
        return D
    return math.comb(D + l - 1, l) - math.comb(D + l - 3, l - 2)


def normalized_gegenbauer(max_degree: int) -> list[list[Fraction]]:
    # G_0=1, G_1=x and
    # G_{n+1}=2(n+lambda)/(n+2lambda) x G_n - n/(n+2lambda) G_{n-1}.
    lam = Fraction(D - 2, 2)
    out = [[Fraction(1)], [Fraction(0), Fraction(1)]]
    for n in range(1, max_degree):
        first = [Fraction(0)] + pscale(out[n], 2 * (Fraction(n) + lam) / (Fraction(n) + 2 * lam))
        second = pscale(out[n - 1], Fraction(n, 1) / (Fraction(n) + 2 * lam))
        out.append(psub(first, second))
    return out


def decimal(q: Fraction, digits: int = 70) -> str:
    return format(Decimal(q.numerator) / Decimal(q.denominator), f".{digits}E")


def main() -> None:
    cert = json.loads(CERT_PATH.read_text())
    assert cert["scope"]["dimension"] == D
    assert cert["scope"]["maximum_nodes"] == N
    components = cert["components"]

    # Repeated-r witness for atomic strictness.
    by_s: dict[int, list[dict]] = {}
    for row in components:
        by_s.setdefault(int(row["s"]), []).append(row)
    duplicates = {
        s: rows for s, rows in by_s.items()
        if len({Fraction(row["r"]) for row in rows}) >= 2
    }
    assert 3 in duplicates
    r3 = sorted({Fraction(row["r"]) for row in duplicates[3]})
    assert len(r3) >= 2 and r3[0] != r3[1]

    G = normalized_gegenbauer(5)
    assert pgcd(G[3], G[4]) == [Fraction(1)]
    assert pgcd(G[4], G[5]) == [Fraction(1)]

    # One global shared matrix can attain all abstract component floors.
    active_degrees = sorted({int(row["s"]) for row in components} |
                            {int(row["s"]) + 1 for row in components})
    dim_checks = {l: harmonic_dim(l) for l in active_degrees}
    assert min(dim_checks.values()) >= N
    component_abstract_checks = []
    for row in components:
        s = int(row["s"])
        r = Fraction(row["r"])
        ds, dt = harmonic_dim(s), harmonic_dim(s + 1)
        T = ds + r * dt
        S2 = ds + r * r * dt
        declared_floor = T * T / N - S2
        # In the explicit shared construction, N orthogonal factor columns each
        # have squared norm T/N, hence Frobenius discrepancy T^2/N-S2.
        constructed_floor = N * (T / N) ** 2 - S2
        assert declared_floor == constructed_floor and declared_floor > 0
        component_abstract_checks.append({
            "s": s,
            "r": str(r),
            "rank_floor": str(declared_floor),
        })

    # Exact degree-10 Sturm gap for nonnegative rules, using the four s=3,4 rows.
    q = [Fraction(0)]
    A = Fraction(0)
    selected = []
    for row in components:
        s = int(row["s"])
        if s not in (3, 4):
            continue
        r = Fraction(row["r"])
        y = Fraction(Decimal(row["y"]))
        ds, dt = harmonic_dim(s), harmonic_dim(s + 1)
        B = (ds + r * dt) ** 2 / N - ds - r * r * dt
        c = y / B
        L = padd(pscale(G[s], ds), pscale(G[s + 1], r * dt))
        q = padd(q, pscale(pmul(L, L), c))
        A += c * (ds + r * dt) ** 2
        selected.append({"s": s, "r": str(r), "y": str(y), "c": str(c)})
    assert len(selected) == 4
    assert len(q) - 1 == 10
    assert A > Q0

    q_minus = list(q)
    q_minus[0] -= Q0
    sturm = sturm_sequence(q_minus)
    signs_minus = [sign(peval(p, Fraction(-1))) for p in sturm]
    signs_plus = [sign(peval(p, Fraction(1))) for p in sturm]
    vminus, vplus = variations(signs_minus), variations(signs_plus)
    assert vminus == vplus  # no real root in (-1,1)
    assert peval(q_minus, Fraction(0)) > 0
    assert peval(q_minus, Fraction(-1)) > 0
    assert peval(q_minus, Fraction(1)) > 0

    epsilon_positive = Q0 * Fraction(N - 1, N)
    old_floor = Fraction(Decimal(cert["certified_result"]["mse_lower_bound"]))

    # Arbitrary-total-mass extension. The comparison's used degree-zero
    # coefficient is sum c_j S2_j. The residual kernel contributes the
    # remaining constant harmonic coefficient times (1-sum w)^2.
    jet = json.loads(JET_PATH.read_text())
    maclaurin_lower = [Fraction(Decimal(row[0])) for row in jet["maclaurin_intervals"]]

    def sphere_moment(n: int) -> Fraction:
        if n % 2:
            return Fraction(0)
        k = n // 2
        out = Fraction(1)
        for j in range(k):
            out *= Fraction(2 * j + 1, D + 2 * j)
        return out

    k0_lower = sum(a * sphere_moment(n) for n, a in enumerate(maclaurin_lower))
    total_A = Fraction(0)  # coefficient of total_mass^2
    used0 = Fraction(0)
    for row in components:
        s = int(row["s"]); r = Fraction(row["r"]); y = Fraction(Decimal(row["y"]))
        ds, dt = harmonic_dim(s), harmonic_dim(s + 1)
        T = ds + r * dt; S2 = ds + r * r * dt
        B = T * T / N - S2
        c = y / B
        total_A += c * T * T / N
        used0 += c * S2
    assert total_A - used0 == old_floor
    residual0 = k0_lower - used0
    assert residual0 > 0
    quad = total_A + residual0
    optimum_mass = k0_lower / quad
    arbitrary_mass_floor = k0_lower - k0_lower * k0_lower / quad
    kerdock_upper = Fraction(Decimal(cert["certified_result"]["kerdock_mse_upper_bound"]))
    improved_floor = old_floor + epsilon_positive

    report = {
        "status": "PASS",
        "scope": {
            "dimension": D,
            "node_budget": N,
            "source_certificate_components": len(components),
        },
        "shared_abstract_relaxation": {
            "active_harmonic_blocks": len(active_degrees),
            "minimum_active_block_dimension": min(dim_checks.values()),
            "all_component_floors_simultaneously_attainable": True,
            "construction": "For every block choose N orthonormal columns E_l and set global block factor Z_l=sqrt(d_l/N) E_l. Every profile-scaled principal moment matrix has N equal eigenvalues T_profile/N.",
        },
        "atomic_equality": {
            "possible": False,
            "witness_degree": 3,
            "distinct_r": [str(r3[0]), str(r3[1])],
            "reason": "Equality would force every off-diagonal inner product to be a common zero of G_3 and G_4; their exact polynomial gcd is 1.",
            "duplicate_s_groups": len(duplicates),
        },
        "arbitrary_total_mass": {
            "k0_lower_partial_order320": decimal(k0_lower),
            "comparison_degree0_used": decimal(used0),
            "optimal_total_mass_in_relaxation": decimal(optimum_mass),
            "mse_floor": decimal(arbitrary_mass_floor),
            "fraction_of_kerdock_upper": decimal(arbitrary_mass_floor / kerdock_upper),
            "same_cost_cap": decimal(kerdock_upper / arbitrary_mass_floor),
            "mass_one_loss": decimal(old_floor - arbitrary_mass_floor),
        },
        "bounded_variation_consequence": {
            "statement": "For every finite V, the minimum risk over mass-one signed rules with l1 weight norm at most V is strictly above the v21 floor. Hence any sequence approaching the abstract floor must have l1 norm tending to infinity.",
        },
        "nonnegative_sturm_gap": {
            "selected_profiles": selected,
            "q_degree": len(q) - 1,
            "certified_q_lower": decimal(Q0),
            "sturm_length": len(sturm),
            "variations_at_minus_one": vminus,
            "variations_at_plus_one": vplus,
            "root_count_open_interval": vminus - vplus,
            "epsilon_absolute": decimal(epsilon_positive),
            "old_floor": decimal(old_floor),
            "improved_nonnegative_floor": decimal(improved_floor),
            "fraction_of_kerdock_upper": decimal(improved_floor / kerdock_upper),
            "same_cost_cap": decimal(kerdock_upper / improved_floor),
            "note": "This quantitative corollary is weaker than the existing T22 positive-weight theorem; its value is as an independent shared-profile realizability check.",
        },
    }
    out = ROOT / "results" / "SHARED_PROFILE_RELAXATION_AND_STURM_VERIFICATION.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
