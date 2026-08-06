#!/usr/bin/env python3
"""Close the WHestBench T16 all-degree auxiliary LP by primal-dual equality.

This script supplements prove_t16_all_degree.py.  The prior certificate proves
strict dual reduced-cost negativity for every Gegenbauer degree >= 6.  Here we:

1. recover the exact three algebraic dual nodes (roots of the orthogonal cubic);
2. define the degree-5 primal h_* as the Hermite interpolant of K_32 at those
   nodes, matching both value and first derivative;
3. interval-enclose the six normalized-Gegenbauer coefficients and prove the
   five constrained coefficients are positive;
4. prove K_32^(6)(t)>0 on (-1,1), using an analytic Bell-polynomial reduction,
   an exact rational Bernstein certificate, and a small interval certificate;
5. invoke the Hermite remainder formula to prove h_* <= K_32;
6. obtain exact primal-dual equality from moment matching and contact.

Trust base for the new numerical enclosures: CPython Fraction arithmetic and
mpmath 1.3.0 interval arithmetic at 80 decimal digits.  The polynomial
Bernstein portions are exact rational computations.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import platform
import sys

import mpmath as mp
import sympy as sp

D = 256
DEPTH = 32
OUTER_DEPTH = 31
N = 66048
IV_DPS = 80
ROOT_BISECTIONS = 220

mp.mp.dps = IV_DPS
mp.iv.dps = IV_DPS
iv = mp.iv


def fstr(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}" if x.denominator != 1 else str(x.numerator)


def poly_add(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    n = max(len(a), len(b))
    out = [Fraction(0) for _ in range(n)]
    for i, z in enumerate(a):
        out[i] += z
    for i, z in enumerate(b):
        out[i] += z
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def poly_scale(a: list[Fraction], c: Fraction) -> list[Fraction]:
    return [c * z for z in a]


def poly_mul_x(a: list[Fraction]) -> list[Fraction]:
    return [Fraction(0)] + a


def normalized_gegenbauer(max_degree: int = 5) -> list[list[Fraction]]:
    G = [[Fraction(1)]]
    if max_degree == 0:
        return G
    G.append([Fraction(0), Fraction(1)])
    for ell in range(1, max_degree):
        A = Fraction(2 * ell + D - 2, ell + D - 2)
        B = Fraction(ell, ell + D - 2)
        G.append(poly_add(poly_scale(poly_mul_x(G[ell]), A), poly_scale(G[ell - 1], -B)))
    return G


G = normalized_gegenbauer(5)
GP = [[Fraction(k) * p[k] for k in range(1, len(p))] for p in G]


def P(x: Fraction) -> Fraction:
    return 22102 * x**3 + 21930 * x**2 - 87 * x - 85


def refine_root(a: Fraction, b: Fraction, iterations: int = ROOT_BISECTIONS) -> tuple[Fraction, Fraction]:
    fa, fb = P(a), P(b)
    if fa * fb >= 0:
        raise AssertionError("initial interval lacks strict sign change")
    for _ in range(iterations):
        m = (a + b) / 2
        fm = P(m)
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return a, b


INITIAL_ROOTS = [
    (Fraction(-992278935, 10**9), Fraction(-992278934, 10**9)),
    (Fraction(-62224856, 10**9), Fraction(-62224855, 10**9)),
    (Fraction(62285891, 10**9), Fraction(62285892, 10**9)),
]


def frac_interval(a: Fraction, b: Fraction):
    return iv.mpf([fstr(a), fstr(b)])


def endpoint_text(x, upper: bool = False) -> str:
    endpoint = x.b if upper else x.a
    raw = str(endpoint).strip("[]")
    parts = [p.strip() for p in raw.split(",")]
    return parts[-1] if upper else parts[0]


def interval_pair(x) -> dict[str, str]:
    return {"lower": endpoint_text(x), "upper": endpoint_text(x, True)}


def iv_poly_eval(poly: list[Fraction], x):
    y = iv.mpf(0)
    for z in reversed(poly):
        y = y * x + iv.mpf(fstr(z))
    return y


def iv_asin(x):
    # Valid on [-1,1], and monotone. atan2(x,sqrt(1-x^2)) avoids the absence
    # of a direct inverse-sine method in mpmath.iv.
    return iv.atan2(x, iv.sqrt(1 - x * x))


def kappa_point(x):
    s = iv.sqrt(1 - x * x)
    kp = iv.mpf("1/2") + iv_asin(x) / iv.pi
    return s / iv.pi + kp * x


def kappa_range(x):
    # kappa is increasing, so endpoint evaluation gives a much tighter range
    # than a dependency-heavy direct interval expression.
    lo = kappa_point(x.a)
    hi = kappa_point(x.b)
    return iv.mpf([lo.a, hi.b])


def kernel_and_prime_interval(x, depth: int = DEPTH):
    p = iv.mpf(1)
    y = x
    for _ in range(depth):
        s = iv.sqrt(1 - y * y)
        kp = iv.mpf("1/2") + iv_asin(y) / iv.pi
        p *= kp
        y = kappa_range(y)
    return y, p


def hermite_coefficient_intervals(root_intervals: list[tuple[Fraction, Fraction]]):
    A = iv.matrix(6, 6)
    b = iv.matrix(6, 1)
    for j, (ra, rb) in enumerate(root_intervals):
        x = frac_interval(ra, rb)
        kval, kp = kernel_and_prime_interval(x)
        for ell in range(6):
            A[2 * j, ell] = iv_poly_eval(G[ell], x)
            A[2 * j + 1, ell] = iv_poly_eval(GP[ell], x)
        b[2 * j] = kval
        b[2 * j + 1] = kp
    c = iv.lu_solve(A, b)
    return [c[i] for i in range(6)]


def bernstein_coefficients(poly: sp.Poly) -> list[sp.Rational]:
    n = poly.degree()
    power = [sp.Rational(poly.nth(k)) for k in range(n + 1)]
    return [
        sp.factor(sum(power[k] * sp.binomial(i, k) / sp.binomial(n, k) for k in range(i + 1)))
        for i in range(n + 1)
    ]


def b62_exact_certificate():
    """Prove B_{6,2}(kappa) >= -kappa^(6)/4 for t<0.

    Put t=-cos(phi), c=cos(phi), s=sin(phi), 0<phi<pi/2.  The classical
    inequality phi*cot(phi) <= (1+2c)/3 follows from differentiating
      sin(phi)(1+2cos(phi))-3phi cos(phi)
    and reducing to 3a cos(a)-sin(3a)>=0, a=phi/2.

    After this substitution it is enough to prove
      3D(c) >= 4(1-c^2)^(3/2) R(c),
    D=24c^4+72c^2+9, R=24c^3-28c^2+36c+3.
    Squaring is safe because both sides are positive.  Exact Bernstein
    coefficients on four quarter intervals prove the resulting polynomial
    is strictly positive.
    """
    c, y = sp.symbols("c y")
    Dp = 24 * c**4 + 72 * c**2 + 9
    Rp = 24 * c**3 - 28 * c**2 + 36 * c + 3
    Fp = sp.Poly(sp.expand(9 * Dp**2 - 16 * (1 - c**2) ** 3 * Rp**2), c, domain=sp.QQ)
    r_bernstein = bernstein_coefficients(sp.Poly(Rp, c, domain=sp.QQ))
    if min(r_bernstein) <= 0:
        raise AssertionError(("R polynomial not positive", r_bernstein))
    quarters = [
        (sp.Rational(0), sp.Rational(1, 4)),
        (sp.Rational(1, 4), sp.Rational(1, 2)),
        (sp.Rational(1, 2), sp.Rational(3, 4)),
        (sp.Rational(3, 4), sp.Rational(1)),
    ]
    rows = []
    for a, b in quarters:
        transformed = sp.Poly(sp.expand(Fp.as_expr().subs(c, a + (b - a) * y)), y, domain=sp.QQ)
        coeffs = bernstein_coefficients(transformed)
        minimum = min(coeffs)
        if minimum <= 0:
            raise AssertionError((a, b, minimum))
        rows.append({
            "interval": [str(a), str(b)],
            "minimum_bernstein_coefficient": str(minimum),
            "all_positive": True,
        })
    return {
        "claim": "B_6,2(kappa)(t) >= -kappa^(6)(t)/4 for -1<t<0",
        "trigonometric_inequality": "phi*cot(phi) <= (1+2*cos(phi))/3 on [0,pi/2]",
        "R_polynomial": str(Rp),
        "R_bernstein_coefficients_on_0_1": [str(z) for z in r_bernstein],
        "squared_polynomial": str(Fp.as_expr()),
        "bernstein_subintervals": rows,
        "exact": True,
    }


def outer_log_derivative_ratio_box(a: str, b: str | None):
    upper = (1 / iv.pi).b if b is None else b
    x = iv.mpf([a, upper])
    p = iv.mpf(1)  # F'
    r = iv.mpf(0)  # F''/F'
    for _ in range(OUTER_DEPTH):
        s = iv.sqrt(1 - x * x)
        kp = iv.mpf("1/2") + iv_asin(x) / iv.pi
        kpp = 1 / (iv.pi * s)
        r = r + (kpp / kp) * p
        p = p * kp
        x = kappa_range(x)
    return {
        "input_interval": [a, "1/pi" if b is None else b],
        "ratio_interval": interval_pair(r),
    }


def outer_ratio_certificate():
    rows = [
        outer_log_derivative_ratio_box("0", "0.1"),
        outer_log_derivative_ratio_box("0.1", "0.2"),
        outer_log_derivative_ratio_box("0.2", "0.3"),
        outer_log_derivative_ratio_box("0.3", None),
    ]
    max_upper = max(mp.mpf(row["ratio_interval"]["upper"]) for row in rows)
    if not max_upper < mp.mpf(9) / 4:
        raise AssertionError(max_upper)
    return {
        "claim": "For F=kappa composed 31 times, F''(u)/F'(u) < 9/4 on 0<=u<=1/pi",
        "boxes": rows,
        "maximum_certified_upper": mp.nstr(max_upper, 85),
        "comparison_bound": "9/4",
        "interval_precision_decimal_digits": IV_DPS,
        "passed": True,
    }


def main() -> None:
    root_intervals = [refine_root(a, b) for a, b in INITIAL_ROOTS]
    coeffs = hermite_coefficient_intervals(root_intervals)
    if not all(mp.mpf(endpoint_text(coeffs[i])) > 0 for i in range(1, 6)):
        raise AssertionError("a constrained Gegenbauer coefficient was not certified positive")

    b62 = b62_exact_certificate()
    outer = outer_ratio_certificate()

    # The remaining Bell terms are analytic:
    # B63 >= 0 by Q/phi^2=(r-2c)^2+1-2c^2 with r=sin(phi)/phi<=1;
    # B64 >= 0 from sin(phi)>=phi cos(phi); B65,B66>0.
    conclusion = (
        "For t>=0 every derivative of kappa through order 6 is nonnegative. "
        "For t<0, B63,B64,B65,B66 are nonnegative, B62>=-B61/4, and "
        "F''/F'<9/4. Hence K32^(6)>=F'B61*(1-9/16)>0."
    )

    theorem = (
        "Let h_* be the unique degree-at-most-5 polynomial satisfying "
        "h_*(t_j)=K_32(t_j) and h_*'(t_j)=K_32'(t_j) at the three roots "
        "of 22102t^3+21930t^2-87t-85. Then h_*<=K_32 on [-1,1], its "
        "nonconstant normalized-Gegenbauer coefficients are positive, and it "
        "is an optimizer of the unrestricted all-degree Delsarte auxiliary LP."
    )

    payload = {
        "title": "T16 primal-dual closure certificate",
        "status": "PROVED UNDER EXPLICIT INTERVAL-ARITHMETIC TRUST BASE",
        "dimension": D,
        "depth": DEPTH,
        "node_budget": N,
        "orthogonal_cubic": "22102*t^3 + 21930*t^2 - 87*t - 85",
        "root_intervals": [
            {
                "exact": [fstr(a), fstr(b)],
                "decimal": interval_pair(frac_interval(a, b)),
                "sign_change": P(a) * P(b) < 0,
            }
            for a, b in root_intervals
        ],
        "hermite_primal": {
            "definition": "degree<=5 Hermite interpolant matching K32 and K32' at all three dual nodes",
            "gegenbauer_coefficient_intervals_c0_to_c5": [interval_pair(z) for z in coeffs],
            "all_nonconstant_coefficients_strictly_positive": True,
        },
        "sixth_derivative_certificate": {
            "bell_decomposition": (
                "(F o kappa)^(6)=sum_{k=1}^6 F^(k)(kappa(t))*B_{6,k}; "
                "F=kappa^31"
            ),
            "b62_bound": b62,
            "outer_ratio_bound": outer,
            "other_bell_terms": {
                "B63_nonnegative": "analytic proof using r=sin(phi)/phi<=1",
                "B64_nonnegative": "analytic proof using sin(phi)>=phi*cos(phi)",
                "B65_B66_positive": True,
            },
            "conclusion": conclusion,
        },
        "hermite_remainder": (
            "K32(t)-h_*(t)=K32^(6)(xi)/6!*product_j(t-t_j)^2 >=0 for t in (-1,1); "
            "endpoint values follow by continuity."
        ),
        "primal_dual_equality": (
            "Moment matching gives q_l=sum_j lambda_j G_l(t_j) for l=0..5. "
            "Contact gives sum_l q_l c_l=sum_j lambda_j K32(t_j), exactly the dual objective. "
            "The prior T16 certificate gives strict dual feasibility for every l>=6."
        ),
        "theorem": theorem,
        "scope": {
            "proved": "all-degree auxiliary-LP optimality of h_* for d=256, depth=32, N=66048",
            "not_implied": [
                "exact optimality of the Kerdock cubature rule",
                "finite-width optimality",
                "signed-weight arbitrary-node optimality",
                "network-adaptive or nonlinear estimator impossibility",
            ],
        },
        "trust_base": {
            "python": sys.version,
            "platform": platform.platform(),
            "mpmath": mp.__version__,
            "sympy": sp.__version__,
            "interval_decimal_digits": IV_DPS,
            "exact_parts": "Fraction root bisection and rational Bernstein coefficients",
        },
    }

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["certificate_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    out = Path(__file__).with_name("T16_PRIMAL_DUAL_CLOSURE_CERTIFICATE.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
