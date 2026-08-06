#!/usr/bin/env python3
"""Exact/analytic certificate for the WHestBench T16 reduced-cost claim.

Target:
  d = 256, alpha = (d-2)/2 = 127, N = 66048.
  q_0 = 1 - 1/N and q_l = -1/N for l >= 1.

The script proves r_l = q_l - sum_j lambda_j G_l(t_j) < 0 for every l >= 6.
It uses:
  * exact Fraction arithmetic to derive the 3-node dual quadrature;
  * exact rational root-isolation sign checks;
  * exact integer arithmetic for degrees 6..14658;
  * an analytic normalized-Gegenbauer tail bound for l >= 14659.

Only the Python standard library is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import factorial
from pathlib import Path
import hashlib
import json
import time

D = 256
ALPHA = 127
N = 66048
CUTOFF = 14659
DELTA_NUM = 13951
DELTA_DEN = 10**6


def fstr(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}" if x.denominator != 1 else str(x.numerator)


def poly_add(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    n = max(len(a), len(b))
    out = [Fraction(0) for _ in range(n)]
    for i, x in enumerate(a): out[i] += x
    for i, x in enumerate(b): out[i] += x
    while len(out) > 1 and out[-1] == 0: out.pop()
    return out


def poly_scale(a: list[Fraction], c: Fraction) -> list[Fraction]:
    return [c*x for x in a]


def poly_mul_x(a: list[Fraction]) -> list[Fraction]:
    return [Fraction(0)] + a


def normalized_gegenbauer_polynomials(max_degree: int) -> list[list[Fraction]]:
    """Exact normalized G_l in ascending monomial coefficients."""
    G = [[Fraction(1)]]
    if max_degree == 0:
        return G
    G.append([Fraction(0), Fraction(1)])
    for l in range(1, max_degree):
        A = Fraction(2*l + D - 2, l + D - 2)
        B = Fraction(l, l + D - 2)
        G.append(poly_add(poly_scale(poly_mul_x(G[l]), A), poly_scale(G[l-1], -B)))
    return G


def derive_monomial_moments() -> list[Fraction]:
    """Recover m_k=L[t^k] from L[G_0]=1-1/N and L[G_l]=-1/N."""
    G = normalized_gegenbauer_polynomials(5)
    q = [Fraction(N-1, N)] + [Fraction(-1, N)]*5
    moments: list[Fraction] = []
    for l in range(6):
        known = sum(G[l][k] * moments[k] for k in range(l))
        leading = G[l][l]
        moments.append((q[l] - known) / leading)
    # Independent exact audit: recompute L[G_l].
    for l in range(6):
        got = sum(G[l][k]*moments[k] for k in range(l+1))
        assert got == q[l], (l, got, q[l])
    return moments


def solve_linear_fraction(A: list[list[Fraction]], b: list[Fraction]) -> list[Fraction]:
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot = next(i for i in range(col, n) if M[i][col] != 0)
        M[col], M[pivot] = M[pivot], M[col]
        p = M[col][col]
        M[col] = [x/p for x in M[col]]
        for i in range(n):
            if i == col: continue
            c = M[i][col]
            if c:
                M[i] = [x-c*y for x, y in zip(M[i], M[col])]
    return [M[i][-1] for i in range(n)]


def derive_orthogonal_cubic(m: list[Fraction]) -> tuple[Fraction, Fraction, Fraction]:
    """Find p=t^3+a t^2+b t+c with L[t^k p]=0, k=0,1,2."""
    # Unknown order is (c,b,a).
    A = [[m[k], m[k+1], m[k+2]] for k in range(3)]
    rhs = [-m[k+3] for k in range(3)]
    c, b, a = solve_linear_fraction(A, rhs)
    for k in range(3):
        assert m[k+3] + a*m[k+2] + b*m[k+1] + c*m[k] == 0
    return a, b, c


def P(x: Fraction) -> Fraction:
    """Integer-scaled exact orthogonal cubic."""
    return 22102*x**3 + 21930*x**2 - 87*x - 85


@dataclass(frozen=True)
class RI:
    lo: Fraction
    hi: Fraction
    def __post_init__(self):
        if self.lo > self.hi: raise ValueError(self)


def ri_add(a: RI, b: RI) -> RI: return RI(a.lo+b.lo, a.hi+b.hi)
def ri_neg(a: RI) -> RI: return RI(-a.hi, -a.lo)
def ri_sub(a: RI, b: RI) -> RI: return ri_add(a, ri_neg(b))
def ri_mul(a: RI, b: RI) -> RI:
    vals = (a.lo*b.lo, a.lo*b.hi, a.hi*b.lo, a.hi*b.hi)
    return RI(min(vals), max(vals))
def ri_div(a: RI, b: RI) -> RI:
    if b.lo <= 0 <= b.hi: raise ZeroDivisionError(b)
    vals = (a.lo/b.lo, a.lo/b.hi, a.hi/b.lo, a.hi/b.hi)
    return RI(min(vals), max(vals))
def ri_point(x: Fraction) -> RI: return RI(x, x)


def isolate_roots_and_weights(m: list[Fraction]):
    # Each interval has an exact sign change. Since P is cubic and intervals are
    # disjoint, these are all three roots, one per interval.
    roots = [
        RI(Fraction(-992278935, 10**9), Fraction(-992278934, 10**9)),
        RI(Fraction(-62224856, 10**9), Fraction(-62224855, 10**9)),
        RI(Fraction(62285891, 10**9), Fraction(62285892, 10**9)),
    ]
    sign_rows = []
    for I in roots:
        pl, ph = P(I.lo), P(I.hi)
        assert pl*ph < 0, (I, pl, ph)
        sign_rows.append({"left": fstr(I.lo), "right": fstr(I.hi),
                          "P_left": fstr(pl), "P_right": fstr(ph)})

    # Lagrange-weight formula using only m0,m1,m2. Exact rational intervals
    # prove all three algebraic weights are positive.
    weights: list[RI] = []
    for j in range(3):
        k, h = [u for u in range(3) if u != j]
        numerator = ri_add(
            ri_point(m[2]),
            ri_add(
                ri_neg(ri_mul(ri_add(roots[k], roots[h]), ri_point(m[1]))),
                ri_mul(ri_mul(roots[k], roots[h]), ri_point(m[0])),
            ),
        )
        denominator = ri_mul(ri_sub(roots[j], roots[k]), ri_sub(roots[j], roots[h]))
        w = ri_div(numerator, denominator)
        assert w.lo > 0, (j, w)
        weights.append(w)

    # Coarse exact root enclosure used in the tail: |t_j| < 0.993.
    assert all(I.lo > Fraction(-993,1000) and I.hi < Fraction(993,1000) for I in roots)
    return roots, weights, sign_rows


def exact_finite_sweep(cutoff: int):
    """Exact signs via a denominator-scaled quotient-ring recurrence.

    P=22102 t^3+21930 t^2-87t-85, so multiplication by t modulo P maps
      (v0,v1,v2) -> (85v2, 22102v0+87v2, 22102v1-21930v2) / 22102.
    """
    scale_p = 22102
    aint = 21930

    def mul_t_num(v: tuple[int,int,int]) -> tuple[int,int,int]:
        v0, v1, v2 = v
        return (85*v2, scale_p*v0 + 87*v2, scale_p*v1 - aint*v2)

    v_prev = (1,0,0)  # G0 remainder numerator; denominator 1
    v_cur = (0,1,0)   # G1 remainder numerator; denominator 1
    den_cur = 1

    best_num = None
    best_den = None
    best_l = None
    first_rows = []
    rolling = hashlib.sha256()
    start = time.perf_counter()

    for l in range(1, cutoff):
        if l >= 6:
            qnum = (N-1)*v_cur[0] - v_cur[1] + 257*v_cur[2]
            rnum = -den_cur - qnum
            rden = N*den_cur
            if rnum >= 0:
                raise AssertionError(("nonnegative reduced cost", l, rnum, rden))
            if best_num is None or rnum*best_den > best_num*rden:
                best_num, best_den, best_l = rnum, rden, l
            if l <= 20:
                first_rows.append({"degree": l, "exact": f"{rnum}/{rden}",
                                   "decimal": float(Fraction(rnum,rden))})
            # Compact reproducibility digest of each exact sign state.
            for z in (l, rnum, rden):
                sign = b'+' if z >= 0 else b'-'
                zz = abs(z)
                raw = zz.to_bytes(max(1,(zz.bit_length()+7)//8), 'big')
                rolling.update(sign); rolling.update(len(raw).to_bytes(4,'big')); rolling.update(raw)

        mt = mul_t_num(v_cur)
        anum = 2*l + 254
        prev_multiplier = scale_p if l == 1 else scale_p*scale_p*(l+253)
        v_next = tuple(anum*mt[i] - l*prev_multiplier*v_prev[i] for i in range(3))
        den_next = scale_p*(l+254)*den_cur
        v_prev, v_cur = v_cur, v_next
        den_cur = den_next

    _elapsed = time.perf_counter()-start
    best = Fraction(best_num,best_den)
    return {
        "degrees_checked": [6, cutoff-1],
        "all_strictly_negative": True,
        "largest_reduced_cost_degree": best_l,
        "largest_reduced_cost_exact": fstr(best),
        "largest_reduced_cost_decimal": float(best),
        "first_reduced_costs": first_rows,
        "final_denominator_bits": den_cur.bit_length(),
        "rolling_sha256": rolling.hexdigest(),
    }


def tail_certificate():
    # From the Laplace integral representation, for alpha=127 and delta=1-t^2:
    # |G_l(t)| <= 254! / [127! (l delta)^127].
    # The exact root enclosure |t|<0.993 gives delta>13951/10^6.
    lhs = N*factorial(254)*(DELTA_DEN**ALPHA)
    rhs = factorial(127)*((CUTOFF*DELTA_NUM)**ALPHA)
    rhs_prev = factorial(127)*(((CUTOFF-1)*DELTA_NUM)**ALPHA)
    assert lhs < rhs
    assert not (lhs < rhs_prev)  # minimal cutoff for this conservative root bound
    B = Fraction(factorial(254)*(DELTA_DEN**ALPHA), factorial(127)*((CUTOFF*DELTA_NUM)**ALPHA))
    assert B < Fraction(1,N)
    return {
        "alpha": ALPHA,
        "root_absolute_upper": "993/1000",
        "delta_lower": f"{DELTA_NUM}/{DELTA_DEN}",
        "bound": "|G_l(t)| <= 254! / (127! * (l*(1-t^2))^127)",
        "cutoff": CUTOFF,
        "cutoff_is_minimal_for_this_delta_bound": True,
        "bound_at_cutoff_exact": fstr(B),
        "bound_at_cutoff_decimal": float(B),
        "one_over_N_decimal": float(Fraction(1,N)),
        "strict_margin_decimal": float(Fraction(1,N)-B),
        "integer_inequality_sha256": hashlib.sha256((str(lhs)+"<"+str(rhs)).encode()).hexdigest(),
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    moments = derive_monomial_moments()
    a,b,c = derive_orthogonal_cubic(moments)
    assert (a,b,c) == (Fraction(255,257), Fraction(-87,22102), Fraction(-85,22102))
    roots, weights, sign_rows = isolate_roots_and_weights(moments)
    finite = exact_finite_sweep(CUTOFF)
    tail = tail_certificate()

    cert = {
        "claim": "For d=256, N=66048, every dual reduced cost r_l is strictly negative for l>=6.",
        "status": "PROVED (exact finite arithmetic plus analytic tail)",
        "normalization": "G_l=C_l^127/C_l^127(1), G_l(1)=1",
        "q": {"q0": "66047/66048", "q_l_l_ge_1": "-1/66048"},
        "monomial_moments_m0_to_m5": [fstr(x) for x in moments],
        "orthogonal_cubic_monic": [fstr(c), fstr(b), fstr(a), "1"],
        "orthogonal_cubic_integer": "22102*t^3 + 21930*t^2 - 87*t - 85",
        "root_sign_certificates": sign_rows,
        "root_intervals_decimal": [[float(I.lo),float(I.hi)] for I in roots],
        "weight_intervals_exact": [[fstr(w.lo),fstr(w.hi)] for w in weights],
        "weight_intervals_decimal": [[float(w.lo),float(w.hi)] for w in weights],
        "weight_sum_exact": "66047/66048",
        "finite_certificate": finite,
        "tail_certificate": tail,
        "scope_note": "This proves the all-degree reduced-cost inequalities. Exact primal-dual equality for a particular degree-5 minorant is a separate certificate.",
    }
    path = out_dir/'T16_ALL_DEGREE_CERTIFICATE.json'
    path.write_text(json.dumps(cert, indent=2) + "\n")
    print(json.dumps(cert, indent=2))


if __name__ == '__main__':
    main()
