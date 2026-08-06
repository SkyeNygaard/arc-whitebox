#!/usr/bin/env python3
"""Independent sanity checks for the salvaged theorem package.

These computations check algebraic identities and representative numerical cases.
They are not substitutes for the analytic proofs in the Markdown files.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp

OUT = Path(__file__).with_name("SALVAGED_THEOREMS_VERIFICATION.json")
mp.mp.dps = 80


def kappa(t: mp.mpf) -> mp.mpf:
    return (mp.sqrt(1-t*t) + (mp.pi-mp.acos(t))*t) / mp.pi


def kappa_coeff(m: int) -> mp.mpf:
    # coefficient of t^(2m), m>=1
    return mp.binomial(2*m-2, m-1) / (2*m*(2*m-1)*(4**(m-1))*mp.pi)


def convolve(a, b, nmax):
    out = [mp.mpf("0") for _ in range(nmax+1)]
    for i, x in enumerate(a):
        if i > nmax:
            break
        for j, y in enumerate(b):
            if i+j > nmax:
                break
            out[i+j] += x*y
    return out


def compose_outer_kappa(inner, nmax, outer_terms=100):
    # kappa(s)=1/pi + s/2 + sum_m b_m s^(2m)
    out = [mp.mpf("0") for _ in range(nmax+1)]
    out[0] += 1/mp.pi
    for n in range(nmax+1):
        out[n] += inner[n]/2
    power = [mp.mpf("0") for _ in range(nmax+1)]
    power[0] = 1
    inner2 = convolve(inner, inner, nmax)
    power = inner2[:]
    for m in range(1, outer_terms+1):
        b = kappa_coeff(m)
        for n in range(nmax+1):
            out[n] += b*power[n]
        power = convolve(power, inner2, nmax)
    return out


def kappa_series(nmax):
    a = [mp.mpf("0") for _ in range(nmax+1)]
    a[0] = 1/mp.pi
    if nmax >= 1:
        a[1] = mp.mpf("0.5")
    for m in range(1, nmax//2+1):
        a[2*m] = kappa_coeff(m)
    return a


def eval_series(a, t):
    return mp.fsum(a[n]*(t**n) for n in range(len(a)))


def t29_checks():
    nmax = 60
    a = kappa_series(nmax)
    positive_base = all(a[n] > 0 for n in [0,1] + list(range(2,nmax+1,2)))
    zero_odd = all(a[n] == 0 for n in range(3,nmax+1,2))
    k2 = compose_outer_kappa(a, nmax, outer_terms=180)
    all_positive_k2 = all(x > 0 for x in k2)

    # Series accuracy away from endpoints.
    ts = [mp.mpf("-0.8"), mp.mpf("-0.2"), mp.mpf("0.3"), mp.mpf("0.8")]
    base_err = max(abs(eval_series(a,t)-kappa(t)) for t in ts)
    comp_err = max(abs(eval_series(k2,t)-kappa(kappa(t))) for t in ts)

    # Strict-PD numerical example with antipodes and generic points.
    rng = np.random.default_rng(20260730)
    d = 7
    pts = []
    for _ in range(7):
        x = rng.normal(size=d)
        x /= np.linalg.norm(x)
        pts.append(x)
    pts.extend([-pts[0], -pts[2]])
    X = np.array(pts)
    G = np.empty((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            tf = float(X[i] @ X[j])
            tf = min(1.0, max(-1.0, tf))
            t = mp.mpf(str(tf))
            y = t
            for _ in range(32):
                y = kappa(y)
            G[i,j] = float(y)
    eigs = np.linalg.eigvalsh((G+G.T)/2)

    # General minimizer-set counterexample and ridge selector.
    N = 4
    Gone = np.ones((N,N))
    u = np.ones(N)/N
    w = np.array([1.25,-0.75,0.25,0.25])
    risk_u = float((u @ Gone @ u) - 2*np.ones(N)@u + 1)
    risk_w = float((w @ Gone @ w) - 2*np.ones(N)@w + 1)

    return {
        "kappa_base_coefficients_positive": positive_base,
        "kappa_base_odd_above_one_zero": zero_odd,
        "kappa_composed_first_61_coefficients_positive": all_positive_k2,
        "base_series_max_error_on_test_points": mp.nstr(base_err, 20),
        "composed_series_max_error_on_test_points": mp.nstr(comp_err, 20),
        "sample_K32_gram_min_eigenvalue": float(eigs[0]),
        "constant_field_uniform_risk": risk_u,
        "constant_field_nonuniform_mass_one_risk": risk_w,
    }


def t38_checks():
    # Symbolic spectrum formulas.
    A,O,C,d,M = sp.symbols("A O C d M")
    global_expr = sp.simplify((A-O)+d*(O-C)+M*d*C)
    global_target = A+(d-1)*O+(M*d-d)*C
    symbolic_global_ok = sp.simplify(global_expr-global_target) == 0

    # Direct matrix spectrum in a strict example.
    d0, M0 = 5, 4
    A0, O0, C0 = 1.7, 0.4, 0.12
    N = d0*M0
    G = np.full((N,N), C0)
    for b in range(M0):
        sl = slice(b*d0,(b+1)*d0)
        G[sl,sl] = O0
        np.fill_diagonal(G[sl,sl], A0)
    eigs = np.linalg.eigvalsh(G)
    expected = sorted(
        [A0-O0]*(M0*(d0-1))
        + [A0-O0+d0*(O0-C0)]*(M0-1)
        + [A0+(d0-1)*O0+(N-d0)*C0]
    )
    spectrum_err = float(np.max(np.abs(eigs-np.array(expected))))

    # Pure quadratic boundary.
    a2 = 2.3
    A1, O1, C1 = a2, 0.0, a2/d0
    Gq = np.full((N,N), C1)
    for b in range(M0):
        sl=slice(b*d0,(b+1)*d0)
        Gq[sl,sl]=O1
        np.fill_diagonal(Gq[sl,sl],A1)
    eq = np.linalg.eigvalsh(Gq)
    zero_mult = int(np.sum(np.abs(eq) < 1e-10))
    # Arbitrary basis masses, equal within each basis -> zero nonconstant form.
    masses = np.array([1.2,-0.4,0.3,-0.1])
    masses = masses + (1-masses.sum())/M0
    w = np.repeat(masses/d0,d0)
    qform = float(w@Gq@w)
    # Compare to formula: a2 sum_b(sum_i w^2-S_b^2/d) + constant cross part.
    nonconstant = 0.0
    for b in range(M0):
        wb=w[b*d0:(b+1)*d0]
        S=wb.sum()
        nonconstant += a2*(wb@wb-S*S/d0)

    # Exhaustive integer-partition check for P<d: one partial basis is optimal.
    def partitions(n, max_part=None):
        if n == 0:
            yield []
            return
        if max_part is None or max_part > n:
            max_part = n
        for first in range(max_part, 0, -1):
            for rest in partitions(n-first, first):
                yield [first] + rest
    partition_checks = {}
    for P in range(1, d0):
        def H(parts):
            return sum(d0*r/(a2*(d0-r)) for r in parts)
        vals=[(parts,H(parts)) for parts in partitions(P)]
        best=max(vals,key=lambda x:x[1])
        partition_checks[str(P)]={
            "best_partition":best[0],
            "one_basis_is_best":best[0]==[P],
            "best_H":best[1],
        }

    return {
        "symbolic_global_eigenvalue_identity": symbolic_global_ok,
        "strict_example_spectrum_max_error": spectrum_err,
        "pure_quadratic_between_basis_zero_multiplicity": zero_mult,
        "pure_quadratic_expected_zero_multiplicity": M0-1,
        "block_uniform_nonconstant_risk": nonconstant,
        "block_uniform_full_quadratic_form": qform,
        "pure_quadratic_budget_partition_checks": partition_checks,
    }


def replication_checks():
    rng=np.random.default_rng(12345)
    m=6
    trials=500000
    b=0.4
    sigma=1.3
    e=b+sigma*rng.normal(size=(trials,m))
    avg=e.mean(axis=1)
    empirical=float(np.mean(avg*avg))
    theoretical=b*b+sigma*sigma/m
    R0=b*b+sigma*sigma
    beta=b*b/R0
    linear_ratio=m*theoretical/R0
    formula_ratio=1+(m-1)*beta
    return {
        "independent_biased_average_empirical_mse": empirical,
        "independent_biased_average_theoretical_mse": theoretical,
        "absolute_error": abs(empirical-theoretical),
        "linear_cost_ratio_direct": linear_ratio,
        "linear_cost_ratio_formula": formula_ratio,
    }


def relu_checks():
    # Standard normal, small t, compare exact quadrature to L t^3/3.
    sqrt2pi=mp.sqrt(2*mp.pi)
    p=lambda z: mp.e**(-z*z/2)/sqrt2pi
    vals=[]
    for t in [mp.mpf("0.5"),mp.mpf("0.1"),mp.mpf("0.02"),mp.mpf("-0.1")]:
        if t>0:
            exact=mp.quad(lambda z:(z+t)**2*p(z),[-t,0])
        else:
            a=abs(t)
            exact=mp.quad(lambda z:(a-z)**2*p(z),[0,a])
        bound=(1/sqrt2pi)*abs(t)**3/3
        vals.append({"t":str(t),"exact":mp.nstr(exact,25),"bound":mp.nstr(bound,25),"ratio":mp.nstr(exact/bound,20)})
    asym_t=mp.mpf("1e-4")
    asym=mp.quad(lambda z:(z+asym_t)**2*p(z),[-asym_t,0])/(asym_t**3)
    return {
        "normal_cases":vals,
        "small_t_ratio_to_t_cubed":mp.nstr(asym,25),
        "normal_density_at_zero_over_3":mp.nstr(1/(3*sqrt2pi),25),
    }


def haar_checks():
    # Finite cyclic group analogue. Haar is uniform; e has zero uniform mean.
    e=np.array([2.0,-1.0,0.5,-1.5])
    assert abs(e.mean())<1e-15
    mu=np.array([0.4,0.2,0.2,0.2])
    h=np.ones(4)/4
    mean=float(mu@e)
    chi=float(np.sum((mu/h-1)**2*h))
    orient=float(h@(e*e))
    return {
        "conditional_mean":mean,
        "chi_square":chi,
        "orientation_risk":orient,
        "mean_squared":mean*mean,
        "chi_square_times_risk":chi*orient,
        "bound_holds":mean*mean <= chi*orient+1e-15,
    }


def main():
    result={
        "status":"PASS",
        "scope":"symbolic identities and representative numerical checks; analytic proofs are in the Markdown theorem files",
        "t29":t29_checks(),
        "t38":t38_checks(),
        "replication":replication_checks(),
        "relu":relu_checks(),
        "haar":haar_checks(),
    }
    checks=[
        result["t29"]["kappa_base_coefficients_positive"],
        result["t29"]["kappa_composed_first_61_coefficients_positive"],
        result["t29"]["sample_K32_gram_min_eigenvalue"]>0,
        result["t38"]["symbolic_global_eigenvalue_identity"],
        result["t38"]["strict_example_spectrum_max_error"]<1e-10,
        result["t38"]["pure_quadratic_between_basis_zero_multiplicity"]==result["t38"]["pure_quadratic_expected_zero_multiplicity"],
        abs(result["t38"]["block_uniform_nonconstant_risk"])<1e-12,
        all(v["one_basis_is_best"] for v in result["t38"]["pure_quadratic_budget_partition_checks"].values()),
        result["replication"]["absolute_error"]<0.01,
        result["haar"]["bound_holds"],
    ]
    if not all(checks):
        result["status"]="FAIL"
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["status"]=="PASS" else 1)

if __name__=="__main__":
    main()
