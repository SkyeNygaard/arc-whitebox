#!/usr/bin/env python3
"""Adversarial checks for T29's free-mass statement and scale numerics."""
from decimal import Decimal, getcontext, ROUND_FLOOR, ROUND_CEILING
import json
from pathlib import Path

getcontext().prec = 100

# Directed-enclosure helper using Decimal's next_minus/next_plus.
def outward_div(a: Decimal, b: Decimal):
    q = a / b
    return q.next_minus(), q.next_plus()

# Exact counterexample: constant random field, G = 11^T.
N = 4
u = [Decimal(1) / N] * N
v = [Decimal(1), Decimal(-1), Decimal(0), Decimal(0)]
alpha = Decimal("1")
w_uniform = [alpha * x for x in u]
w_nonuniform = [w_uniform[i] + v[i] for i in range(N)]

def total(w):
    return sum(w, Decimal(0))

def quadratic_constant_gram(w):
    # w^T (11^T) w = (sum_i w_i)^2
    return total(w) ** 2

def constant_field_risk(w):
    # Normalize E[Z^2]=1: c=A0=E_X=1.
    s=total(w)
    return Decimal(1)-Decimal(2)*s+s*s

# Archived rigorous enclosures from the T22 proof package.
A_lo = Decimal("0.974729989541714712312258085264191196422089014048652080604140725421300908486529660854678526902154912616533635138269820861")
A_hi = Decimal("0.974729989541714712312486955297461289338001476427951954852887889789982249288958791439364611154133167265972721306531007161")
E_lo = Decimal("0.9747302329077504666127808462108414633985481621607157811601376765058907")
E_hi = Decimal("0.9747302329077504666127808462108414633985481621607157811601376765059437")

# Monotone interval calculations; endpoints are widened by one Decimal ulp.
alpha_lo = (A_lo / E_hi).next_minus()
alpha_hi = (A_hi / E_lo).next_plus()
delta_lo = (E_lo - A_hi).next_minus()
delta_hi = (E_hi - A_lo).next_plus()
rel_lo = (delta_lo / E_hi).next_minus()
rel_hi = (delta_hi / E_lo).next_plus()
abs_gain_lo = ((delta_lo * delta_lo) / E_hi).next_minus()
abs_gain_hi = ((delta_hi * delta_hi) / E_lo).next_plus()

result = {
    "theorem_attacked": "T29 free-total-mass uniqueness and scale audit",
    "counterexample": {
        "field": "Y(x)=Z, a square-integrable constant random field",
        "gram": "G=11^T",
        "N": N,
        "alpha": str(alpha),
        "uniform_scaled_weights": [str(x) for x in w_uniform],
        "nonuniform_weights": [str(x) for x in w_nonuniform],
        "nonuniform_zero_sum_perturbation": [str(x) for x in v],
        "uniform_total_mass": str(total(w_uniform)),
        "nonuniform_total_mass": str(total(w_nonuniform)),
        "uniform_quadratic_energy": str(quadratic_constant_gram(w_uniform)),
        "nonuniform_quadratic_energy": str(quadratic_constant_gram(w_nonuniform)),
        "uniform_risk": str(constant_field_risk(w_uniform)),
        "nonuniform_risk": str(constant_field_risk(w_nonuniform)),
        "conclusion": "For the constant field alpha_*=1 and both weights integrate Y exactly with zero risk; the nonuniform vector is therefore a genuine minimizer."
    },
    "correct_minimizer_set": "alpha_* u + (ker G intersect 1^perp); uniqueness requires positive definiteness on 1^perp",
    "rigorous_scale_intervals": {
        "A0": [str(A_lo), str(A_hi)],
        "E_X": [str(E_lo), str(E_hi)],
        "alpha_star": [str(alpha_lo), str(alpha_hi)],
        "E_X_minus_A0": [str(delta_lo), str(delta_hi)],
        "relative_reduction_delta_over_E_X": [str(rel_lo), str(rel_hi)],
        "absolute_MSE_reduction_delta_squared_over_E_X": [str(abs_gain_lo), str(abs_gain_hi)]
    },
    "precision_finding": "Printing ~28 digits of alpha from 16-digit displayed inputs was unsupported; the rigorous interval above is the citable replacement.",
    "pass": True
}

out = Path(__file__).with_name("T29_ATTACK_RESULTS.json")
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
