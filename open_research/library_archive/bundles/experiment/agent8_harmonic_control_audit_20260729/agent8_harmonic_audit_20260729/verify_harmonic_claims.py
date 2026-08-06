#!/usr/bin/env python3
"""Independent numerical checks for the Agent-8 harmonic-control audit.

This script is not the proof. It checks two exact algebraic claims numerically:
1. A bias-free one-hidden-layer ReLU Stein ridge field has zero average on
   every antipodal orthonormal-basis block after exact Gaussian radialization.
2. The symmetrized spherical Poisson kernel has exact mean one and nonzero
   high even Gegenbauer coefficients, giving an analytically integrable,
   nonpolynomial counterexample to 'analytically integrable implies low degree'.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.special import roots_jacobi


def normalized_gegenbauer_values(t: np.ndarray, max_degree: int, lam: float) -> list[np.ndarray]:
    """P_l(t)=C_l^lam(t)/C_l^lam(1), using a stable normalized recurrence."""
    vals = [np.ones_like(t)]
    if max_degree == 0:
        return vals
    vals.append(t.copy())
    for n in range(1, max_degree):
        a = 2.0 * (n + lam) / (n + 2.0 * lam)
        b = n / (n + 2.0 * lam)
        vals.append(a * t * vals[-1] - b * vals[-2])
    return vals


def check_relu_stein_block(d: int = 256, hidden: int = 17, seed: int = 20260729) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    v = rng.standard_normal((d, hidden))
    a = rng.standard_normal((d, hidden))

    # Each block contains ±q_i on the unit sphere. Exact radialization to a
    # radius-sqrt(d) Gaussian shell changes x·phi to d*(u·a)ReLU(v·u).
    points = np.concatenate([q.T, -q.T], axis=0)
    proj = points @ v
    gates = proj > 0.0
    divergence = gates @ np.sum(a * v, axis=0)
    phi = np.maximum(proj, 0.0) @ a.T
    x_dot_phi = d * np.sum(points * phi, axis=1)
    control = divergence - x_dot_phi

    # Whole block average, and pairwise basis contributions.
    block_mean = float(np.mean(control))
    pair_means = 0.5 * (control[:d] + control[d:])

    # The two analytic terms should each average to 1/2 sum_j a_j·v_j.
    target = 0.5 * float(np.sum(a * v))
    return {
        "dimension": float(d),
        "hidden_units": float(hidden),
        "block_mean_abs": abs(block_mean),
        "max_pair_mean_abs": float(np.max(np.abs(pair_means))),
        "divergence_mean_error_abs": abs(float(np.mean(divergence)) - target),
        "x_dot_phi_mean_error_abs": abs(float(np.mean(x_dot_phi)) - target),
    }


def symmetrized_poisson_kernel(t: np.ndarray, d: int, r: float) -> np.ndarray:
    p_plus = (1.0 - r * r) / np.power(1.0 - 2.0 * r * t + r * r, d / 2.0)
    p_minus = (1.0 - r * r) / np.power(1.0 + 2.0 * r * t + r * r, d / 2.0)
    return 0.5 * (p_plus + p_minus)


def check_poisson_counterexample(d: int = 256, r: float = 0.12, quadrature_order: int = 1024) -> dict:
    # For U uniform on S^{d-1}, T=v·U has density proportional to
    # (1-t^2)^((d-3)/2). Gauss-Jacobi exactly matches this weight.
    alpha = (d - 3.0) / 2.0
    t, w = roots_jacobi(quadrature_order, alpha, alpha)
    w = w / np.sum(w)
    f = symmetrized_poisson_kernel(t, d=d, r=r)

    lam = (d - 2.0) / 2.0
    polys = normalized_gegenbauer_values(t, 12, lam)
    coeffs = {}
    moments = {}
    for ell, p in enumerate(polys):
        inner = float(np.sum(w * f * p))
        norm2 = float(np.sum(w * p * p))
        coeffs[str(ell)] = inner / norm2
        moments[str(ell)] = inner

    even_nonzero = {str(ell): coeffs[str(ell)] for ell in (6, 8, 10, 12)}
    odd_max = max(abs(coeffs[str(ell)]) for ell in (1, 3, 5, 7, 9, 11))
    return {
        "dimension": d,
        "r": r,
        "quadrature_order": quadrature_order,
        "sphere_mean": float(np.sum(w * f)),
        "sphere_mean_error_abs": abs(float(np.sum(w * f)) - 1.0),
        "gegenbauer_coefficients_0_to_12": coeffs,
        "high_even_coefficients": even_nonzero,
        "max_abs_odd_coefficient": odd_max,
        "analytic_harmonic_multipliers_even": {str(ell): r**ell for ell in (0, 2, 4, 6, 8, 10, 12)},
    }


def main() -> None:
    results = {
        "relu_stein_block_check": check_relu_stein_block(),
        "poisson_counterexample_check": check_poisson_counterexample(),
    }
    output = Path(__file__).with_name("verification_results.json")
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))

    assert results["relu_stein_block_check"]["block_mean_abs"] < 1e-11
    assert results["poisson_counterexample_check"]["sphere_mean_error_abs"] < 1e-10
    for value in results["poisson_counterexample_check"]["high_even_coefficients"].values():
        assert abs(value) > 1e-12


if __name__ == "__main__":
    main()
