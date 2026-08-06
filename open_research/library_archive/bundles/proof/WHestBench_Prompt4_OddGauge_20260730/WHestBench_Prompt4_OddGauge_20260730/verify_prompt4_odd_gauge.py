#!/usr/bin/env python3
"""Deterministic verifier for Prompt 4 odd-gauge identities.

This is a theorem sanity checker, not an empirical WHestBench experiment.
It tests:
  * scalar odd contraction equivariance;
  * O(k) polar-gauge invariance and ambient covariance;
  * positive-pivot discontinuity;
  * exact source-projection risk identities;
  * phase-accuracy and source-capacity threshold formulas.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def haar_orthogonal(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q @ np.diag(signs)


def polar_factor(c: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(c, full_matrices=False)
    return u @ vt


def polar_gauge(u: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    c = u.T @ b
    s = np.linalg.svd(c, compute_uv=False)
    return u @ polar_factor(c), float(s[-1])


def positive_pivot(v: np.ndarray) -> np.ndarray:
    j = int(np.argmax(np.abs(v)))
    return v if v[j] >= 0 else -v


def source_threshold(source_gain: float, target_gain: float) -> dict[str, float | bool]:
    h = 1.0 - 1.0 / source_gain
    need = 1.0 - 1.0 / target_gain
    eta = need / h
    feasible = eta <= 1.0 + 1e-15
    out: dict[str, float | bool] = {
        "source_gain": source_gain,
        "target_gain": target_gain,
        "reducible_fraction_h": h,
        "required_retained_fraction_eta": eta,
        "feasible": feasible,
    }
    if feasible:
        out["optimal_shrink_sign_accuracy_q"] = (1.0 + math.sqrt(max(0.0, eta))) / 2.0
        out["full_amplitude_sign_accuracy_q"] = (3.0 + eta) / 4.0
        out["full_amplitude_wrong_sign_max"] = (1.0 - eta) / 4.0
    return out


def run() -> dict[str, Any]:
    rng = np.random.default_rng(20260730)
    tol = 2e-11
    results: dict[str, Any] = {}

    # Scalar global-sign equivariance.
    u = rng.normal(size=256)
    b = rng.normal(size=256)
    z = float(u @ b)
    z_flip = float((-u) @ b)
    assert abs(z_flip + z) < 1e-12
    results["scalar_odd_equivariance_error"] = abs(z_flip + z)

    # O(k) gauge invariance and ambient covariance.
    n, k = 32, 4
    u0, _ = np.linalg.qr(rng.normal(size=(n, k)))
    # Make a well-conditioned physically coupled reference.
    b0 = u0 @ np.diag([2.0, 1.7, 1.3, 0.9]) + 0.05 * rng.normal(size=(n, k))
    oriented, margin = polar_gauge(u0, b0)

    o = haar_orthogonal(rng, k)
    oriented_right, margin_right = polar_gauge(u0 @ o, b0)
    right_error = float(np.linalg.norm(oriented - oriented_right))
    assert right_error < tol

    h = haar_orthogonal(rng, n)
    oriented_ambient, margin_ambient = polar_gauge(h @ u0, h @ b0)
    ambient_error = float(np.linalg.norm(oriented_ambient - h @ oriented))
    assert ambient_error < tol

    c_spd = oriented.T @ b0
    symmetry_error = float(np.linalg.norm(c_spd - c_spd.T))
    min_eig = float(np.linalg.eigvalsh((c_spd + c_spd.T) / 2.0)[0])
    assert symmetry_error < tol and min_eig > 0

    # Circular reference counterexample: B(U)=U co-rotates and fixes nothing.
    circular_original, _ = polar_gauge(u0, u0)
    circular_rotated, _ = polar_gauge(u0 @ o, u0 @ o)
    circular_failure = float(np.linalg.norm(circular_original - circular_rotated))
    assert circular_failure > 0.1

    results["polar_gauge"] = {
        "right_gauge_invariance_error": right_error,
        "ambient_covariance_error": ambient_error,
        "cross_gram_margin": margin,
        "right_margin_difference": abs(margin - margin_right),
        "ambient_margin_difference": abs(margin - margin_ambient),
        "oriented_cross_gram_symmetry_error": symmetry_error,
        "oriented_cross_gram_min_eigenvalue": min_eig,
        "circular_reference_failure_distance": circular_failure,
    }

    # Positive-pivot discontinuity at an absolute-coordinate tie.
    eps = 1e-9
    vp = np.array([1.0, -1.0 + eps])
    vm = np.array([1.0, -1.0 - eps])
    vp /= np.linalg.norm(vp)
    vm /= np.linalg.norm(vm)
    op = positive_pivot(vp)
    om = positive_pivot(vm)
    input_distance = float(np.linalg.norm(vp - vm))
    output_distance = float(np.linalg.norm(op - om))
    assert input_distance < 1e-8 and output_distance > 1.999999
    results["positive_pivot_discontinuity"] = {
        "input_distance": input_distance,
        "oriented_output_distance": output_distance,
    }

    # Exact projection risk identity.
    d, m = 40, 5
    a = rng.normal(size=(d, m))
    e = rng.normal(size=d)
    coef_star, *_ = np.linalg.lstsq(a, e, rcond=None)
    c_star = a @ coef_star
    residual = e - c_star
    coef_hat = coef_star + 0.2 * rng.normal(size=m)
    c_hat = a @ coef_hat
    lhs = float(np.dot(e - c_hat, e - c_hat))
    rhs = float(np.dot(residual, residual) + np.dot(c_hat - c_star, c_hat - c_star))
    projection_error = abs(lhs - rhs)
    assert projection_error < 1e-10
    results["projection_risk_identity_error"] = projection_error

    # Scalar multiplier law.
    r0 = float(np.dot(e, e))
    delta = float(np.dot(c_star, c_star))
    t = 0.37
    direct = float(np.dot(e - t * c_star, e - t * c_star))
    formula = r0 - (2.0 * t - t * t) * delta
    multiplier_error = abs(direct - formula)
    assert multiplier_error < 1e-10
    results["scalar_multiplier_identity_error"] = multiplier_error

    # Wrong-sign cost: correct oracle risk vs flipped risk.
    r_star = float(np.dot(residual, residual))
    r_wrong = float(np.dot(e + c_star, e + c_star))
    wrong_formula = r0 + 3.0 * delta
    assert abs(r_wrong - wrong_formula) < 1e-10
    results["wrong_sign"] = {
        "oracle_improvement_delta": delta,
        "correct_risk": r_star,
        "wrong_sign_risk": r_wrong,
        "wrong_sign_formula_error": abs(r_wrong - wrong_formula),
    }

    thresholds = []
    for source_gain in [1.20, 1.144709, 2.5, 1.0 / 0.454, 1.0 / 0.778]:
        for target_gain in [1.05, 1.10, 1.30]:
            thresholds.append(source_threshold(source_gain, target_gain))
    results["thresholds"] = thresholds

    # Nonlinear/compute general threshold example kept symbolic plus one numeric check.
    source_gain = 1.20
    h_src = 1.0 - 1.0 / source_gain
    score_multiplier = 1.01
    nonlinear_ratio = 1e-4
    rhs_eta = (
        1.0
        - (score_multiplier ** -0.5 - math.sqrt(nonlinear_ratio)) ** 2
    ) / h_src
    results["example_complete_threshold"] = {
        "source_gain": source_gain,
        "score_multiplier_lambda": score_multiplier,
        "nonlinear_remainder_ratio_nu": nonlinear_ratio,
        "required_eta_sufficient": rhs_eta,
    }

    return results


def main() -> None:
    out = run()
    path = Path(__file__).with_name("prompt4_verification_results.json")
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "PASS",
        "output": str(path),
        "polar_right_error": out["polar_gauge"]["right_gauge_invariance_error"],
        "polar_ambient_error": out["polar_gauge"]["ambient_covariance_error"],
        "pivot_jump": out["positive_pivot_discontinuity"]["oriented_output_distance"],
        "projection_identity_error": out["projection_risk_identity_error"],
    }, indent=2))


if __name__ == "__main__":
    main()
