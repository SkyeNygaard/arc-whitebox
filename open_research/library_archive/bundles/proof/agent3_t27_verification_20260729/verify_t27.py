#!/usr/bin/env python3
"""Independent numerical and algebraic checks for WHestBench theorem T27.

This script does not use the original T27 implementation.  It reconstructs the
normalized depth-32 ReLU kernel, the three Kerdock line-pair constants, the
spherical kernel mean, the risk decomposition, the fixed-support optimizer,
and the concentrated-basis budget optimizer.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.special import gammaln, roots_jacobi

D = 256
DEPTH = 32
B = 129
M = 256
TOL = 1e-10


def relu_kernel(t: np.ndarray | float) -> np.ndarray:
    x = np.asarray(t, dtype=np.float64)
    x = np.clip(x, -1.0, 1.0)
    return (
        np.sqrt(np.maximum(1.0 - x * x, 0.0))
        + x * (np.pi - np.arccos(x))
    ) / np.pi


def deep_kernel(t: np.ndarray | float, depth: int = DEPTH) -> np.ndarray:
    x = np.asarray(t, dtype=np.float64)
    for _ in range(depth):
        x = relu_kernel(x)
    return x


def spherical_kernel_mean(d: int = D, quadrature_order: int = 512) -> float:
    alpha = (d - 3.0) / 2.0
    x, w = roots_jacobi(quadrature_order, alpha, alpha)
    return float(np.dot(w, deep_kernel(x)) / np.sum(w))


def constants() -> dict[str, float]:
    k1 = float(deep_kernel(1.0))
    km1 = float(deep_kernel(-1.0))
    k0 = float(deep_kernel(0.0))
    kp = float(deep_kernel(1.0 / 16.0))
    km = float(deep_kernel(-1.0 / 16.0))
    A = 0.5 * (k1 + km1)
    O = k0
    C = 0.5 * (kp + km)
    A0 = spherical_kernel_mean()
    return {
        "K(1)": k1,
        "K(-1)": km1,
        "K(0)": k0,
        "K(1/16)": kp,
        "K(-1/16)": km,
        "A": A,
        "O": O,
        "C": C,
        "A0": A0,
        "A-O": A - O,
        "O-C": O - C,
        "C-A0": C - A0,
    }


def risk_formula(weights: list[np.ndarray], c: dict[str, float]) -> float:
    s2 = sum(float(np.sum(w)) ** 2 for w in weights)
    w2 = sum(float(np.dot(w, w)) for w in weights)
    return c["C-A0"] + c["O-C"] * s2 + c["A-O"] * w2


def risk_direct(weights: list[np.ndarray], c: dict[str, float]) -> float:
    labels: list[tuple[int, int]] = []
    flat: list[float] = []
    for b, w in enumerate(weights):
        for i, value in enumerate(w):
            labels.append((b, i))
            flat.append(float(value))
    ww = np.asarray(flat)
    n = len(ww)
    G = np.empty((n, n), dtype=np.float64)
    for j, (bj, ij) in enumerate(labels):
        for k, (bk, ik) in enumerate(labels):
            if bj == bk and ij == ik:
                G[j, k] = c["A"]
            elif bj == bk:
                G[j, k] = c["O"]
            else:
                G[j, k] = c["C"]
    return float(ww @ G @ ww - c["A0"])


def c_of_r(r: int, c: dict[str, float]) -> float:
    if not 1 <= r <= M:
        raise ValueError(r)
    return c["O-C"] + c["A-O"] / r


def h_of_r(r: int | float, c: dict[str, float]) -> float:
    if r == 0:
        return 0.0
    return float(r / (c["A-O"] + c["O-C"] * r))


def fixed_support_optimum(counts: Iterable[int], c: dict[str, float]):
    counts = list(counts)
    active = [r for r in counts if r > 0]
    if not active:
        raise ValueError("empty support cannot have total mass one")
    inv_c = np.asarray([1.0 / c_of_r(r, c) for r in active])
    H = float(np.sum(inv_c))
    masses = inv_c / H
    weights: list[np.ndarray] = []
    j = 0
    for r in counts:
        if r == 0:
            weights.append(np.zeros(0))
        else:
            weights.append(np.full(r, masses[j] / r))
            j += 1
    risk = c["C-A0"] + 1.0 / H
    return weights, risk, masses


def concentrated_counts(P: int) -> list[int]:
    if not 1 <= P <= B * M:
        raise ValueError(P)
    q, s = divmod(P, M)
    counts = [M] * q
    if s:
        counts.append(s)
    counts.extend([0] * (B - len(counts)))
    return counts


def compositions(total: int, slots: int, cap: int):
    if slots == 1:
        if 0 <= total <= cap:
            yield (total,)
        return
    for x in range(min(cap, total) + 1):
        for rest in compositions(total - x, slots - 1, cap):
            yield (x,) + rest


def main() -> None:
    rng = np.random.default_rng(20260729)
    c = constants()

    # Sign and positivity checks.
    assert c["A-O"] > 0
    assert c["O-C"] < 0
    assert c_of_r(M, c) > 0
    assert all(c_of_r(r, c) > 0 for r in range(1, M + 1))

    # Independent direct Gram-matrix checks of the risk reduction.
    max_decomposition_error = 0.0
    for _ in range(500):
        nb = int(rng.integers(1, 9))
        counts = [int(rng.integers(1, 9)) for _ in range(nb)]
        raw = [rng.normal(size=r) for r in counts]
        total = sum(float(np.sum(w)) for w in raw)
        raw[0][0] += 1.0 - total
        weights = raw
        rf = risk_formula(weights, c)
        rd = risk_direct(weights, c)
        max_decomposition_error = max(max_decomposition_error, abs(rf - rd))
    assert max_decomposition_error < TOL

    # Fixed-support optimality against arbitrary signed perturbations.
    min_signed_gap = math.inf
    max_opt_formula_error = 0.0
    for _ in range(3000):
        nb = int(rng.integers(1, 20))
        counts = [int(rng.integers(0, M + 1)) for _ in range(nb)]
        if sum(counts) == 0:
            counts[0] = 1
        opt_w, opt_risk, _ = fixed_support_optimum(counts, c)
        max_opt_formula_error = max(
            max_opt_formula_error, abs(risk_formula(opt_w, c) - opt_risk)
        )
        trial: list[np.ndarray] = []
        for r in counts:
            trial.append(rng.normal(size=r) if r else np.zeros(0))
        total = sum(float(np.sum(w)) for w in trial)
        k = next(i for i, r in enumerate(counts) if r)
        trial[k][0] += 1.0 - total
        gap = risk_formula(trial, c) - opt_risk
        min_signed_gap = min(min_signed_gap, gap)
        assert gap >= -2e-11
    assert max_opt_formula_error < TOL

    # Exhaustive integer support-allocation verification in a smaller box,
    # using the actual T27 h(r).  This checks the discrete exchange conclusion.
    exhaustive = []
    slots, cap = 5, 8
    for P in range(1, slots * cap + 1):
        all_counts = list(compositions(P, slots, cap))
        values = np.asarray([sum(h_of_r(r, c) for r in x) for x in all_counts])
        best = float(values.max())
        q, s = divmod(P, cap)
        concentrated = tuple([cap] * q + ([s] if s else []) + [0] * (slots - q - (1 if s else 0)))
        conc_value = sum(h_of_r(r, c) for r in concentrated)
        assert abs(best - conc_value) < 1e-9
        exhaustive.append({"P": P, "best_H": best, "concentrated_H": conc_value})

    # Actual boundary budgets and random allocation challenges.
    boundaries = [1, 2, 255, 256, 257, 511, 512, 513, 33023, 33024]
    boundary_rows = []
    min_random_allocation_gap = math.inf
    for P in boundaries:
        counts = concentrated_counts(P)
        _, rr, masses = fixed_support_optimum(counts, c)
        q, s = divmod(P, M)
        boundary_rows.append(
            {
                "P_lines": P,
                "N_antipodal_points": 2 * P,
                "full_bases": q,
                "partial_lines": s,
                "risk": rr,
                "active_basis_masses": masses.tolist(),
            }
        )
        for _ in range(200):
            # Start with P singleton balls and distribute across 129 capped bins.
            trial_counts = np.zeros(B, dtype=int)
            remaining = P
            while remaining:
                available = np.flatnonzero(trial_counts < M)
                bidx = int(rng.choice(available))
                add = int(rng.integers(1, min(remaining, M - trial_counts[bidx]) + 1))
                trial_counts[bidx] += add
                remaining -= add
            H_trial = sum(h_of_r(int(r), c) for r in trial_counts)
            H_opt = sum(h_of_r(r, c) for r in counts)
            min_random_allocation_gap = min(min_random_allocation_gap, H_opt - H_trial)
            assert H_trial <= H_opt + 1e-8

    # An explicit zero-total signed cancellation is strictly harmful.
    z = np.array([1.0, -1.0])
    cancellation_penalty = c["A-O"] * float(z @ z)
    assert cancellation_penalty > 0

    # Outside-universe algebra check: a pair at |t|=1/2 has a fourth pair value.
    outside_pair_value = float(0.5 * (deep_kernel(0.5) + deep_kernel(-0.5)))
    assert min(abs(outside_pair_value - c[k]) for k in ("A", "O", "C")) > 1e-6

    # Explicit nonlinear finite-network counterexample to over-broad extrapolation.
    # For f_a(u)=ReLU(a.u), line averages along an ONB are |a_i|/2.
    # The nonlinear L2 reconstruction recovers the spherical mean exactly.
    a_vec = rng.normal(size=D)
    line_means = np.abs(a_vec) / 2.0
    E_abs_u1 = math.exp(
        gammaln(D / 2.0) - 0.5 * math.log(math.pi) - gammaln((D + 1.0) / 2.0)
    )
    kappa_d = E_abs_u1 / 2.0
    true_one_relu_mean = kappa_d * float(np.linalg.norm(a_vec))
    nonlinear_reconstruction = kappa_d * 2.0 * float(np.linalg.norm(line_means))
    nonlinear_error = abs(nonlinear_reconstruction - true_one_relu_mean)
    assert nonlinear_error < 1e-14

    result = {
        "verdict": "VERIFIED_AFTER_SPECIFIED_CORRECTIONS",
        "constants": c,
        "c_256": c_of_r(256, c),
        "denominator_at_256": c["A-O"] + 256 * c["O-C"],
        "h_derivative_sign": "positive on [0,256]",
        "h_second_derivative_sign": "positive on [0,256]",
        "max_risk_decomposition_abs_error": max_decomposition_error,
        "max_fixed_support_formula_abs_error": max_opt_formula_error,
        "minimum_random_signed_rule_minus_optimum": min_signed_gap,
        "minimum_optimal_H_minus_random_allocation_H": min_random_allocation_gap,
        "zero_total_signed_cancellation_penalty_example": cancellation_penalty,
        "outside_universe_even_kernel_at_t_half": outside_pair_value,
        "nonlinear_one_relu_reconstruction_abs_error": nonlinear_error,
        "boundaries": boundary_rows,
        "exhaustive_small_box": exhaustive,
        "corrections": [
            "Use 'line budget P' or '2P paired point evaluations', not unrestricted 'point budget'.",
            "Define h(0)=0 and require S_b=0 when r_b=0.",
            "State P in 1..33024; P=0 is infeasible and larger budgets are capped by the universe.",
            "Clarify that weights are total weights on symmetrized antipodal line evaluations.",
        ],
    }
    out = Path(__file__).with_name("verification_results.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps({"script_sha256": digest, **result}, indent=2))


if __name__ == "__main__":
    main()
