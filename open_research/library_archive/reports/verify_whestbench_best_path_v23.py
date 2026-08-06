#!/usr/bin/env python3
"""Independent numerical checks for WHestBench T70-T73."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np

OUT = Path("/mnt/data/WHestBench_Best_Path_v23_Verification.json")

def antipodal_sphere(rng: np.random.Generator, pairs: int, dim: int) -> np.ndarray:
    x = rng.normal(size=(pairs, dim))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return np.concatenate([x, -x], axis=0)

def forward(nodes: np.ndarray, weights: list[np.ndarray]):
    h = nodes
    hs = [h]
    zs = []
    for w in weights:
        z = h @ w.T
        h = np.maximum(z, 0.0)
        zs.append(z)
        hs.append(h)
    return hs, zs

def mean_diff(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return p.mean(axis=0) - q.mean(axis=0)

def verify() -> dict:
    rng = np.random.default_rng(20260730)
    dim = 7
    width = 9
    depth = 4
    weights = [
        rng.normal(scale=math.sqrt(2 / (dim if l == 0 else width)),
                   size=(width, dim if l == 0 else width))
        for l in range(depth)
    ]

    p_nodes = antipodal_sphere(rng, 1500, dim)
    q_nodes = antipodal_sphere(rng, 140, dim)
    p_h, p_z = forward(p_nodes, weights)
    q_h, q_z = forward(q_nodes, weights)

    u = rng.normal(size=width)
    delta_L = mean_diff(p_h[-1], q_h[-1])

    qvecs = []
    for ell in range(depth):
        qv = u.copy()
        for k in range(depth - 1, ell, -1):
            qv = 0.5 * weights[k].T @ qv
        qvecs.append(qv)

    phi_p = np.zeros(len(p_nodes))
    phi_q = np.zeros(len(q_nodes))
    band_p = []
    band_q = []
    bands = [(0, 1), (1, 3), (3, 4)]
    for ell in range(depth):
        phi_p += 0.5 * (np.abs(p_z[ell]) @ qvecs[ell])
        phi_q += 0.5 * (np.abs(q_z[ell]) @ qvecs[ell])
    for lo, hi in bands:
        bp = np.zeros(len(p_nodes))
        bq = np.zeros(len(q_nodes))
        for ell in range(lo, hi):
            bp += 0.5 * (np.abs(p_z[ell]) @ qvecs[ell])
            bq += 0.5 * (np.abs(q_z[ell]) @ qvecs[ell])
        band_p.append(bp.mean())
        band_q.append(bq.mean())

    lhs = float(u @ delta_L)
    rhs = float(phi_p.mean() - phi_q.mean())
    band_rhs = float(sum(np.array(band_p) - np.array(band_q)))

    linear = p_nodes @ (weights[0].T)
    # Pointwise linear term vector coefficient for u^T W_L...W_1 x / 2^L.
    coeff = u.copy()
    for k in range(depth - 1, -1, -1):
        coeff = weights[k].T @ coeff
    pointwise_rhs_p = p_h[-1] @ u - (p_nodes @ coeff) / (2 ** depth)
    pointwise_rhs_q = q_h[-1] @ u - (q_nodes @ coeff) / (2 ** depth)
    no_free_lunch_error = max(
        float(np.max(np.abs(phi_p - pointwise_rhs_p))),
        float(np.max(np.abs(phi_q - pointwise_rhs_q))),
    )

    # Source-space corollary.
    A = rng.normal(size=(width, 4))
    b_direct = A.T @ delta_L
    b_potential = []
    for j in range(A.shape[1]):
        a = A[:, j]
        qvs = []
        for ell in range(depth):
            qv = a.copy()
            for k in range(depth - 1, ell, -1):
                qv = 0.5 * weights[k].T @ qv
            qvs.append(qv)
        pp = sum(0.5 * (np.abs(p_z[ell]) @ qvs[ell]) for ell in range(depth))
        qq = sum(0.5 * (np.abs(q_z[ell]) @ qvs[ell]) for ell in range(depth))
        b_potential.append(pp.mean() - qq.mean())
    source_error = float(np.max(np.abs(b_direct - np.asarray(b_potential))))

    # T72.
    rstar = 0.12
    v = np.array([0.04, 0.09, 0.01, 0.025])
    gamma = np.array([2e-4, 5e-4, 1e-4, 3e-4])
    S = float(np.sum(np.sqrt(v * gamma)))
    nstar = np.sqrt(v / gamma) / math.sqrt(rstar)
    direct_obj = float((rstar + np.sum(v / nstar)) * (1 + np.sum(gamma * nstar)))
    closed_obj = float((math.sqrt(rstar) + S) ** 2)

    # T73.
    n = 1000
    h = rng.normal(size=(n, width))
    w_final = rng.normal(scale=math.sqrt(2 / width), size=(width, width))
    delta = rng.normal(scale=0.15, size=width)
    z = h @ w_final.T
    shifted = np.maximum(z + (w_final @ delta)[None, :], 0.0)
    base = np.maximum(z, 0.0)
    direct = (shifted - base).mean(axis=0)

    s = w_final @ delta
    g = np.empty(width)
    g_sorted = np.empty(width)
    for j in range(width):
        g[j] = np.mean(np.maximum(z[:, j] + s[j], 0.0) - np.maximum(z[:, j], 0.0))
        order = np.argsort(-z[:, j])
        thresholds = (-z[:, j])[order]
        zsort = z[:, j][order]
        prefix = np.concatenate([[0.0], np.cumsum(zsort)])
        k = int(np.searchsorted(thresholds, s[j], side="left"))
        active_sum = prefix[k]
        baseline_sum = np.maximum(z[:, j], 0.0).sum()
        g_sorted[j] = (k * s[j] + active_sum - baseline_sum) / n

    p = (z > 0).mean(axis=0)
    remainder_direct = direct - p * s
    crossing = np.where(
        z * (z + s[None, :]) < 0,
        np.abs(z + s[None, :]),
        0.0,
    ).mean(axis=0)

    return {
        "seed": 20260730,
        "T70_projection_error": abs(lhs - rhs),
        "T70_source_vector_max_error": source_error,
        "T71_band_sum_error": abs(lhs - band_rhs),
        "T71_pointwise_no_free_lunch_max_error": no_free_lunch_error,
        "T72_direct_optimum": direct_obj,
        "T72_closed_form": closed_obj,
        "T72_absolute_error": abs(direct_obj - closed_obj),
        "T72_optimal_sample_allocations": nstar.tolist(),
        "T73_direct_vs_formula_max_error": float(np.max(np.abs(direct - g))),
        "T73_sorted_evaluator_max_error": float(np.max(np.abs(direct - g_sorted))),
        "T73_crossing_remainder_max_error": float(np.max(np.abs(remainder_direct - crossing))),
        "all_checks_pass": bool(
            abs(lhs - rhs) < 1e-11
            and source_error < 1e-11
            and abs(lhs - band_rhs) < 1e-11
            and no_free_lunch_error < 1e-11
            and abs(direct_obj - closed_obj) < 1e-13
            and np.max(np.abs(direct - g)) < 1e-13
            and np.max(np.abs(direct - g_sorted)) < 1e-13
            and np.max(np.abs(remainder_direct - crossing)) < 1e-13
        ),
    }

if __name__ == "__main__":
    result = verify()
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
