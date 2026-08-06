#!/usr/bin/env python3
"""WHestBench Agent 6 score-economics engine.

The engine evaluates source-capacity, coefficient-error, and compute frontiers in
the physical source Gram metric. It supports:
  * biased and correlated contraction errors;
  * non-orthogonal / rank-deficient sources;
  * deterministic or capped scalable estimator cost;
  * independent GLS design arms with vector-valued shared samples;
  * integer sample allocations with setup costs;
  * empirical random-cost/risk aggregation and grouped bootstrap.

Input is JSON. See agent6_candidate_inputs.json for a complete example.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover - deterministic fallback below
    minimize = None

EPS = 1e-12


def as_array(x: Any, *, ndim: int | None = None) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"expected {ndim} dimensions, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("array contains non-finite values")
    return a


def psd_pinv(a: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """Symmetric Moore-Penrose inverse with negative numerical modes clipped."""
    a = 0.5 * (a + a.T)
    vals, vecs = np.linalg.eigh(a)
    scale = max(float(np.max(np.abs(vals))), 1.0)
    keep = vals > rtol * scale
    if np.any(vals < -rtol * scale):
        raise ValueError(f"matrix is not PSD; minimum eigenvalue={vals.min():.6g}")
    if not np.any(keep):
        return np.zeros_like(a)
    return (vecs[:, keep] / vals[keep]) @ vecs[:, keep].T


def source_risk_ratio(
    baseline_risk: float,
    gram: np.ndarray,
    contraction: np.ndarray,
) -> float:
    """Oracle residual ratio for a frozen linear source."""
    if baseline_risk <= 0:
        raise ValueError("baseline_risk must be positive")
    ginv = psd_pinv(gram)
    captured = float(contraction.T @ ginv @ contraction)
    return max(0.0, (baseline_risk - captured) / baseline_risk)


def contraction_error_penalty(
    gram: np.ndarray,
    bias: np.ndarray | None,
    covariance: np.ndarray | None,
    baseline_risk: float = 1.0,
) -> dict[str, float]:
    """Exact linear-source penalty for b-hat=b+bias+noise and theta=G^+ b-hat."""
    if baseline_risk <= 0:
        raise ValueError("baseline_risk must be positive")
    ginv = psd_pinv(gram)
    k = gram.shape[0]
    mu = np.zeros(k) if bias is None else as_array(bias, ndim=1)
    sigma = np.zeros((k, k)) if covariance is None else as_array(covariance, ndim=2)
    if mu.shape != (k,) or sigma.shape != (k, k):
        raise ValueError("bias/covariance dimensions do not match Gram matrix")
    sigma = 0.5 * (sigma + sigma.T)
    bias_pen = float(mu.T @ ginv @ mu) / baseline_risk
    var_pen = float(np.trace(ginv @ sigma)) / baseline_risk
    return {
        "bias_penalty_ratio": bias_pen,
        "variance_penalty_ratio": var_pen,
        "total_penalty_ratio": bias_pen + var_pen,
    }


def score_ratio(cost_ratio: float, risk_ratio: float) -> float:
    if cost_ratio < 0 or risk_ratio < 0:
        raise ValueError("cost_ratio and risk_ratio must be nonnegative")
    return cost_ratio * risk_ratio


def physical_penalty_allowance(target: float, fixed_cost: float, source_ratio: float) -> float:
    """Maximum added normalized physical MSE at zero further compute."""
    return target / fixed_cost - source_ratio


def scalable_score(fixed_cost: float, source_plus_fixed_risk: float, kappa: float, x: float) -> float:
    """(c0+x)(a+kappa/x), where x is added cost fraction."""
    if x <= 0:
        return math.inf if kappa > 0 else fixed_cost * source_plus_fixed_risk
    return (fixed_cost + x) * (source_plus_fixed_risk + kappa / x)


def optimal_scalable_frontier(
    target: float,
    fixed_cost: float,
    source_ratio: float,
    kappa: float | None = None,
    fixed_penalty: float = 0.0,
    max_total_cost: float | None = None,
) -> dict[str, Any]:
    """Exact scalar source-noise-compute frontier, optionally cost-capped."""
    a = source_ratio + fixed_penalty
    if min(target, fixed_cost, a) <= 0:
        raise ValueError("target, fixed_cost, and source+fixed risk must be positive")
    xmax = math.inf if max_total_cost is None else max(0.0, max_total_cost - fixed_cost)
    oracle_adjusted = fixed_cost * a
    x_boundary_unclipped = math.sqrt(target * fixed_cost / a) - fixed_cost
    x_for_kappa_limit = max(0.0, min(x_boundary_unclipped, xmax))
    if x_for_kappa_limit <= 0:
        kappa_max = 0.0
    else:
        kappa_max = x_for_kappa_limit * (target / (fixed_cost + x_for_kappa_limit) - a)
        kappa_max = max(0.0, kappa_max)

    out: dict[str, Any] = {
        "source_plus_fixed_risk_ratio": a,
        "oracle_adjusted_ratio": oracle_adjusted,
        "zero_noise_pass": oracle_adjusted < target,
        "physical_penalty_allowance_at_fixed_cost": target / fixed_cost - a,
        "max_added_cost_ratio": None if math.isinf(xmax) else xmax,
        "boundary_optimal_added_cost_unclipped": x_boundary_unclipped,
        "kappa_max": kappa_max,
    }
    if kappa is not None:
        if kappa < 0:
            raise ValueError("kappa must be nonnegative")
        x_unclipped = math.sqrt(fixed_cost * kappa / a) if kappa > 0 else 0.0
        x_opt = min(x_unclipped, xmax)
        if kappa > 0 and x_opt <= 0:
            j = math.inf
        else:
            j = scalable_score(fixed_cost, a, kappa, x_opt)
        out.update({
            "kappa": kappa,
            "optimal_added_cost_unclipped": x_unclipped,
            "optimal_added_cost": x_opt,
            "optimal_total_cost": fixed_cost + x_opt,
            "optimal_adjusted_ratio": j,
            "pass": j < target,
            "budget_binding": bool(not math.isinf(xmax) and x_unclipped > xmax + 1e-12),
        })
    return out


def grouped_bootstrap(
    rows: list[dict[str, Any]],
    group_key: str,
    value_key: str,
    weight_key: str | None = None,
    reps: int = 20000,
    seed: int = 20260730,
) -> dict[str, float]:
    """Grouped nonparametric bootstrap for a weighted mean."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[group_key]), []).append(row)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("at least two groups are required")

    def stat(selected: Iterable[str]) -> float:
        num = den = 0.0
        for g in selected:
            for row in groups[g]:
                w = 1.0 if weight_key is None else float(row[weight_key])
                num += w * float(row[value_key])
                den += w
        return num / den

    rng = np.random.default_rng(seed)
    vals = np.empty(reps)
    for i in range(reps):
        sample = rng.choice(keys, size=len(keys), replace=True)
        vals[i] = stat(sample)
    point = stat(keys)
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return {"point": float(point), "low95": float(lo), "high95": float(hi)}


@dataclass(frozen=True)
class DesignArm:
    name: str
    h: np.ndarray
    covariance: np.ndarray
    sample_cost: float
    setup_cost: float
    min_samples: int
    max_samples: int


def parse_arm(d: dict[str, Any], k: int) -> DesignArm:
    h = as_array(d.get("H", np.eye(k)), ndim=2)
    v = as_array(d["covariance"], ndim=2)
    if h.shape[1] != k or v.shape != (h.shape[0], h.shape[0]):
        raise ValueError(f"arm {d.get('name')} dimensions are inconsistent")
    return DesignArm(
        name=str(d["name"]),
        h=h,
        covariance=v,
        sample_cost=float(d["sample_cost_ratio"]),
        setup_cost=float(d.get("setup_cost_ratio", 0.0)),
        min_samples=int(d.get("min_samples", 0)),
        max_samples=int(d.get("max_samples", 10000)),
    )


def gls_covariance(arms: list[DesignArm], counts: np.ndarray, k: int) -> np.ndarray:
    info = np.zeros((k, k))
    for arm, n in zip(arms, counts):
        if n <= 0:
            continue
        vinv = psd_pinv(arm.covariance)
        info += float(n) * arm.h.T @ vinv @ arm.h
    return psd_pinv(info)


def allocation_cost(fixed_cost: float, arms: list[DesignArm], counts: np.ndarray) -> float:
    total = fixed_cost
    for arm, n in zip(arms, counts):
        if n > 0:
            total += arm.setup_cost + float(n) * arm.sample_cost
    return total


def evaluate_allocation(
    source_ratio: float,
    fixed_cost: float,
    gram: np.ndarray,
    arms: list[DesignArm],
    counts: np.ndarray,
    baseline_risk: float,
    fixed_bias: np.ndarray | None = None,
    replay_penalty: float = 0.0,
) -> dict[str, float]:
    k = gram.shape[0]
    cov = gls_covariance(arms, counts, k)
    penalties = contraction_error_penalty(gram, fixed_bias, cov, baseline_risk)
    risk = source_ratio + replay_penalty + penalties["total_penalty_ratio"]
    cost = allocation_cost(fixed_cost, arms, counts)
    return {
        **penalties,
        "risk_ratio": risk,
        "cost_ratio": cost,
        "adjusted_ratio": cost * risk,
    }


def optimize_integer_design(d: dict[str, Any], target: float) -> dict[str, Any]:
    gram = as_array(d["gram"], ndim=2)
    if gram.shape[0] != gram.shape[1]:
        raise ValueError("Gram matrix must be square")
    k = gram.shape[0]
    arms = [parse_arm(x, k) for x in d["arms"]]
    source_ratio = float(d["source_ratio"])
    fixed_cost = float(d["fixed_cost_ratio"])
    baseline_risk = float(d.get("baseline_risk", 1.0))
    max_total = float(d.get("max_total_cost_ratio", math.inf))
    fixed_bias = None if "fixed_bias" not in d else as_array(d["fixed_bias"], ndim=1)
    replay_penalty = float(d.get("replay_penalty_ratio", 0.0))

    def objective(x: np.ndarray) -> float:
        x = np.maximum(x, 0.0)
        counts = x
        cost = fixed_cost + sum(
            (arm.setup_cost if n > EPS else 0.0) + n * arm.sample_cost
            for arm, n in zip(arms, counts)
        )
        if cost > max_total + 1e-10:
            return 1e6 + 1e3 * (cost - max_total)
        cov = gls_covariance(arms, counts, k)
        pen = contraction_error_penalty(gram, fixed_bias, cov, baseline_risk)["total_penalty_ratio"]
        return cost * (source_ratio + replay_penalty + pen)

    lower = np.array([a.min_samples for a in arms], dtype=float)
    upper = np.array([a.max_samples for a in arms], dtype=float)
    start = np.maximum(lower, 1.0)
    # Keep the start inside the cost cap.
    while allocation_cost(fixed_cost, arms, start) > max_total and np.any(start > lower):
        j = int(np.argmax(start - lower))
        start[j] = max(lower[j], math.floor(start[j] / 2))

    cont = start.copy()
    if minimize is not None:
        constraints = [{
            "type": "ineq",
            "fun": lambda x: max_total - (
                fixed_cost + sum(
                    (arm.setup_cost if n > EPS else 0.0) + n * arm.sample_cost
                    for arm, n in zip(arms, x)
                )
            ),
        }]
        res = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=list(zip(lower, upper)),
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        if res.success and np.all(np.isfinite(res.x)):
            cont = res.x

    # Integer neighborhood plus a deterministic marginal-improvement search.
    options = []
    for value, arm in zip(cont, arms):
        cand = {
            int(max(arm.min_samples, min(arm.max_samples, math.floor(value)))),
            int(max(arm.min_samples, min(arm.max_samples, math.ceil(value)))),
            int(max(arm.min_samples, min(arm.max_samples, round(value)))),
        }
        options.append(sorted(cand))
    combos = list(itertools.product(*options))
    best_counts = np.array([a.min_samples for a in arms], dtype=int)
    best = math.inf
    for combo in combos:
        counts = np.asarray(combo, dtype=int)
        if allocation_cost(fixed_cost, arms, counts) <= max_total + 1e-12:
            val = objective(counts.astype(float))
            if val < best:
                best, best_counts = val, counts

    # Greedy additions in chunks, then unit local search.
    counts = best_counts.copy()
    chunk_sizes = [1024, 256, 64, 16, 4, 1]
    for chunk in chunk_sizes:
        improved = True
        while improved:
            improved = False
            current = objective(counts.astype(float))
            best_trial = current
            best_j = None
            for j, arm in enumerate(arms):
                if counts[j] + chunk > arm.max_samples:
                    continue
                trial = counts.copy(); trial[j] += chunk
                if allocation_cost(fixed_cost, arms, trial) > max_total + 1e-12:
                    continue
                val = objective(trial.astype(float))
                if val < best_trial - 1e-14:
                    best_trial, best_j = val, j
            if best_j is not None:
                counts[best_j] += chunk
                improved = True
    best_counts = counts
    metrics = evaluate_allocation(
        source_ratio, fixed_cost, gram, arms, best_counts, baseline_risk,
        fixed_bias=fixed_bias, replay_penalty=replay_penalty,
    )
    return {
        "name": str(d.get("name", "general_design")),
        "counts": {arm.name: int(n) for arm, n in zip(arms, best_counts)},
        **metrics,
        "target_ratio": target,
        "pass": metrics["adjusted_ratio"] < target,
        "dominant_bottleneck": (
            "source capacity" if fixed_cost * source_ratio >= target else
            "bias" if metrics["bias_penalty_ratio"] >= metrics["variance_penalty_ratio"] and metrics["bias_penalty_ratio"] > 0 else
            "variance / information" if metrics["variance_penalty_ratio"] > 0 else
            "compute"
        ),
    }


def candidate_result(c: dict[str, Any], global_cfg: dict[str, Any]) -> dict[str, Any]:
    target = float(global_cfg["target_ratio"])
    baseline_nodes = float(global_cfg.get("baseline_nodes", 0))
    direct_sample_cost = c.get(
        "direct_sample_cost_ratio", global_cfg.get("direct_sample_cost_ratio")
    )
    max_total = c.get("max_total_cost_ratio", global_cfg.get("max_total_cost_ratio"))
    frontier = optimal_scalable_frontier(
        target=target,
        fixed_cost=float(c["fixed_cost_ratio"]),
        source_ratio=float(c["source_ratio"]),
        fixed_penalty=float(c.get("fixed_penalty_ratio", 0.0)),
        kappa=c.get("kappa"),
        max_total_cost=None if max_total is None else float(max_total),
    )
    out = {"name": c["name"], **frontier}
    if direct_sample_cost is not None and float(direct_sample_cost) > 0:
        out["direct_sample_cost_ratio"] = float(direct_sample_cost)
        out["max_direct_shared_projected_variance_over_R0"] = (
            frontier["kappa_max"] / float(direct_sample_cost)
        )
    elif baseline_nodes > 0:
        # Backward-compatible equal-row-cost convention: one sample costs 1/N.
        out["max_direct_shared_projected_variance_over_R0"] = baseline_nodes * frontier["kappa_max"]
    for key in (
        "dimension", "evidence", "bootstrap_low", "bootstrap_high",
        "worst_network_ratio", "worst_case_ratio", "notes",
    ):
        if key in c:
            out[key] = c[key]
    if "bootstrap_high" in c:
        out["bootstrap_high_adjusted_oracle"] = float(c["bootstrap_high"]) * float(c["fixed_cost_ratio"])
    if "worst_network_ratio" in c:
        out["worst_network_adjusted_oracle"] = float(c["worst_network_ratio"]) * float(c["fixed_cost_ratio"])
    return out


def aggregate_random_rows(spec: dict[str, Any]) -> dict[str, Any]:
    rows = spec["rows"]
    costs = np.array([float(r["cost_ratio"]) for r in rows])
    risks = np.array([float(r["risk_ratio"]) for r in rows])
    weights = np.array([float(r.get("weight", 1.0)) for r in rows])
    weights /= weights.sum()
    mean_product = float(np.sum(weights * costs * risks))
    product_means = float(np.sum(weights * costs) * np.sum(weights * risks))
    return {
        "name": spec.get("name", "random_compute_rows"),
        "mean_of_products": mean_product,
        "product_of_means": product_means,
        "cost_risk_covariance": mean_product - product_means,
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    target = float(config["target_ratio"])
    results: dict[str, Any] = {
        "schema_version": "agent6-score-economics-v1",
        "target_ratio": target,
        "baseline_nodes": config.get("baseline_nodes"),
        "max_total_cost_ratio": config.get("max_total_cost_ratio"),
        "candidates": [candidate_result(c, config) for c in config.get("candidates", [])],
        "general_designs": [optimize_integer_design(d, target) for d in config.get("general_designs", [])],
        "random_compute_aggregations": [aggregate_random_rows(x) for x in config.get("random_compute_rows", [])],
    }
    passing = [c for c in results["candidates"] if c["zero_noise_pass"]]
    if passing:
        results["best_threshold_candidate"] = max(passing, key=lambda x: x["kappa_max"])["name"]
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    result = run(config)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
