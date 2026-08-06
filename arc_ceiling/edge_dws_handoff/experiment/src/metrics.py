from __future__ import annotations

from typing import Any
import numpy as np


def replay_error(baseline_error: np.ndarray, jacobian: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    return baseline_error + np.einsum("nod,nd->no", jacobian, coeffs)


def mse_rows(error: np.ndarray) -> np.ndarray:
    return np.mean(np.square(error), axis=1)


def aggregate_by_base(values: np.ndarray, base_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.asarray(sorted({str(v) for v in base_ids}), dtype=object)
    out = np.array([values[np.asarray([str(v) == u for v in base_ids])].mean() for u in unique])
    return unique, out


def grouped_bootstrap_gain(
    base_mse: np.ndarray,
    cand_mse: np.ndarray,
    base_ids: np.ndarray,
    seed: int = 9017,
    n_boot: int = 10000,
) -> tuple[float, list[float]]:
    ids, b = aggregate_by_base(base_mse, base_ids)
    _, c = aggregate_by_base(cand_mse, base_ids)
    point = float(b.sum() / max(c.sum(), 1e-300))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.integers(0, len(ids), len(ids))
        draws[i] = b[ix].sum() / max(c[ix].sum(), 1e-300)
    return point, [float(v) for v in np.quantile(draws, [0.025, 0.5, 0.975])]


def expected_calibration_error(prob: np.ndarray, label: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if mask.any():
            ece += mask.mean() * abs(prob[mask].mean() - label[mask].mean())
    return float(ece)


def adjusted_score(raw_mse: float, effective_compute_b: float, budget_b: float = 272.0) -> float:
    return float(raw_mse * max(0.1, effective_compute_b / budget_b))


def evaluate(
    arrays: dict[str, np.ndarray],
    idx: np.ndarray,
    coeffs: np.ndarray,
    confidence: np.ndarray,
    baseline_compute_b: float,
    candidate_compute_b: float,
    bootstrap_seed: int = 9017,
) -> dict[str, Any]:
    e0 = arrays["baseline_error"][idx]
    j = arrays["replay_jacobian"][idx]
    err = replay_error(e0, j, coeffs)
    base = mse_rows(e0)
    cand = mse_rows(err)
    base_ids = arrays["base_network_id"][idx]
    gain, ci = grouped_bootstrap_gain(base, cand, base_ids, seed=bootstrap_seed)
    ids, bb = aggregate_by_base(base, base_ids)
    _, cc = aggregate_by_base(cand, base_ids)
    ratios = cc / np.maximum(bb, 1e-300)
    improvement = (cand < base).astype(float)
    oracle_resid = arrays["target_coeffs"][idx] - arrays["anchor_coeffs"][idx]
    pred_resid = coeffs - arrays["anchor_coeffs"][idx]
    den = np.linalg.norm(oracle_resid.ravel()) * np.linalg.norm(pred_resid.ravel())
    cosine = float(np.dot(oracle_resid.ravel(), pred_resid.ravel()) / den) if den else 0.0
    brier = float(np.mean((confidence - improvement) ** 2))
    ece = expected_calibration_error(confidence, improvement)
    base_raw = float(base.mean())
    cand_raw = float(cand.mean())
    base_adj = adjusted_score(base_raw, baseline_compute_b)
    cand_adj = adjusted_score(cand_raw, candidate_compute_b)
    adjusted_gain = base_adj / max(cand_adj, 1e-300)
    adjusted_ci = [x * baseline_compute_b / candidate_compute_b for x in ci]
    return {
        "examples": int(len(idx)),
        "base_networks": int(len(ids)),
        "baseline_raw_mse": base_raw,
        "candidate_raw_mse": cand_raw,
        "raw_gain_baseline_over_candidate": gain,
        "raw_gain_group_bootstrap_ci95": ci,
        "adjusted_gain_baseline_over_candidate": float(adjusted_gain),
        "adjusted_gain_group_bootstrap_ci95": [float(x) for x in adjusted_ci],
        "wins_base_networks": int(np.sum(cc < bb)),
        "median_candidate_over_baseline": float(np.median(ratios)),
        "worst_candidate_over_baseline": float(np.max(ratios)),
        "correction_cosine": cosine,
        "confidence_brier": brier,
        "confidence_ece10": ece,
        "baseline_effective_compute_B": float(baseline_compute_b),
        "candidate_effective_compute_B": float(candidate_compute_b),
    }
