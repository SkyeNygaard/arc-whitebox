#!/usr/bin/env python3
"""Direct signed lower-anchor estimation via reanchored structured pilots.

The estimator never forms an independent absolute Gaussian target and subtracts
Kerdock at the end.  Instead, two disjoint four-basis Kerdock pilots estimate,
at every ReLU, the local Gaussianization source around the already-computed
full Kerdock cloud.  A coupled mean/covariance defect is propagated to the
frozen target layer.  The two pilot estimates are averaged; disagreement is a
runtime confidence observable.

This is a development harness on architecture-matched synthetic networks.  It
uses all 129 frozen Kerdock bases and the frozen 128 sample-row radial-Hermite
control.  High-sample Sobol references are evaluation-only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frozen_reference_impl as fr

D = fr.D
TARGET = fr.TARGET
N_BASES = fr.N_BASES
ROWS_PER_BASIS = fr.ROWS_PER_BASIS
FOLDS = fr.FOLDS
RIDGE = fr.RIDGE
RADIAL_SCALE = D / (fr.chi_mean(D) ** 2)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

# Disjoint, evenly spread complete bases.  Each basis includes its antipodal
# 512-row block, so every pilot is exactly centered at the input.
PILOT_GROUPS = (
    (0, 32, 64, 96),
    (16, 48, 80, 112),
)


def pilot_rows(group: tuple[int, ...]) -> np.ndarray:
    return np.concatenate([
        np.arange(b * ROWS_PER_BASIS, (b + 1) * ROWS_PER_BASIS, dtype=np.int64)
        for b in group
    ])


def raw_moments(x: torch.Tensor, rows: np.ndarray | None, full_diag_only: bool = False):
    """Radially corrected mean/covariance of homogeneous degree-one rows."""
    if rows is not None:
        z = x[torch.from_numpy(rows)]
    else:
        z = x
    zd = z.double()
    mean = zd.mean(0).cpu().numpy()
    second_diag = (zd * zd).mean(0).cpu().numpy() * RADIAL_SCALE
    var = np.maximum(second_diag - mean * mean, 1e-18)
    if full_diag_only:
        return mean, var, None
    second = (zd.T @ zd / len(zd)).cpu().numpy() * RADIAL_SCALE
    cov = 0.5 * (second + second.T) - np.outer(mean, mean)
    np.fill_diagonal(cov, var)
    return mean, var, cov


def relu_gaussian(mu: np.ndarray, var: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sd = np.sqrt(np.maximum(var, 1e-18))
    t = mu / sd
    p = ndtr(t)
    phi = np.exp(-0.5 * t * t) * INV_SQRT_2PI
    mean = mu * p + sd * phi
    second = (mu * mu + var) * p + mu * sd * phi
    post_var = np.maximum(second - mean * mean, 1e-18)
    return mean, post_var, p


def update_defect(
    delta_mu: np.ndarray,
    delta_cov: np.ndarray,
    w: np.ndarray,
    full_pre_mean: np.ndarray,
    full_pre_var: np.ndarray,
    full_post_mean: np.ndarray,
    full_post_var: np.ndarray,
    pilot_pre_cov: np.ndarray,
    pilot_post_cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """One internally-cancelled source/transport update.

    The base is the observed Kerdock preactivation law.  The candidate branch is
    base plus the carried signed defect.  Their nonlinear difference and the
    observed pilot source are combined before leaving the layer.
    """
    dmu_pre = delta_mu @ w
    dcov_pre = w.T @ delta_cov @ w
    dcov_pre = 0.5 * (dcov_pre + dcov_pre.T)

    pred_mu_pre = full_pre_mean + dmu_pre
    pred_var_pre = np.maximum(full_pre_var + np.diag(dcov_pre), 1e-18)
    pred_mean, pred_var, gate = relu_gaussian(pred_mu_pre, pred_var_pre)

    pred_cov_pre = pilot_pre_cov + dcov_pre
    pred_cov = pred_cov_pre * np.outer(gate, gate)
    np.fill_diagonal(pred_cov, pred_var)

    next_mu = pred_mean - full_post_mean
    next_cov = pred_cov - pilot_post_cov
    # Pilot off-diagonal source, full-cloud diagonal source.
    np.fill_diagonal(next_cov, pred_var - full_post_var)
    next_cov = 0.5 * (next_cov + next_cov.T)

    return next_mu, next_cov, {
        'delta_mu_norm': float(np.linalg.norm(next_mu)),
        'delta_cov_fro': float(np.linalg.norm(next_cov)),
        'gate_min': float(np.min(gate)),
        'gate_max': float(np.max(gate)),
    }


def lower_anchor_selected(
    center: np.ndarray,
    mean: np.ndarray,
    diag_second: np.ndarray,
    rowdir_second: np.ndarray,
    indices: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """Selected contractions of the exact lower-order recentering term."""
    d = mean - center
    vi_d = directions @ d
    vi_mean = directions @ mean
    i = indices
    return (
        diag_second[i] * vi_d
        + 2.0 * d[i] * rowdir_second
        + 2.0 * (center[i] * center[i] - mean[i] * mean[i]) * vi_mean
    ) / (D + 1.0)


def bootstrap_ratio(base: np.ndarray, cand: np.ndarray, seed: int = 20260729, draws: int = 20000):
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(base), size=(draws, len(base)))
    ratios = cand[ix].sum(1) / np.maximum(base[ix].sum(1), 1e-300)
    return [float(x) for x in np.quantile(ratios, [0.025, 0.975])]


def run_one(network_id: int, xk_np: np.ndarray, truth_n: int, chunk: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    ws, weight_hash, weight_seed = fr.make_weights(network_id)
    x = torch.from_numpy(xk_np.copy())
    row_groups = [pilot_rows(g) for g in PILOT_GROUPS]
    states = [(np.zeros(D), np.zeros((D, D))) for _ in row_groups]
    # Each complete antipodal basis has radially corrected input covariance I.
    pilot_post_covs = [np.eye(D, dtype=np.float64) for _ in row_groups]
    target_h = None
    traces: list[list[dict[str, float]]] = [[] for _ in row_groups]

    with torch.no_grad():
        for layer, wt in enumerate(ws):
            pre = x @ wt
            post = torch.relu(pre)
            pre_mean, pre_var, _ = raw_moments(pre, None, True)
            post_mean, post_var, _ = raw_moments(post, None, True)
            w = wt.double().numpy()
            new_states = []
            new_pilot_post_covs = []
            for gi, rows in enumerate(row_groups):
                # Exact reuse: pre = previous_post @ W, so the pilot pre-covariance
                # is W.T @ C_post @ W.  This removes one 2048x256 covariance
                # contraction per pilot and layer without changing the estimator.
                pre_cov = w.T @ pilot_post_covs[gi] @ w
                pre_cov = 0.5 * (pre_cov + pre_cov.T)
                _, _, post_cov = raw_moments(post, rows, False)
                dm, dc, tr = update_defect(
                    states[gi][0], states[gi][1], w,
                    pre_mean, pre_var, post_mean, post_var,
                    pre_cov, post_cov,
                )
                tr['layer'] = layer
                traces[gi].append(tr)
                new_states.append((dm, dc))
                new_pilot_post_covs.append(post_cov)
            states = new_states
            pilot_post_covs = new_pilot_post_covs
            x = post
            if layer == TARGET:
                target_h = x.clone()
                target_states = [(dm.copy(), dc.copy()) for dm, dc in states]
    assert target_h is not None
    final_y = x.double().mean(0).numpy()
    H = target_h.double().numpy()
    m = H.mean(0)
    rho = fr.chi_mean(D)

    # Frozen pointwise feature/control construction.
    Q = fr.sample_anchor_matrix(H, m, rho)
    indices, directions = fr.sample_row_probes(Q)
    X = fr.radial_features_sample_rows(H, m, indices, directions, rho)
    fit = fr.fit_crossfit(X, final_y.reshape(1, -1) if False else x.double().numpy())
    sample_anchor = fr.contract_rows(Q, indices, directions)

    # Full-cloud selected Kerdock raw second moments, radially corrected.
    hs = H[:, indices]
    hr = H @ directions.T
    sample_diag = (H ** 2).mean(0) * RADIAL_SCALE
    sample_rowdir = (hs * hr).mean(0) * RADIAL_SCALE

    candidate_lower = []
    for dm, dc in target_states:
        mu = m + dm
        diag_second = sample_diag + np.diag(dc) + mu ** 2 - m ** 2
        dc_rowdir = np.sum(dc[indices] * directions, axis=1)
        rowdir_second = sample_rowdir + dc_rowdir + mu[indices] * (directions @ mu) - m[indices] * (directions @ m)
        candidate_lower.append(lower_anchor_selected(m, mu, diag_second, rowdir_second, indices, directions))
    candidate_lower = np.asarray(candidate_lower)
    mean_lower = candidate_lower.mean(0)
    disagreement = float(np.linalg.norm(candidate_lower[0] - candidate_lower[1]) / max(np.linalg.norm(mean_lower), 1e-30))

    # Independent evaluation references.  These are never visible to candidate construction.
    ref1 = fr.stream_reference(ws, truth_n, 31_000_000 + 2 * network_id, chunk)
    ref2 = fr.stream_reference(ws, truth_n, 31_000_001 + 2 * network_id, chunk)
    y1, y2 = ref1['y'], ref2['y']
    truth_mu = 0.5 * (ref1['mu'] + ref2['mu'])
    truth_M = 0.5 * (ref1['M'] + ref2['M'])
    truth_diag = np.diag(truth_M)
    truth_rowdir = np.sum(truth_M[indices] * directions, axis=1)
    oracle_lower = lower_anchor_selected(m, truth_mu, truth_diag, truth_rowdir, indices, directions)

    base_pred, _ = fr.estimate_from_fit(fit, sample_anchor)
    # Grid is retained as development evidence; split-level freezing occurs in summarize().
    alpha_grid = np.asarray([-1.0, -0.5, -0.25, 0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.3])
    pred_grid = []
    for alpha in alpha_grid:
        pred, _ = fr.estimate_from_fit(fit, sample_anchor + alpha * mean_lower)
        pred_grid.append(pred)
    oracle_pred, _ = fr.estimate_from_fit(fit, sample_anchor + oracle_lower)

    def mse(pred: np.ndarray) -> float:
        return float(np.mean((pred - 0.5 * (y1 + y2)) ** 2))
    def unbiased(pred: np.ndarray) -> float:
        return float(np.mean((pred - y1) * (pred - y2)))

    base_mse = mse(base_pred)
    return {
        'network_id': network_id,
        'weight_seed': weight_seed,
        'weight_sha256': weight_hash,
        'truth_n_per_half': truth_n,
        'baseline_mse': base_mse,
        'baseline_unbiased_mse': unbiased(base_pred),
        'alpha_grid': alpha_grid.tolist(),
        'candidate_mse_grid': [mse(p) for p in pred_grid],
        'candidate_unbiased_mse_grid': [unbiased(p) for p in pred_grid],
        'oracle_lower_mse': mse(oracle_pred),
        'oracle_lower_unbiased_mse': unbiased(oracle_pred),
        'anchor': {
            'candidate_relative_error': float(np.linalg.norm(mean_lower - oracle_lower) / max(np.linalg.norm(oracle_lower), 1e-30)),
            'candidate_cosine': float(np.dot(mean_lower, oracle_lower) / max(np.linalg.norm(mean_lower) * np.linalg.norm(oracle_lower), 1e-30)),
            'pilot_disagreement': disagreement,
            'candidate_norm': float(np.linalg.norm(mean_lower)),
            'oracle_norm': float(np.linalg.norm(oracle_lower)),
            'pilot_cosines': [float(np.dot(v, oracle_lower) / max(np.linalg.norm(v) * np.linalg.norm(oracle_lower), 1e-30)) for v in candidate_lower],
        },
        'target_delta': {
            'pilot_mean_norms': [float(np.linalg.norm(s[0])) for s in target_states],
            'pilot_cov_fro': [float(np.linalg.norm(s[1])) for s in target_states],
        },
        'traces': traces,
        'runtime_seconds': float(time.perf_counter() - t0),
    }


def summarize(records: list[dict[str, Any]], tune_n: int) -> dict[str, Any]:
    base = np.asarray([r['baseline_mse'] for r in records])
    grid = np.asarray(records[0]['alpha_grid'])
    cm = np.asarray([r['candidate_mse_grid'] for r in records])
    tune = np.arange(min(tune_n, len(records)))
    val = np.arange(min(tune_n, len(records)), len(records))
    tune_ratios = cm[tune].sum(0) / base[tune].sum()
    best_idx = int(np.argmin(tune_ratios))
    alpha = float(grid[best_idx])

    # One bounded abstention rescue: confidence threshold and shrink are chosen
    # only on tuning networks, then frozen.  Fallback is the exact baseline.
    disagreements = np.asarray([r['anchor']['pilot_disagreement'] for r in records])
    thresholds = np.unique(np.r_[np.quantile(disagreements[tune], [0.25, 0.5, 0.75]), np.inf])
    rescues = []
    for th in thresholds:
        applied = disagreements[tune] <= th
        mixed = np.where(applied, cm[tune, best_idx], base[tune])
        rescues.append((mixed.sum() / base[tune].sum(), float(th), int(applied.sum())))
    _, threshold, tune_applied = min(rescues)

    def block(ix: np.ndarray) -> dict[str, Any]:
        if len(ix) == 0:
            return {}
        cand = cm[ix, best_idx]
        applied = disagreements[ix] <= threshold
        gated = np.where(applied, cand, base[ix])
        oracle = np.asarray([records[i]['oracle_lower_mse'] for i in ix])
        return {
            'n': int(len(ix)),
            'alpha': alpha,
            'candidate_over_base': float(cand.sum() / base[ix].sum()),
            'candidate_ci95': bootstrap_ratio(base[ix], cand),
            'candidate_wins': int(np.sum(cand < base[ix])),
            'candidate_worst': float(np.max(cand / base[ix])),
            'gated_over_base': float(gated.sum() / base[ix].sum()),
            'gated_ci95': bootstrap_ratio(base[ix], gated, seed=20260730),
            'gated_wins': int(np.sum(gated < base[ix])),
            'gated_worst': float(np.max(gated / base[ix])),
            'gated_applied': int(np.sum(applied)),
            'oracle_lower_over_base': float(oracle.sum() / base[ix].sum()),
            'mean_anchor_cosine': float(np.mean([records[i]['anchor']['candidate_cosine'] for i in ix])),
            'median_anchor_cosine': float(np.median([records[i]['anchor']['candidate_cosine'] for i in ix])),
            'mean_anchor_relative_error': float(np.mean([records[i]['anchor']['candidate_relative_error'] for i in ix])),
            'median_disagreement': float(np.median(disagreements[ix])),
            'per_network_candidate_ratio': (cand / base[ix]).tolist(),
            'per_network_gated_ratio': (gated / base[ix]).tolist(),
        }

    return {
        'protocol': {
            'all_129_bases': True,
            'probe_count': fr.PROBES,
            'target_layer_zero_based': TARGET,
            'pilot_groups': [list(g) for g in PILOT_GROUPS],
            'tune_network_ids': [records[i]['network_id'] for i in tune],
            'validation_network_ids': [records[i]['network_id'] for i in val],
            'alpha_grid': grid.tolist(),
            'frozen_alpha': alpha,
            'frozen_disagreement_threshold': threshold,
            'tune_applied': tune_applied,
            'candidate_uses_reference': False,
            'reference_is_evaluation_only': True,
        },
        'tuning': block(tune),
        'validation': block(val),
        'all': block(np.arange(len(records))),
        'records': records,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--network-start', type=int, default=3000)
    ap.add_argument('--networks', type=int, default=8)
    ap.add_argument('--tune-networks', type=int, default=4)
    ap.add_argument('--truth-n', type=int, default=32768)
    ap.add_argument('--chunk', type=int, default=8192)
    ap.add_argument('--threads', type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    xk, meta = fr.make_kerdock()
    records = []
    for nid in range(args.network_start, args.network_start + args.networks):
        rec = run_one(nid, xk, args.truth_n, args.chunk)
        records.append(rec)
        print(json.dumps({
            'network': nid,
            'runtime': rec['runtime_seconds'],
            'cos': rec['anchor']['candidate_cosine'],
            'relerr': rec['anchor']['candidate_relative_error'],
            'disagreement': rec['anchor']['pilot_disagreement'],
            'oracle_ratio': rec['oracle_lower_mse'] / rec['baseline_mse'],
            'best_network_ratio': min(rec['candidate_mse_grid']) / rec['baseline_mse'],
        }), flush=True)
    payload = summarize(records, args.tune_networks)
    payload['kerdock'] = meta
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload['validation'], indent=2))


if __name__ == '__main__':
    main()
