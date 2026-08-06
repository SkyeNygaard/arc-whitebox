#!/usr/bin/env python3
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
RADIAL_SCALE = D / (fr.chi_mean(D) ** 2)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
GL_X, GL_W = np.polynomial.legendre.leggauss(12)


def phi(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) * INV_SQRT_2PI


def relu_univariate(mu: np.ndarray, var: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    var = np.maximum(var, 1e-18)
    sigma = np.sqrt(var)
    a = mu / sigma
    p = ndtr(a)
    f = phi(a)
    mean = mu * p + sigma * f
    second = (mu * mu + var) * p + mu * sigma * f
    return mean, np.maximum(second - mean * mean, 0.0), p


def bvn_cdf_varying(a: np.ndarray, b: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Phi_2(a,b;rho) using d/drho Phi_2 = bivariate pdf.

    Fixed Gauss-Legendre quadrature is deterministic, vectorized, and accurate
    enough for the closure screen. Inputs may be matrices of the same shape.
    """
    rho = np.clip(rho, -0.995, 0.995)
    out = ndtr(a) * ndtr(b)
    # Integral from 0 to rho; signed rho is handled by the affine map.
    for x, w in zip(GL_X, GL_W):
        t = 0.5 * rho * (x + 1.0)
        one = np.maximum(1.0 - t * t, 1e-12)
        expo = -(a * a - 2.0 * t * a * b + b * b) / (2.0 * one)
        pdf = np.exp(expo) / (2.0 * math.pi * np.sqrt(one))
        out += 0.5 * rho * w * pdf
    return np.clip(out, 0.0, 1.0)


def relu_bivariate(mu: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact Gaussian ReLU mean/covariance, up to deterministic Phi2 quadrature."""
    var = np.maximum(np.diag(cov), 1e-18)
    sigma = np.sqrt(var)
    a = mu / sigma
    rho = cov / np.maximum(np.outer(sigma, sigma), 1e-30)
    rho = np.clip(rho, -0.995, 0.995)
    ai = a[:, None]
    aj = a[None, :]
    s = np.sqrt(np.maximum(1.0 - rho * rho, 1e-12))
    P = bvn_cdf_varying(ai, aj, rho)
    A = phi(ai) * ndtr((aj - rho * ai) / s)
    B = phi(aj) * ndtr((ai - rho * aj) / s)
    pdf2 = np.exp(-(ai * ai - 2.0 * rho * ai * aj + aj * aj) / (2.0 * np.maximum(1.0 - rho * rho, 1e-12)))
    pdf2 /= 2.0 * math.pi * s
    uv = rho * P + (1.0 - rho * rho) * pdf2
    second = (
        mu[:, None] * mu[None, :] * P
        + mu[:, None] * sigma[None, :] * B
        + mu[None, :] * sigma[:, None] * A
        + sigma[:, None] * sigma[None, :] * uv
    )
    mean, post_var, _ = relu_univariate(mu, var)
    out_cov = second - np.outer(mean, mean)
    out_cov = 0.5 * (out_cov + out_cov.T)
    np.fill_diagonal(out_cov, post_var)
    return mean, out_cov


def observed_marginals(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    xd = x.double()
    mean = xd.mean(0).cpu().numpy()
    second = (xd * xd).mean(0).cpu().numpy() * RADIAL_SCALE
    return mean, np.maximum(second - mean * mean, 1e-18)


def observed_covariance(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    xd = x.double()
    mean = xd.mean(0).cpu().numpy()
    second = (xd.T @ xd / len(xd)).cpu().numpy() * RADIAL_SCALE
    cov = second - np.outer(mean, mean)
    cov = 0.5 * (cov + cov.T)
    return mean, cov


def lower_anchor_selected(center: np.ndarray, mean: np.ndarray, diag_second: np.ndarray,
                          rowdir_second: np.ndarray, indices: np.ndarray,
                          directions: np.ndarray) -> np.ndarray:
    d = mean - center
    vi_d = directions @ d
    vi_mean = directions @ mean
    i = indices
    return (
        diag_second[i] * vi_d
        + 2.0 * d[i] * rowdir_second
        + 2.0 * (center[i] * center[i] - mean[i] * mean[i]) * vi_mean
    ) / (D + 1.0)


def anchor_from_state(H: np.ndarray, mean: np.ndarray, cov: np.ndarray,
                      indices: np.ndarray, directions: np.ndarray) -> np.ndarray:
    m = H.mean(0)
    second_diag = np.diag(cov) + mean * mean
    rowdir = np.sum(cov[indices] * directions, axis=1) + mean[indices] * (directions @ mean)
    return lower_anchor_selected(m, mean, second_diag, rowdir, indices, directions)


def anchor_from_defect(H: np.ndarray, delta_mean: np.ndarray, delta_cov: np.ndarray,
                       indices: np.ndarray, directions: np.ndarray) -> np.ndarray:
    m = H.mean(0)
    hs = H[:, indices]
    hr = H @ directions.T
    sample_diag = (H * H).mean(0) * RADIAL_SCALE
    sample_rowdir = (hs * hr).mean(0) * RADIAL_SCALE
    mean = m + delta_mean
    diag_second = sample_diag + np.diag(delta_cov) + mean * mean - m * m
    rowdir = sample_rowdir + np.sum(delta_cov[indices] * directions, axis=1)
    rowdir += mean[indices] * (directions @ mean) - m[indices] * (directions @ m)
    return lower_anchor_selected(m, mean, diag_second, rowdir, indices, directions)


def propagate(network_id: int, xk_np: np.ndarray) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, dict[str, np.ndarray], list[dict[str, float]]]:
    ws, _, _ = fr.make_weights(network_id)
    x = torch.from_numpy(xk_np.copy())

    # Absolute deterministic bivariate-Gaussian closure.
    g_mean = np.zeros(D, dtype=np.float64)
    g_cov = np.eye(D, dtype=np.float64)

    # Internally centered marginal-reanchored covariance defect.
    d_mean = np.zeros(D, dtype=np.float64)
    d_cov = np.zeros((D, D), dtype=np.float64)

    trace: list[dict[str, float]] = []
    target_h = None
    target_final = None
    target_states: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for layer, wt in enumerate(ws):
            w = wt.double().numpy()
            pre_k = x @ wt
            post_k = torch.relu(pre_k)

            # Absolute Gaussian closure.
            g_pre_mean = g_mean @ w
            g_pre_cov = w.T @ g_cov @ w
            g_mean, g_cov = relu_bivariate(g_pre_mean, g_pre_cov)

            # Internally centered recurrence. Layer 0 is exact because Gaussian
            # input through a linear map is jointly Gaussian.
            if layer == 0:
                true_mean, true_cov = relu_bivariate(np.zeros(D), w.T @ w)
                k_mean, k_cov = observed_covariance(post_k)
                d_mean = true_mean - k_mean
                d_cov = true_cov - k_cov
            else:
                k_pre_mean, k_pre_var = observed_marginals(pre_k)
                k_post_mean, k_post_var = observed_marginals(post_k)
                pre_dm = d_mean @ w
                pre_dc = w.T @ d_cov @ w
                target_pre_mean = k_pre_mean + pre_dm
                target_pre_var = np.maximum(k_pre_var + np.diag(pre_dc), 1e-18)
                target_post_mean, target_post_var, gate = relu_univariate(target_pre_mean, target_pre_var)
                d_mean = target_post_mean - k_post_mean
                d_cov = pre_dc * np.outer(gate, gate)
                np.fill_diagonal(d_cov, target_post_var - k_post_var)
                d_cov = 0.5 * (d_cov + d_cov.T)

            trace.append({
                'layer': float(layer),
                'abs_mean_norm': float(np.linalg.norm(g_mean)),
                'abs_cov_fro': float(np.linalg.norm(g_cov)),
                'defect_mean_norm': float(np.linalg.norm(d_mean)),
                'defect_cov_fro': float(np.linalg.norm(d_cov)),
            })
            x = post_k
            if layer == TARGET:
                target_h = x.clone()
                target_states = {
                    'absolute_mean': g_mean.copy(),
                    'absolute_cov': g_cov.copy(),
                    'centered_delta_mean': d_mean.copy(),
                    'centered_delta_cov': d_cov.copy(),
                }
            if layer == len(ws) - 1:
                target_final = x.clone()

    assert target_h is not None and target_final is not None
    return ws, target_h, target_final, target_states, trace


def run_one(network_id: int, xk_np: np.ndarray, truth_n: int, chunk: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    ws, weight_hash, weight_seed = fr.make_weights(network_id)
    # Recreate once in propagate to keep interface simple; deterministic hashes verify identity.
    ws2, target_h, final_k, states, trace = propagate(network_id, xk_np)
    assert all(torch.equal(a, b) for a, b in zip(ws, ws2))
    H = target_h.double().numpy()
    Yk = final_k.double().numpy()
    m = H.mean(0)
    rho = fr.chi_mean(D)
    Q = fr.sample_anchor_matrix(H, m, rho)
    indices, directions = fr.sample_row_probes(Q)
    X = fr.radial_features_sample_rows(H, m, indices, directions, rho)
    fit = fr.fit_crossfit(X, Yk)
    sample_anchor = fr.contract_rows(Q, indices, directions)

    cand_abs = anchor_from_state(H, states['absolute_mean'], states['absolute_cov'], indices, directions)
    cand_center = anchor_from_defect(H, states['centered_delta_mean'], states['centered_delta_cov'], indices, directions)

    ref1 = fr.stream_reference(ws, truth_n, 51_000_000 + 2 * network_id, chunk)
    ref2 = fr.stream_reference(ws, truth_n, 51_000_001 + 2 * network_id, chunk)
    truth_y = 0.5 * (ref1['y'] + ref2['y'])
    truth_mu = 0.5 * (ref1['mu'] + ref2['mu'])
    truth_M = 0.5 * (ref1['M'] + ref2['M'])
    truth_diag = np.diag(truth_M)
    truth_rowdir = np.sum(truth_M[indices] * directions, axis=1)
    oracle = lower_anchor_selected(m, truth_mu, truth_diag, truth_rowdir, indices, directions)

    base, _ = fr.estimate_from_fit(fit, sample_anchor)
    alpha_grid = np.asarray([-1.0, -0.5, -0.25, 0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
    methods = {'absolute_bivariate': cand_abs, 'centered_marginal': cand_center}

    def mse(pred: np.ndarray) -> float:
        return float(np.mean((pred - truth_y) ** 2))

    def unbiased(pred: np.ndarray) -> float:
        return float(np.mean((pred - ref1['y']) * (pred - ref2['y'])))

    out_methods = {}
    for name, anchor in methods.items():
        preds = [fr.estimate_from_fit(fit, sample_anchor + a * anchor)[0] for a in alpha_grid]
        out_methods[name] = {
            'anchor': anchor.tolist(),
            'relative_error': float(np.linalg.norm(anchor - oracle) / max(np.linalg.norm(oracle), 1e-30)),
            'cosine': float(np.dot(anchor, oracle) / max(np.linalg.norm(anchor) * np.linalg.norm(oracle), 1e-30)),
            'norm': float(np.linalg.norm(anchor)),
            'mse_grid': [mse(p) for p in preds],
            'unbiased_mse_grid': [unbiased(p) for p in preds],
        }
    oracle_pred, _ = fr.estimate_from_fit(fit, sample_anchor + oracle)
    return {
        'network_id': network_id,
        'weight_seed': weight_seed,
        'weight_sha256': weight_hash,
        'truth_n_per_half': truth_n,
        'baseline_mse': mse(base),
        'baseline_unbiased_mse': unbiased(base),
        'oracle_lower_mse': mse(oracle_pred),
        'oracle_lower_unbiased_mse': unbiased(oracle_pred),
        'alpha_grid': alpha_grid.tolist(),
        'methods': out_methods,
        'oracle_anchor': oracle.tolist(),
        'trace': trace,
        'runtime_seconds': float(time.perf_counter() - t0),
    }


def bootstrap_ratio(base: np.ndarray, cand: np.ndarray, seed: int, draws: int = 20000) -> list[float]:
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(base), size=(draws, len(base)))
    ratios = cand[ix].sum(1) / np.maximum(base[ix].sum(1), 1e-300)
    return [float(x) for x in np.quantile(ratios, [0.025, 0.975])]


def summarize(records: list[dict[str, Any]], tune_n: int) -> dict[str, Any]:
    base = np.asarray([r['baseline_mse'] for r in records])
    grid = np.asarray(records[0]['alpha_grid'])
    tune = np.arange(min(tune_n, len(records)))
    val = np.arange(min(tune_n, len(records)), len(records))
    out = {'tune_ids': [records[i]['network_id'] for i in tune], 'validation_ids': [records[i]['network_id'] for i in val], 'methods': {}}
    for method in records[0]['methods']:
        cm = np.asarray([r['methods'][method]['mse_grid'] for r in records])
        ti = int(np.argmin(cm[tune].sum(0) / base[tune].sum()))
        alpha = float(grid[ti])
        def block(ix: np.ndarray, seed: int) -> dict[str, Any]:
            cand = cm[ix, ti]
            ratios = cand / base[ix]
            return {
                'n': int(len(ix)), 'alpha': alpha,
                'candidate_over_base': float(cand.sum() / base[ix].sum()),
                'ci95': bootstrap_ratio(base[ix], cand, seed),
                'wins': int(np.sum(cand < base[ix])),
                'worst': float(np.max(ratios)),
                'per_network': ratios.tolist(),
                'mean_cosine': float(np.mean([records[i]['methods'][method]['cosine'] for i in ix])),
                'median_cosine': float(np.median([records[i]['methods'][method]['cosine'] for i in ix])),
                'mean_relative_error': float(np.mean([records[i]['methods'][method]['relative_error'] for i in ix])),
            }
        out['methods'][method] = {'tuning': block(tune, 20260731), 'validation': block(val, 20260732) if len(val) else {}}
    oracle = np.asarray([r['oracle_lower_mse'] for r in records])
    out['oracle_tuning_over_base'] = float(oracle[tune].sum() / base[tune].sum())
    if len(val): out['oracle_validation_over_base'] = float(oracle[val].sum() / base[val].sum())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--network', type=int)
    p.add_argument('--networks', type=int, nargs='*')
    p.add_argument('--truth-n', type=int, default=16384)
    p.add_argument('--chunk', type=int, default=4096)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--records-dir', type=Path)
    p.add_argument('--tune-n', type=int, default=6)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    xk, meta = fr.make_kerdock()
    if args.records_dir:
        recs = [json.loads(q.read_text()) for q in sorted(args.records_dir.glob('network_*.json'))]
        payload = summarize(recs, args.tune_n)
        payload['kerdock_sha256'] = meta['points_sha256']
        args.out.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        return
    ids = args.networks or ([args.network] if args.network is not None else [])
    if not ids: raise SystemExit('provide --network/--networks or --records-dir')
    if len(ids) == 1:
        payload = run_one(ids[0], xk, args.truth_n, args.chunk)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(json.dumps({'network': ids[0], 'runtime': payload['runtime_seconds'], 'base': payload['baseline_mse'], 'methods': {k: {'cos': v['cosine'], 'best_ratio': min(v['mse_grid'])/payload['baseline_mse']} for k,v in payload['methods'].items()}}, indent=2))
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        for n in ids:
            q = args.out / f'network_{n}.json'
            if q.exists(): continue
            rec = run_one(n, xk, args.truth_n, args.chunk)
            q.write_text(json.dumps(rec, indent=2))
            print(json.dumps({'network': n, 'runtime': rec['runtime_seconds'], 'methods': {k: {'cos': v['cosine'], 'best_ratio': min(v['mse_grid'])/rec['baseline_mse']} for k,v in rec['methods'].items()}}, indent=2), flush=True)

if __name__ == '__main__':
    main()
