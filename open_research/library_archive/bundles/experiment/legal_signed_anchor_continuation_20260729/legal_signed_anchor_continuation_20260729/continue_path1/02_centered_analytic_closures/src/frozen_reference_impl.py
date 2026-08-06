#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

D = 256
DEPTH = 32
TARGET = 29
N_BASES = 129
ROWS_PER_BASIS = 512
N_ROWS = N_BASES * ROWS_PER_BASIS
PROBES = 128
FOLDS = 6
RIDGE = 0.1
ROTATION_SEED = 3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(x: np.ndarray) -> str:
    y = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(y.dtype).encode())
    h.update(str(y.shape).encode())
    h.update(y.tobytes())
    return h.hexdigest()


def chi_mean(d: int = D) -> float:
    return math.sqrt(2.0) * math.exp(math.lgamma((d + 1.0) / 2.0) - math.lgamma(d / 2.0))


def gf128_mul(a: int, b: int) -> int:
    result, left, right = 0, a, b
    while right:
        if right & 1:
            result ^= left
        right >>= 1
        carry = left & 0x40
        left = (left << 1) & 0x7F
        if carry:
            left ^= 0x03
    return result


def gf128_square(a: int) -> int:
    return gf128_mul(a, a)


def gf128_pow(a: int, power: int) -> int:
    result, base, exponent = 1, a, power
    while exponent:
        if exponent & 1:
            result = gf128_mul(result, base)
        base = gf128_square(base)
        exponent >>= 1
    return result


def gf128_trace(a: int) -> int:
    total = term = a
    for _ in range(1, 7):
        term = gf128_square(term)
        total ^= term
    if total not in (0, 1):
        raise AssertionError(total)
    return total


def kerdock_chirp(u: int) -> np.ndarray:
    bits = np.empty(D, dtype=np.uint8)
    for coordinate in range(D):
        x = coordinate & 0x7F
        xn = coordinate >> 7
        ux = gf128_mul(u, x)
        polynomial = gf128_pow(ux, 3) ^ gf128_pow(ux, 5) ^ gf128_pow(ux, 9)
        bits[coordinate] = gf128_trace(polynomial) ^ (xn & gf128_trace(ux))
    return (1 - 2 * bits.astype(np.int16)).astype(np.float32)


def walsh_hadamard() -> np.ndarray:
    indices = np.arange(D, dtype=np.uint16)
    parity = np.asarray([int(i).bit_count() & 1 for i in range(D)], dtype=np.uint8)
    bits = parity[np.bitwise_and(indices[:, None], indices[None, :])]
    return (1 - 2 * bits.astype(np.int16)).astype(np.float32)


def haar_rotation(seed: int = ROTATION_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((D, D))
    q, r = np.linalg.qr(gaussian)
    q *= np.where(np.diag(r) < 0.0, -1.0, 1.0)[None, :]
    return q.astype(np.float32)


def make_kerdock() -> tuple[np.ndarray, dict[str, Any]]:
    radius = chi_mean(D)
    H = walsh_hadamard() / math.sqrt(D)
    rotation = haar_rotation(ROTATION_SEED)
    blocks: list[np.ndarray] = []
    chirps = []
    for u in range(128):
        chirp = kerdock_chirp(u)
        chirps.append(chirp)
        basis = (H * chirp[None, :]) @ rotation
        blocks.append((radius * basis).astype(np.float32))
        blocks.append((-radius * basis).astype(np.float32))
    coordinate = (radius * rotation).astype(np.float32)
    blocks.append(coordinate)
    blocks.append(-coordinate)
    x = np.concatenate(blocks, axis=0)
    if x.shape != (N_ROWS, D):
        raise AssertionError(x.shape)
    block_ids = np.repeat(np.arange(N_BASES), ROWS_PER_BASIS)
    signs_in_block = np.tile(np.r_[np.ones(D, dtype=np.int8), -np.ones(D, dtype=np.int8)], N_BASES)
    metadata = {
        'ordering': 'u=0..127 chirp-Walsh bases, then rotated coordinate basis; within each basis 256 positive rows followed by their 256 negatives',
        'rotation_seed': ROTATION_SEED,
        'radius': radius,
        'points_sha256': sha256_array(x),
        'rotation_sha256': sha256_array(rotation),
        'chirps_sha256': sha256_array(np.asarray(chirps, dtype=np.float32)),
        'block_ids_sha256': sha256_array(block_ids),
        'signs_sha256': sha256_array(signs_in_block),
    }
    return x, metadata


def make_weights(network_id: int) -> tuple[list[torch.Tensor], str, int]:
    seed = 51000 + network_id
    rng = np.random.default_rng(seed)
    scale = math.sqrt(2.0 / D)
    ws_np = [(rng.standard_normal((D, D)) * scale).astype(np.float32) for _ in range(DEPTH)]
    digest = hashlib.sha256()
    for w in ws_np:
        digest.update(w.tobytes())
    ws = [torch.from_numpy(w) for w in ws_np]
    return ws, digest.hexdigest(), seed


def forward_target_final(x: torch.Tensor, ws: list[torch.Tensor], target: int = TARGET) -> tuple[torch.Tensor, torch.Tensor]:
    ht = None
    with torch.no_grad():
        for layer, w in enumerate(ws):
            x = torch.relu(x @ w)
            if layer == target:
                ht = x.clone()
    assert ht is not None
    return ht, x


def stream_reference(ws: list[torch.Tensor], n: int, seed: int, chunk: int) -> dict[str, np.ndarray]:
    engine = torch.quasirandom.SobolEngine(D, scramble=True, seed=seed)
    y_sum = np.zeros(D, dtype=np.float64)
    mu_sum = np.zeros(D, dtype=np.float64)
    M_sum = np.zeros((D, D), dtype=np.float64)
    raw_sum = np.zeros((D, D), dtype=np.float64)
    done = 0
    with torch.no_grad():
        while done < n:
            b = min(chunk, n - done)
            u = engine.draw(b, dtype=torch.float32).clamp_(1e-7, 1 - 1e-7)
            x = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
            h, y = forward_target_final(x, ws)
            H = h.double().numpy()
            Y = y.double().numpy()
            y_sum += Y.sum(axis=0)
            mu_sum += H.sum(axis=0)
            M_sum += H.T @ H
            raw_sum += (H * H).T @ H
            done += b
    return {'y': y_sum / n, 'mu': mu_sum / n, 'M': M_sum / n, 'raw': raw_sum / n}


def sample_anchor_matrix(H: np.ndarray, m: np.ndarray, rho: float) -> np.ndarray:
    raw = (H * H).T @ H / len(H)
    M = H.T @ H / len(H)
    m2 = np.diag(M)
    return (
        raw / (rho * rho)
        - D / (D + 1.0) * m2[:, None] * m[None, :] / (rho * rho)
        - 2.0 * D / (D + 1.0) * m[:, None] * M / (rho * rho)
        + 2.0 / (D + 1.0) * (m * m)[:, None] * m[None, :]
    )


def exact_anchor_matrix(mu: np.ndarray, M: np.ndarray, raw: np.ndarray, m: np.ndarray) -> np.ndarray:
    return (raw - np.diag(M)[:, None] * m[None, :] - 2.0 * m[:, None] * M + 2.0 * (m * m)[:, None] * mu[None, :]) / (D + 1.0)


def connected_anchor_matrix(mu: np.ndarray, M: np.ndarray, raw: np.ndarray) -> np.ndarray:
    c21 = raw - np.diag(M)[:, None] * mu[None, :] - 2.0 * mu[:, None] * M + 2.0 * (mu * mu)[:, None] * mu[None, :]
    return c21 / (D + 1.0)


def anchor_component_matrices(
    mu: np.ndarray,
    M: np.ndarray,
    raw: np.ndarray,
    m: np.ndarray,
    sample_M_scaled: np.ndarray,
    sample_raw_scaled: np.ndarray,
) -> dict[str, np.ndarray]:
    # Decompose the *correction relative to the same-cloud anchor*.  This is
    # the convention used by the prior lower/connected ablations: each
    # component candidate starts at Q, then adds only its exact defect channel.
    complete = exact_anchor_matrix(mu, M, raw, m)
    exact_connected = connected_anchor_matrix(mu, M, raw)
    sample_connected = connected_anchor_matrix(m, sample_M_scaled, sample_raw_scaled)
    sample_complete = exact_anchor_matrix(m, sample_M_scaled, sample_raw_scaled, m)
    exact_lower = complete - exact_connected
    sample_lower = sample_complete - sample_connected
    return {
        'complete_exact': complete,
        'lower_only': sample_complete + (exact_lower - sample_lower),
        'connected_only': sample_complete + (exact_connected - sample_connected),
        # Omit oracle mean information by substituting the observable sample
        # center m for mu while retaining exact pair and raw third moments.
        'complete_mean_omitted': exact_anchor_matrix(m, M, raw, m),
        # Omit oracle pair-moment information by substituting the rescaled
        # same-cloud pair matrix while retaining exact mean and raw third moment.
        'complete_pair_moments_omitted': exact_anchor_matrix(mu, sample_M_scaled, raw, m),
    }


def sample_row_probes(Q: np.ndarray, p: int = PROBES) -> tuple[np.ndarray, np.ndarray]:
    score = np.linalg.norm(Q, axis=1)
    indices = np.argsort(score)[::-1][:p]
    directions = Q[indices].copy()
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-30)
    return indices.astype(np.int64), directions


def radial_features_sample_rows(H: np.ndarray, m: np.ndarray, indices: np.ndarray, directions: np.ndarray, rho: float) -> np.ndarray:
    hs = H[:, indices]
    hr = H @ directions.T
    mr = directions @ m
    x = (
        (hs * hs) * hr / (rho * rho)
        - (hs * hs) * mr[None, :] * (D / (D + 1.0)) / (rho * rho)
        - (hs * m[indices][None, :]) * hr * (2.0 * D / (D + 1.0)) / (rho * rho)
        + (m[indices] * m[indices])[None, :] * hr * (2.0 / (D + 1.0))
    )
    return x


def contract_rows(A: np.ndarray, indices: np.ndarray, directions: np.ndarray) -> np.ndarray:
    return np.sum(A[indices] * directions, axis=1)


def fold_groups() -> list[np.ndarray]:
    return [np.asarray(g, dtype=np.int64) for g in np.array_split(np.arange(N_BASES), FOLDS)]


def fit_crossfit(X: np.ndarray, Y: np.ndarray) -> dict[str, Any]:
    # Algebraically identical to explicit centering, but uses sufficient
    # statistics so each held-out block is processed once and large centered
    # training copies are never materialized.
    bid = np.repeat(np.arange(N_BASES), ROWS_PER_BASIS)
    groups = fold_groups()
    total_sx = X.sum(axis=0)
    total_sy = Y.sum(axis=0)
    total_xx = X.T @ X
    total_xy = X.T @ Y
    betas = []
    fold_y_mean = []
    fold_x_mean = []
    fold_sizes = []
    ridge_scales = []
    for group in groups:
        te = np.isin(bid, group)
        xe = X[te]
        ye = Y[te]
        nte = len(xe)
        ntr = len(X) - nte
        sx_e = xe.sum(axis=0)
        sy_e = ye.sum(axis=0)
        xx_e = xe.T @ xe
        xy_e = xe.T @ ye
        sx = total_sx - sx_e
        sy = total_sy - sy_e
        gram = (total_xx - xx_e) - np.outer(sx, sx) / ntr
        cross = (total_xy - xy_e) - np.outer(sx, sy) / ntr
        scale = max(float(np.trace(gram) / X.shape[1]), 1e-12)
        beta = np.linalg.solve(gram + RIDGE * scale * np.eye(X.shape[1]), cross)
        betas.append(beta)
        fold_y_mean.append(sy_e / nte)
        fold_x_mean.append(sx_e / nte)
        fold_sizes.append(nte)
        ridge_scales.append(scale)
    return {
        'groups': groups,
        'betas': np.asarray(betas),
        'fold_y_mean': np.asarray(fold_y_mean),
        'fold_x_mean': np.asarray(fold_x_mean),
        'fold_sizes': np.asarray(fold_sizes, dtype=np.int64),
        'ridge_scales': np.asarray(ridge_scales),
    }


def estimate_from_fit(fit: dict[str, Any], anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fold_pred = fit['fold_y_mean'] - np.einsum('fp,fpd->fd', fit['fold_x_mean'] - anchor[None, :], fit['betas'])
    weights = fit['fold_sizes'] / fit['fold_sizes'].sum()
    return weights @ fold_pred, fold_pred


def pooled_mse(pred: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> float:
    truth = 0.5 * (y1 + y2)
    return float(np.mean((pred - truth) ** 2))


def unbiased_mse(pred: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> float:
    return float(np.mean((pred - y1) * (pred - y2)))


def relevant_spectrum(defect: np.ndarray, indices: np.ndarray, fit: dict[str, Any]) -> dict[str, Any]:
    fold_weights = fit['fold_sizes'] / fit['fold_sizes'].sum()
    beta_bar = np.einsum('f,fpd->pd', fold_weights, fit['betas'])
    weights = np.sum(beta_bar * beta_bar, axis=1)
    weighted_rows = np.sqrt(np.maximum(weights, 0.0))[:, None] * defect[indices]
    s = np.linalg.svd(weighted_rows, compute_uv=False)
    energy = np.cumsum(s * s) / max(float(np.sum(s * s)), 1e-30)
    return {
        'singular_values': s,
        'energy_curve': energy,
        'rank16_energy': float(energy[min(15, len(energy) - 1)]),
        'beta_bar': beta_bar,
        'probe_output_weights': weights,
        'weighted_defect_rows': weighted_rows,
    }


def run_one(network_id: int, xk: np.ndarray, outdir: Path, truth_n: int, chunk: int, overwrite: bool, reference_file: Path | None = None) -> dict[str, Any]:
    outfile = outdir / 'vectors' / f'network_{network_id:04d}.npz'
    recordfile = outdir / 'records' / f'network_{network_id:04d}.json'
    if recordfile.exists() and outfile.exists() and not overwrite:
        return json.loads(recordfile.read_text())
    t0 = time.perf_counter()
    ws, weight_hash, weight_seed = make_weights(network_id)

    # Generate the independent references before materializing the full Kerdock
    # target/final clouds.  This is mathematically identical and avoids a
    # pathological CPU allocator/thread interaction in the constrained runner.
    ref_seed_1 = 9_100_000 + 2 * network_id
    ref_seed_2 = 9_100_001 + 2 * network_id
    if reference_file is None:
        ref1 = stream_reference(ws, truth_n, ref_seed_1, chunk)
        ref2 = stream_reference(ws, truth_n, ref_seed_2, chunk)
    else:
        with np.load(reference_file) as z:
            stored_n = int(z['n_per_half'])
            if stored_n != truth_n:
                raise ValueError(f'{reference_file}: n_per_half={stored_n}, expected {truth_n}')
            ref1 = {k: z[f'{k}_half1'].copy() for k in ('y','mu','M','raw')}
            ref2 = {k: z[f'{k}_half2'].copy() for k in ('y','mu','M','raw')}
    pooled = {k: 0.5 * (ref1[k] + ref2[k]) for k in ref1}

    xkt = torch.from_numpy(xk)
    hk, yk = forward_target_final(xkt, ws)
    H = hk.double().numpy()
    Y = yk.double().numpy()
    m = H.mean(axis=0)
    base = Y.mean(axis=0)
    rho = chi_mean(D)

    Q = sample_anchor_matrix(H, m, rho)
    indices, directions = sample_row_probes(Q)
    X = radial_features_sample_rows(H, m, indices, directions, rho)
    feature_sample_anchor = X.mean(axis=0)
    Q_anchor = contract_rows(Q, indices, directions)
    sample_M_raw = H.T @ H / len(H)
    sample_raw_raw = (H * H).T @ H / len(H)
    sample_M_scaled = (D / (rho * rho)) * sample_M_raw
    sample_raw_scaled = ((D + 1.0) / (rho * rho)) * sample_raw_raw
    components = anchor_component_matrices(pooled['mu'], pooled['M'], pooled['raw'], m, sample_M_scaled, sample_raw_scaled)
    anchors = {'sample_anchor': Q_anchor}
    anchors.update({k: contract_rows(v, indices, directions) for k, v in components.items()})
    # Half anchors preserve oracle noise information.
    c1 = anchor_component_matrices(ref1['mu'], ref1['M'], ref1['raw'], m, sample_M_scaled, sample_raw_scaled)
    c2 = anchor_component_matrices(ref2['mu'], ref2['M'], ref2['raw'], m, sample_M_scaled, sample_raw_scaled)
    half_anchors_1 = {k: contract_rows(v, indices, directions) for k, v in c1.items()}
    half_anchors_2 = {k: contract_rows(v, indices, directions) for k, v in c2.items()}

    fit = fit_crossfit(X, Y)
    predictions: dict[str, np.ndarray] = {'baseline': base}
    fold_predictions: dict[str, np.ndarray] = {}
    for label, anchor in anchors.items():
        predictions[label], fold_predictions[label] = estimate_from_fit(fit, anchor)

    y1, y2 = ref1['y'], ref2['y']
    metrics = {}
    base_pooled = pooled_mse(base, y1, y2)
    base_unbiased = unbiased_mse(base, y1, y2)
    for label, pred in predictions.items():
        pm = pooled_mse(pred, y1, y2)
        um = unbiased_mse(pred, y1, y2)
        metrics[label] = {
            'pooled_mse': pm,
            'pooled_ratio': pm / max(base_pooled, 1e-300),
            'unbiased_mse': um,
            'unbiased_ratio': um / max(base_unbiased, 1e-300) if base_unbiased > 0 else None,
        }

    defect = components['complete_exact'] - Q
    spectrum = relevant_spectrum(defect, indices, fit)
    correction_vectors = {label: pred - base for label, pred in predictions.items() if label != 'baseline'}
    pred_labels = list(predictions)
    anchor_labels = list(anchors)
    groups_matrix = np.zeros((FOLDS, N_BASES), dtype=np.uint8)
    for i, group in enumerate(fit['groups']):
        groups_matrix[i, group] = 1

    np.savez_compressed(
        outfile,
        network_id=np.asarray(network_id),
        weight_seed=np.asarray(weight_seed),
        weight_sha256=np.asarray(weight_hash),
        probe_indices=indices,
        probe_directions=directions,
        feature_sample_anchor=feature_sample_anchor,
        matrix_sample_anchor=Q_anchor,
        feature_anchor_max_abs_difference=np.asarray(float(np.max(np.abs(feature_sample_anchor - Q_anchor)))),
        anchor_labels=np.asarray(anchor_labels),
        anchors=np.stack([anchors[k] for k in anchor_labels]),
        half_anchors_1=np.stack([half_anchors_1.get(k, np.full(PROBES, np.nan)) for k in anchor_labels]),
        half_anchors_2=np.stack([half_anchors_2.get(k, np.full(PROBES, np.nan)) for k in anchor_labels]),
        prediction_labels=np.asarray(pred_labels),
        predictions=np.stack([predictions[k] for k in pred_labels]),
        correction_labels=np.asarray(list(correction_vectors)),
        correction_vectors=np.stack([correction_vectors[k] for k in correction_vectors]),
        fold_prediction_labels=np.asarray(list(fold_predictions)),
        fold_predictions=np.stack([fold_predictions[k] for k in fold_predictions]),
        fold_betas=fit['betas'],
        fold_basis_membership=groups_matrix,
        fold_sizes=fit['fold_sizes'],
        ridge_scales=fit['ridge_scales'],
        reference_y_half1=ref1['y'], reference_y_half2=ref2['y'],
        reference_mu_half1=ref1['mu'], reference_mu_half2=ref2['mu'],
        reference_M_half1=ref1['M'], reference_M_half2=ref2['M'],
        reference_raw_half1=ref1['raw'], reference_raw_half2=ref2['raw'],
        sample_center=m,
        baseline_output=base,
        defect_matrix=defect,
        downstream_singular_values=spectrum['singular_values'],
        downstream_energy_curve=spectrum['energy_curve'],
        beta_bar=spectrum['beta_bar'],
        probe_output_weights=spectrum['probe_output_weights'],
        weighted_defect_rows=spectrum['weighted_defect_rows'],
    )
    runtime = time.perf_counter() - t0
    record = {
        'network_id': network_id,
        'weight_seed': weight_seed,
        'weight_sha256': weight_hash,
        'reference': {
            'n_per_half': truth_n,
            'seeds': [ref_seed_1, ref_seed_2],
            'pooled_reference_noise_mse': float(0.25 * np.mean((y1 - y2) ** 2)),
            'base_pooled_mse': base_pooled,
            'base_unbiased_mse': base_unbiased,
        },
        'metrics': metrics,
        'probe_indices': indices.tolist(),
        'feature_anchor_max_abs_difference': float(np.max(np.abs(feature_sample_anchor - Q_anchor))),
        'anchor_noise': {
            k: float(0.5 * np.linalg.norm(half_anchors_1[k] - half_anchors_2[k]) / max(np.linalg.norm(anchors[k] - feature_sample_anchor), 1e-30))
            for k in components
        },
        'spectrum': {
            'top32_singular_values': spectrum['singular_values'][:32].tolist(),
            'rank16_energy': spectrum['rank16_energy'],
        },
        'vectors_file': str(outfile.name),
        'vectors_sha256': sha256_file(outfile),
        'runtime_seconds': runtime,
    }
    recordfile.write_text(json.dumps(record, indent=2))
    del H, Y, X, Q, defect, hk, yk, xkt, ws
    gc.collect()
    return record


def summarize(records: list[dict[str, Any]], bootstrap_draws: int = 20000) -> dict[str, Any]:
    labels = list(records[0]['metrics'])
    base = np.asarray([r['reference']['base_pooled_mse'] for r in records])
    summary: dict[str, Any] = {}
    rng = np.random.default_rng(20260729)
    n = len(records)
    bootstrap_indices = rng.integers(0, n, size=(bootstrap_draws, n))
    for label in labels:
        mse = np.asarray([r['metrics'][label]['pooled_mse'] for r in records])
        ratios = mse / base
        bs = mse[bootstrap_indices].sum(axis=1) / np.maximum(base[bootstrap_indices].sum(axis=1), 1e-300)
        summary[label] = {
            'aggregate_pooled_ratio': float(mse.sum() / base.sum()),
            'bootstrap_95_interval': [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            'wins': int(np.sum(mse < base)),
            'median_ratio': float(np.median(ratios)),
            'worst_ratio': float(np.max(ratios)),
            'p90_ratio': float(np.quantile(ratios, 0.9)),
            'per_network_ratios': ratios.tolist(),
            'aggregate_unbiased_mse': float(np.sum([r['metrics'][label]['unbiased_mse'] for r in records])),
        }
    return summary


def prior_reproduction(inputs: Path) -> dict[str, Any]:
    high = json.loads((inputs / 'sparse_radial_highref8_merged.json').read_text())
    low = json.loads((inputs / 'lowerpilot_screen8_merged.json').read_text())
    reported = {
        'sample_rows': high['summary']['sample_rows'],
        'diag': high['summary']['diag'],
        'complete_32_exact': low['summary']['exact'],
        'lower_only': low['summary']['oracle_lower'],
        'connected_only': low['summary']['oracle_connected'],
    }
    recomputed = {}
    for label in ['sample_rows', 'diag']:
        mse = sum(r['variants'][label]['mse'] for r in high['records'])
        base = sum(r['baseline_mse'] for r in high['records'])
        ratios = [r['variants'][label]['mse_ratio'] for r in high['records']]
        recomputed[label] = {'aggregate_ratio': mse / base, 'wins': sum(x < 1 for x in ratios), 'worst': max(ratios)}
    for outlabel, inlabel in [('complete_32_exact', 'exact'), ('lower_only', 'oracle_lower'), ('connected_only', 'oracle_connected')]:
        mse = sum(r['mse'][inlabel] for r in low['records'])
        base = sum(r['baseline_mse'] for r in low['records'])
        ratios = [r['ratio'][inlabel] for r in low['records']]
        recomputed[outlabel] = {'aggregate_ratio': mse / base, 'wins': sum(x < 1 for x in ratios), 'worst': max(ratios)}
    return {'reported': reported, 'recomputed_from_records': recomputed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', type=Path, default=Path('/mnt/data/sparse_radial_validation_inputs'))
    ap.add_argument('--outdir', type=Path, default=Path('/mnt/data/sparse_radial_fresh_validation'))
    ap.add_argument('--network-start', type=int, default=1000)
    ap.add_argument('--networks', type=int, default=24)
    ap.add_argument('--truth-n', type=int, default=131072)
    ap.add_argument('--chunk', type=int, default=8192)
    ap.add_argument('--threads', type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / 'vectors').mkdir(exist_ok=True)
    (args.outdir / 'records').mkdir(exist_ok=True)

    input_names = [
        'sparse_radial_cubic_control.py', 'sparse_radial_highref8_merged.json',
        'lowerpilot_screen8_merged.json', 'sparse_cubic_center_and_channel_report.md',
        'arc_code.zip', 'whestbench_canonical_research_ledger_20260729_merged.xlsx',
    ]
    input_hashes = {name: sha256_file(args.inputs / name) for name in input_names}
    xk, kmeta = make_kerdock()
    records = []
    for network_id in range(args.network_start, args.network_start + args.networks):
        r = run_one(network_id, xk, args.outdir, args.truth_n, args.chunk, args.overwrite)
        records.append(r)
        print(json.dumps({
            'network': network_id,
            'complete_ratio': r['metrics']['complete_exact']['pooled_ratio'],
            'lower_ratio': r['metrics']['lower_only']['pooled_ratio'],
            'connected_ratio': r['metrics']['connected_only']['pooled_ratio'],
            'rank16_energy': r['spectrum']['rank16_energy'],
            'runtime_seconds': r['runtime_seconds'],
        }), flush=True)

    summary = summarize(records)
    # Pooled downstream spectrum across network-local weighted defect rows.
    weighted = []
    corrections = []
    for r in records:
        with np.load(args.outdir / 'vectors' / r['vectors_file']) as z:
            weighted.append(z['weighted_defect_rows'].copy())
            labels = list(z['correction_labels'])
            corrections.append(z['correction_vectors'][labels.index('complete_exact')].copy())
    pooled_weighted = np.concatenate(weighted, axis=0)
    pooled_s = np.linalg.svd(pooled_weighted, compute_uv=False)
    pooled_energy = np.cumsum(pooled_s * pooled_s) / max(float(np.sum(pooled_s * pooled_s)), 1e-30)
    correction_matrix = np.asarray(corrections)
    centered_correction = correction_matrix - correction_matrix.mean(axis=0)
    correction_s = np.linalg.svd(centered_correction, compute_uv=False)
    correction_energy = np.cumsum(correction_s * correction_s) / max(float(np.sum(correction_s * correction_s)), 1e-30)
    spectrum = {
        'definition': 'Per-network complete anchor defect E-Q, restricted to frozen selected rows and left-scaled by sqrt(||mean crossfit beta_p||_2^2); pooled by vertical concatenation.',
        'pooled_top64_singular_values': pooled_s[:64].tolist(),
        'pooled_energy_curve_first64': pooled_energy[:64].tolist(),
        'pooled_rank16_energy': float(pooled_energy[15]),
        'rank16_model_allowed': bool(pooled_energy[15] >= 0.90),
        'cross_network_final_correction_top24_singular_values': correction_s[:24].tolist(),
        'cross_network_final_correction_rank16_energy': float(correction_energy[min(15, len(correction_energy)-1)]),
    }
    primary = summary['complete_exact']
    lower = summary['lower_only']
    connected = summary['connected_only']
    complete_benefit = max(1.0 - primary['aggregate_pooled_ratio'], 1e-30)
    lower_fraction = (1.0 - lower['aggregate_pooled_ratio']) / complete_benefit
    certificate = {
        'status': 'PASS' if primary['aggregate_pooled_ratio'] <= 0.50 else 'FAIL',
        'primary_numeric_gate': {'threshold': 0.50, 'observed': primary['aggregate_pooled_ratio'], 'passed': primary['aggregate_pooled_ratio'] <= 0.50},
        'preferred_gate': {'threshold': 0.30, 'observed': primary['aggregate_pooled_ratio'], 'passed': primary['aggregate_pooled_ratio'] <= 0.30},
        'confidence_interval': primary['bootstrap_95_interval'],
        'tail_metrics': {'wins': primary['wins'], 'networks': len(records), 'median_ratio': primary['median_ratio'], 'p90_ratio': primary['p90_ratio'], 'worst_ratio': primary['worst_ratio']},
        'lower_only_fraction_of_complete_benefit': float(lower_fraction),
        'lower_only_material_at_20pct': bool(lower_fraction >= 0.20),
        'connected_only_ratio': connected['aggregate_pooled_ratio'],
        'rank16_energy': spectrum['pooled_rank16_energy'],
        'rank16_model_allowed': spectrum['rank16_model_allowed'],
        'closure_scope_on_failure': 'Only the exact frozen 128-sample-row radial-Hermite construction is closed; sparse cubic controls as a family are not closed.',
        'oracle_warning': 'The complete exact anchor uses independent high-sample reference moments and is not deployable.',
    }
    prior = prior_reproduction(args.inputs)
    manifest = {
        'schema_version': 1,
        'title': 'Fresh sparse radial-Hermite exact-anchor validation',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'environment': {
            'python': sys.version,
            'platform': platform.platform(),
            'numpy': np.__version__,
            'torch': torch.__version__,
            'torch_threads': args.threads,
        },
        'input_sha256': input_hashes,
        'prior_reproduction': prior,
        'frozen_construction': {
            'dimension': D, 'depth': DEPTH, 'target_layer_zero_based': TARGET,
            'target_layer_human': TARGET + 1, 'probe_count': PROBES,
            'probe_selector': 'top 128 rows by descending Euclidean row norm of observable same-cloud sample anchor Q; stable NumPy argsort convention; right direction is normalized selected Q row; left direction is selected coordinate basis vector',
            'folds': FOLDS, 'fold_definition': 'np.array_split basis IDs 0..128 into six sequential groups; hold out complete 512-row basis blocks',
            'ridge': RIDGE, 'ridge_normalization': 'gram + 0.1 * max(trace(gram)/128,1e-12) * I',
            'pointwise_feature': 'Phi_ij(h;m)=h_i^2 h_j/rho^2 - d/(rho^2(d+1))*(m_j h_i^2+2 m_i h_i h_j)+2 m_i^2 h_j/(d+1)',
            'complete_exact_anchor': '(raw - diag(M) m^T - 2 diag(m) M + 2 (m^2) mu^T)/(d+1)',
            'component_definitions': {
                'sample_anchor': 'same-cloud radial anchor Q; observable',
                'complete_exact': 'full independent high-sample oracle anchor',
                'lower_only': 'Q plus exact-minus-same-cloud lower recentering defect',
                'connected_only': 'Q plus exact-minus-same-cloud connected-c21 defect',
                'complete_mean_omitted': 'complete anchor with oracle mu replaced by observable sample center m',
                'complete_pair_moments_omitted': 'complete anchor with oracle M replaced by radially rescaled same-cloud pair matrix',
            },
            'kerdock': kmeta,
            'network_generation': 'np.random.default_rng(51000+network_id), 32 independent 256x256 He-Gaussian matrices; row-forward h@W',
            'network_ids': [r['network_id'] for r in records],
            'reference': {'design': 'two independent scrambled Sobol Gaussian halves', 'samples_per_half': args.truth_n, 'chunk': args.chunk},
            'oracle_not_deployable': True,
        },
        'summary': summary,
        'downstream_weighted_spectrum': spectrum,
        'certificate': certificate,
        'records': records,
    }
    manifest_path = args.outdir / 'SPARSE_RADIAL_PROBE_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (args.outdir / 'FROZEN_VALIDATION_RESULTS.json').write_text(json.dumps({'summary': summary, 'records': records}, indent=2))
    (args.outdir / 'COMPONENT_ABLATION.json').write_text(json.dumps({k: summary[k] for k in summary if k != 'baseline'}, indent=2))
    (args.outdir / 'PASS_FAIL_CERTIFICATE.json').write_text(json.dumps(certificate, indent=2))
    np.savez_compressed(args.outdir / 'DOWNSTREAM_WEIGHTED_SPECTRUM.npz', pooled_singular_values=pooled_s, pooled_energy_curve=pooled_energy, correction_singular_values=correction_s, correction_energy_curve=correction_energy)
    print(json.dumps({
        'manifest': str(manifest_path),
        'primary_ratio': primary['aggregate_pooled_ratio'],
        'primary_ci': primary['bootstrap_95_interval'],
        'wins': primary['wins'],
        'worst': primary['worst_ratio'],
        'rank16_energy': spectrum['pooled_rank16_energy'],
        'certificate': certificate['status'],
    }), flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
