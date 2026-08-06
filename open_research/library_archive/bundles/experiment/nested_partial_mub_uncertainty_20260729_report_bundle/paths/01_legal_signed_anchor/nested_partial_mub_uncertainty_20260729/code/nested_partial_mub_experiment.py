#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frozen_reference_impl as fr

D = fr.D
TARGET = fr.TARGET
ROWS = fr.ROWS_PER_BASIS
RHO = fr.chi_mean(D)
MAIN_IDS = np.r_[np.arange(111), 128].astype(np.int64)
PREFIXES = (2, 4, 8, 12, 17)
ALPHA = 0.20
FOLDS = 6
RIDGE = 0.1


def sha256_array(x: np.ndarray) -> str:
    y = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(y.dtype).encode())
    h.update(str(y.shape).encode())
    h.update(y.tobytes())
    return h.hexdigest()


def apply_input_rotation(ws: list[torch.Tensor], seed: int | None) -> list[torch.Tensor]:
    if seed is None or seed < 0:
        return ws
    q = fr.haar_rotation(seed)
    return [torch.from_numpy(q @ ws[0].numpy())] + ws[1:]


def unrotated_chirp_blocks(ids: np.ndarray) -> np.ndarray:
    h = fr.walsh_hadamard() / math.sqrt(D)
    blocks = []
    for u in ids:
        basis = (h * fr.kerdock_chirp(int(u))[None, :]).astype(np.float32)
        block = np.concatenate((basis, -basis), axis=0) * np.float32(RHO)
        blocks.append(block)
    return np.ascontiguousarray(np.concatenate(blocks, axis=0))


BASE_CHIRP17 = unrotated_chirp_blocks(np.arange(17))


def companion_target_blocks(ws_rot: list[torch.Tensor], companion_seed: int) -> np.ndarray:
    # Equivalent to first_activation_blocks in the surviving package:
    # base chirp-Walsh blocks are right-rotated, then propagated to layer 29.
    q = fr.haar_rotation(companion_seed)
    pts = (BASE_CHIRP17 @ q).astype(np.float32, copy=False)
    x = torch.from_numpy(pts)
    with torch.no_grad():
        for layer, w in enumerate(ws_rot[: TARGET + 1]):
            x = torch.relu(x @ w)
    return x.double().numpy().reshape(17, ROWS, D)


def fit_crossfit_subset(X: np.ndarray, Y: np.ndarray, block_ids: np.ndarray) -> dict[str, Any]:
    unique = np.asarray(sorted(np.unique(block_ids)), dtype=np.int64)
    groups = [np.asarray(g, dtype=np.int64) for g in np.array_split(unique, FOLDS)]
    total_sx = X.sum(axis=0)
    total_sy = Y.sum(axis=0)
    total_xx = X.T @ X
    total_xy = X.T @ Y
    betas, fold_y_mean, fold_x_mean, fold_sizes, ridge_scales = [], [], [], [], []
    for group in groups:
        te = np.isin(block_ids, group)
        xe, ye = X[te], Y[te]
        nte = len(xe)
        ntr = len(X) - nte
        sx_e, sy_e = xe.sum(axis=0), ye.sum(axis=0)
        xx_e, xy_e = xe.T @ xe, xe.T @ ye
        sx, sy = total_sx - sx_e, total_sy - sy_e
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
        "groups": groups,
        "betas": np.asarray(betas),
        "fold_y_mean": np.asarray(fold_y_mean),
        "fold_x_mean": np.asarray(fold_x_mean),
        "fold_sizes": np.asarray(fold_sizes, dtype=np.int64),
        "ridge_scales": np.asarray(ridge_scales),
    }


def estimate(fit: dict[str, Any], anchor: np.ndarray) -> np.ndarray:
    fold_pred = fit["fold_y_mean"] - np.einsum(
        "fp,fpd->fd", fit["fold_x_mean"] - anchor[None, :], fit["betas"]
    )
    weights = fit["fold_sizes"] / fit["fold_sizes"].sum()
    return weights @ fold_pred


def lower_anchor(
    center: np.ndarray,
    target_mean: np.ndarray,
    pair_scaled: np.ndarray,
    indices: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    diag_second = np.diag(pair_scaled)
    rowdir_second = np.sum(pair_scaled[indices] * directions, axis=1)
    d = target_mean - center
    vi_d = directions @ d
    vi_mean = directions @ target_mean
    i = indices
    return (
        diag_second[i] * vi_d
        + 2.0 * d[i] * rowdir_second
        + 2.0 * (center[i] * center[i] - target_mean[i] * target_mean[i]) * vi_mean
    ) / (D + 1.0)


def correction_from_mean(
    target_mean: np.ndarray,
    *,
    m: np.ndarray,
    pair_scaled: np.ndarray,
    indices: np.ndarray,
    directions: np.ndarray,
    fit: dict[str, Any],
    sample_anchor: np.ndarray,
    base: np.ndarray,
    alpha: float = ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    a = lower_anchor(m, target_mean, pair_scaled, indices, directions)
    pred = estimate(fit, sample_anchor + alpha * a)
    return a, pred - base


def mse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(x, float) - np.asarray(y, float))))


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-300))


def process_rotation(
    network_id: int,
    rotation_index: int,
    rotation_seed: int | None,
    ws: list[torch.Tensor],
    xk: np.ndarray,
    truth_y: np.ndarray,
    truth_mu: np.ndarray,
    truth_M: np.ndarray,
    ref1_y: np.ndarray,
    ref2_y: np.ndarray,
    out_vectors: Path,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    ws_rot = apply_input_rotation(ws, rotation_seed)
    xkt = torch.from_numpy(xk)
    with torch.no_grad():
        hk, yk = fr.forward_target_final(xkt, ws_rot)
    Hfull = hk.double().numpy()
    Yfull = yk.double().numpy()
    basefull = Yfull.mean(axis=0)
    bid_full = np.repeat(np.arange(fr.N_BASES), ROWS)
    take = np.isin(bid_full, MAIN_IDS)
    H = Hfull[take]
    Y = Yfull[take]
    bid = bid_full[take]
    m = H.mean(axis=0)
    pair_scaled = (D / (RHO * RHO)) * (H.T @ H / len(H))
    Q = fr.sample_anchor_matrix(H, m, RHO)
    indices, directions = fr.sample_row_probes(Q)
    X = fr.radial_features_sample_rows(H, m, indices, directions, RHO)
    fit = fit_crossfit_subset(X, Y, bid)
    sample_anchor = X.mean(axis=0)
    y0 = estimate(fit, sample_anchor)

    comp_seed = 1011 + 100 * network_id
    C = companion_target_blocks(ws_rot, comp_seed)
    ext_means = C.mean(axis=1)
    cum = np.cumsum(ext_means, axis=0)
    prefix_means = cum / np.arange(1, 18)[:, None]

    anchors = []
    corrections = []
    for mu in prefix_means:
        a, c = correction_from_mean(
            mu, m=m, pair_scaled=pair_scaled, indices=indices,
            directions=directions, fit=fit, sample_anchor=sample_anchor, base=y0
        )
        anchors.append(a)
        corrections.append(c)
    anchors = np.asarray(anchors)
    corrections = np.asarray(corrections)

    # Original-vs-external paired difference for the first two blocks; both
    # sides are already present in the 112+17 package, so this adds no paths.
    H_blocks = H.reshape(len(MAIN_IDS), ROWS, D)
    orig_lookup = {int(b): H_blocks[j].mean(axis=0) for j, b in enumerate(MAIN_IDS)}
    orig2 = 0.5 * (orig_lookup[0] + orig_lookup[1])
    ext2 = ext_means[:2].mean(axis=0)
    paired_delta = (2.0 / 129.0) * (orig2 - ext2)
    paired_mean = m - paired_delta
    paired_anchor, paired_corr = correction_from_mean(
        paired_mean, m=m, pair_scaled=pair_scaled, indices=indices,
        directions=directions, fit=fit, sample_anchor=sample_anchor, base=y0,
        alpha=1.0,
    )

    # Robust same-budget alternatives.
    sorted_ext = np.sort(ext_means, axis=0)
    trimmed_mean = sorted_ext[2:-2].mean(axis=0)
    group_means = np.stack([
        ext_means[0:4].mean(0), ext_means[4:8].mean(0),
        ext_means[8:12].mean(0), ext_means[12:17].mean(0)
    ])
    mom_mean = np.median(group_means, axis=0)
    robust = {}
    for name, mu in (("trimmed", trimmed_mean), ("median_of_groups", mom_mean)):
        a, c = correction_from_mean(
            mu, m=m, pair_scaled=pair_scaled, indices=indices,
            directions=directions, fit=fit, sample_anchor=sample_anchor, base=y0
        )
        robust[name] = (a, c)

    # Leave-one-basis influence around c17.
    loo_corr = []
    for j in range(17):
        mu = (ext_means.sum(axis=0) - ext_means[j]) / 16.0
        _, c = correction_from_mean(
            mu, m=m, pair_scaled=pair_scaled, indices=indices,
            directions=directions, fit=fit, sample_anchor=sample_anchor, base=y0
        )
        loo_corr.append(c)
    loo_corr = np.asarray(loo_corr)

    # Exact lower-order oracle for mechanism diagnostics only.
    truth_pair_rowdir = np.sum(truth_M[indices] * directions, axis=1)
    oracle_anchor = lower_anchor(m, truth_mu, truth_M, indices, directions)
    oracle_pred = estimate(fit, sample_anchor + oracle_anchor)
    oracle_corr = oracle_pred - y0

    full_base_mse = mse(basefull, truth_y)
    reduced_mse = mse(y0, truth_y)
    c17 = corrections[16]
    cand = y0 + c17
    cand_mse = mse(cand, truth_y)
    ideal = truth_y - y0
    err = y0 - truth_y
    correction_metrics = {
        "error_correction_inner_product": float(np.dot(err, c17) / D),
        "ideal_correction_inner_product": float(np.dot(ideal, c17) / D),
        "correction_norm": float(np.linalg.norm(c17)),
        "correction_cosine": cosine(c17, ideal),
        "oracle_lower_correction_cosine": cosine(c17, oracle_corr),
    }

    prefix_metrics = {}
    for k in PREFIXES:
        c = corrections[k - 1]
        p = y0 + c
        prefix_metrics[str(k)] = {
            "mse": mse(p, truth_y),
            "ratio_to_full_baseline": mse(p, truth_y) / max(full_base_mse, 1e-300),
            "correction_norm": float(np.linalg.norm(c)),
            "correction_cosine": cosine(c, ideal),
        }

    vector_path = out_vectors / f"network_{network_id:04d}_rotation_{rotation_index}.npz"
    np.savez_compressed(
        vector_path,
        network_id=np.asarray(network_id), rotation_index=np.asarray(rotation_index),
        rotation_seed=np.asarray(-1 if rotation_seed is None else rotation_seed),
        truth_y=truth_y, ref1_y=ref1_y, ref2_y=ref2_y,
        basefull=basefull, reduced_base=y0, sample_center=m,
        probe_indices=indices, probe_directions=directions,
        sample_anchor=sample_anchor, pair_scaled=pair_scaled,
        fit_betas=fit["betas"], fit_fold_sizes=fit["fold_sizes"],
        ext_block_means=ext_means,
        anchors=anchors, corrections=corrections,
        paired_anchor=paired_anchor, paired_correction=paired_corr,
        robust_names=np.asarray(list(robust)),
        robust_anchors=np.stack([robust[n][0] for n in robust]),
        robust_corrections=np.stack([robust[n][1] for n in robust]),
        loo_corrections=loo_corr,
        oracle_anchor=oracle_anchor, oracle_correction=oracle_corr,
    )
    record = {
        "network_id": network_id,
        "rotation_index": rotation_index,
        "rotation_seed": -1 if rotation_seed is None else rotation_seed,
        "companion_seed": comp_seed,
        "vectors_file": vector_path.name,
        "vectors_sha256": fr.sha256_file(vector_path),
        "main_ids": MAIN_IDS.tolist(),
        "prefixes": list(PREFIXES),
        "alpha": ALPHA,
        "full_baseline_mse": full_base_mse,
        "full_baseline_unbiased_mse": fr.unbiased_mse(basefull, ref1_y, ref2_y),
        "reduced_base_mse": reduced_mse,
        "reduced_ratio": reduced_mse / max(full_base_mse, 1e-300),
        "c17_mse": cand_mse,
        "c17_ratio": cand_mse / max(full_base_mse, 1e-300),
        "prefix_metrics": prefix_metrics,
        "correction_metrics": correction_metrics,
        "paired": {
            "correction_norm": float(np.linalg.norm(paired_corr)),
            "cosine_with_c17": cosine(paired_corr, c17),
            "cosine_with_ideal": cosine(paired_corr, ideal),
            "norm_ratio_to_c17": float(np.linalg.norm(paired_corr) / max(np.linalg.norm(c17), 1e-300)),
        },
        "jackknife": {
            "max_relative_influence": float(np.max(np.linalg.norm(loo_corr - c17[None, :], axis=1)) / max(np.linalg.norm(c17), 1e-300)),
            "median_relative_influence": float(np.median(np.linalg.norm(loo_corr - c17[None, :], axis=1)) / max(np.linalg.norm(c17), 1e-300)),
        },
        "oracle_lower_ratio": mse(oracle_pred, truth_y) / max(full_base_mse, 1e-300),
        "reference_noise_mse": float(0.25 * mse(ref1_y, ref2_y)),
        "reference_noise_fraction": float(0.25 * mse(ref1_y, ref2_y) / max(full_base_mse, 1e-300)),
        "runtime_seconds": float(time.perf_counter() - t0),
        "peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    del Hfull, Yfull, H, Y, X, Q, C, hk, yk, xkt
    gc.collect()
    return record


def run_base(network_id: int, rotation_seeds: list[int], truth_n: int, chunk: int, out: Path, overwrite: bool) -> list[dict[str, Any]]:
    out_records = out / "results" / "records"
    out_vectors = out / "results" / "vectors"
    out_records.mkdir(parents=True, exist_ok=True)
    out_vectors.mkdir(parents=True, exist_ok=True)
    ws, weight_hash, weight_seed = fr.make_weights(network_id)
    ref1 = fr.stream_reference(ws, truth_n, 71_000_000 + 2 * network_id, chunk)
    ref2 = fr.stream_reference(ws, truth_n, 71_000_001 + 2 * network_id, chunk)
    truth_y = 0.5 * (ref1["y"] + ref2["y"])
    truth_mu = 0.5 * (ref1["mu"] + ref2["mu"])
    truth_M = 0.5 * (ref1["M"] + ref2["M"])
    xk, meta = fr.make_kerdock()
    records = []
    for ri, seed in enumerate(rotation_seeds):
        path = out_records / f"network_{network_id:04d}_rotation_{ri}.json"
        if path.exists() and not overwrite:
            records.append(json.loads(path.read_text()))
            continue
        rseed = None if seed < 0 else seed + 1009 * network_id
        rec = process_rotation(
            network_id, ri, rseed, ws, xk, truth_y, truth_mu, truth_M,
            ref1["y"], ref2["y"], out_vectors
        )
        rec.update({
            "weight_seed": weight_seed, "weight_sha256": weight_hash,
            "truth_n_per_half": truth_n,
            "reference_seeds": [71_000_000 + 2 * network_id, 71_000_001 + 2 * network_id],
            "kerdock_sha256": meta["points_sha256"],
        })
        path.write_text(json.dumps(rec, indent=2) + "\n")
        records.append(rec)
        print(json.dumps({"network": network_id, "rotation": ri, "ratio": rec["c17_ratio"], "runtime": rec["runtime_seconds"]}), flush=True)
    return records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--network", type=int, required=True)
    p.add_argument("--rotation-seeds", nargs="+", type=int, default=[-1, 7001, 7003])
    p.add_argument("--truth-n", type=int, default=16384)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    run_base(args.network, args.rotation_seeds, args.truth_n, args.chunk, args.out, args.overwrite)


if __name__ == "__main__":
    main()
