#!/usr/bin/env python3
"""Set-level Kerdock coreset screen using basiswise feasibility scores.

The runtime selector is a fixed support-library lookup.  It does not run NNLS,
exchange, herding, or any iterative support optimizer.  Each candidate support
has fixed per-basis quotas.  A low-cost output-coordinate sketch is whitened,
and candidates are scored by whether each basis's full sketch mean can be
represented by the selected rows with small bounded relative reweighting.

Oracle output calibration is used only after selection as the preregistered
same-support diagnostic.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "5")
os.environ.setdefault("OMP_NUM_THREADS", "5")
os.environ.setdefault("MKL_NUM_THREADS", "5")

import numpy as np

import exact_kerdock_coreset_diagnostic as base

D = base.WIDTH
B = base.ALL_BASES
P = base.PAIRS_PER_BASIS
NPAIRS = base.TOTAL_PAIRS


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def fixed_random_library(count: int, quota: np.ndarray, seed: int) -> np.ndarray:
    out = np.empty((count, int(quota.sum())), dtype=np.int32)
    for c in range(count):
        rng = np.random.default_rng(splitmix64(seed + c))
        pos = 0
        for b, q in enumerate(quota):
            local = rng.choice(P, size=int(q), replace=False)
            out[c, pos:pos + q] = b * P + np.sort(local)
            pos += int(q)
    return out


def affine_stratified_library(count: int, quota: np.ndarray, seed: int) -> np.ndarray:
    """Fixed equispaced supports with basis-specific affine permutations."""
    out = np.empty((count, int(quota.sum())), dtype=np.int32)
    odds = np.arange(1, P, 2, dtype=np.int64)
    for c in range(count):
        pos = 0
        for b, q0 in enumerate(quota):
            q = int(q0)
            # Midpoint lattice; all entries are distinct for q=31/32.
            lattice = np.floor((np.arange(q) + 0.5) * P / q).astype(np.int64)
            h = splitmix64(seed + 1315423911 * c + 2654435761 * b)
            a = int(odds[h % len(odds)])
            shift = int((h >> 16) % P)
            local = (a * lattice + shift) % P
            if np.unique(local).size != q:
                raise RuntimeError("affine support collision")
            out[c, pos:pos + q] = b * P + np.sort(local)
            pos += q
    return out


def pilot_output_coordinates(a31: np.ndarray, w32: np.ndarray, q: int) -> np.ndarray:
    idx = base.pilot_rows(8)
    z = a31[idx].astype(np.float64) @ w32.astype(np.float64)
    a = np.maximum(z, 0.0)
    # Prefer coordinates with both variance and gate uncertainty.
    p = (z > 0).mean(axis=0)
    score = a.var(axis=0) * (0.25 + 3.0 * p * (1.0 - p))
    return np.argsort(score)[-q:].astype(np.int32)


def whiten(features: np.ndarray, rel_floor: float = 1e-4) -> tuple[np.ndarray, dict[str, Any]]:
    x = features.astype(np.float64)
    x -= x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / x.shape[0]
    eig, vec = np.linalg.eigh(cov)
    top = float(max(eig[-1], 1e-30))
    keep = eig > rel_floor * top
    if not np.any(keep):
        keep[-1] = True
    z = x @ (vec[:, keep] / np.sqrt(eig[keep])[None, :])
    return z.astype(np.float32), {
        "input_dim": int(features.shape[1]),
        "kept_dim": int(keep.sum()),
        "eig_min_kept": float(eig[keep].min()),
        "eig_max": float(eig[-1]),
    }


def score_library(
    z: np.ndarray,
    library: np.ndarray,
    quota: np.ndarray,
    batch: int = 16,
    variance_floor: float = 0.05,
    span_penalty: float = 0.10,
) -> dict[str, np.ndarray]:
    """Score fixed supports with basiswise mean/span diagnostics.

    For one whitened feature within one basis, d^2 / var(selected) is the
    diagonal approximation to the minimum squared relative-weight adjustment
    needed to move the selected mean to the full-basis mean.  This directly
    targets bounded positive reweightability under per-basis mass constraints.
    """
    full = z.reshape(B, P, z.shape[1]).astype(np.float64)
    mu = full.mean(axis=1)
    var_full = full.var(axis=1) + 1e-8

    ctot = library.shape[0]
    global_mean = np.empty(ctot, dtype=np.float64)
    basis_mean = np.empty(ctot, dtype=np.float64)
    basis_effort = np.empty(ctot, dtype=np.float64)
    basis_span = np.empty(ctot, dtype=np.float64)

    # Precompute per-basis slices because quotas are 31/32.
    slices: list[slice] = []
    pos = 0
    for q0 in quota:
        q = int(q0)
        slices.append(slice(pos, pos + q))
        pos += q

    for lo in range(0, ctot, batch):
        hi = min(ctot, lo + batch)
        for c in range(lo, hi):
            idx = library[c]
            weighted_global = np.zeros(z.shape[1], dtype=np.float64)
            bm = 0.0
            be = 0.0
            bs = 0.0
            for b, sl in enumerate(slices):
                s = z[idx[sl]].astype(np.float64)
                ms = s.mean(axis=0)
                vs = s.var(axis=0)
                d = ms - mu[b]
                weighted_global += d / B
                bm += float(np.mean(d * d))
                denom = vs + variance_floor * var_full[b] + 1e-8
                be += float(np.mean((d * d) / denom))
                ratio = (vs + 1e-8) / var_full[b]
                bs += float(np.mean(np.log(np.clip(ratio, 1e-4, 1e4)) ** 2))
            global_mean[c] = float(weighted_global @ weighted_global / z.shape[1])
            basis_mean[c] = bm / B
            basis_effort[c] = be / B
            basis_span[c] = bs / B

    return {
        "global_mean": global_mean,
        "basis_mean": basis_mean,
        "basis_effort": basis_effort,
        "basis_effort_span": basis_effort + span_penalty * basis_span,
        "basis_span": basis_span,
    }


def oracle_diagnostic(
    outputs: np.ndarray,
    selection: np.ndarray,
    quota: np.ndarray,
    full_mean: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    oracle_features = base.standardize(outputs)
    weights, info = base.calibrated_weights(
        oracle_features[selection], selection, quota
    )
    mse = base.added_mse(outputs, selection, weights, full_mean)
    return mse, info


def make_selection_result(
    name: str,
    library: np.ndarray,
    score_name: str,
    scores: dict[str, np.ndarray],
    outputs: np.ndarray,
    quota: np.ndarray,
    full_mean: np.ndarray,
) -> dict[str, Any]:
    ci = int(np.argmin(scores[score_name]))
    sel = library[ci]
    mse, info = oracle_diagnostic(outputs, sel, quota, full_mean)
    return {
        "name": name,
        "candidate_index": ci,
        "score_name": score_name,
        "score": float(scores[score_name][ci]),
        "same_support_oracle_added_mse": mse,
        "pass_1.1e-8": bool(mse <= 1.1e-8),
        "pass_2.2e-8": bool(mse <= 2.2e-8),
        "oracle_weight_info": info,
        "support_sha256_like": int(np.bitwise_xor.reduce(sel.astype(np.int64) * 1000003)),
    }


def run_network(
    seed: int,
    chirps: np.ndarray,
    rotation: np.ndarray,
    qcoords: int,
    library_count: int,
    selected_pairs: int,
    library_seed: int,
    compute_oracle_support: bool,
) -> dict[str, Any]:
    started = time.time()
    weights = base.gen_weights(seed)
    anchor = base.propagate_to_anchor(weights, chirps, rotation, 28)
    t_anchor = time.time()

    h = anchor
    for w in weights[28:31]:
        h = base.relu(h @ w)
    a31 = h
    coords = pilot_output_coordinates(a31, weights[31], qcoords)
    sketch_rows = base.relu(a31 @ weights[31][:, coords])
    sketch_pairs = base.pair_average(sketch_rows)
    z, whiten_info = whiten(sketch_pairs)
    t_sketch = time.time()

    # Full outputs are research labels only; a runtime selector would not form
    # unselected final coordinates.
    final_rows = base.relu(a31 @ weights[31])
    outputs = base.pair_average(final_rows)
    full_mean = outputs.mean(axis=0)
    t_full = time.time()

    quota = base.quotas(selected_pairs)
    random_lib = fixed_random_library(library_count, quota, library_seed)
    affine_lib = affine_stratified_library(library_count, quota, library_seed + 99173)
    union_lib = np.concatenate([random_lib, affine_lib], axis=0)

    random_scores = score_library(z, random_lib, quota)
    affine_scores = score_library(z, affine_lib, quota)
    union_scores = {
        key: np.concatenate([random_scores[key], affine_scores[key]])
        for key in random_scores
    }
    t_score = time.time()

    methods = []
    # Control: old-style global discrepancy over a fixed library.
    methods.append(make_selection_result(
        "union_global_mean_control", union_lib, "global_mean", union_scores,
        outputs, quota, full_mean
    ))
    methods.append(make_selection_result(
        "random_library_basis_effort", random_lib, "basis_effort", random_scores,
        outputs, quota, full_mean
    ))
    methods.append(make_selection_result(
        "affine_library_basis_effort", affine_lib, "basis_effort", affine_scores,
        outputs, quota, full_mean
    ))
    methods.append(make_selection_result(
        "union_basis_effort", union_lib, "basis_effort", union_scores,
        outputs, quota, full_mean
    ))
    methods.append(make_selection_result(
        "union_basis_effort_span", union_lib, "basis_effort_span", union_scores,
        outputs, quota, full_mean
    ))

    # One fixed random support control, evaluated with the same oracle weights.
    fixed_mse, fixed_info = oracle_diagnostic(
        outputs, random_lib[0], quota, full_mean
    )
    methods.append({
        "name": "fixed_random_control",
        "candidate_index": 0,
        "score_name": None,
        "score": None,
        "same_support_oracle_added_mse": fixed_mse,
        "pass_1.1e-8": bool(fixed_mse <= 1.1e-8),
        "pass_2.2e-8": bool(fixed_mse <= 2.2e-8),
        "oracle_weight_info": fixed_info,
    })

    oracle_support = None
    if compute_oracle_support:
        of = base.standardize(outputs)
        osel = base.best_multistart(of, quota, 4, seed + 300)
        osel, iters = base.exchange_selection(of, osel, quota, 32)
        ow, oinfo = base.calibrated_weights(of[osel], osel, quota)
        oracle_support = {
            "same_support_oracle_added_mse": base.added_mse(outputs, osel, ow, full_mean),
            "iterations": iters,
            "oracle_weight_info": oinfo,
        }

    # Arithmetic count relative to the four-layer tail from layer 28.
    nrows = base.TOTAL_ROWS
    mrows = 2 * selected_pairs
    full_layer_flops = 2 * nrows * D * D
    sketch_flops = 2 * nrows * D * qcoords
    selected_final_flops = 2 * mrows * D * D
    dense_final_flops = full_layer_flops
    score_ops = 2 * union_lib.shape[0] * selected_pairs * z.shape[1]
    whitening_ops = 2 * NPAIRS * qcoords * qcoords
    candidate_final_cost = sketch_flops + selected_final_flops + score_ops + whitening_ops
    saved_final = dense_final_flops - candidate_final_cost

    return {
        "seed": seed,
        "qcoords": qcoords,
        "selected_coords": coords.tolist(),
        "whiten": whiten_info,
        "library_count_each": library_count,
        "methods": methods,
        "oracle_support": oracle_support,
        "timing_seconds": {
            "anchor_prefix": t_anchor - started,
            "sketch": t_sketch - t_anchor,
            "research_full_output": t_full - t_sketch,
            "library_scoring": t_score - t_full,
            "oracle_diagnostics": time.time() - t_score,
            "total": time.time() - started,
        },
        "selector_cost": {
            "full_final_layer_flops": int(dense_final_flops),
            "sketch_flops": int(sketch_flops),
            "selected_final_flops": int(selected_final_flops),
            "score_ops_approx": int(score_ops),
            "whitening_ops_approx": int(whitening_ops),
            "candidate_final_cost_approx": int(candidate_final_cost),
            "net_final_layer_flops_saved_approx": int(saved_final),
            "candidate_vs_dense_final_fraction": float(candidate_final_cost / dense_final_flops),
        },
    }


def parse_seeds(value: str) -> list[int]:
    if ":" in value:
        a, b = value.split(":", 1)
        return list(range(int(a), int(b)))
    return [int(x) for x in value.split(",") if x.strip()]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    names = [m["name"] for m in records[0]["methods"]]
    out: dict[str, Any] = {}
    for name in names:
        vals = np.array([
            next(m for m in r["methods"] if m["name"] == name)["same_support_oracle_added_mse"]
            for r in records
        ], dtype=np.float64)
        out[name] = {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "worst": float(vals.max()),
            "passes_1.1e-8": int(np.sum(vals <= 1.1e-8)),
            "passes_2.2e-8": int(np.sum(vals <= 2.2e-8)),
            "values": vals.tolist(),
        }
    if all(r["oracle_support"] is not None for r in records):
        vals = np.array([r["oracle_support"]["same_support_oracle_added_mse"] for r in records])
        out["full_oracle_support"] = {
            "mean": float(vals.mean()), "median": float(np.median(vals)),
            "worst": float(vals.max()),
            "passes_1.1e-8": int(np.sum(vals <= 1.1e-8)),
            "passes_2.2e-8": int(np.sum(vals <= 2.2e-8)),
            "values": vals.tolist(),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", type=Path, required=True)
    ap.add_argument("--seeds", default="63998")
    ap.add_argument("--qcoords", type=int, default=128)
    ap.add_argument("--library-count", type=int, default=64)
    ap.add_argument("--selected-pairs", type=int, default=4096)
    ap.add_argument("--library-seed", type=int, default=202607295)
    ap.add_argument("--oracle-support", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    with np.load(args.asset, allow_pickle=False) as asset:
        chirps = asset["chirps"].astype(np.float32)
        rotation = asset["rotation"].astype(np.float32)

    records = []
    for i, seed in enumerate(parse_seeds(args.seeds), 1):
        rec = run_network(seed, chirps, rotation, args.qcoords,
                          args.library_count, args.selected_pairs,
                          args.library_seed, args.oracle_support)
        records.append(rec)
        payload = {"config": vars(args) | {"asset": str(args.asset), "output": str(args.output)},
                   "records": records, "summary": summarize(records)}
        args.output.write_text(json.dumps(payload, indent=2))
        best = min(rec["methods"], key=lambda x: x["same_support_oracle_added_mse"])
        print(f"[{i}] seed={seed} best={best['name']} {best['same_support_oracle_added_mse']:.3e} "
              f"fixed={next(m for m in rec['methods'] if m['name']=='fixed_random_control')['same_support_oracle_added_mse']:.3e} "
              f"time={rec['timing_seconds']['total']:.1f}s", flush=True)
    print(json.dumps(summarize(records), indent=2))


if __name__ == "__main__":
    main()
