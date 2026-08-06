#!/usr/bin/env python3
"""Train the bounded x1/x1a covariance-closure model on the 10k joint-feature corpus.

The script deliberately uses MLP-level train/validation/test splits. Pair rows from
one MLP never cross a split. The minimal default feature set is exchange invariant:

    layer, a_i+a_j, a_i*a_j, |a_i-a_j|, rho_ij, x1_ij, (a_i-a_j)*x1a_ij

This encodes the synthetic ablation result that x1 and x1a carry nearly all of the
useful joint information while enforcing i<->j symmetry.

Required local files from keenanpepper/whestbench-relu-mlp-jointfeats-10k:
    a_train.npy, rho_train.npy, rn_train.npy, x1_train.npy, x1a_train.npy

Optional contraction evaluation also uses, from whestbench-relu-mlp-moments-10k:
    weights_train.npy, data_train.npz
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

WIDTH = 256
DEPTH = 32
PAIRS = WIDTH * (WIDTH - 1) // 2
IU, JU = np.triu_indices(WIDTH, 1)


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray


def parse_layers(text: str) -> np.ndarray:
    if text.strip().lower() == "all":
        return np.arange(31, dtype=np.int64)
    out = np.array(sorted({int(x) for x in text.split(",") if x.strip()}), dtype=np.int64)
    if out.size == 0 or np.any((out < 0) | (out >= 31)):
        raise ValueError("layers must be 'all' or comma-separated indices in [0, 30]")
    return out


def require_files(root: Path, names: Sequence[str]) -> dict[str, Path]:
    paths = {name: root / name for name in names}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n  " + "\n  ".join(missing))
    return paths


def make_split(n: int, n_train: int, n_valid: int, n_test: int, seed: int) -> Split:
    if min(n_train, n_valid, n_test) <= 0 or n_train + n_valid + n_test > n:
        raise ValueError("invalid split sizes")
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)[: n_train + n_valid + n_test]
    return Split(p[:n_train], p[n_train:n_train+n_valid], p[n_train+n_valid:])


def invariant_features(layer: int, a: np.ndarray, rho: np.ndarray,
                       x1: np.ndarray, x1a: np.ndarray, pair_idx: np.ndarray) -> np.ndarray:
    i = IU[pair_idx]
    j = JU[pair_idx]
    ai = np.asarray(a[i], dtype=np.float32)
    aj = np.asarray(a[j], dtype=np.float32)
    d = ai - aj
    return np.column_stack((
        np.full(pair_idx.size, (layer + 1) / DEPTH, dtype=np.float32),
        ai + aj,
        ai * aj,
        np.abs(d),
        np.asarray(rho[pair_idx], dtype=np.float32),
        np.asarray(x1[pair_idx], dtype=np.float32),
        d * np.asarray(x1a[pair_idx], dtype=np.float32),
    )).astype(np.float32, copy=False)


def sample_rows(arrays: dict[str, np.ndarray], networks: np.ndarray, layers: np.ndarray,
                rows_per_net_layer: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = len(networks) * len(layers) * rows_per_net_layer
    X = np.empty((total, 7), dtype=np.float32)
    y = np.empty(total, dtype=np.float32)
    groups = np.empty(total, dtype=np.int32)
    cursor = 0
    for ordinal, net in enumerate(networks):
        for layer in layers:
            idx = rng.choice(PAIRS, size=rows_per_net_layer, replace=False)
            sl = slice(cursor, cursor + rows_per_net_layer)
            X[sl] = invariant_features(
                int(layer), arrays["a"][net, layer], arrays["rho"][net, layer],
                arrays["x1"][net, layer], arrays["x1a"][net, layer], idx,
            )
            y[sl] = arrays["rn"][net, layer, idx]
            groups[sl] = ordinal
            cursor += rows_per_net_layer
        if (ordinal + 1) % 50 == 0 or ordinal + 1 == len(networks):
            print(json.dumps({"loaded_networks": ordinal + 1, "rows": cursor}), flush=True)
    return X, y, groups


def evaluate_pairs(model, arrays: dict[str, np.ndarray], networks: np.ndarray,
                   layers: np.ndarray, rows_per_net_layer: int, seed: int) -> dict:
    X, y, groups = sample_rows(arrays, networks, layers, rows_per_net_layer, seed)
    pred = model.predict(X).astype(np.float64)
    y64 = y.astype(np.float64)
    base_mse = float(np.mean(y64 * y64))
    model_mse = float(np.mean((pred - y64) ** 2))
    per_net = []
    for g in range(len(networks)):
        mask = groups == g
        bm = float(np.mean(y64[mask] ** 2))
        mm = float(np.mean((pred[mask] - y64[mask]) ** 2))
        per_net.append({"network": int(networks[g]), "base_mse": bm, "model_mse": mm,
                        "gain": bm / max(mm, 1e-30)})
    return {
        "rows": int(len(y)),
        "base_mse": base_mse,
        "model_mse": model_mse,
        "gain": base_mse / max(model_mse, 1e-30),
        "r2": float(r2_score(y64, pred)),
        "fraction_networks_improved": float(np.mean([r["model_mse"] < r["base_mse"] for r in per_net])),
        "per_network": per_net,
    }


def contraction_eval(model, arrays: dict[str, np.ndarray], networks: np.ndarray,
                     layers: np.ndarray, weights_path: Path, moments_path: Path,
                     max_networks: int) -> dict:
    """Evaluate only the variance error caused by the learned residual closure.

    The target rn is normalized by sigma_i sigma_j. This reconstructs the residual
    covariance error and contracts it through the actual next-layer weights. It does
    not claim to be a full free-rollout score; it is the correct one-step metric.
    """
    W = np.load(weights_path, mmap_mode="r")
    moments = np.load(moments_path)
    pre_var = moments["pre_var"]
    rows = []
    for net in networks[:max_networks]:
        for layer in layers:
            if layer >= 30:
                continue
            a = arrays["a"][net, layer]
            rho = arrays["rho"][net, layer]
            x1 = arrays["x1"][net, layer]
            x1a = arrays["x1a"][net, layer]
            all_idx = np.arange(PAIRS, dtype=np.int64)
            pred = model.predict(invariant_features(int(layer), a, rho, x1, x1a, all_idx))
            truth = np.asarray(arrays["rn"][net, layer], dtype=np.float64)
            sigma = np.sqrt(np.maximum(np.asarray(pre_var[net, layer], dtype=np.float64), 1e-20))
            scale = sigma[IU] * sigma[JU]
            def matrix_from_pair(v: np.ndarray) -> np.ndarray:
                M = np.zeros((WIDTH, WIDTH), dtype=np.float64)
                q = np.asarray(v, dtype=np.float64) * scale
                M[IU, JU] = q
                M[JU, IU] = q
                return M
            E0 = matrix_from_pair(-truth)
            E1 = matrix_from_pair(pred - truth)
            w = np.asarray(W[net, layer + 1], dtype=np.float64)
            dv0 = np.diag(w.T @ E0 @ w)
            dv1 = np.diag(w.T @ E1 @ w)
            denom = np.maximum(np.asarray(pre_var[net, layer + 1], dtype=np.float64), 1e-12)
            r0 = float(np.sqrt(np.mean((dv0 / denom) ** 2)))
            r1 = float(np.sqrt(np.mean((dv1 / denom) ** 2)))
            rows.append({"network": int(net), "layer": int(layer), "base_rms": r0,
                         "model_rms": r1, "gain": r0 / max(r1, 1e-30)})
            print(json.dumps(rows[-1]), flush=True)
    b = np.array([r["base_rms"] for r in rows])
    m = np.array([r["model_rms"] for r in rows])
    return {"rows": rows, "base_rms_mean": float(b.mean()), "model_rms_mean": float(m.mean()),
            "gain": float(b.mean() / max(m.mean(), 1e-30)),
            "fraction_improved": float(np.mean(m < b))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--train-networks", type=int, default=8000)
    ap.add_argument("--valid-networks", type=int, default=1000)
    ap.add_argument("--test-networks", type=int, default=1000)
    ap.add_argument("--train-rows-per-net-layer", type=int, default=8)
    ap.add_argument("--eval-rows-per-net-layer", type=int, default=64)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--max-leaf-nodes", type=int, default=127)
    ap.add_argument("--min-samples-leaf", type=int, default=100)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--moments", type=Path)
    ap.add_argument("--contraction-networks", type=int, default=20)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = require_files(args.data_dir, ["a_train.npy", "rho_train.npy", "rn_train.npy",
                                             "x1_train.npy", "x1a_train.npy"])
    arrays = {k: np.load(paths[f"{k}_train.npy"], mmap_mode="r") for k in ("a", "rho", "rn", "x1", "x1a")}
    n = arrays["a"].shape[0]
    layers = parse_layers(args.layers)
    split = make_split(n, args.train_networks, args.valid_networks, args.test_networks, args.seed)

    print(json.dumps({"stage": "load_train", "networks": len(split.train), "layers": layers.tolist()}), flush=True)
    Xtr, ytr, _ = sample_rows(arrays, split.train, layers, args.train_rows_per_net_layer, args.seed + 1)
    model = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=args.max_iter, learning_rate=0.05,
        max_leaf_nodes=args.max_leaf_nodes, min_samples_leaf=args.min_samples_leaf,
        l2_regularization=5.0, early_stopping=True, validation_fraction=0.08,
        n_iter_no_change=30, random_state=args.seed,
    )
    t0 = time.time(); model.fit(Xtr, ytr); train_seconds = time.time() - t0
    del Xtr, ytr
    with open(args.out_dir / "x1_closure_histgb.pkl", "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    valid = evaluate_pairs(model, arrays, split.valid, layers, args.eval_rows_per_net_layer, args.seed + 2)
    test = evaluate_pairs(model, arrays, split.test, layers, args.eval_rows_per_net_layer, args.seed + 3)
    result = {
        "config": {**vars(args), "data_dir": str(args.data_dir), "out_dir": str(args.out_dir),
                   "weights": str(args.weights) if args.weights else None,
                   "moments": str(args.moments) if args.moments else None},
        "feature_names": ["layer", "a_sum", "a_product", "abs_a_difference", "rho", "x1", "a_difference_times_x1a"],
        "split": {k: v.tolist() for k, v in asdict(split).items()},
        "train_seconds": train_seconds,
        "iterations": int(model.n_iter_),
        "valid": valid,
        "test": test,
    }
    if (args.weights is None) ^ (args.moments is None):
        raise ValueError("--weights and --moments must be supplied together")
    if args.weights is not None:
        result["contraction_test"] = contraction_eval(
            model, arrays, split.test, layers, args.weights, args.moments,
            args.contraction_networks,
        )
    (args.out_dir / "x1_closure_results.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({"valid_gain": valid["gain"], "test_gain": test["gain"],
                      "test_r2": test["r2"], "model_bytes": (args.out_dir / "x1_closure_histgb.pkl").stat().st_size}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
