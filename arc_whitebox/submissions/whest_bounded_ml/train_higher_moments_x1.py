#!/usr/bin/env python3
"""Train the compact x1/x1a covariance closure on real WhestBench moments.

Expected data: one ``mlp_XXXXX.npz`` file per network from
``keenanpepper/arc-whestbench-higher-moments-2026``.

The model predicts the normalized residual

    (Cov[ReLU(z_i), ReLU(z_j)] - GaussianCov_i,j) / (sigma_i sigma_j)

using only exchange-invariant pair features and the symmetric/antisymmetric
(2,1) third-cumulant slices x1/x1a. Train/validation/test splits are by MLP,
never by pair, to prevent leakage.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.special import ndtr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

WIDTH = 256
DEPTH = 32
IU, JU = np.triu_indices(WIDTH, 1)
PAIR_COUNT = len(IU)


def phi(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bivariate_normal_cdf(a: np.ndarray, b: np.ndarray, rho: np.ndarray,
                         nodes: int = 12) -> np.ndarray:
    """Drezner-Wesolowsky integral after rho=sin(theta)-style smoothing."""
    a, b, rho = np.broadcast_arrays(a, b, rho)
    rho = np.clip(rho.astype(np.float64), -0.999999, 0.999999)
    x, w = np.polynomial.legendre.leggauss(nodes)
    rr = 0.5 * rho[..., None] * (x + 1.0)
    den = np.maximum(1.0 - rr * rr, 1e-14)
    exponent = -(a[..., None] ** 2 - 2 * rr * a[..., None] * b[..., None]
                 + b[..., None] ** 2) / (2 * den)
    density = np.exp(exponent) / (2 * math.pi * np.sqrt(den))
    return np.clip(ndtr(a) * ndtr(b) + 0.5 * rho * np.sum(density * w, axis=-1), 0, 1)


def gaussian_relu_covariance_pairs(mu: np.ndarray, sigma: np.ndarray,
                                   rho: np.ndarray, i: np.ndarray,
                                   j: np.ndarray) -> np.ndarray:
    a = mu / sigma
    ai, aj = a[i], a[j]
    si, sj = sigma[i], sigma[j]
    r = np.clip(rho, -0.999999, 0.999999)
    root = np.sqrt(np.maximum(1.0 - r * r, 1e-14))
    q = bivariate_normal_cdf(ai, aj, r)
    p2 = np.exp(-(ai * ai - 2 * r * ai * aj + aj * aj)
                / (2 * np.maximum(1 - r * r, 1e-14))) / (2 * math.pi * root)
    second = (
        mu[i] * mu[j] * q
        + mu[i] * sj * phi(aj) * ndtr((ai - r * aj) / root)
        + mu[j] * si * phi(ai) * ndtr((aj - r * ai) / root)
        + si * sj * (r * q + (1 - r * r) * p2)
    )
    post_mean = mu * ndtr(a) + sigma * phi(a)
    return second - post_mean[i] * post_mean[j]


def connected_21(mu: np.ndarray, m11: np.ndarray, m21: np.ndarray,
                 m2: np.ndarray) -> np.ndarray:
    return (m21 - 2 * mu[:, None] * m11 - m2[:, None] * mu[None, :]
            + 2 * (mu[:, None] ** 2) * mu[None, :])


def rows_from_file(path: Path, layers: np.ndarray, pairs_per_layer: int,
                   seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(path) as data:
        global_index = int(data["global_index"])
        rng = np.random.default_rng(seed + global_index)
        feature_rows: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        for layer in layers:
            pair_idx = rng.choice(PAIR_COUNT, pairs_per_layer, replace=False)
            i, j = IU[pair_idx], JU[pair_idx]

            pre_mean = np.asarray(data["pre_mean"][layer], np.float64)
            pre_m11 = np.asarray(data["pre_M11"][layer], np.float64)
            pre_m21 = np.asarray(data["pre_M21"][layer], np.float64)
            pre_m2 = np.asarray(data["pre_m2"][layer], np.float64)

            pre_cov = pre_m11 - np.outer(pre_mean, pre_mean)
            variance = np.maximum(np.diag(pre_cov), 1e-12)
            sigma = np.sqrt(variance)
            pair_rho = np.clip(pre_cov[i, j] / (sigma[i] * sigma[j]), -1, 1)
            standardized_mean = pre_mean / sigma

            k21 = connected_21(pre_mean, pre_m11, pre_m21, pre_m2)
            # This exactly matches the public joint-feature corpus convention.
            denom = np.maximum(sigma[i] ** 3 + sigma[j] ** 3, 1e-12)
            x1 = (k21[i, j] + k21[j, i]) / denom
            x1a = (k21[i, j] - k21[j, i]) / denom
            difference = standardized_mean[i] - standardized_mean[j]

            features = np.column_stack([
                np.full(len(i), (layer + 1) / DEPTH),
                standardized_mean[i] + standardized_mean[j],
                standardized_mean[i] * standardized_mean[j],
                np.abs(difference),
                pair_rho,
                x1,
                difference * x1a,
            ]).astype(np.float32)

            post_mean = np.asarray(data["mean"][layer], np.float64)
            true_cov = (np.asarray(data["M11"][layer], np.float64)[i, j]
                        - post_mean[i] * post_mean[j])
            gaussian_cov = gaussian_relu_covariance_pairs(
                pre_mean, sigma, pair_rho, i, j
            )
            target = ((true_cov - gaussian_cov) / (sigma[i] * sigma[j])).astype(np.float32)
            feature_rows.append(features)
            targets.append(target)

    return np.concatenate(feature_rows), np.concatenate(targets), global_index


class CoefficientNet(nn.Module):
    """Symmetry-constrained closure: c_s*x1 + c_a*(a_i-a_j)*x1a."""

    def __init__(self, hidden: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(5, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        coefficients = self.body(rows[:, :5])
        return coefficients[:, 0] * rows[:, 5] + coefficients[:, 1] * rows[:, 6]


def load_split(files: Iterable[Path], layers: np.ndarray, pairs_per_layer: int,
               seed: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    all_x, all_y, ids = [], [], []
    files = list(files)
    for index, path in enumerate(files):
        x, y, global_index = rows_from_file(path, layers, pairs_per_layer, seed)
        all_x.append(x)
        all_y.append(y)
        ids.append(global_index)
        if (index + 1) % 25 == 0 or index + 1 == len(files):
            print(json.dumps({"loaded": index + 1,
                              "rows": sum(len(v) for v in all_y)}), flush=True)
    return np.concatenate(all_x), np.concatenate(all_y), ids


def evaluate(model: CoefficientNet, x: np.ndarray, y: np.ndarray,
             mean: np.ndarray, std: np.ndarray,
             device: torch.device) -> dict[str, float]:
    normalized = x.copy()
    normalized[:, :5] = (normalized[:, :5] - mean) / std
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(normalized), 131072):
            batch = torch.from_numpy(normalized[start:start + 131072]).to(device)
            predictions.append(model(batch).cpu().numpy())
    prediction = np.concatenate(predictions).astype(np.float64)
    target = y.astype(np.float64)
    base_mse = float(np.mean(target * target))
    model_mse = float(np.mean((prediction - target) ** 2))
    return {
        "rows": len(y),
        "base_mse": base_mse,
        "model_mse": model_mse,
        "gain": base_mse / max(model_mse, 1e-30),
        "r2": 1.0 - model_mse / max(base_mse, 1e-30),
    }


def export_numpy(checkpoint: dict, destination: Path) -> None:
    state = checkpoint["state_dict"]
    linear_ids = sorted({int(key.split(".")[1]) for key in state
                         if key.startswith("body.") and key.endswith(".weight")})
    arrays: dict[str, np.ndarray] = {
        "mean": np.asarray(checkpoint["mean"], np.float32),
        "std": np.asarray(checkpoint["std"], np.float32),
        "hidden": np.int64(checkpoint["hidden"]),
    }
    for output_index, layer_id in enumerate(linear_ids):
        arrays[f"W{output_index}"] = state[f"body.{layer_id}.weight"].numpy().astype(np.float32)
        arrays[f"b{output_index}"] = state[f"body.{layer_id}.bias"].numpy().astype(np.float32)
    np.savez_compressed(destination, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-files", type=int, default=700)
    parser.add_argument("--valid-files", type=int, default=150)
    parser.add_argument("--test-files", type=int, default=150)
    parser.add_argument("--pairs-per-layer", type=int, default=128)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.data_dir.glob("mlp_*.npz"))
    required = args.train_files + args.valid_files + args.test_files
    if len(files) < required:
        raise FileNotFoundError(f"need {required} mlp_*.npz files, found {len(files)}")

    rng = np.random.default_rng(args.seed)
    files = [files[index] for index in rng.permutation(len(files))[:required]]
    train_files = files[:args.train_files]
    valid_files = files[args.train_files:args.train_files + args.valid_files]
    test_files = files[-args.test_files:]
    layers = (np.arange(31) if args.layers == "all"
              else np.array([int(value) for value in args.layers.split(",")]))

    print("loading training rows", flush=True)
    train_x, train_y, train_ids = load_split(
        train_files, layers, args.pairs_per_layer, args.seed
    )
    print("loading validation rows", flush=True)
    valid_x, valid_y, valid_ids = load_split(
        valid_files, layers, max(args.pairs_per_layer, 256), args.seed + 1
    )
    print("loading test rows", flush=True)
    test_x, test_y, test_ids = load_split(
        test_files, layers, max(args.pairs_per_layer, 256), args.seed + 2
    )

    mean = train_x[:, :5].mean(axis=0).astype(np.float32)
    std = (train_x[:, :5].std(axis=0) + 1e-6).astype(np.float32)
    train_x[:, :5] = (train_x[:, :5] - mean) / std

    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=16384, shuffle=True,
    )
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = CoefficientNet(args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-5)

    best_state = None
    best_loss = float("inf")
    stale_epochs = 0
    history = []
    started = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for rows, target in loader:
            rows, target = rows.to(device), target.to(device)
            prediction = model(rows)
            loss = torch.mean((prediction - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        validation = evaluate(model, valid_x, valid_y, mean, std, device)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "valid": validation,
        }
        history.append(record)
        print(json.dumps(record), flush=True)

        if validation["model_mse"] < best_loss:
            best_loss = validation["model_mse"]
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 12:
            break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    test = evaluate(model, test_x, test_y, mean, std, device)
    checkpoint = {
        "state_dict": best_state,
        "mean": mean,
        "std": std,
        "hidden": args.hidden,
    }
    checkpoint_path = args.out_dir / "higher_moments_x1_coefnet.pt"
    numpy_path = args.out_dir / "higher_moments_x1_coefnet.npz"
    torch.save(checkpoint, checkpoint_path)
    export_numpy(checkpoint, numpy_path)

    results = {
        "config": {**vars(args), "data_dir": str(args.data_dir),
                   "out_dir": str(args.out_dir)},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_ids": train_ids,
        "valid_ids": valid_ids,
        "test_ids": test_ids,
        "history": history,
        "test": test,
        "seconds": time.time() - started,
        "checkpoint": str(checkpoint_path),
        "numpy_model": str(numpy_path),
    }
    results_path = args.out_dir / "higher_moments_x1_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps({"test": test, "parameters": results["parameters"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
