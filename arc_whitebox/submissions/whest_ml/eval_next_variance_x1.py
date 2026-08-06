#!/usr/bin/env python3
"""Evaluate whether the learned pair closure survives the next-weight contraction.

This is the decisive diagnostic after pair-residual evaluation. For source layer l:

  C_base  = Gaussian bivariate ReLU covariance
  C_model = C_base + alpha * learned off-diagonal residual
  v_next  = diag(W[l+1]^T C W[l+1])

The target is the 1e8-sample next pre-activation variance stored in the higher-
moment file. `oracle` diagonal mode gives both methods the true current post-ReLU
diagonal, isolating the learned off-diagonal closure. `gaussian` uses the Gaussian
marginal diagonal and measures the combined baseline as-is.

Correction strength alpha is fitted only on validation MLPs and then frozen for
the test MLPs. All reported test metrics therefore remain held out.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coefnet_numpy_runtime import NumpyCoefNet

WIDTH = 256
DEPTH = 32


def phi(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bivariate_normal_cdf(a: np.ndarray, b: np.ndarray, rho: np.ndarray,
                         nodes: int = 16) -> np.ndarray:
    a, b, rho = np.broadcast_arrays(a, b, rho)
    rho = np.clip(rho.astype(np.float64), -0.999999, 0.999999)
    x, w = np.polynomial.legendre.leggauss(nodes)
    rr = 0.5 * rho[..., None] * (x + 1.0)
    den = np.maximum(1.0 - rr * rr, 1e-14)
    exponent = -(a[..., None] ** 2 - 2 * rr * a[..., None] * b[..., None]
                 + b[..., None] ** 2) / (2 * den)
    density = np.exp(exponent) / (2 * math.pi * np.sqrt(den))
    return np.clip(
        ndtr(a) * ndtr(b) + 0.5 * rho * np.sum(density * w, axis=-1),
        0.0, 1.0,
    )


def gaussian_relu_covariance(mu: np.ndarray, sigma: np.ndarray,
                             rho: np.ndarray) -> np.ndarray:
    sigma = np.maximum(np.asarray(sigma, np.float64), 1e-12)
    mu = np.asarray(mu, np.float64)
    a = mu / sigma
    rho = np.clip(np.asarray(rho, np.float64), -0.999999, 0.999999)
    root = np.sqrt(np.maximum(1.0 - rho * rho, 1e-14))
    ai, aj = a[:, None], a[None, :]
    si, sj = sigma[:, None], sigma[None, :]
    mui, muj = mu[:, None], mu[None, :]
    probability = bivariate_normal_cdf(ai, aj, rho)
    density2 = np.exp(
        -(ai * ai - 2 * rho * ai * aj + aj * aj)
        / (2 * np.maximum(1.0 - rho * rho, 1e-14))
    ) / (2 * math.pi * root)
    second = (
        mui * muj * probability
        + mui * sj * phi(aj) * ndtr((ai - rho * aj) / root)
        + muj * si * phi(ai) * ndtr((aj - rho * ai) / root)
        + si * sj * (rho * probability + (1.0 - rho * rho) * density2)
    )
    post_mean = mu * ndtr(a) + sigma * phi(a)
    post_second_diag = (mu * mu + sigma * sigma) * ndtr(a) + mu * sigma * phi(a)
    np.fill_diagonal(second, post_second_diag)
    covariance = second - np.outer(post_mean, post_mean)
    return (covariance + covariance.T) * 0.5


def connected_21(mu: np.ndarray, m11: np.ndarray, m21: np.ndarray,
                 m2: np.ndarray) -> np.ndarray:
    return (
        m21 - 2 * mu[:, None] * m11 - m2[:, None] * mu[None, :]
        + 2 * (mu[:, None] ** 2) * mu[None, :]
    )


def contracted_variance(covariance: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """diag(W.T @ C @ W) for forward convention h @ W."""
    return np.sum(weights * (covariance @ weights), axis=0)


def load_moment_arrays(path: Path) -> dict[str, np.ndarray]:
    keys = [
        "global_index", "pre_mean", "pre_m2", "pre_M11", "pre_M21",
        "mean", "M11",
    ]
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in keys}


def load_weights(weights_dir: Path, global_index: int) -> np.ndarray:
    candidates = [
        weights_dir / f"mlp_{global_index:05d}.npy",
        weights_dir / f"mlp_{global_index:05d}.npz",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".npy":
            weights = np.load(path, mmap_mode="r")
        else:
            with np.load(path) as data:
                key = "weights" if "weights" in data else data.files[0]
                weights = np.asarray(data[key])
        weights = np.asarray(weights, np.float64)
        if weights.shape != (DEPTH, WIDTH, WIDTH):
            raise ValueError(f"{path}: expected {(DEPTH,WIDTH,WIDTH)}, got {weights.shape}")
        return weights
    raise FileNotFoundError(
        f"No weights for global index {global_index} in {weights_dir}"
    )


def model_residual(model: NumpyCoefNet, layer: int, pre_mean: np.ndarray,
                   pre_covariance: np.ndarray, pre_m11: np.ndarray,
                   pre_m21: np.ndarray, pre_m2: np.ndarray) -> np.ndarray:
    width = len(pre_mean)
    iu, ju = np.triu_indices(width, 1)
    variance = np.maximum(np.diag(pre_covariance), 1e-12)
    sigma = np.sqrt(variance)
    rho = np.clip(pre_covariance / np.maximum(np.outer(sigma, sigma), 1e-12), -1, 1)
    np.fill_diagonal(rho, 1.0)
    standardized_mean = pre_mean / sigma
    k21 = connected_21(pre_mean, pre_m11, pre_m21, pre_m2)
    denom = np.maximum(sigma[iu] ** 3 + sigma[ju] ** 3, 1e-12)
    x1 = (k21[iu, ju] + k21[ju, iu]) / denom
    x1a = (k21[iu, ju] - k21[ju, iu]) / denom
    difference = standardized_mean[iu] - standardized_mean[ju]
    base_features = np.column_stack([
        np.full(len(iu), (layer + 1) / DEPTH),
        standardized_mean[iu] + standardized_mean[ju],
        standardized_mean[iu] * standardized_mean[ju],
        np.abs(difference),
        rho[iu, ju],
    ]).astype(np.float32)
    normalized = model.predict_invariant(
        base_features,
        difference.astype(np.float32),
        x1.astype(np.float32),
        x1a.astype(np.float32),
    ).astype(np.float64)
    scale = sigma[iu] * sigma[ju]
    residual = np.zeros((width, width), dtype=np.float64)
    residual[iu, ju] = normalized * scale
    residual[ju, iu] = normalized * scale
    return residual


@dataclass
class Case:
    global_index: int
    layer: int
    target: np.ndarray
    base: np.ndarray
    delta: np.ndarray
    orientation_relative_rms: float
    min_eigenvalue_base: float
    min_eigenvalue_alpha1: float


def build_cases(files: Iterable[Path], weights_dir: Path, model: NumpyCoefNet,
                layers: list[int], diagonal_mode: str,
                eig_diagnostics: bool) -> list[Case]:
    cases: list[Case] = []
    for file_index, path in enumerate(files):
        data = load_moment_arrays(path)
        global_index = int(data["global_index"])
        weights = load_weights(weights_dir, global_index)
        for layer in layers:
            if layer >= DEPTH - 1:
                continue
            pre_mean = np.asarray(data["pre_mean"][layer], np.float64)
            pre_m11 = np.asarray(data["pre_M11"][layer], np.float64)
            pre_m21 = np.asarray(data["pre_M21"][layer], np.float64)
            pre_m2 = np.asarray(data["pre_m2"][layer], np.float64)
            pre_cov = pre_m11 - np.outer(pre_mean, pre_mean)
            pre_var = np.maximum(np.diag(pre_cov), 1e-12)
            sigma = np.sqrt(pre_var)
            rho = np.clip(pre_cov / np.maximum(np.outer(sigma, sigma), 1e-12), -1, 1)
            np.fill_diagonal(rho, 1.0)

            post_mean = np.asarray(data["mean"][layer], np.float64)
            post_m11 = np.asarray(data["M11"][layer], np.float64)
            true_post_cov = post_m11 - np.outer(post_mean, post_mean)
            base_post_cov = gaussian_relu_covariance(pre_mean, sigma, rho)
            residual = model_residual(
                model, layer, pre_mean, pre_cov, pre_m11, pre_m21, pre_m2
            )
            if diagonal_mode == "oracle":
                diagonal = np.diag(true_post_cov)
                np.fill_diagonal(base_post_cov, diagonal)
            elif diagonal_mode != "gaussian":
                raise ValueError(f"unknown diagonal mode {diagonal_mode}")

            next_weights = np.asarray(weights[layer + 1], np.float64)
            next_mean = np.asarray(data["pre_mean"][layer + 1], np.float64)
            next_m11 = np.asarray(data["pre_M11"][layer + 1], np.float64)
            next_true_cov = next_m11 - np.outer(next_mean, next_mean)
            target = np.maximum(np.diag(next_true_cov), 1e-12)

            true_contracted = contracted_variance(true_post_cov, next_weights)
            orientation_relative_rms = float(np.sqrt(np.mean(
                ((true_contracted - target) / target) ** 2
            )))
            base = contracted_variance(base_post_cov, next_weights)
            delta = contracted_variance(residual, next_weights)
            min_base = float("nan")
            min_model = float("nan")
            if eig_diagnostics:
                min_base = float(np.linalg.eigvalsh(base_post_cov)[0])
                min_model = float(np.linalg.eigvalsh(base_post_cov + residual)[0])
            cases.append(Case(
                global_index=global_index,
                layer=layer,
                target=target,
                base=base,
                delta=delta,
                orientation_relative_rms=orientation_relative_rms,
                min_eigenvalue_base=min_base,
                min_eigenvalue_alpha1=min_model,
            ))
        print(json.dumps({
            "loaded": file_index + 1,
            "global_index": global_index,
            "cases": len(cases),
            "diagonal_mode": diagonal_mode,
        }), flush=True)
    return cases


def fit_alpha(cases: list[Case], metric: str, lo: float, hi: float) -> float:
    if not cases:
        return 1.0
    numerator = 0.0
    denominator = 0.0
    for case in cases:
        error = case.base - case.target
        delta = case.delta
        if metric == "relative_variance":
            error = error / case.target
            delta = delta / case.target
        elif metric != "absolute_variance":
            raise ValueError(metric)
        numerator += float(np.sum(error * delta))
        denominator += float(np.sum(delta * delta))
    if denominator <= 1e-30:
        return 0.0
    return float(np.clip(-numerator / denominator, lo, hi))


def metric_row(case: Case, alpha: float) -> dict[str, float | int]:
    prediction = case.base + alpha * case.delta
    target = case.target
    base_error = case.base - target
    model_error = prediction - target
    base_abs = float(np.mean(base_error ** 2))
    model_abs = float(np.mean(model_error ** 2))
    base_rel = float(np.mean((base_error / target) ** 2))
    model_rel = float(np.mean((model_error / target) ** 2))
    base_sigma = np.sqrt(np.maximum(case.base, 1e-12))
    model_sigma = np.sqrt(np.maximum(prediction, 1e-12))
    target_sigma = np.sqrt(target)
    base_sigma_mse = float(np.mean(((base_sigma - target_sigma) / target_sigma) ** 2))
    model_sigma_mse = float(np.mean(((model_sigma - target_sigma) / target_sigma) ** 2))
    return {
        "global_index": case.global_index,
        "layer": case.layer,
        "alpha": alpha,
        "base_variance_mse": base_abs,
        "model_variance_mse": model_abs,
        "variance_gain": base_abs / max(model_abs, 1e-30),
        "base_relative_variance_mse": base_rel,
        "model_relative_variance_mse": model_rel,
        "relative_variance_gain": base_rel / max(model_rel, 1e-30),
        "base_sigma_relative_mse": base_sigma_mse,
        "model_sigma_relative_mse": model_sigma_mse,
        "sigma_gain": base_sigma_mse / max(model_sigma_mse, 1e-30),
        "base_sigma_relative_rms": math.sqrt(base_sigma_mse),
        "model_sigma_relative_rms": math.sqrt(model_sigma_mse),
        "negative_variance_fraction": float(np.mean(prediction <= 0)),
        "orientation_relative_rms": case.orientation_relative_rms,
        "min_eigenvalue_base": case.min_eigenvalue_base,
        "min_eigenvalue_alpha1": case.min_eigenvalue_alpha1,
    }


def aggregate(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not rows:
        return {}
    keys = [
        ("variance", "base_variance_mse", "model_variance_mse"),
        ("relative_variance", "base_relative_variance_mse", "model_relative_variance_mse"),
        ("sigma", "base_sigma_relative_mse", "model_sigma_relative_mse"),
    ]
    output: dict[str, float | int] = {"cases": len(rows)}
    for name, base_key, model_key in keys:
        base = np.array([float(row[base_key]) for row in rows])
        model = np.array([float(row[model_key]) for row in rows])
        output[f"{name}_gain"] = float(base.mean() / max(model.mean(), 1e-30))
        output[f"{name}_fraction_improved"] = float(np.mean(model < base))
        output[f"{name}_median_case_gain"] = float(np.median(base / np.maximum(model, 1e-30)))
        output[f"base_{name}_mse"] = float(base.mean())
        output[f"model_{name}_mse"] = float(model.mean())
    output["base_sigma_relative_rms"] = float(math.sqrt(output["base_sigma_mse"]))
    output["model_sigma_relative_rms"] = float(math.sqrt(output["model_sigma_mse"]))
    output["negative_variance_fraction"] = float(np.mean([
        float(row["negative_variance_fraction"]) for row in rows
    ]))
    output["max_orientation_relative_rms"] = float(max(
        float(row["orientation_relative_rms"]) for row in rows
    ))
    output["mean_orientation_relative_rms"] = float(np.mean([
        float(row["orientation_relative_rms"]) for row in rows
    ]))
    return output


def grouped(rows: list[dict[str, float | int]], key: str) -> dict[str, dict[str, float | int]]:
    values = sorted({int(row[key]) for row in rows})
    return {
        str(value): aggregate([row for row in rows if int(row[key]) == value])
        for value in values
    }


def files_for_ids(data_dir: Path, ids: Iterable[int]) -> list[Path]:
    paths = [data_dir / f"mlp_{int(i):05d}.npz" for i in ids]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing moment files: {missing[:5]}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--layers", default="4,8,12,16,20,24,28,30")
    parser.add_argument("--diagonal-modes", default="oracle,gaussian")
    parser.add_argument("--alpha", default="auto",
                        help="float, or auto (fit on validation only)")
    parser.add_argument("--alpha-metric", choices=["relative_variance", "absolute_variance"],
                        default="relative_variance")
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--skip-eigen-diagnostics", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("next_variance_x1_eval.json"))
    args = parser.parse_args()

    result = json.loads(args.results_json.read_text())
    valid_ids = list(map(int, result["valid_ids"]))
    test_ids = list(map(int, result["test_ids"]))
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    modes = [value.strip() for value in args.diagonal_modes.split(",") if value.strip()]
    model = NumpyCoefNet(args.model)

    output: dict[str, object] = {
        "config": {
            "model": str(args.model),
            "results_json": str(args.results_json),
            "data_dir": str(args.data_dir),
            "weights_dir": str(args.weights_dir),
            "layers": layers,
            "alpha": args.alpha,
            "alpha_metric": args.alpha_metric,
            "valid_ids": valid_ids,
            "test_ids": test_ids,
        },
        "modes": {},
    }

    for mode in modes:
        print(json.dumps({"stage": "validation", "diagonal_mode": mode}), flush=True)
        valid_cases = build_cases(
            files_for_ids(args.data_dir, valid_ids), args.weights_dir, model,
            layers, mode, not args.skip_eigen_diagnostics,
        )
        alpha = (
            fit_alpha(valid_cases, args.alpha_metric, args.alpha_min, args.alpha_max)
            if args.alpha == "auto" else float(args.alpha)
        )
        valid_rows = [metric_row(case, alpha) for case in valid_cases]
        valid_rows_alpha1 = [metric_row(case, 1.0) for case in valid_cases]
        print(json.dumps({
            "stage": "test", "diagonal_mode": mode, "alpha": alpha,
            "validation": aggregate(valid_rows),
            "validation_alpha1": aggregate(valid_rows_alpha1),
        }), flush=True)
        test_cases = build_cases(
            files_for_ids(args.data_dir, test_ids), args.weights_dir, model,
            layers, mode, not args.skip_eigen_diagnostics,
        )
        test_rows = [metric_row(case, alpha) for case in test_cases]
        test_rows_alpha1 = [metric_row(case, 1.0) for case in test_cases]
        mode_result = {
            "alpha": alpha,
            "validation": {
                "summary": aggregate(valid_rows),
                "by_layer": grouped(valid_rows, "layer"),
                "by_network": grouped(valid_rows, "global_index"),
            },
            "validation_alpha1": {
                "summary": aggregate(valid_rows_alpha1),
                "by_layer": grouped(valid_rows_alpha1, "layer"),
                "by_network": grouped(valid_rows_alpha1, "global_index"),
            },
            "test": {
                "summary": aggregate(test_rows),
                "by_layer": grouped(test_rows, "layer"),
                "by_network": grouped(test_rows, "global_index"),
                "rows": test_rows,
            },
            "test_alpha1": {
                "summary": aggregate(test_rows_alpha1),
                "by_layer": grouped(test_rows_alpha1, "layer"),
                "by_network": grouped(test_rows_alpha1, "global_index"),
                "rows": test_rows_alpha1,
            },
        }
        output["modes"][mode] = mode_result
        print(json.dumps({
            "diagonal_mode": mode,
            "alpha": alpha,
            "test": mode_result["test"]["summary"],
        }), flush=True)

    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps({"output": str(args.out)}))


def _self_test() -> None:
    rng = np.random.default_rng(0)
    n = 12
    a = rng.standard_normal((n, n))
    covariance = a @ a.T
    weights = rng.standard_normal((n, n))
    expected = np.diag(weights.T @ covariance @ weights)
    actual = contracted_variance(covariance, weights)
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
    target = rng.uniform(0.5, 2.0, n)
    base = target + rng.normal(0, 0.2, n)
    delta = rng.normal(0, 0.1, n)
    cases = [Case(0, 0, target, base, delta, 0.0, 0.0, 0.0)]
    alpha = fit_alpha(cases, "absolute_variance", -10, 10)
    grid = np.linspace(alpha - 1, alpha + 1, 1001)
    losses = [np.mean((base + x * delta - target) ** 2) for x in grid]
    assert abs(grid[int(np.argmin(losses))] - alpha) < 0.01
    print("self-test passed")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
