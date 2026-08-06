#!/usr/bin/env python3
"""Real-data recursive rollout benchmark for the learned x1/x1a closure.

This is the stage after the one-step contraction test. It starts from the exact
Gaussian input state, recursively propagates means/covariances through all 32
layers, and evaluates final-output mean and variance against the 1e8-sample
higher-moment targets.

Two feature modes are provided:

* oracle_x1: use the true normalized k21 slices x1/x1a at each layer, while
  recursively propagating the mean and covariance. This isolates whether the
  learned covariance closure remains dynamically useful through depth.
* teacher_marginals: additionally replace each recursive pre-activation mean and
  diagonal variance with their true values before the nonlinear step. This is a
  diagnostic for off-diagonal stability only and is not deployable.

The deployable factorized-K3 stage should only be attempted if oracle_x1 passes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coefnet_numpy_runtime_stable import NumpyCoefNet


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


def gaussian_relu_moments(mu: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    covariance = (np.asarray(covariance, np.float64) + np.asarray(covariance, np.float64).T) * 0.5
    variance = np.maximum(np.diag(covariance), 1e-12)
    sigma = np.sqrt(variance)
    a = np.asarray(mu, np.float64) / sigma
    rho = np.clip(covariance / np.maximum(np.outer(sigma, sigma), 1e-12), -0.999999, 0.999999)
    np.fill_diagonal(rho, 1.0)
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
    post_second_diag = (mu * mu + variance) * ndtr(a) + mu * sigma * phi(a)
    np.fill_diagonal(second, post_second_diag)
    post_covariance = second - np.outer(post_mean, post_mean)
    return post_mean, (post_covariance + post_covariance.T) * 0.5


def connected_21(mu: np.ndarray, m11: np.ndarray, m21: np.ndarray,
                 m2: np.ndarray) -> np.ndarray:
    return (
        m21 - 2 * mu[:, None] * m11 - m2[:, None] * mu[None, :]
        + 2 * (mu[:, None] ** 2) * mu[None, :]
    )


def nearest_psd_preserve_diagonal(covariance: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, float]:
    covariance = (covariance + covariance.T) * 0.5
    diagonal = np.maximum(np.diag(covariance), 1e-12)
    scale = np.sqrt(diagonal)
    correlation = covariance / np.maximum(np.outer(scale, scale), 1e-30)
    correlation = (correlation + correlation.T) * 0.5
    np.fill_diagonal(correlation, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    minimum = float(eigenvalues[0])
    if minimum >= eps:
        return covariance, minimum
    eigenvalues = np.maximum(eigenvalues, eps)
    repaired = (eigenvectors * eigenvalues) @ eigenvectors.T
    renorm = np.sqrt(np.maximum(np.diag(repaired), eps))
    repaired /= np.outer(renorm, renorm)
    np.fill_diagonal(repaired, 1.0)
    output = repaired * np.outer(scale, scale)
    np.fill_diagonal(output, diagonal)
    return (output + output.T) * 0.5, minimum


def load_moments(path: Path) -> dict[str, np.ndarray]:
    keys = ["global_index", "pre_mean", "pre_m2", "pre_M11", "pre_M21"]
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in keys}


def load_weights(weights_dir: Path, global_index: int) -> np.ndarray:
    for suffix in (".npy", ".npz"):
        path = weights_dir / f"mlp_{global_index:05d}{suffix}"
        if not path.exists():
            continue
        if suffix == ".npy":
            array = np.load(path, mmap_mode="r")
        else:
            with np.load(path) as data:
                key = "weights" if "weights" in data else data.files[0]
                array = np.asarray(data[key])
        array = np.asarray(array, np.float64)
        if array.ndim != 3 or array.shape[1] != array.shape[2]:
            raise ValueError(f"bad weights shape {array.shape} in {path}")
        return array
    raise FileNotFoundError(f"weights for MLP {global_index} not found in {weights_dir}")


def true_x1_features(data: dict[str, np.ndarray], layer: int) -> tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(data["pre_mean"][layer], np.float64)
    m11 = np.asarray(data["pre_M11"][layer], np.float64)
    m21 = np.asarray(data["pre_M21"][layer], np.float64)
    m2 = np.asarray(data["pre_m2"][layer], np.float64)
    covariance = m11 - np.outer(mu, mu)
    sigma = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    k21 = connected_21(mu, m11, m21, m2)
    denominator = np.maximum(sigma[:, None] ** 3 + sigma[None, :] ** 3, 1e-12)
    return (k21 + k21.T) / denominator, (k21 - k21.T) / denominator


def predicted_residual(model: NumpyCoefNet, layer: int, depth: int,
                       mu: np.ndarray, covariance: np.ndarray,
                       x1_full: np.ndarray, x1a_full: np.ndarray) -> tuple[np.ndarray, dict]:
    width = len(mu)
    iu, ju = np.triu_indices(width, 1)
    variance = np.maximum(np.diag(covariance), 1e-12)
    sigma = np.sqrt(variance)
    rho = np.clip(covariance / np.maximum(np.outer(sigma, sigma), 1e-12), -1.0, 1.0)
    np.fill_diagonal(rho, 1.0)
    a = mu / sigma
    difference = a[iu] - a[ju]
    base = np.column_stack([
        np.full(len(iu), (layer + 1) / depth),
        a[iu] + a[ju],
        a[iu] * a[ju],
        np.abs(difference),
        rho[iu, ju],
    ])
    normalized = model.predict_invariant(
        base, difference, x1_full[iu, ju], x1a_full[iu, ju]
    )
    residual = np.zeros((width, width), np.float64)
    residual[iu, ju] = normalized * sigma[iu] * sigma[ju]
    residual[ju, iu] = residual[iu, ju]
    return residual, dict(model.last_diagnostics)


@dataclass
class RolloutResult:
    global_index: int
    alpha: float
    feature_mode: str
    psd_mode: str
    layers: list[dict]
    final_mean_mse: float
    final_relative_variance_mse: float
    final_sigma_mse: float
    psd_repairs: int
    minimum_pre_repair_eigenvalue: float
    runtime_feature_diagnostics: dict[str, int]


def run_rollout(path: Path, weights_dir: Path, model: NumpyCoefNet,
                alpha: float, feature_mode: str, psd_mode: str) -> RolloutResult:
    data = load_moments(path)
    global_index = int(data["global_index"])
    weights = load_weights(weights_dir, global_index)
    depth, width, _ = weights.shape
    if data["pre_mean"].shape[0] < depth:
        raise ValueError(f"moment file has only {data['pre_mean'].shape[0]} layers; weights have {depth}")

    # Exact first pre-activation state for x ~ N(0, I), forward convention x @ W.
    mu = np.zeros(width, np.float64)
    covariance = weights[0].T @ weights[0]
    layers: list[dict] = []
    repairs = 0
    minimum_eigenvalue = float("inf")
    diagnostic_totals = {
        "nonfinite_normalized_features": 0,
        "clipped_normalized_features": 0,
        "nonfinite_predictions_replaced": 0,
        "rows": 0,
    }

    for layer in range(depth):
        true_mu = np.asarray(data["pre_mean"][layer], np.float64)
        true_cov = np.asarray(data["pre_M11"][layer], np.float64) - np.outer(true_mu, true_mu)
        true_variance = np.maximum(np.diag(true_cov), 1e-12)
        predicted_variance = np.maximum(np.diag(covariance), 1e-12)
        mean_mse = float(np.mean((mu - true_mu) ** 2))
        relative_variance_mse = float(np.mean(((predicted_variance - true_variance) / true_variance) ** 2))
        sigma_mse = float(np.mean(((np.sqrt(predicted_variance) - np.sqrt(true_variance)) / np.sqrt(true_variance)) ** 2))
        layers.append({
            "layer": layer,
            "mean_mse": mean_mse,
            "relative_variance_mse": relative_variance_mse,
            "sigma_relative_mse": sigma_mse,
            "mean_relative_rms": float(np.sqrt(mean_mse / max(np.mean(true_mu ** 2), 1e-30))),
            "variance_relative_rms": float(np.sqrt(relative_variance_mse)),
            "sigma_relative_rms": float(np.sqrt(sigma_mse)),
        })
        if layer == depth - 1:
            break

        state_mu = mu.copy()
        state_covariance = covariance.copy()
        if feature_mode == "teacher_marginals":
            state_mu = true_mu.copy()
            np.fill_diagonal(state_covariance, true_variance)
        elif feature_mode != "oracle_x1":
            raise ValueError(f"unknown feature mode {feature_mode}")

        post_mean, post_covariance = gaussian_relu_moments(state_mu, state_covariance)
        x1_full, x1a_full = true_x1_features(data, layer)
        residual, diagnostics = predicted_residual(
            model, layer, depth, state_mu, state_covariance, x1_full, x1a_full
        )
        post_covariance += alpha * residual
        post_covariance = (post_covariance + post_covariance.T) * 0.5

        if psd_mode == "clip":
            post_covariance, before = nearest_psd_preserve_diagonal(post_covariance)
            minimum_eigenvalue = min(minimum_eigenvalue, before)
            repairs += int(before < 1e-8)
        elif psd_mode == "diagnose":
            before = float(np.linalg.eigvalsh(post_covariance)[0])
            minimum_eigenvalue = min(minimum_eigenvalue, before)
        elif psd_mode != "none":
            raise ValueError(f"unknown PSD mode {psd_mode}")

        for key in diagnostic_totals:
            diagnostic_totals[key] += int(diagnostics.get(key, 0))

        next_weights = weights[layer + 1]
        mu = post_mean @ next_weights
        covariance = next_weights.T @ post_covariance @ next_weights
        covariance = (covariance + covariance.T) * 0.5

    final = layers[-1]
    return RolloutResult(
        global_index=global_index,
        alpha=float(alpha),
        feature_mode=feature_mode,
        psd_mode=psd_mode,
        layers=layers,
        final_mean_mse=final["mean_mse"],
        final_relative_variance_mse=final["relative_variance_mse"],
        final_sigma_mse=final["sigma_relative_mse"],
        psd_repairs=repairs,
        minimum_pre_repair_eigenvalue=(minimum_eigenvalue if np.isfinite(minimum_eigenvalue) else float("nan")),
        runtime_feature_diagnostics=diagnostic_totals,
    )


def files_for_ids(moment_dir: Path, ids: Iterable[int]) -> list[Path]:
    output = []
    for global_index in ids:
        path = moment_dir / f"mlp_{int(global_index):05d}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        output.append(path)
    return output


def aggregate(rows: list[RolloutResult], baseline: Optional[list[RolloutResult]] = None) -> dict:
    mean_mse = float(np.mean([row.final_mean_mse for row in rows]))
    variance_mse = float(np.mean([row.final_relative_variance_mse for row in rows]))
    sigma_mse = float(np.mean([row.final_sigma_mse for row in rows]))
    result = {
        "mlps": len(rows),
        "final_mean_mse": mean_mse,
        "final_relative_variance_mse": variance_mse,
        "final_sigma_relative_mse": sigma_mse,
        "final_variance_relative_rms": float(np.sqrt(variance_mse)),
        "final_sigma_relative_rms": float(np.sqrt(sigma_mse)),
        "psd_repairs": int(sum(row.psd_repairs for row in rows)),
        "minimum_pre_repair_eigenvalue": float(min(row.minimum_pre_repair_eigenvalue for row in rows)),
    }
    if baseline is not None:
        base_mean = float(np.mean([row.final_mean_mse for row in baseline]))
        base_var = float(np.mean([row.final_relative_variance_mse for row in baseline]))
        base_sigma = float(np.mean([row.final_sigma_mse for row in baseline]))
        result.update({
            "mean_mse_gain": base_mean / max(mean_mse, 1e-30),
            "relative_variance_mse_gain": base_var / max(variance_mse, 1e-30),
            "sigma_mse_gain": base_sigma / max(sigma_mse, 1e-30),
            "fraction_mlps_mean_improved": float(np.mean([
                row.final_mean_mse < base.final_mean_mse for row, base in zip(rows, baseline)
            ])),
            "fraction_mlps_variance_improved": float(np.mean([
                row.final_relative_variance_mse < base.final_relative_variance_mse
                for row, base in zip(rows, baseline)
            ])),
        })
    return result


def fit_alpha(validation_files: list[Path], weights_dir: Path, model: NumpyCoefNet,
              alpha_grid: list[float], feature_mode: str, psd_mode: str,
              metric: str) -> tuple[float, list[dict]]:
    records = []
    for alpha in alpha_grid:
        rows = [run_rollout(path, weights_dir, model, alpha, feature_mode, psd_mode)
                for path in validation_files]
        if metric == "final_mean_mse":
            objective = float(np.mean([row.final_mean_mse for row in rows]))
        elif metric == "final_sigma_mse":
            objective = float(np.mean([row.final_sigma_mse for row in rows]))
        else:
            raise ValueError(metric)
        record = {"alpha": alpha, "objective": objective, "summary": aggregate(rows)}
        records.append(record)
        print(json.dumps({"stage": "alpha_grid", "alpha": alpha, "objective": objective}), flush=True)
    best = min(records, key=lambda record: record["objective"])
    return float(best["alpha"]), records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--moments-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-mode", choices=["oracle_x1", "teacher_marginals"], default="oracle_x1")
    parser.add_argument("--psd-mode", choices=["none", "diagnose", "clip"], default="clip")
    parser.add_argument("--alpha-grid", default="0,0.35,0.5,0.65,0.8,1.0")
    parser.add_argument("--fit-metric", choices=["final_mean_mse", "final_sigma_mse"], default="final_mean_mse")
    parser.add_argument("--validation-limit", type=int, default=5,
                        help="Number of validation MLPs used for alpha search; 0 means all.")
    parser.add_argument("--feature-clip", type=float, default=30.0)
    parser.add_argument("--fixed-alpha", type=float, default=None,
                        help="Skip validation search and evaluate this fixed alpha.")
    args = parser.parse_args()

    result_spec = json.loads(args.results_json.read_text())
    valid_ids = [int(v) for v in result_spec["valid_ids"]]
    test_ids = [int(v) for v in result_spec["test_ids"]]
    if args.validation_limit > 0:
        valid_ids = valid_ids[:args.validation_limit]
    validation_files = files_for_ids(args.moments_dir, valid_ids)
    test_files = files_for_ids(args.moments_dir, test_ids)
    model = NumpyCoefNet(args.model, feature_clip=args.feature_clip)
    alpha_grid = [float(value) for value in args.alpha_grid.split(",")]

    if args.fixed_alpha is None:
        best_alpha, grid = fit_alpha(
            validation_files, args.weights_dir, model, alpha_grid,
            args.feature_mode, args.psd_mode, args.fit_metric,
        )
    else:
        best_alpha = float(args.fixed_alpha)
        grid = []
    print(json.dumps({"stage": "test", "alpha": best_alpha}), flush=True)

    baseline = [run_rollout(path, args.weights_dir, model, 0.0,
                            args.feature_mode, args.psd_mode) for path in test_files]
    corrected = [run_rollout(path, args.weights_dir, model, best_alpha,
                             args.feature_mode, args.psd_mode) for path in test_files]
    output = {
        "config": {
            "feature_mode": args.feature_mode,
            "psd_mode": args.psd_mode,
            "fit_metric": args.fit_metric,
            "validation_ids": valid_ids,
            "test_ids": test_ids,
            "alpha_grid": alpha_grid,
            "feature_clip": args.feature_clip,
        },
        "best_alpha": best_alpha,
        "alpha_search": grid,
        "test_baseline": aggregate(baseline),
        "test_corrected": aggregate(corrected, baseline),
        "baseline_rows": [row.__dict__ for row in baseline],
        "corrected_rows": [row.__dict__ for row in corrected],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({
        "output": str(args.output),
        "best_alpha": best_alpha,
        "test": output["test_corrected"],
    }), flush=True)


if __name__ == "__main__":
    main()
