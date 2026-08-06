"""Compare random-plane angular Rao--Blackwellization with sphere sampling.

For a bias-free ReLU MLP, f(r u) = r f(u). A Haar-random 2D plane followed by
a uniform circle angle has a uniform marginal direction on the sphere. We
randomize the phase of an equally spaced angular grid, so every finite-K plane
estimator is unbiased while increasingly Rao--Blackwellizing the angle.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
from scipy.special import gammaln, ndtri
from scipy.stats import qmc


def expected_chi(width: int) -> float:
    return float(
        math.sqrt(2.0)
        * math.exp(
            gammaln((width + 1) / 2.0)
            - gammaln(width / 2.0)
        )
    )


def load_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return (
        np.asarray(data["weights"], dtype=np.float32),
        np.asarray(data["means"], dtype=np.float64)[-1],
    )


def forward_mean(
    directions: np.ndarray,
    weights: np.ndarray,
    *,
    block_rows: int,
) -> np.ndarray:
    total = np.zeros(weights.shape[1], dtype=np.float64)
    for start in range(0, len(directions), block_rows):
        activations = np.asarray(
            directions[start : start + block_rows], dtype=np.float32
        )
        for weight in weights:
            activations = activations @ weight
            np.maximum(activations, 0.0, out=activations)
        total += activations.sum(axis=0, dtype=np.float64)
    return total / len(directions)


def plane_directions(
    width: int,
    total: int,
    angles_per_plane: int,
    seed: int,
) -> np.ndarray:
    if total % angles_per_plane:
        raise ValueError("total must be divisible by angles_per_plane")
    planes = total // angles_per_plane
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((planes, width, 2))
    bases, _ = np.linalg.qr(gaussian)
    phase = rng.uniform(
        0.0, 2.0 * math.pi / angles_per_plane, size=planes
    )
    angles = (
        phase[:, None]
        + 2.0
        * math.pi
        * np.arange(angles_per_plane)[None, :]
        / angles_per_plane
    )
    directions = (
        bases[:, :, 0, None] * np.cos(angles)[:, None, :]
        + bases[:, :, 1, None] * np.sin(angles)[:, None, :]
    )
    return directions.transpose(0, 2, 1).reshape(total, width).astype(
        np.float32
    )


def iid_antithetic_directions(
    width: int, total: int, seed: int
) -> np.ndarray:
    if total % 2:
        raise ValueError("antithetic total must be even")
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((total // 2, width))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    return np.concatenate([base, -base], axis=0).astype(np.float32)


def sobol_antithetic_directions(
    width: int, total: int, seed: int
) -> np.ndarray:
    if total % 2:
        raise ValueError("antithetic total must be even")
    unit = qmc.Sobol(d=width, scramble=True, seed=seed).random(total // 2)
    eps = np.finfo(np.float64).eps
    gaussian = ndtri(np.clip(unit, eps, 1.0 - eps))
    gaussian /= np.linalg.norm(gaussian, axis=1, keepdims=True)
    return np.concatenate([gaussian, -gaussian], axis=0).astype(
        np.float32
    )


def run_estimator(
    weights: np.ndarray,
    *,
    method: str,
    total: int,
    angles_per_plane: int,
    seed: int,
    block_rows: int,
) -> tuple[np.ndarray, float]:
    width = weights.shape[1]
    start = time.perf_counter()
    if method == "plane":
        directions = plane_directions(
            width, total, angles_per_plane, seed
        )
    elif method == "iid_antithetic":
        directions = iid_antithetic_directions(width, total, seed)
    elif method == "sobol_antithetic":
        directions = sobol_antithetic_directions(width, total, seed)
    else:
        raise ValueError(method)
    directions *= expected_chi(width)
    prediction = forward_mean(
        directions, weights, block_rows=block_rows
    )
    return prediction, time.perf_counter() - start


def evaluate_method(
    weights: np.ndarray,
    target: np.ndarray,
    *,
    method: str,
    total: int,
    angles_per_plane: int,
    seeds: int,
    block_rows: int,
) -> dict:
    predictions = []
    seconds = []
    for seed in range(seeds):
        prediction, elapsed = run_estimator(
            weights,
            method=method,
            total=total,
            angles_per_plane=angles_per_plane,
            seed=seed,
            block_rows=block_rows,
        )
        predictions.append(prediction)
        seconds.append(elapsed)
    stacked = np.stack(predictions)
    ensemble = stacked.mean(axis=0)
    single_mses = ((stacked - target[None, :]) ** 2).mean(axis=1)
    variance = ((stacked - ensemble[None, :]) ** 2).mean()
    bias2 = ((ensemble - target) ** 2).mean()
    return {
        "method": method,
        "angles_per_plane": (
            angles_per_plane if method == "plane" else None
        ),
        "planes": (
            total // angles_per_plane if method == "plane" else None
        ),
        "total_forward_directions": total,
        "seeds": seeds,
        "mean_single_target_mse": float(single_mses.mean()),
        "sd_single_target_mse": float(single_mses.std(ddof=1)),
        "ensemble_target_mse": float(bias2),
        "across_seed_prediction_variance": float(variance),
        "mean_seconds": statistics.fmean(seconds),
        "median_seconds": statistics.median(seconds),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz")
    parser.add_argument("--total", type=int, default=8192)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument(
        "--angles", nargs="+", type=int, default=[16, 64, 256, 1024]
    )
    parser.add_argument("--block-rows", type=int, default=8192)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    weights, target = load_npz(args.npz)
    records = [
        evaluate_method(
            weights,
            target,
            method=method,
            total=args.total,
            angles_per_plane=args.angles[0],
            seeds=args.seeds,
            block_rows=args.block_rows,
        )
        for method in ("iid_antithetic", "sobol_antithetic")
    ]
    records.extend(
        evaluate_method(
            weights,
            target,
            method="plane",
            total=args.total,
            angles_per_plane=angles,
            seeds=args.seeds,
            block_rows=args.block_rows,
        )
        for angles in args.angles
    )
    baseline_variance = next(
        r["across_seed_prediction_variance"]
        for r in records
        if r["method"] == "sobol_antithetic"
    )
    for record in records:
        record["variance_ratio_vs_sobol_antithetic"] = (
            record["across_seed_prediction_variance"]
            / baseline_variance
        )
    artifact = {
        "npz": args.npz,
        "total_forward_directions": args.total,
        "records": records,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
