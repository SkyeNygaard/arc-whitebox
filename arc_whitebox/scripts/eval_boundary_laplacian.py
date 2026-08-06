"""Falsify or validate a boundary-integral estimator for homogeneous ReLU MLPs.

For a degree-one homogeneous continuous piecewise-linear scalar function ``f``
and ``X ~ N(0, I_d)``, second-order Gaussian integration by parts and radial
homogeneity give the exact distributional identity

    E[f(X)] = E[Delta f(X)].

For a ReLU composition, a directional second derivative obeys

    D_v^2 relu(h) = 1[h > 0] D_v^2 h + delta(h) (D_v h)^2.

Thus a forward pass carrying a JVP and a second directional derivative samples
the network's activation-boundary curvature.  This script replaces ``delta``
by a narrow Gaussian kernel and tests whether the resulting estimator has
remotely competitive variance before investing in exact coarea/root sampling.

The experiment uses points on a fixed sphere, exploiting homogeneity.  If its
radius is ``r`` and ``U`` is uniform on that sphere, then

    E[f(X)] = r^2 / (d - 1) * E[Delta f(U)]

when ``r = E[||X||]``.  A Rademacher direction has identity covariance, so one
direction per point is an unbiased Hutchinson trace estimator away from the
kernel approximation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from eval_sampling_official import DEFAULT_DATA, _load_rows
from eval_spherical_stein_cv import make_points, sphere_radius_mean


def gaussian_delta(h: np.ndarray, bandwidth: np.ndarray) -> np.ndarray:
    z = h / bandwidth
    return (
        np.exp(-0.5 * np.square(z), dtype=np.float32)
        / (np.float32(math.sqrt(2.0 * math.pi)) * bandwidth)
    )


def estimate_boundary(
    weights: np.ndarray,
    samples: int,
    point_seed: int,
    direction_seed: int,
    bandwidth_fraction: float,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return all-layer boundary and same-row direct estimates."""
    depth, width, _ = weights.shape
    radius = sphere_radius_mean(width)
    points = make_points(width, samples, point_seed)
    rng = np.random.default_rng(direction_seed)

    boundary_sum = np.zeros((depth, width), dtype=np.float64)
    direct_sum = np.zeros((depth, width), dtype=np.float64)
    boundary_sq_sum = np.zeros((depth, width), dtype=np.float64)
    kernel_mass_sum = 0.0
    kernel_active = 0

    start = time.perf_counter()
    for offset in range(0, samples, chunk):
        x = points[offset : offset + chunk]
        # Independent Rademacher probes have E[v v^T] = I exactly.
        tangent = rng.integers(0, 2, size=x.shape, dtype=np.int8)
        tangent = (2 * tangent - 1).astype(np.float32)
        second = np.zeros_like(x)
        activation = x

        for layer, weight in enumerate(weights):
            pre = activation @ weight
            pre_tangent = tangent @ weight
            pre_second = second @ weight

            # Per-neuron scale normalization makes one bandwidth fraction
            # meaningful despite finite-width variance drift across layers.
            rms = np.sqrt(
                np.mean(np.square(pre, dtype=np.float64), axis=0)
            ).astype(np.float32)
            bandwidth = np.maximum(
                np.float32(bandwidth_fraction) * rms,
                np.float32(1e-5),
            )
            delta = gaussian_delta(pre, bandwidth)
            gate = pre > 0.0

            activation = np.maximum(pre, 0.0)
            tangent = pre_tangent * gate
            second = pre_second * gate + delta * np.square(pre_tangent)

            direct_sum[layer] += activation.sum(axis=0, dtype=np.float64)
            boundary_sum[layer] += second.sum(axis=0, dtype=np.float64)
            boundary_sq_sum[layer] += np.square(
                second, dtype=np.float64
            ).sum(axis=0)
            kernel_mass_sum += float(delta.sum(dtype=np.float64))
            kernel_active += delta.size

    elapsed = time.perf_counter() - start
    shell_to_gaussian = radius * radius / (width - 1)
    boundary = shell_to_gaussian * boundary_sum / samples
    direct = direct_sum / samples
    boundary_var = np.maximum(
        boundary_sq_sum / samples - np.square(boundary_sum / samples),
        0.0,
    )
    diagnostics = {
        "seconds": elapsed,
        "radius": radius,
        "shell_to_gaussian": shell_to_gaussian,
        "mean_kernel_density": kernel_mass_sum / kernel_active,
        "mean_boundary_standard_error": float(
            shell_to_gaussian * np.mean(np.sqrt(boundary_var / samples))
        ),
        "mean_abs_boundary": float(np.mean(np.abs(boundary))),
        "mean_direct": float(np.mean(direct)),
    }
    return boundary, direct, diagnostics


def estimate_direct_equal_cost(
    weights: np.ndarray,
    samples: int,
    seed: int,
    chunk: int,
) -> tuple[np.ndarray, float]:
    """Three-forward-equivalent baseline for the boundary pass."""
    depth, width, _ = weights.shape
    points = make_points(width, 3 * samples, seed)
    sums = np.zeros((depth, width), dtype=np.float64)
    start = time.perf_counter()
    for offset in range(0, len(points), chunk):
        activation = points[offset : offset + chunk]
        for layer, weight in enumerate(weights):
            activation = np.maximum(activation @ weight, 0.0)
            sums[layer] += activation.sum(axis=0, dtype=np.float64)
    return sums / len(points), time.perf_counter() - start


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(prediction - target)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument(
        "--bandwidth-fractions",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.4],
    )
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = _load_rows(args.data, args.indices)
    records: list[dict[str, float | int | str]] = []
    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        baseline, baseline_seconds = estimate_direct_equal_cost(
            weights,
            args.samples,
            seed=7001 + 997 * index,
            chunk=args.chunk,
        )
        baseline_final_mse = mse(baseline[-1], targets[-1])
        for fraction in args.bandwidth_fractions:
            boundary, direct, diagnostics = estimate_boundary(
                weights,
                samples=args.samples,
                point_seed=101 + 997 * index,
                direction_seed=9001 + 997 * index,
                bandwidth_fraction=fraction,
                chunk=args.chunk,
            )
            record: dict[str, float | int | str] = {
                "index": index,
                "name": name,
                "samples": args.samples,
                "bandwidth_fraction": fraction,
                "baseline_final_mse": baseline_final_mse,
                "baseline_all_mse": mse(baseline, targets),
                "baseline_seconds": baseline_seconds,
                "boundary_final_mse": mse(boundary[-1], targets[-1]),
                "boundary_all_mse": mse(boundary, targets),
                "same_rows_direct_final_mse": mse(direct[-1], targets[-1]),
                "mean_boundary_final": float(np.mean(boundary[-1])),
                "mean_target_final": float(np.mean(targets[-1])),
                **diagnostics,
            }
            records.append(record)
            print(record, flush=True)

    summaries = []
    for fraction in args.bandwidth_fractions:
        selected = [r for r in records if r["bandwidth_fraction"] == fraction]
        summary = {
            "bandwidth_fraction": fraction,
            "mean_baseline_final_mse": float(
                np.mean([r["baseline_final_mse"] for r in selected])
            ),
            "mean_boundary_final_mse": float(
                np.mean([r["boundary_final_mse"] for r in selected])
            ),
            "mean_boundary_all_mse": float(
                np.mean([r["boundary_all_mse"] for r in selected])
            ),
            "ratio_to_equal_cost": float(
                np.mean([r["boundary_final_mse"] for r in selected])
                / np.mean([r["baseline_final_mse"] for r in selected])
            ),
            "mean_boundary_standard_error": float(
                np.mean([r["mean_boundary_standard_error"] for r in selected])
            ),
        }
        summaries.append(summary)
    result = {
        "identity": "E[f(X)] = E[Delta f(X)] for degree-one homogeneous CPWL f",
        "cost_model": "boundary pass = activation + JVP + second-direction matmul",
        "summaries": summaries,
        "records": records,
    }
    print({"summaries": summaries})
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
