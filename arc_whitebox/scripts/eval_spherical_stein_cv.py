"""Test a spherical Stein control variate for the homogeneous ReLU MLP.

For ``u`` uniform on the unit sphere and any degree-one homogeneous scalar
network output ``f``, integration by parts on the sphere gives

    E[d (v.T u) f(u) - v.T grad f(u)] = 0.

The network has no biases, so this identity applies output-by-output.  A
directional derivative through all layers costs one extra forward-equivalent.
This script compares a cross-fitted Stein estimator with an equal-cost RQMC
baseline on disjoint randomized scrambles.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import gammaln

from eval_sampling_official import DEFAULT_DATA, Design, _load_rows


def sphere_radius_mean(width: int) -> float:
    return float(
        math.sqrt(2.0)
        * math.exp(gammaln((width + 1) / 2.0) - gammaln(width / 2.0))
    )


def make_points(width: int, samples: int, seed: int) -> np.ndarray:
    design = Design(
        kind="sobol",
        n=width,
        total=samples,
        seed=seed,
        antithetic=True,
        sphere=True,
    )
    blocks = []
    used = 0
    while used < samples:
        block = design.next(min(4096, samples - used))
        if not len(block):
            raise RuntimeError(f"design stopped after {used}/{samples}")
        blocks.append(block)
        used += len(block)
    return np.concatenate(blocks, axis=0)


def forward_final(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    a = x
    for weight in weights:
        a = np.maximum(a @ weight, 0.0)
    return a.astype(np.float64)


def forward_final_with_jvp(
    weights: np.ndarray,
    x: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    a = x
    da = np.broadcast_to(direction.astype(np.float32), x.shape).copy()
    for weight in weights:
        h = a @ weight
        dh = da @ weight
        gate = h > 0.0
        a = np.maximum(h, 0.0)
        da = dh * gate
    return a.astype(np.float64), da.astype(np.float64)


def fit_beta(values: np.ndarray, controls: np.ndarray, ridge: float) -> np.ndarray:
    centered_values = values - values.mean(axis=0)
    centered_controls = controls - controls.mean(axis=0)
    covariance = np.mean(centered_values * centered_controls, axis=0)
    variance = np.mean(np.square(centered_controls), axis=0)
    return covariance / np.maximum(variance + ridge, 1e-30)


def evaluate_one(
    weights: np.ndarray,
    target: np.ndarray,
    mlp_seed: int,
    samples_per_stream: int,
    ridge_fraction: float,
) -> dict[str, float]:
    width = weights.shape[-1]
    radius = sphere_radius_mean(width)
    direction_rng = np.random.default_rng(mlp_seed + 90_001)
    direction = direction_rng.standard_normal(width)
    direction /= np.linalg.norm(direction)

    # Two independent scrambles for cross-fitting the coefficient.
    stein_values = []
    stein_controls = []
    for seed in (101, 202):
        x = make_points(width, samples_per_stream, seed + mlp_seed * 997)
        values, derivative = forward_final_with_jvp(weights, x, direction)
        u_dot_v = (x.astype(np.float64) @ direction) / radius
        controls = (
            width * u_dot_v[:, None] * values
            - radius * derivative
        )
        stein_values.append(values)
        stein_controls.append(controls)

    control_scale = float(
        np.mean(np.square(np.concatenate(stein_controls, axis=0)))
    )
    ridge = ridge_fraction * control_scale
    beta_0 = fit_beta(stein_values[0], stein_controls[0], ridge)
    beta_1 = fit_beta(stein_values[1], stein_controls[1], ridge)
    cross_prediction = 0.5 * (
        stein_values[0].mean(axis=0)
        - beta_1 * stein_controls[0].mean(axis=0)
        + stein_values[1].mean(axis=0)
        - beta_0 * stein_controls[1].mean(axis=0)
    )
    pooled_values = np.concatenate(stein_values, axis=0)
    pooled_controls = np.concatenate(stein_controls, axis=0)
    pooled_beta = fit_beta(pooled_values, pooled_controls, ridge)
    pooled_prediction = (
        pooled_values.mean(axis=0)
        - pooled_beta * pooled_controls.mean(axis=0)
    )

    # The JVP doubles the dominant layer matmuls.  Give the baseline twice as
    # many rows, split over the same number of independent scrambles.
    baseline_means = []
    for seed in (303, 404):
        x = make_points(width, 2 * samples_per_stream, seed + mlp_seed * 997)
        baseline_means.append(forward_final(weights, x).mean(axis=0))
    baseline_prediction = np.mean(baseline_means, axis=0)

    raw_prediction = np.mean(
        [values.mean(axis=0) for values in stein_values],
        axis=0,
    )

    def mse(prediction: np.ndarray) -> float:
        return float(np.mean(np.square(prediction - target)))

    return {
        "baseline_mse": mse(baseline_prediction),
        "stein_crossfit_mse": mse(cross_prediction),
        "stein_pooled_mse": mse(pooled_prediction),
        "stein_raw_halfcost_mse": mse(raw_prediction),
        "mean_abs_beta": float(np.mean(np.abs(pooled_beta))),
        "mean_control_rms": float(np.sqrt(control_scale)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--samples-per-stream", type=int, default=2048)
    parser.add_argument("--ridge-fraction", type=float, default=1e-6)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = _load_rows(args.data, args.indices)
    records = []
    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        record = {
            "index": index,
            "name": name,
            **evaluate_one(
                weights,
                targets[-1],
                index,
                args.samples_per_stream,
                args.ridge_fraction,
            ),
        }
        records.append(record)
        print(record, flush=True)

    summary = {}
    for key in (
        "baseline_mse",
        "stein_crossfit_mse",
        "stein_pooled_mse",
        "stein_raw_halfcost_mse",
    ):
        summary[key] = float(np.mean([record[key] for record in records]))
    summary["crossfit_ratio_to_equal_cost"] = (
        summary["stein_crossfit_mse"] / summary["baseline_mse"]
    )
    summary["pooled_ratio_to_equal_cost"] = (
        summary["stein_pooled_mse"] / summary["baseline_mse"]
    )
    result = {
        "method": {
            "identity": "E[d*(v.u)*f - v.grad(f)] = 0 on S^(d-1)",
            "samples_per_stream": args.samples_per_stream,
            "stein_streams": 2,
            "equal_cost_baseline_rows_per_stream": 2 * args.samples_per_stream,
            "ridge_fraction": args.ridge_fraction,
        },
        "summary": summary,
        "records": records,
    }
    print({"summary": summary})
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
