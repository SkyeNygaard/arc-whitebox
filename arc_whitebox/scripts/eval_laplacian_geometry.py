"""Falsify or validate a boundary-Laplacian estimator for homogeneous ReLU MLPs.

For a degree-one homogeneous continuous piecewise-linear scalar function and
X ~ N(0, I_d), Gaussian integration by parts plus Euler homogeneity gives

    E[f(X)] = E[Delta f(X)].

The Laplacian is a distribution supported on ReLU kink surfaces.  This script
uses a Gaussian kernel in place of each Dirac delta and a Rademacher
Hutchinson direction to propagate a stochastic forward Laplacian:

    h  = a W
    dh = da W
    Lh = La W
    a  = relu(h)
    da = 1[h > 0] dh
    La = 1[h > 0] Lh + delta_eps(h) dh^2.

One row costs three dense layer matmuls, so the equal-matmul baseline gets
three times as many ordinary forward rows.  Bandwidth is selected only on
official Mini IDs 0--9, then frozen and checked once on IDs 10--19.  The
challenge holdout IDs 50--99 are never loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_sampling_official import DEFAULT_DATA, Design, _load_rows  # noqa: E402


WIDTH = 256
DEPTH = 32
SQRT_2PI = math.sqrt(2.0 * math.pi)
DEFAULT_OUT = ROOT / "results" / "laplacian_geometry_proxy.json"
DEFAULT_EPSILONS = (0.4, 0.2, 0.1, 0.05, 0.025)
SELECTION_IDS = tuple(range(10))
VALIDATION_IDS = tuple(range(10, 20))


def make_gaussian_points(samples: int, seed: int) -> np.ndarray:
    design = Design(
        kind="sobol",
        n=WIDTH,
        total=samples,
        seed=seed,
        antithetic=True,
        sphere=False,
    )
    blocks = []
    used = 0
    while used < samples:
        block = design.next(min(8192, samples - used))
        if not len(block):
            raise RuntimeError(f"Gaussian design stopped at {used}/{samples}")
        blocks.append(block)
        used += len(block)
    return np.concatenate(blocks, axis=0)


def make_sphere_points(samples: int, seed: int) -> np.ndarray:
    design = Design(
        kind="sobol",
        n=WIDTH,
        total=samples,
        seed=seed,
        antithetic=True,
        sphere=True,
    )
    blocks = []
    used = 0
    while used < samples:
        block = design.next(min(8192, samples - used))
        if not len(block):
            raise RuntimeError(f"sphere design stopped at {used}/{samples}")
        blocks.append(block)
        used += len(block)
    return np.concatenate(blocks, axis=0)


def forward_mean(weights: np.ndarray, points: np.ndarray) -> np.ndarray:
    activation = points
    for weight in weights:
        activation = np.maximum(activation @ weight, 0.0)
    return activation.mean(axis=0, dtype=np.float64)


def stochastic_laplacian_mean(
    weights: np.ndarray,
    points: np.ndarray,
    probes: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, dict[str, float]]:
    activation = points.copy()
    tangent = probes.copy()
    laplacian = np.zeros_like(points)
    maximum_abs_laplacian = 0.0
    kernel_hit_fractions = []

    for weight in weights:
        preactivation = activation @ weight
        tangent_h = tangent @ weight
        laplacian_h = laplacian @ weight
        gate = preactivation > 0.0
        standardized = preactivation / np.float32(epsilon)
        delta = np.exp(
            -0.5 * np.square(standardized),
            dtype=np.float32,
        ) / np.float32(SQRT_2PI * epsilon)

        activation = np.maximum(preactivation, 0.0)
        tangent = tangent_h * gate
        laplacian = (
            laplacian_h * gate
            + delta * np.square(tangent_h)
        )
        maximum_abs_laplacian = max(
            maximum_abs_laplacian,
            float(np.max(np.abs(laplacian))),
        )
        kernel_hit_fractions.append(
            float(np.mean(np.abs(preactivation) < epsilon))
        )

    return laplacian.mean(axis=0, dtype=np.float64), {
        "maximum_abs_laplacian": maximum_abs_laplacian,
        "mean_layer_kernel_hit_fraction": float(
            np.mean(kernel_hit_fractions)
        ),
        "final_layer_kernel_hit_fraction": kernel_hit_fractions[-1],
    }


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(prediction - target)))


def cost(samples: int) -> dict[str, int]:
    one_forward = 2 * samples * DEPTH * WIDTH**2
    return {
        "samples": samples,
        "dense_matmul_flops_fma2": 3 * one_forward,
        "activation_forward_flops_fma2": one_forward,
        "jvp_forward_flops_fma2": one_forward,
        "laplacian_forward_flops_fma2": one_forward,
        "equal_cost_baseline_samples": 3 * samples,
        "equal_cost_baseline_dense_matmul_flops_fma2": 3 * one_forward,
    }


def evaluate_rows(
    rows: list[tuple[str, np.ndarray, np.ndarray]],
    ids: tuple[int, ...],
    samples: int,
    epsilons: tuple[float, ...],
    seed_offset: int,
) -> list[dict[str, object]]:
    records = []
    for mlp_id, (name, weights, targets) in zip(ids, rows, strict=True):
        seed = seed_offset + 1009 * mlp_id
        gaussian = make_gaussian_points(samples, seed)
        rng = np.random.default_rng(seed + 7_919)
        probes = rng.choice(
            np.asarray([-1.0, 1.0], dtype=np.float32),
            size=gaussian.shape,
        )
        sphere = make_sphere_points(3 * samples, seed + 31_337)

        start = time.perf_counter()
        baseline = forward_mean(weights, sphere)
        baseline_seconds = time.perf_counter() - start
        target = targets[-1]
        epsilon_records = []
        for epsilon in epsilons:
            start = time.perf_counter()
            prediction, diagnostics = stochastic_laplacian_mean(
                weights,
                gaussian,
                probes,
                epsilon,
            )
            epsilon_records.append(
                {
                    "epsilon": epsilon,
                    "mse": mse(prediction, target),
                    "prediction_mean": float(np.mean(prediction)),
                    "prediction_min": float(np.min(prediction)),
                    "prediction_max": float(np.max(prediction)),
                    "elapsed_seconds": time.perf_counter() - start,
                    **diagnostics,
                }
            )
        record = {
            "id": mlp_id,
            "name": name,
            "target_mean": float(np.mean(target)),
            "equal_cost_baseline_mse": mse(baseline, target),
            "equal_cost_baseline_seconds": baseline_seconds,
            "epsilons": epsilon_records,
        }
        records.append(record)
        best = min(epsilon_records, key=lambda item: item["mse"])
        print(
            f"id={mlp_id:02d} baseline={record['equal_cost_baseline_mse']:.3e} "
            f"best_eps={best['epsilon']:.4g} lap={best['mse']:.3e}",
            flush=True,
        )
    return records


def aggregate(
    records: list[dict[str, object]],
    epsilons: tuple[float, ...],
) -> dict[str, object]:
    baseline = float(
        np.mean([record["equal_cost_baseline_mse"] for record in records])
    )
    by_epsilon = {}
    for epsilon in epsilons:
        values = []
        for record in records:
            epsilon_record = next(
                item
                for item in record["epsilons"]
                if item["epsilon"] == epsilon
            )
            values.append(epsilon_record["mse"])
        mean_mse = float(np.mean(values))
        by_epsilon[str(epsilon)] = {
            "mean_mse": mean_mse,
            "median_mlp_mse": float(np.median(values)),
            "ratio_to_equal_cost_baseline": mean_mse / baseline,
        }
    return {
        "equal_cost_baseline_mean_mse": baseline,
        "by_epsilon": by_epsilon,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=list(DEFAULT_EPSILONS),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    epsilons = tuple(args.epsilons)

    selection_rows = _load_rows(args.data, list(SELECTION_IDS))
    selection_records = evaluate_rows(
        selection_rows,
        SELECTION_IDS,
        args.samples,
        epsilons,
        seed_offset=41,
    )
    selection_summary = aggregate(selection_records, epsilons)
    selected_epsilon = min(
        epsilons,
        key=lambda epsilon: selection_summary["by_epsilon"][str(epsilon)][
            "mean_mse"
        ],
    )
    print(f"FROZEN epsilon={selected_epsilon}", flush=True)

    # Only the frozen bandwidth is evaluated on the disjoint validation IDs.
    validation_rows = _load_rows(args.data, list(VALIDATION_IDS))
    validation_records = evaluate_rows(
        validation_rows,
        VALIDATION_IDS,
        args.samples,
        (selected_epsilon,),
        seed_offset=83,
    )
    validation_summary = aggregate(
        validation_records,
        (selected_epsilon,),
    )

    result = {
        "identity": {
            "statement": "E[f(X)] = E[Delta f(X)] for degree-1 homogeneous f and X~N(0,I)",
            "derivation": (
                "Gaussian integration by parts gives E[Delta f]="
                "E[(||X||^2-d)f]. With f(RU)=R f(U) and "
                "E[R^3]=(d+1)E[R], the right side equals E[f]."
            ),
            "proxy": (
                "Gaussian mollifier for every ReLU delta and one independent "
                "Rademacher Hutchinson direction per input point"
            ),
        },
        "protocol": {
            "selection_ids": [SELECTION_IDS[0], SELECTION_IDS[-1]],
            "validation_ids": [VALIDATION_IDS[0], VALIDATION_IDS[-1]],
            "challenge_holdout_loaded": False,
            "samples": args.samples,
            "epsilon_grid": list(epsilons),
            "selected_epsilon": selected_epsilon,
            "baseline": "sphere Sobol antithetic at 3x rows",
        },
        "cost": cost(args.samples),
        "selection": {
            "summary": selection_summary,
            "records": selection_records,
        },
        "validation": {
            "summary": validation_summary,
            "records": validation_records,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "selected_epsilon": selected_epsilon,
                "cost": result["cost"],
                "selection_summary": selection_summary,
                "validation_summary": validation_summary,
                "out": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
