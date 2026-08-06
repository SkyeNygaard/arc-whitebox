"""Strict train/test optimization of one-stream Sobol RQMC compute.

Predeclared candidates:

    N in {6144, 8192, 12288, 16384, 24576, 32768}
    scrambled Sobol seed in {0, ..., 7}

Every estimator uses one reused scrambled Sobol stream, antithetic pairs, exact
radial integration by sphere normalization, and no frame whitening.  Nested
prefixes let the research harness evaluate all six N values in a single
N=32768 pass for a given seed and MLP; the prediction at N is exactly the first
N/2 Sobol directions and their antipodes.

Only mini IDs 0--49 select the (N, seed) configuration by adjusted mean MSE.
After freezing it, the script loads and evaluates IDs 50--99 once.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import gammaln, ndtri
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_sampling_official import _load_rows  # noqa: E402


DEFAULT_DATA = ROOT / "data" / "official_phase1_mini" / "data"
DEFAULT_OUT = ROOT / "results" / "optimize_rqmc_compute.json"
WIDTH = 256
DEPTH = 32
BUDGET = 272_000_000_000
SAMPLE_GRID = (6144, 8192, 12288, 16384, 24576, 32768)
SEED_GRID = tuple(range(8))


def sphere_sobol_base(seed: int, max_samples: int) -> np.ndarray:
    base_rows = max_samples // 2
    engine = qmc.Sobol(d=WIDTH, scramble=True, seed=seed)
    u = engine.random(base_rows)
    z = ndtri(np.clip(u, 1e-7, 1.0 - 1e-7)).astype(np.float32)
    expected_radius = float(
        math.sqrt(2.0)
        * math.exp(
            gammaln((WIDTH + 1) / 2.0) - gammaln(WIDTH / 2.0)
        )
    )
    z *= (
        expected_radius / np.linalg.norm(z, axis=1, keepdims=True)
    ).astype(np.float32)
    return z


def nested_estimates(
    weights: np.ndarray,
    base: np.ndarray,
    sample_grid: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Return exact antithetic prefix estimates for every requested N."""
    checkpoints = [samples // 2 for samples in sample_grid]
    if checkpoints[-1] > len(base):
        raise ValueError("base design is shorter than largest checkpoint")
    running_sum = np.zeros(WIDTH, dtype=np.float64)
    estimates: dict[int, np.ndarray] = {}
    previous = 0
    for samples, endpoint in zip(sample_grid, checkpoints, strict=True):
        directions = base[previous:endpoint]
        activation = np.concatenate((directions, -directions), axis=0)
        for weight in weights:
            activation = np.maximum(activation @ weight, 0.0)
        running_sum += activation.sum(axis=0, dtype=np.float64)
        estimates[samples] = running_sum / samples
        previous = endpoint
    return estimates


def cost(samples: int) -> dict[str, float | int]:
    # Exact dense matmul FLOPs under the requested convention.
    matmul = 2 * samples * DEPTH * WIDTH**2
    # Pointwise estimate:
    #   DEPTH ReLU comparisons per activation,
    #   final float64 accumulation,
    #   roughly 4n+1 operations per base-row sphere normalization.
    relu = samples * DEPTH * WIDTH
    accumulation = samples * WIDTH
    sphere = (samples // 2) * (4 * WIDTH + 1)
    pointwise = relu + accumulation + sphere
    estimated_total = matmul + pointwise
    budget_fraction = estimated_total / BUDGET
    return {
        "samples": samples,
        "matmul_flops_exact_fma2": matmul,
        "relu_pointwise_estimate": relu,
        "accumulation_pointwise_estimate": accumulation,
        "sphere_pointwise_estimate": sphere,
        "pointwise_estimate_total": pointwise,
        "estimated_total_flops": estimated_total,
        "estimated_budget_fraction": budget_fraction,
        "score_multiplier_with_floor": max(0.1, budget_fraction),
        "floor_active": budget_fraction < 0.1,
    }


def summarize_mses(mses: np.ndarray, score_multiplier: float) -> dict[str, float]:
    return {
        "mean_mse": float(np.mean(mses)),
        "median_mlp_mse": float(np.median(mses)),
        "p90_mlp_mse": float(np.quantile(mses, 0.9)),
        "max_mlp_mse": float(np.max(mses)),
        "adjusted_mean": float(np.mean(mses) * score_multiplier),
    }


def evaluate_training(
    rows: list[tuple[str, np.ndarray, np.ndarray]],
    samples: tuple[int, ...],
    seeds: tuple[int, ...],
) -> tuple[dict[tuple[int, int], np.ndarray], dict[int, float]]:
    per_config: dict[tuple[int, int], np.ndarray] = {}
    elapsed_by_seed: dict[int, float] = {}
    for seed in seeds:
        base = sphere_sobol_base(seed, max(samples))
        errors = {n: [] for n in samples}
        start = time.perf_counter()
        for position, (_, weights, targets) in enumerate(rows, start=1):
            estimates = nested_estimates(weights, base, samples)
            for n in samples:
                errors[n].append(
                    float(np.mean(np.square(estimates[n] - targets[-1])))
                )
            if position % 10 == 0 or position == len(rows):
                print(
                    f"seed={seed} [{position:2d}/{len(rows)}] "
                    f"N={samples[-1]} mse={errors[samples[-1]][-1]:.3e}",
                    flush=True,
                )
        elapsed_by_seed[seed] = time.perf_counter() - start
        for n in samples:
            per_config[(n, seed)] = np.asarray(errors[n], dtype=np.float64)
    return per_config, elapsed_by_seed


def across_seed_summary(
    per_config: dict[tuple[int, int], np.ndarray],
    samples: tuple[int, ...],
    seeds: tuple[int, ...],
) -> dict[str, object]:
    result = {}
    for n in samples:
        means = np.asarray(
            [np.mean(per_config[(n, seed)]) for seed in seeds],
            dtype=np.float64,
        )
        adjusted = means * float(cost(n)["score_multiplier_with_floor"])
        result[str(n)] = {
            "seed_mean_mse": {
                str(seed): float(value)
                for seed, value in zip(seeds, means, strict=True)
            },
            "mean_across_seeds": float(np.mean(means)),
            "std_across_seeds": float(np.std(means)),
            "coefficient_of_variation": float(
                np.std(means) / np.mean(means)
            ),
            "min_seed_mean_mse": float(np.min(means)),
            "max_seed_mean_mse": float(np.max(means)),
            "max_to_min_ratio": float(np.max(means) / np.min(means)),
            "best_seed": int(seeds[int(np.argmin(adjusted))]),
            "best_adjusted_mean": float(np.min(adjusted)),
            "median_adjusted_mean": float(np.median(adjusted)),
        }
    return result


def evaluate_test_once(
    rows: list[tuple[str, np.ndarray, np.ndarray]],
    samples: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float | int | str]], float]:
    base = sphere_sobol_base(seed, samples)
    mses = []
    records = []
    start = time.perf_counter()
    for position, (name, weights, targets) in enumerate(rows, start=1):
        estimate = nested_estimates(weights, base, (samples,))[samples]
        mse = float(np.mean(np.square(estimate - targets[-1])))
        mses.append(mse)
        records.append(
            {
                "id": position + 49,
                "name": name,
                "mse": mse,
            }
        )
        if position % 10 == 0 or position == len(rows):
            print(
                f"test [{position:2d}/{len(rows)}] id={position + 49} "
                f"mse={mse:.3e}",
                flush=True,
            )
    return np.asarray(mses), records, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    costs = {n: cost(n) for n in SAMPLE_GRID}
    # Test rows are deliberately not loaded until after selection.
    train_rows = _load_rows(args.data, list(range(50)))
    per_config, elapsed_by_seed = evaluate_training(
        train_rows,
        SAMPLE_GRID,
        SEED_GRID,
    )
    configurations = []
    for n in SAMPLE_GRID:
        multiplier = float(costs[n]["score_multiplier_with_floor"])
        for seed in SEED_GRID:
            summary = summarize_mses(per_config[(n, seed)], multiplier)
            configurations.append(
                {
                    "samples": n,
                    "seed": seed,
                    **summary,
                }
            )
    selected = min(
        configurations,
        key=lambda item: item["adjusted_mean"],
    )
    selected_n = int(selected["samples"])
    selected_seed = int(selected["seed"])
    print(
        f"FROZEN selection N={selected_n} seed={selected_seed} "
        f"train_adjusted={selected['adjusted_mean']:.4e}",
        flush=True,
    )

    # The holdout is opened exactly once, for the frozen configuration.
    test_rows = _load_rows(args.data, list(range(50, 100)))
    test_mses, test_records, test_elapsed = evaluate_test_once(
        test_rows,
        selected_n,
        selected_seed,
    )
    selected_cost = costs[selected_n]
    test_summary = summarize_mses(
        test_mses,
        float(selected_cost["score_multiplier_with_floor"]),
    )
    worst = sorted(test_records, key=lambda record: record["mse"], reverse=True)[:8]

    seed_summary = across_seed_summary(
        per_config,
        SAMPLE_GRID,
        SEED_GRID,
    )
    selected_seed_values = seed_summary[str(selected_n)]["seed_mean_mse"]
    selected_seed_mean = float(selected_seed_values[str(selected_seed)])
    seed_means = np.asarray(
        [float(selected_seed_values[str(seed)]) for seed in SEED_GRID]
    )
    stability = {
        "selected_seed_train_rank_of_8": int(
            np.argsort(np.argsort(seed_means))[selected_seed] + 1
        ),
        "selected_seed_vs_median_mse_ratio": float(
            selected_seed_mean / np.median(seed_means)
        ),
        "selected_N_seed_max_to_min_ratio": float(
            np.max(seed_means) / np.min(seed_means)
        ),
        "interpretation": (
            "A selected/median ratio far below one or a large max/min ratio "
            "indicates selection may be exploiting a lucky scramble."
        ),
    }

    result = {
        "protocol": {
            "train_ids": [0, 49],
            "test_ids": [50, 99],
            "sample_grid": list(SAMPLE_GRID),
            "seed_grid": list(SEED_GRID),
            "design": "single reused scrambled Sobol stream, sphere, antithetic",
            "frame_whitening": False,
            "selection": "minimum train adjusted mean over the predeclared grid",
            "test_opened_configs": 1,
        },
        "costs": {str(n): value for n, value in costs.items()},
        "train_configurations": configurations,
        "train_across_seed_stability": seed_summary,
        "nested_research_elapsed_seconds_by_seed": {
            str(seed): elapsed for seed, elapsed in elapsed_by_seed.items()
        },
        "selected": selected,
        "selected_cost": selected_cost,
        "selected_seed_stability": stability,
        "test": {
            **test_summary,
            "elapsed_seconds": test_elapsed,
            "worst_mlps": worst,
        },
        "cost_note": (
            "Adjusted means use exact dense matmul FMA=2 FLOPs plus the stated "
            "pointwise estimate. Sobol generation and inverse-normal "
            "transcendentals are not assigned synthetic FLOP constants. The "
            "0.1 scoring floor remains active only where explicitly shown."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print("\nRESULT_SUMMARY")
    print(
        json.dumps(
            {
                "selected": selected,
                "selected_cost": selected_cost,
                "selected_seed_stability": stability,
                "test": result["test"],
                "out": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
