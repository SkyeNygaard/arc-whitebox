"""Target-free diagnostics for choosing a Kerdock orientation from the weights.

The final 66,048-point Kerdock/MUB rule is expensive, but its first layer is
cheap because it is evaluated by Walsh-Hadamard transforms.  Moreover, the
exact Gaussian mean after the first ReLU is known:

    E[relu(X @ w_j)] = ||w_j|| / sqrt(2 pi).

This script asks whether the discrepancy between that exact mean and the
Kerdock first-layer mean predicts which fixed rotation will have the smallest
final-layer error.  It uses official target values only *after* constructing
the target-free diagnostics, and is hard-restricted to selection IDs 0--49.

Several downstream salience weightings are included:

* raw layer-1 squared error;
* a diagonal squared-path approximation propagated backwards through the
  remaining weights with an independent 1/2 ReLU gate probability;
* a signed linearization that propagates the layer-1 mean error through
  half-open gates.

The script consumes the already-computed dense Kerdock rotation results, so it
does not rerun the 31 expensive layers.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval_kerdock_design import (
    WIDTH,
    make_kerdock_design,
    random_rotation,
)
from eval_sampling_official import DEFAULT_DATA, _load_rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = (
    ROOT / "results" / "kerdock_design_selection.json",
    ROOT / "results" / "kerdock_design_selection_10_49.json",
)
DEFAULT_OUT = ROOT / "results" / "kerdock_adaptive_orientation.json"
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def load_final_errors(paths: tuple[Path, ...]) -> dict[tuple[int, int], float]:
    errors: dict[tuple[int, int], float] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        for record in payload["records"]:
            key = (int(record["index"]), int(record["rotation_seed"]))
            errors[key] = float(record["final_mse"])
    return errors


def backward_squared_salience(weights: np.ndarray) -> np.ndarray:
    """Diagonal approximation to mean squared final sensitivity."""
    salience = np.full(WIDTH, 1.0 / WIDTH, dtype=np.float64)
    for weight in weights[:0:-1]:
        salience = 0.5 * (
            np.square(weight.astype(np.float64)) @ salience
        )
    mean = float(np.mean(salience))
    if not np.isfinite(mean) or mean <= 0.0:
        raise AssertionError("invalid backward salience")
    return salience / mean


def half_gate_linearized_error(
    error: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    propagated = error.astype(np.float64)
    for weight in weights[1:]:
        propagated = 0.5 * (propagated @ weight.astype(np.float64))
    return propagated


def diagnostic_records(
    rows: list[tuple[str, np.ndarray, np.ndarray]],
    indices: list[int],
    seeds: list[int],
    final_errors: dict[tuple[int, int], float],
) -> list[dict[str, float | int | str]]:
    points = make_kerdock_design()
    rotations = {seed: random_rotation(WIDTH, seed) for seed in seeds}
    records: list[dict[str, float | int | str]] = []
    for index, (name, weights, _targets) in zip(indices, rows, strict=True):
        exact = (
            np.linalg.norm(weights[0].astype(np.float64), axis=0)
            * INV_SQRT_2PI
        )
        salience = backward_squared_salience(weights)
        for seed, rotation in rotations.items():
            effective_weight = rotation @ weights[0]
            activation = np.maximum(points @ effective_weight, 0.0)
            estimate = activation.mean(axis=0, dtype=np.float64)
            error = estimate - exact
            linearized = half_gate_linearized_error(error, weights)
            record: dict[str, float | int | str] = {
                "index": index,
                "name": name,
                "rotation_seed": seed,
                "layer1_mse": float(np.mean(np.square(error))),
                "layer1_salience_mse": float(
                    np.mean(np.square(error) * salience)
                ),
                "layer1_max_abs": float(np.max(np.abs(error))),
                "layer1_mean_bias_sq": float(np.square(np.mean(error))),
                "linearized_final_mse": float(
                    np.mean(np.square(linearized))
                ),
            }
            final = final_errors.get((index, seed))
            if final is not None:
                record["final_mse"] = final
            records.append(record)
        print({"index": index, "diagnostics": len(seeds)}, flush=True)
    return records


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def summarize(
    records: list[dict[str, float | int | str]],
    candidate_seeds: list[int],
) -> dict[str, object]:
    features = (
        "layer1_mse",
        "layer1_salience_mse",
        "layer1_max_abs",
        "layer1_mean_bias_sq",
        "linearized_final_mse",
    )
    by_index: dict[int, list[dict[str, float | int | str]]] = defaultdict(list)
    for record in records:
        if (
            int(record["rotation_seed"]) in candidate_seeds
            and "final_mse" in record
        ):
            by_index[int(record["index"])].append(record)

    eligible = {
        index: rows
        for index, rows in by_index.items()
        if {int(row["rotation_seed"]) for row in rows}
        == set(candidate_seeds)
    }
    fixed = {}
    for seed in candidate_seeds:
        fixed[str(seed)] = float(
            np.mean(
                [
                    float(
                        next(
                            row["final_mse"]
                            for row in rows
                            if int(row["rotation_seed"]) == seed
                        )
                    )
                    for rows in eligible.values()
                ]
            )
        )

    selectors: dict[str, object] = {}
    for feature in features:
        selected_errors = []
        selected_seeds = []
        oracle_errors = []
        rank_correlations = []
        for rows in eligible.values():
            chosen = min(rows, key=lambda row: float(row[feature]))
            selected_errors.append(float(chosen["final_mse"]))
            selected_seeds.append(int(chosen["rotation_seed"]))
            oracle_errors.append(
                min(float(row["final_mse"]) for row in rows)
            )
            if len(rows) > 1:
                feature_values = np.asarray(
                    [float(row[feature]) for row in rows]
                )
                final_values = np.asarray(
                    [float(row["final_mse"]) for row in rows]
                )
                corr = np.corrcoef(
                    rankdata(feature_values),
                    rankdata(final_values),
                )[0, 1]
                rank_correlations.append(float(corr))
        counts = {
            str(seed): selected_seeds.count(seed)
            for seed in candidate_seeds
        }
        selectors[feature] = {
            "mean_final_mse": float(np.mean(selected_errors)),
            "median_final_mse": float(np.median(selected_errors)),
            "mean_within_network_spearman": float(
                np.mean(rank_correlations)
            ),
            "selection_counts": counts,
            "oracle_mean_final_mse": float(np.mean(oracle_errors)),
        }
    return {
        "candidate_seeds": candidate_seeds,
        "eligible_networks": len(eligible),
        "fixed_seed_mean_final_mse": fixed,
        "selectors": selectors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=list(range(50)),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(8)),
    )
    parser.add_argument(
        "--candidate-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 3, 5],
    )
    parser.add_argument(
        "--result-inputs",
        type=Path,
        nargs="+",
        default=list(DEFAULT_RESULTS),
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.indices or min(args.indices) < 0 or max(args.indices) >= 50:
        raise ValueError("adaptive diagnostics are restricted to IDs 0--49")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("this diagnostic accepts nonnegative rotation seeds")

    errors = load_final_errors(tuple(args.result_inputs))
    rows = _load_rows(args.data, args.indices)
    records = diagnostic_records(
        rows,
        args.indices,
        args.seeds,
        errors,
    )
    summaries = [
        summarize(records, args.candidate_seeds),
    ]
    if set(range(8)).issubset(args.seeds):
        summaries.append(summarize(records, list(range(8))))
    payload = {
        "protocol": {
            "selection_indices": args.indices,
            "holdout_loaded": False,
            "diagnostics_use_targets": False,
            "targets_used_only_for_retrospective_final_mse": True,
        },
        "summaries": summaries,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print({"out": str(args.out), "summaries": summaries}, flush=True)


if __name__ == "__main__":
    main()
