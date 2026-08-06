"""Low-rank quadratic layer-1 control functionals for antithetic sphere RQMC.

Antithetic pairing turns the first-layer activation into ``|x W1| / 2``.
Its mean and covariance are analytic under the sphere measure.  After
diagonalizing that covariance, each squared principal coordinate

    phi_k = z_k**2 - eigenvalue_k

has exactly zero mean and is therefore a valid nonlinear control variate.
This script learns final-output regressions on 8--64 such controls on one
scrambled net and applies them only to the other net.  It also evaluates a
joint estimator with the full linear pair control from
``eval_crossfit_layer1_cv.py``.

The construction is an unbiased randomized control functional, with fixed
coefficient selection on IDs 0--49.  IDs 50--99 are only evaluated when
explicitly requested.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))

import eval_multistream_rqmc as rq  # noqa: E402
from eval_crossfit_layer1_cv import (  # noqa: E402
    exact_layer1_sphere_moments,
)


TOTALS = {"A": 32_768, "D": 30_000}
SEEDS = {"A": 101, "D": 404}
RANKS = (8, 16, 32, 64)
QUADRATIC_RIDGES = (1e-2, 1e-1, 1.0)
LINEAR_RIDGE = 1.0


def regress(
    features: np.ndarray,
    outputs: np.ndarray,
    ridge_fraction: float,
) -> np.ndarray:
    centered_outputs = outputs - outputs.mean(axis=0, dtype=np.float64)
    gram = features.T @ features / len(features)
    cross = features.T @ centered_outputs / len(features)
    ridge = ridge_fraction * float(np.trace(gram) / len(gram))
    return np.linalg.solve(
        gram + ridge * np.eye(len(gram)),
        cross,
    )


def forward_pairs(
    directions: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = directions @ weights[0]
    np.maximum(first, 0.0, out=first)
    activations = first
    for weight in weights[1:]:
        activations = activations @ weight
        np.maximum(activations, 0.0, out=activations)
    pair_first = first.reshape(-1, 2, first.shape[1]).mean(axis=1)
    pair_final = activations.reshape(
        -1, 2, activations.shape[1]
    ).mean(axis=1)
    return (
        pair_final.mean(axis=0, dtype=np.float64),
        pair_first.astype(np.float64),
        pair_final.astype(np.float64),
    )


def linear_coefficient(
    pair_first: np.ndarray,
    pair_final: np.ndarray,
    exact_mean: np.ndarray,
    exact_covariance: np.ndarray,
) -> np.ndarray:
    centered_first = pair_first - exact_mean[None, :]
    centered_final = pair_final - pair_final.mean(axis=0)[None, :]
    cross = centered_first.T @ centered_final / len(pair_first)
    ridge = LINEAR_RIDGE * float(
        np.trace(exact_covariance) / len(exact_covariance)
    )
    return np.linalg.solve(
        exact_covariance + ridge * np.eye(len(exact_covariance)),
        cross,
    )


def collect_records(
    ids: set[int],
    streams: dict[str, np.ndarray],
) -> list[dict]:
    records = []
    max_rank = max(RANKS)
    for mlp_id, name, weights, target in rq.iter_rows():
        if mlp_id not in ids:
            continue
        start = time.perf_counter()
        exact_mean, exact_covariance = exact_layer1_sphere_moments(
            weights[0], pair_averaged=True
        )
        eigenvalues, eigenvectors = np.linalg.eigh(exact_covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order][:max_rank], 0.0)
        eigenvectors = eigenvectors[:, order[:max_rank]]

        mean_a, first_a, final_a = forward_pairs(streams["A"], weights)
        mean_d, first_d, final_d = forward_pairs(streams["D"], weights)

        coefficient_a = linear_coefficient(
            first_a, final_a, exact_mean, exact_covariance
        )
        coefficient_d = linear_coefficient(
            first_d, final_d, exact_mean, exact_covariance
        )
        linear_correction_a = (
            exact_mean - first_a.mean(axis=0)
        ) @ coefficient_d
        linear_correction_d = (
            exact_mean - first_d.mean(axis=0)
        ) @ coefficient_a

        coordinate_a = (first_a - exact_mean[None, :]) @ eigenvectors
        coordinate_d = (first_d - exact_mean[None, :]) @ eigenvectors
        feature_a_all = coordinate_a * coordinate_a - eigenvalues[None, :]
        feature_d_all = coordinate_d * coordinate_d - eigenvalues[None, :]

        quadratic = {}
        for rank in RANKS:
            feature_a = feature_a_all[:, :rank]
            feature_d = feature_d_all[:, :rank]
            for ridge in QUADRATIC_RIDGES:
                coefficient_a = regress(feature_a, final_a, ridge)
                coefficient_d = regress(feature_d, final_d, ridge)
                key = f"r{rank}_g{ridge:g}"
                quadratic[key] = {
                    "A": -feature_a.mean(axis=0) @ coefficient_d,
                    "D": -feature_d.mean(axis=0) @ coefficient_a,
                }

        records.append(
            {
                "mlp_id": mlp_id,
                "mlp_name": name,
                "target": target,
                "A": mean_a,
                "D": mean_d,
                "linear": {
                    "A": linear_correction_a,
                    "D": linear_correction_d,
                },
                "quadratic": quadratic,
                "seconds": time.perf_counter() - start,
            }
        )
        print(
            {
                "mlp_id": mlp_id,
                "seconds": records[-1]["seconds"],
            },
            flush=True,
        )
    records.sort(key=lambda record: record["mlp_id"])
    return records


def flatten(records: list[dict], getter) -> np.ndarray:
    return np.concatenate([getter(record) for record in records])


def candidate_specs() -> dict[str, dict]:
    specs = {"baseline": {"quadratic": None, "linear": False}}
    for rank in RANKS:
        for ridge in QUADRATIC_RIDGES:
            key = f"r{rank}_g{ridge:g}"
            specs[f"quadratic_{key}"] = {
                "quadratic": key,
                "linear": False,
            }
            specs[f"joint_{key}"] = {
                "quadratic": key,
                "linear": True,
            }
    return specs


def design_matrix(records: list[dict], spec: dict) -> np.ndarray:
    a = flatten(records, lambda record: record["A"])
    columns = [
        flatten(records, lambda record: record["D"] - record["A"])
    ]
    if spec["linear"]:
        columns.extend(
            [
                flatten(records, lambda record: record["linear"]["A"]),
                flatten(records, lambda record: record["linear"]["D"]),
            ]
        )
    if spec["quadratic"] is not None:
        key = spec["quadratic"]
        columns.extend(
            [
                flatten(
                    records,
                    lambda record: record["quadratic"][key]["A"],
                ),
                flatten(
                    records,
                    lambda record: record["quadratic"][key]["D"],
                ),
            ]
        )
    return a, np.stack(columns, axis=1)


def fit(records: list[dict], spec: dict) -> np.ndarray:
    base, design = design_matrix(records, spec)
    target = flatten(records, lambda record: record["target"])
    gram = design.T @ design
    ridge = 1e-8 * np.trace(gram) / len(gram)
    return np.linalg.solve(
        gram + ridge * np.eye(len(gram)),
        design.T @ (target - base),
    )


def predict(record: dict, spec: dict, coefficients: np.ndarray) -> np.ndarray:
    columns = [record["D"] - record["A"]]
    if spec["linear"]:
        columns.extend([record["linear"]["A"], record["linear"]["D"]])
    if spec["quadratic"] is not None:
        correction = record["quadratic"][spec["quadratic"]]
        columns.extend([correction["A"], correction["D"]])
    return record["A"] + np.stack(columns, axis=1) @ coefficients


def evaluate(
    records: list[dict],
    specs: dict[str, dict],
    coefficients: dict[str, np.ndarray],
) -> dict:
    output = {}
    for name, spec in specs.items():
        rows = []
        for record in records:
            estimate = predict(record, spec, coefficients[name])
            rows.append(
                {
                    "mlp_id": record["mlp_id"],
                    "mse": rq.mse(estimate, record["target"]),
                }
            )
        output[name] = {
            "spec": spec,
            "coefficients": coefficients[name].tolist(),
            "mean_raw_mse": statistics.fmean(row["mse"] for row in rows),
            "median_mlp_mse": statistics.median(
                row["mse"] for row in rows
            ),
            "rows": rows,
        }
    baseline = output["baseline"]["mean_raw_mse"]
    for result in output.values():
        result["ratio_to_baseline"] = result["mean_raw_mse"] / baseline
    return output


def cross_validated(
    records: list[dict],
    specs: dict[str, dict],
    folds: int = 5,
) -> dict:
    errors = {name: [] for name in specs}
    for fold in range(folds):
        train = [
            record for record in records if record["mlp_id"] % folds != fold
        ]
        test = [
            record for record in records if record["mlp_id"] % folds == fold
        ]
        coefficients = {
            name: fit(train, spec) for name, spec in specs.items()
        }
        for name, spec in specs.items():
            for record in test:
                estimate = predict(record, spec, coefficients[name])
                errors[name].append(
                    float(np.mean((estimate - record["target"]) ** 2))
                )
    scores = {
        name: statistics.fmean(values) for name, values in errors.items()
    }
    baseline = scores["baseline"]
    return {
        name: {
            "mean_raw_mse": value,
            "ratio_to_baseline": value / baseline,
        }
        for name, value in scores.items()
    }


def strip_records(records: list[dict]) -> list[dict]:
    return [
        {
            "mlp_id": record["mlp_id"],
            "mlp_name": record["mlp_name"],
            "seconds": record["seconds"],
        }
        for record in records
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "layer1_quadratic_cv.json",
    )
    parser.add_argument("--evaluate-holdout", action="store_true")
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        streams = {
            name: rq.make_stream(TOTALS[name], SEEDS[name])
            for name in ("A", "D")
        }
    specs = candidate_specs()
    selection = collect_records(set(range(50)), streams)
    frozen_coefficients = {
        name: fit(selection, spec) for name, spec in specs.items()
    }
    selection_evaluation = evaluate(
        selection, specs, frozen_coefficients
    )
    selection_cv = cross_validated(selection, specs)
    chosen = min(
        selection_cv,
        key=lambda name: selection_cv[name]["mean_raw_mse"],
    )

    holdout_evaluation = None
    holdout_records = None
    if args.evaluate_holdout:
        holdout = collect_records(set(range(50, 100)), streams)
        holdout_evaluation = evaluate(
            holdout,
            {
                "baseline": specs["baseline"],
                chosen: specs[chosen],
            },
            {
                "baseline": frozen_coefficients["baseline"],
                chosen: frozen_coefficients[chosen],
            },
        )
        holdout_records = strip_records(holdout)

    artifact = {
        "configuration": {
            "selection_ids": [0, 49],
            "holdout_ids": [50, 99],
            "holdout_evaluated": args.evaluate_holdout,
            "totals": TOTALS,
            "seeds": SEEDS,
            "ranks": RANKS,
            "quadratic_ridges": QUADRATIC_RIDGES,
            "linear_ridge": LINEAR_RIDGE,
        },
        "chosen_by_five_fold_selection": chosen,
        "frozen_spec": specs[chosen],
        "frozen_coefficients": frozen_coefficients[chosen].tolist(),
        "selection": {
            "in_sample": selection_evaluation,
            "five_fold_by_network": selection_cv,
            "records": strip_records(selection),
        },
        "holdout": {
            "evaluation": holdout_evaluation,
            "records": holdout_records,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "chosen": chosen,
                "frozen_spec": specs[chosen],
                "frozen_coefficients": frozen_coefficients[chosen].tolist(),
                "selection_cv_chosen": selection_cv[chosen],
                "selection_cv_baseline": selection_cv["baseline"],
                "selection_chosen_in_sample": {
                    key: value
                    for key, value in selection_evaluation[chosen].items()
                    if key != "rows"
                },
                "holdout": (
                    {
                        name: {
                            key: value
                            for key, value in result.items()
                            if key != "rows"
                        }
                        for name, result in holdout_evaluation.items()
                    }
                    if holdout_evaluation is not None
                    else None
                ),
            },
            indent=2,
        )
    )
    print(args.output)


if __name__ == "__main__":
    main()
