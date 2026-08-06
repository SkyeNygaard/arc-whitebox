"""Cross-fitted full layer-1 control variate for near-full sphere RQMC.

For the first ReLU layer of a bias-free Gaussian MLP, both the activation mean
and covariance are available in closed form (the arc-cosine kernel).  The
network's positive homogeneity lets us use the corresponding fixed-radius
sphere moments after integrating the Gaussian radius exactly.

This script fits a full 256-by-256 regression from centered layer-1
activations to final outputs on each of two independent scrambled nets.  By
default it performs the regression on antithetic pair averages.  This matters:
the estimator cancels all odd input components exactly, so a pointwise
regression wastes capacity fitting variation that can never enter the sample
mean.  Each coefficient matrix is applied only to the other net:

    mu_A* = Q_A f + (E[a1] - Q_A a1) B_D
    mu_D* = Q_D f + (E[a1] - Q_D a1) B_A.

Conditional on the fitted coefficient, the correction on the other scramble
has expectation zero.  Thus this is an unbiased randomized control-variate
construction despite estimating 65,536 coefficients separately for every
network.  It needs no additional network forward; the dominant extra work is
one ``a1.T @ final`` contraction per stream.

IDs 0--49 select the covariance ridge and universal blend coefficients.  IDs
50--99 remain untouched unless ``--evaluate-holdout`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import json
import math
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


TOTALS = {"A": 32_768, "D": 30_000}
SEEDS = {"A": 101, "D": 404}
RIDGES = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def exact_layer1_sphere_moments(
    first_weight: np.ndarray,
    *,
    pair_averaged: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return E[a1] and Cov(a1) for x=E[chi_d] times uniform sphere."""
    weight = first_weight.astype(np.float64)
    gram = weight.T @ weight
    sigma = np.sqrt(np.maximum(np.diag(gram), 1e-30))
    rho = gram / np.maximum(sigma[:, None] * sigma[None, :], 1e-30)
    rho = np.clip(rho, -1.0, 1.0)
    root = np.sqrt(np.maximum(1.0 - rho * rho, 0.0))
    if pair_averaged:
        # For an antithetic pair, (ReLU(h)+ReLU(-h))/2 = |h|/2.
        # E[|X||Y|] has the closed form below.
        gaussian_second = (
            sigma[:, None]
            * sigma[None, :]
            / (2.0 * math.pi)
            * (root + rho * np.arcsin(rho))
        )
    else:
        gaussian_second = (
            sigma[:, None]
            * sigma[None, :]
            / (2.0 * math.pi)
            * (root + (math.pi - np.arccos(rho)) * rho)
        )
    mean = sigma / math.sqrt(2.0 * math.pi)
    radius = rq.expected_chi(rq.WIDTH)
    sphere_second = (radius * radius / rq.WIDTH) * gaussian_second
    covariance = sphere_second - mean[:, None] * mean[None, :]
    covariance = 0.5 * (covariance + covariance.T)
    return mean, covariance


def stream_statistics(
    directions: np.ndarray,
    weights: np.ndarray,
    exact_layer1_mean: np.ndarray,
    *,
    pair_averaged: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return final mean, layer-1 mean, and Cov(a1, final) estimate."""
    first = directions @ weights[0]
    np.maximum(first, 0.0, out=first)
    activations = first
    for weight in weights[1:]:
        activations = activations @ weight
        np.maximum(activations, 0.0, out=activations)
    final_mean = activations.mean(axis=0, dtype=np.float64)
    layer1_mean = first.mean(axis=0, dtype=np.float64)
    if pair_averaged:
        if len(first) % 2:
            raise ValueError("pair regression requires an even stream")
        regression_first = first.reshape(
            -1, 2, first.shape[1]
        ).mean(axis=1)
        regression_final = activations.reshape(
            -1, 2, activations.shape[1]
        ).mean(axis=1)
    else:
        regression_first = first
        regression_final = activations
    # Centering final values removes the otherwise very noisy constant
    # component caused by the finite QMC error of a1.
    centered_first = (
        regression_first.astype(np.float64)
        - exact_layer1_mean[None, :]
    )
    centered_final = (
        regression_final.astype(np.float64)
        - regression_final.mean(axis=0, dtype=np.float64)[None, :]
    )
    cross_covariance = centered_first.T @ centered_final / len(first)
    if pair_averaged:
        # ``len(first)`` counts physical rows; pair regression has half as
        # many observations.
        cross_covariance *= 2.0
    return final_mean, layer1_mean, cross_covariance


def regression_matrix(
    covariance: np.ndarray,
    cross_covariance: np.ndarray,
    ridge_fraction: float,
) -> np.ndarray:
    ridge = ridge_fraction * float(np.trace(covariance) / len(covariance))
    return np.linalg.solve(
        covariance + ridge * np.eye(len(covariance)),
        cross_covariance,
    )


def collect_records(
    ids: set[int],
    streams: dict[str, np.ndarray],
    *,
    pair_averaged: bool,
) -> list[dict]:
    records = []
    for mlp_id, name, weights, target in rq.iter_rows():
        if mlp_id not in ids:
            continue
        start = time.perf_counter()
        exact_mean, exact_covariance = exact_layer1_sphere_moments(
            weights[0], pair_averaged=pair_averaged
        )
        mean_a, layer1_a, cross_a = stream_statistics(
            streams["A"],
            weights,
            exact_mean,
            pair_averaged=pair_averaged,
        )
        mean_d, layer1_d, cross_d = stream_statistics(
            streams["D"],
            weights,
            exact_mean,
            pair_averaged=pair_averaged,
        )
        corrections = {}
        for ridge in RIDGES:
            coefficient_a = regression_matrix(
                exact_covariance, cross_a, ridge
            )
            coefficient_d = regression_matrix(
                exact_covariance, cross_d, ridge
            )
            # Opposite-scramble application is the essential cross-fit.
            correction_a = (exact_mean - layer1_a) @ coefficient_d
            correction_d = (exact_mean - layer1_d) @ coefficient_a
            corrections[str(ridge)] = {
                "A": correction_a,
                "D": correction_d,
            }
        records.append(
            {
                "mlp_id": mlp_id,
                "mlp_name": name,
                "target": target,
                "A": mean_a,
                "D": mean_d,
                "corrections": corrections,
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


def fit_coefficients(
    records: list[dict],
    ridge: float | None,
    *,
    theory: bool,
) -> np.ndarray:
    """Fit A + b0(D-A) [+ b1*cA + b2*cD]."""
    a = flatten(records, lambda record: record["A"])
    delta = flatten(records, lambda record: record["D"] - record["A"])
    target = flatten(records, lambda record: record["target"])
    if theory:
        weight_d = np.dot(delta, target - a) / max(
            np.dot(delta, delta), 1e-30
        )
        return np.asarray([np.clip(weight_d, 0.0, 1.0)])
    if ridge is None:
        raise ValueError("free correction fit needs a ridge choice")
    key = str(ridge)
    design = np.stack(
        [
            delta,
            flatten(
                records,
                lambda record: record["corrections"][key]["A"],
            ),
            flatten(
                records,
                lambda record: record["corrections"][key]["D"],
            ),
        ],
        axis=1,
    )
    gram = design.T @ design
    tiny_ridge = 1e-8 * np.trace(gram) / len(gram)
    return np.linalg.solve(
        gram + tiny_ridge * np.eye(len(gram)),
        design.T @ (target - a),
    )


def prediction(
    record: dict,
    method: str,
    coefficients: np.ndarray,
    ridge: float | None,
) -> np.ndarray:
    base = record["A"] + coefficients[0] * (
        record["D"] - record["A"]
    )
    if method == "baseline":
        return base
    if ridge is None:
        raise ValueError("corrected prediction needs ridge")
    correction = record["corrections"][str(ridge)]
    if method == "theory":
        return (
            base
            + (1.0 - coefficients[0]) * correction["A"]
            + coefficients[0] * correction["D"]
        )
    if method == "free":
        return (
            base
            + coefficients[1] * correction["A"]
            + coefficients[2] * correction["D"]
        )
    raise ValueError(method)


def score(
    records: list[dict],
    method: str,
    coefficients: np.ndarray,
    ridge: float | None,
) -> dict:
    rows = []
    for record in records:
        estimate = prediction(record, method, coefficients, ridge)
        rows.append(
            {
                "mlp_id": record["mlp_id"],
                "mse": rq.mse(estimate, record["target"]),
            }
        )
    return {
        "mean_raw_mse": statistics.fmean(row["mse"] for row in rows),
        "median_mlp_mse": statistics.median(row["mse"] for row in rows),
        "rows": rows,
    }


def candidate_name(method: str, ridge: float | None = None) -> str:
    return method if ridge is None else f"{method}_ridge_{ridge:g}"


def fit_candidates(records: list[dict]) -> dict[str, dict]:
    baseline_coefficients = fit_coefficients(records, None, theory=True)
    candidates = {
        "baseline": {
            "method": "baseline",
            "ridge": None,
            "coefficients": baseline_coefficients,
        }
    }
    for ridge in RIDGES:
        theory_name = candidate_name("theory", ridge)
        free_name = candidate_name("free", ridge)
        candidates[theory_name] = {
            "method": "theory",
            "ridge": ridge,
            "coefficients": baseline_coefficients,
        }
        candidates[free_name] = {
            "method": "free",
            "ridge": ridge,
            "coefficients": fit_coefficients(
                records, ridge, theory=False
            ),
        }
    return candidates


def evaluate_candidates(
    records: list[dict],
    candidates: dict[str, dict],
) -> dict:
    output = {}
    for name, candidate in candidates.items():
        result = score(
            records,
            candidate["method"],
            candidate["coefficients"],
            candidate["ridge"],
        )
        output[name] = {
            "method": candidate["method"],
            "ridge": candidate["ridge"],
            "coefficients": candidate["coefficients"].tolist(),
            **result,
        }
    baseline = output["baseline"]["mean_raw_mse"]
    for result in output.values():
        result["ratio_to_baseline"] = result["mean_raw_mse"] / baseline
    return output


def cross_validated(
    records: list[dict],
    folds: int = 5,
) -> dict[str, dict]:
    errors: dict[str, list[float]] = {}
    for fold in range(folds):
        train = [
            record for record in records if record["mlp_id"] % folds != fold
        ]
        test = [
            record for record in records if record["mlp_id"] % folds == fold
        ]
        candidates = fit_candidates(train)
        for name, candidate in candidates.items():
            errors.setdefault(name, [])
            for record in test:
                estimate = prediction(
                    record,
                    candidate["method"],
                    candidate["coefficients"],
                    candidate["ridge"],
                )
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
            "correction_rms": {
                key: {
                    stream: float(np.sqrt(np.mean(value[stream] ** 2)))
                    for stream in ("A", "D")
                }
                for key, value in record["corrections"].items()
            },
        }
        for record in records
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "crossfit_layer1_pair_cv.json",
    )
    parser.add_argument(
        "--individual-regression",
        action="store_true",
        help="Fit pointwise rather than on antithetic pair averages.",
    )
    parser.add_argument("--evaluate-holdout", action="store_true")
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        streams = {
            name: rq.make_stream(TOTALS[name], SEEDS[name])
            for name in ("A", "D")
        }
    pair_averaged = not args.individual_regression
    selection = collect_records(
        set(range(50)), streams, pair_averaged=pair_averaged
    )
    frozen_candidates = fit_candidates(selection)
    selection_evaluation = evaluate_candidates(
        selection, frozen_candidates
    )
    selection_cv = cross_validated(selection)
    chosen = min(
        selection_cv,
        key=lambda name: selection_cv[name]["mean_raw_mse"],
    )

    holdout_evaluation = None
    holdout_records = None
    if args.evaluate_holdout:
        holdout = collect_records(
            set(range(50, 100)),
            streams,
            pair_averaged=pair_averaged,
        )
        holdout_evaluation = evaluate_candidates(
            holdout, {chosen: frozen_candidates[chosen], "baseline": frozen_candidates["baseline"]}
        )
        holdout_records = strip_records(holdout)

    artifact = {
        "configuration": {
            "selection_ids": [0, 49],
            "holdout_ids": [50, 99],
            "holdout_evaluated": args.evaluate_holdout,
            "totals": TOTALS,
            "seeds": SEEDS,
            "ridges": RIDGES,
            "regression_observations": (
                "antithetic_pair_averages"
                if pair_averaged
                else "individual_rows"
            ),
            "online_dominant_extra_flops": (
                "2 * total_directions * 256 * 256"
            ),
            "unbiasedness": (
                "each learned coefficient is applied only to the independent "
                "opposite scramble, whose layer-1 discrepancy has mean zero"
            ),
        },
        "chosen_by_five_fold_selection": chosen,
        "frozen_candidate": {
            key: (
                value.tolist()
                if isinstance(value, np.ndarray)
                else value
            )
            for key, value in frozen_candidates[chosen].items()
        },
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
                "frozen_candidate": artifact["frozen_candidate"],
                "selection_cv": selection_cv,
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
