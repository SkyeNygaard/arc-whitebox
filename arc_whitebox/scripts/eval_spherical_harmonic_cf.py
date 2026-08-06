"""Cross-scramble spherical-harmonic control functional for near-full RQMC.

The bias-free ReLU network is positively homogeneous, so the Gaussian radial
coordinate can be integrated analytically and only an integral over the unit
sphere remains.  Antithetic sampling integrates every odd spherical harmonic
exactly.  This script targets the first remaining non-constant component,
degree two, with its reproducing kernel

    K_2(u, v) = d (d + 2) / 2 * ((u.T v)**2 - 1 / d).

For independent randomized rules A and D, define

    h_A(v) = Q_A K_2(u, v).

Then E_D[f(v) h_A(v) | A] = Q_A P_2 f, the degree-two contribution
to A's quadrature error.  Consequently

    Q_A f - Q_D[f h_A]

is an unbiased randomized control-functional estimator.  The symmetric
version uses both streams.  Since the point sets are fixed for submission,
h_A evaluated on D (and h_D evaluated on A) can be precomputed; online
overhead is only two weighted reductions of the already-computed outputs.

IDs 0--49 are used for selection and coefficient freezing.  IDs 50--99 are
only evaluated when ``--evaluate-holdout`` is explicitly supplied.
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


STREAM_TOTALS = {"A": 32_768, "D": 30_000}
STREAM_SEEDS = {"A": 101, "D": 404}


def harmonic_cross_weight(
    source_directions: np.ndarray,
    target_directions: np.ndarray,
) -> np.ndarray:
    """Return h_source(v)=Q_source K_2(u,v) on target directions."""
    radius = rq.expected_chi(rq.WIDTH)
    source = source_directions.astype(np.float64) / radius
    target = target_directions.astype(np.float64) / radius
    second_moment = source.T @ source / len(source)
    quadratic = np.einsum(
        "ni,ij,nj->n", target, second_moment, target, optimize=True
    )
    normalization = rq.WIDTH * (rq.WIDTH + 2) / 2
    return normalization * (quadratic - 1.0 / rq.WIDTH)


def stream_statistics(
    directions: np.ndarray,
    weights: np.ndarray,
    cross_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Q[f] and Q[f*h] from one network forward."""
    activations = directions.copy()
    for weight in weights:
        activations = activations @ weight
        np.maximum(activations, 0.0, out=activations)
    mean = activations.mean(axis=0, dtype=np.float64)
    weighted = np.mean(
        activations.astype(np.float64) * cross_weight[:, None], axis=0
    )
    return mean, weighted


def collect_records(
    ids: set[int],
    streams: dict[str, np.ndarray],
    h_a_on_d: np.ndarray,
    h_d_on_a: np.ndarray,
) -> list[dict]:
    records = []
    for mlp_id, name, weights, target in rq.iter_rows():
        if mlp_id not in ids:
            continue
        start = time.perf_counter()
        # t_DA estimates D's degree-two quadrature error using A, while
        # t_AD estimates A's degree-two quadrature error using D.
        mean_a, t_da = stream_statistics(
            streams["A"], weights, h_d_on_a
        )
        mean_d, t_ad = stream_statistics(
            streams["D"], weights, h_a_on_d
        )
        records.append(
            {
                "mlp_id": mlp_id,
                "mlp_name": name,
                "target": target,
                "A": mean_a,
                "D": mean_d,
                "t_AD": t_ad,
                "t_DA": t_da,
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


def flattened(records: list[dict], key: str) -> np.ndarray:
    return np.concatenate([record[key] for record in records])


def fit_baseline_weight(records: list[dict]) -> float:
    """Fit D's weight with A+D constrained to sum to one, clipped to simplex."""
    a = flattened(records, "A")
    d = flattened(records, "D")
    target = flattened(records, "target")
    delta = d - a
    weight_d = np.dot(delta, target - a) / max(np.dot(delta, delta), 1e-30)
    return float(np.clip(weight_d, 0.0, 1.0))


def fit_tempering(records: list[dict], weight_d: float) -> float:
    """Fit one multiplier for the theoretically normalized H2 correction."""
    weight_a = 1.0 - weight_d
    base = (
        weight_a * flattened(records, "A")
        + weight_d * flattened(records, "D")
    )
    correction = (
        weight_a * flattened(records, "t_AD")
        + weight_d * flattened(records, "t_DA")
    )
    target = flattened(records, "target")
    coefficient = np.dot(correction, base - target) / max(
        np.dot(correction, correction), 1e-30
    )
    return float(coefficient)


def fit_free_coefficients(
    records: list[dict],
    ridge_fraction: float = 1e-6,
) -> np.ndarray:
    """Fit A + b0(D-A) + b1*t_AD + b2*t_DA."""
    a = flattened(records, "A")
    design = np.stack(
        [
            flattened(records, "D") - a,
            flattened(records, "t_AD"),
            flattened(records, "t_DA"),
        ],
        axis=1,
    )
    response = flattened(records, "target") - a
    gram = design.T @ design
    ridge = ridge_fraction * np.trace(gram) / len(gram)
    return np.linalg.solve(
        gram + ridge * np.eye(len(gram)), design.T @ response
    )


def predict(
    record: dict,
    method: str,
    *,
    weight_d: float,
    tempering: float,
    free_coefficients: np.ndarray,
) -> np.ndarray:
    weight_a = 1.0 - weight_d
    baseline = weight_a * record["A"] + weight_d * record["D"]
    correction = (
        weight_a * record["t_AD"] + weight_d * record["t_DA"]
    )
    if method == "baseline":
        return baseline
    if method == "h2_theory":
        return baseline - correction
    if method == "h2_tempered":
        return baseline - tempering * correction
    if method == "h2_free":
        features = np.stack(
            [
                record["D"] - record["A"],
                record["t_AD"],
                record["t_DA"],
            ],
            axis=1,
        )
        return record["A"] + features @ free_coefficients
    raise ValueError(method)


def evaluate(
    records: list[dict],
    *,
    weight_d: float,
    tempering: float,
    free_coefficients: np.ndarray,
) -> dict[str, dict]:
    methods = ("baseline", "h2_theory", "h2_tempered", "h2_free")
    result = {}
    for method in methods:
        rows = []
        for record in records:
            prediction = predict(
                record,
                method,
                weight_d=weight_d,
                tempering=tempering,
                free_coefficients=free_coefficients,
            )
            rows.append(
                {
                    "mlp_id": record["mlp_id"],
                    "mse": rq.mse(prediction, record["target"]),
                }
            )
        result[method] = {
            "mean_raw_mse": statistics.fmean(row["mse"] for row in rows),
            "median_mlp_mse": statistics.median(
                row["mse"] for row in rows
            ),
            "rows": rows,
        }
    baseline = result["baseline"]["mean_raw_mse"]
    for value in result.values():
        value["ratio_to_baseline"] = value["mean_raw_mse"] / baseline
    return result


def cross_validated(records: list[dict], folds: int = 5) -> dict[str, float]:
    predictions = {
        method: [] for method in ("baseline", "h2_theory", "h2_tempered", "h2_free")
    }
    targets = []
    for fold in range(folds):
        train = [
            record for record in records if record["mlp_id"] % folds != fold
        ]
        test = [
            record for record in records if record["mlp_id"] % folds == fold
        ]
        weight_d = fit_baseline_weight(train)
        tempering = fit_tempering(train, weight_d)
        free = fit_free_coefficients(train)
        for record in test:
            targets.append(record["target"])
            for method in predictions:
                predictions[method].append(
                    predict(
                        record,
                        method,
                        weight_d=weight_d,
                        tempering=tempering,
                        free_coefficients=free,
                    )
                )
    stacked_target = np.stack(targets)
    scores = {
        method: float(
            np.mean((np.stack(values) - stacked_target) ** 2)
        )
        for method, values in predictions.items()
    }
    baseline = scores["baseline"]
    return {
        method: {
            "mean_raw_mse": score,
            "ratio_to_baseline": score / baseline,
        }
        for method, score in scores.items()
    }


def strip_records(records: list[dict]) -> list[dict]:
    return [
        {
            "mlp_id": record["mlp_id"],
            "mlp_name": record["mlp_name"],
            "seconds": record["seconds"],
            "rms_t_AD": float(np.sqrt(np.mean(record["t_AD"] ** 2))),
            "rms_t_DA": float(np.sqrt(np.mean(record["t_DA"] ** 2))),
        }
        for record in records
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "spherical_harmonic_cf.json",
    )
    parser.add_argument(
        "--evaluate-holdout",
        action="store_true",
        help="After freezing on IDs 0--49, evaluate the untouched IDs 50--99.",
    )
    args = parser.parse_args()

    rq.STREAM_TOTALS.update(STREAM_TOTALS)
    rq.STREAM_SEEDS.update(STREAM_SEEDS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        streams = {
            name: rq.make_stream(STREAM_TOTALS[name], STREAM_SEEDS[name])
            for name in ("A", "D")
        }

    h_a_on_d = harmonic_cross_weight(streams["A"], streams["D"])
    h_d_on_a = harmonic_cross_weight(streams["D"], streams["A"])
    diagnostics = {
        "h_A_on_D_mean": float(np.mean(h_a_on_d)),
        "h_A_on_D_rms": float(np.sqrt(np.mean(h_a_on_d**2))),
        "h_A_on_D_max_abs": float(np.max(np.abs(h_a_on_d))),
        "h_D_on_A_mean": float(np.mean(h_d_on_a)),
        "h_D_on_A_rms": float(np.sqrt(np.mean(h_d_on_a**2))),
        "h_D_on_A_max_abs": float(np.max(np.abs(h_d_on_a))),
    }

    selection = collect_records(
        set(range(50)), streams, h_a_on_d, h_d_on_a
    )
    weight_d = fit_baseline_weight(selection)
    tempering = fit_tempering(selection, weight_d)
    free = fit_free_coefficients(selection)
    selection_evaluation = evaluate(
        selection,
        weight_d=weight_d,
        tempering=tempering,
        free_coefficients=free,
    )
    selection_cv = cross_validated(selection)

    holdout_evaluation = None
    holdout_records = None
    if args.evaluate_holdout:
        holdout = collect_records(
            set(range(50, 100)), streams, h_a_on_d, h_d_on_a
        )
        holdout_evaluation = evaluate(
            holdout,
            weight_d=weight_d,
            tempering=tempering,
            free_coefficients=free,
        )
        holdout_records = strip_records(holdout)

    artifact = {
        "configuration": {
            "selection_ids": [0, 49],
            "holdout_ids": [50, 99],
            "holdout_evaluated": args.evaluate_holdout,
            "stream_totals": STREAM_TOTALS,
            "stream_seeds": STREAM_SEEDS,
            "kernel": "d*(d+2)/2 * ((u.v)^2 - 1/d)",
            "online_extra_work": (
                "two fixed scalar-weighted reductions of final outputs"
            ),
        },
        "kernel_diagnostics": diagnostics,
        "frozen_coefficients": {
            "weight_A": 1.0 - weight_d,
            "weight_D": weight_d,
            "tempering": tempering,
            "free": free.tolist(),
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
                "kernel_diagnostics": diagnostics,
                "frozen_coefficients": artifact["frozen_coefficients"],
                "selection_in_sample": {
                    key: {
                        k: v
                        for k, v in value.items()
                        if k != "rows"
                    }
                    for key, value in selection_evaluation.items()
                },
                "selection_cv": selection_cv,
                "holdout": (
                    {
                        key: {
                            k: v
                            for k, v in value.items()
                            if k != "rows"
                        }
                        for key, value in holdout_evaluation.items()
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
