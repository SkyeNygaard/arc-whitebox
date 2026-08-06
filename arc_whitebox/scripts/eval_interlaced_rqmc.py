"""Evaluate order-two digit-interlaced scrambled nets at near-full budget.

Digit interlacing maps pairs of base-2 coordinates

    0.a1 a2 ... , 0.b1 b2 ... -> 0.a1 b1 a2 b2 ...

and upgrades a suitable digital net to a higher-order digital net.  Such nets
can converge faster for integrands with mixed smoothness.  The present
integrand is only piecewise smooth because of ReLUs and the inverse-normal
sphere map, so this is an empirical regularity test rather than an assumption
that the high-order theorem applies.

Selection uses IDs 0--49.  The disjoint IDs 50--99 are only opened with the
explicit ``--evaluate-holdout`` flag.
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
from scipy.special import ndtri
from scipy.stats import qmc


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))

import eval_multistream_rqmc as rq  # noqa: E402


TOTALS = {"A": 32_768, "D": 30_000}
SEEDS = {"A": 101, "D": 404}


def interlace_pairs(uniform: np.ndarray, bits: int) -> np.ndarray:
    """Interlace ``bits`` binary digits from adjacent coordinate pairs."""
    if uniform.shape[1] != 2 * rq.WIDTH:
        raise ValueError(uniform.shape)
    scale = 1 << bits
    integer = np.floor(uniform * scale).astype(np.uint64)
    integer = integer.reshape(len(integer), rq.WIDTH, 2)
    result = np.zeros((len(integer), rq.WIDTH), dtype=np.uint64)
    for digit in range(bits):
        source_shift = bits - 1 - digit
        first = (integer[:, :, 0] >> source_shift) & np.uint64(1)
        second = (integer[:, :, 1] >> source_shift) & np.uint64(1)
        result |= first << np.uint64(2 * bits - 1 - 2 * digit)
        result |= second << np.uint64(2 * bits - 2 - 2 * digit)
    # Midpoints avoid inverse-normal infinities while changing only digits
    # beyond those supplied by the digital net.
    return (result.astype(np.float64) + 0.5) / float(1 << (2 * bits))


def make_interlaced_stream(
    total: int,
    seed: int,
    *,
    bits: int = 26,
) -> np.ndarray:
    if total % 2:
        raise ValueError("antithetic stream requires an even total")
    base = total // 2
    engine = qmc.Sobol(
        d=2 * rq.WIDTH,
        scramble=True,
        seed=seed,
        bits=bits,
    )
    exponent = int(math.log2(base))
    if 2**exponent == base:
        source = engine.random_base2(exponent)
    else:
        source = engine.random(base)
    uniform = interlace_pairs(source, bits)
    eps = np.finfo(np.float64).eps
    gaussian = ndtri(np.clip(uniform, eps, 1.0 - eps))
    gaussian /= np.linalg.norm(gaussian, axis=1, keepdims=True)
    paired = np.stack([gaussian, -gaussian], axis=1).reshape(
        total, rq.WIDTH
    )
    paired *= rq.expected_chi(rq.WIDTH)
    return paired.astype(np.float32)


def final_mean(directions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    activations = directions.copy()
    for weight in weights:
        activations = activations @ weight
        np.maximum(activations, 0.0, out=activations)
    return activations.mean(axis=0, dtype=np.float64)


def collect_records(
    ids: set[int],
    designs: dict[str, np.ndarray],
) -> list[dict]:
    records = []
    for mlp_id, name, weights, target in rq.iter_rows():
        if mlp_id not in ids:
            continue
        start = time.perf_counter()
        predictions = {
            design: final_mean(directions, weights)
            for design, directions in designs.items()
        }
        records.append(
            {
                "mlp_id": mlp_id,
                "mlp_name": name,
                "target": target,
                "predictions": predictions,
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


def fit_simplex(records: list[dict], names: tuple[str, str]) -> float:
    first = np.concatenate(
        [record["predictions"][names[0]] for record in records]
    )
    second = np.concatenate(
        [record["predictions"][names[1]] for record in records]
    )
    target = np.concatenate([record["target"] for record in records])
    delta = second - first
    weight_second = np.dot(delta, target - first) / max(
        np.dot(delta, delta), 1e-30
    )
    return float(np.clip(weight_second, 0.0, 1.0))


def method_prediction(
    record: dict,
    names: tuple[str, str],
    weight_second: float,
) -> np.ndarray:
    return (
        (1.0 - weight_second) * record["predictions"][names[0]]
        + weight_second * record["predictions"][names[1]]
    )


def evaluate(
    records: list[dict],
    methods: dict[str, tuple[tuple[str, str], float]],
) -> dict:
    output = {}
    for method, (names, weight_second) in methods.items():
        rows = []
        for record in records:
            prediction = method_prediction(record, names, weight_second)
            rows.append(
                {
                    "mlp_id": record["mlp_id"],
                    "mse": rq.mse(prediction, record["target"]),
                }
            )
        output[method] = {
            "designs": list(names),
            "weights": [1.0 - weight_second, weight_second],
            "mean_raw_mse": statistics.fmean(row["mse"] for row in rows),
            "median_mlp_mse": statistics.median(
                row["mse"] for row in rows
            ),
            "rows": rows,
        }
    baseline = output["standard_AD"]["mean_raw_mse"]
    for result in output.values():
        result["ratio_to_standard_AD"] = result["mean_raw_mse"] / baseline
    return output


def cross_validated(
    records: list[dict],
    families: dict[str, tuple[str, str]],
    folds: int = 5,
) -> dict:
    squared_errors = {method: [] for method in families}
    for fold in range(folds):
        train = [
            record for record in records if record["mlp_id"] % folds != fold
        ]
        test = [
            record for record in records if record["mlp_id"] % folds == fold
        ]
        weights = {
            method: fit_simplex(train, names)
            for method, names in families.items()
        }
        for method, names in families.items():
            for record in test:
                prediction = method_prediction(
                    record, names, weights[method]
                )
                squared_errors[method].append(
                    np.mean((prediction - record["target"]) ** 2)
                )
    scores = {
        method: statistics.fmean(errors)
        for method, errors in squared_errors.items()
    }
    baseline = scores["standard_AD"]
    return {
        method: {
            "mean_raw_mse": score,
            "ratio_to_standard_AD": score / baseline,
        }
        for method, score in scores.items()
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
        default=ROOT / "results" / "interlaced_rqmc.json",
    )
    parser.add_argument("--bits", type=int, default=26)
    parser.add_argument("--evaluate-holdout", action="store_true")
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        standard_a = rq.make_stream(TOTALS["A"], SEEDS["A"])
        standard_d = rq.make_stream(TOTALS["D"], SEEDS["D"])
        interlaced_a = make_interlaced_stream(
            TOTALS["A"], SEEDS["A"], bits=args.bits
        )
        interlaced_d = make_interlaced_stream(
            TOTALS["D"], SEEDS["D"], bits=args.bits
        )
    designs = {
        "standard_A": standard_a,
        "standard_D": standard_d,
        "interlaced_A": interlaced_a,
        "interlaced_D": interlaced_d,
    }
    families = {
        "standard_AD": ("standard_A", "standard_D"),
        "interlaced_AD": ("interlaced_A", "interlaced_D"),
        "standardA_interlacedD": ("standard_A", "interlaced_D"),
        "interlacedA_standardD": ("interlaced_A", "standard_D"),
    }

    selection = collect_records(set(range(50)), designs)
    frozen_weights = {
        method: fit_simplex(selection, names)
        for method, names in families.items()
    }
    frozen_methods = {
        method: (names, frozen_weights[method])
        for method, names in families.items()
    }
    selection_evaluation = evaluate(selection, frozen_methods)
    selection_cv = cross_validated(selection, families)

    holdout_evaluation = None
    holdout_records = None
    if args.evaluate_holdout:
        holdout = collect_records(set(range(50, 100)), designs)
        holdout_evaluation = evaluate(holdout, frozen_methods)
        holdout_records = strip_records(holdout)

    artifact = {
        "configuration": {
            "selection_ids": [0, 49],
            "holdout_ids": [50, 99],
            "holdout_evaluated": args.evaluate_holdout,
            "totals": TOTALS,
            "seeds": SEEDS,
            "interlacing_factor": 2,
            "bits_per_source_coordinate": args.bits,
            "all_methods_total_directions": sum(TOTALS.values()),
        },
        "frozen_weight_second": frozen_weights,
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
                "frozen_weight_second": frozen_weights,
                "selection_in_sample": {
                    method: {
                        key: value
                        for key, value in result.items()
                        if key != "rows"
                    }
                    for method, result in selection_evaluation.items()
                },
                "selection_cv": selection_cv,
                "holdout": (
                    {
                        method: {
                            key: value
                            for key, value in result.items()
                            if key != "rows"
                        }
                        for method, result in holdout_evaluation.items()
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
