"""Train/freeze/test multistream antithetic-sphere RQMC allocation.

Selection IDs are 0--49; IDs 50--99 are not evaluated until all weights,
features, rule families, and thresholds have been selected.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.special import gammaln, ndtri
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "official_phase1_mini" / "data"
BUDGET = 272_000_000_000
WIDTH = 256
DEPTH = 32
STREAM_TOTALS = {"A": 32768, "B": 16384, "C": 8192}
STREAM_SEEDS = {"A": 101, "B": 202, "C": 303}
CHECKPOINTS = {
    "A": [16384, 32768],
    "B": [8192, 16384],
    "C": [4096, 8192],
}


def expected_chi(width: int) -> float:
    return float(
        math.sqrt(2.0)
        * math.exp(
            gammaln((width + 1) / 2.0)
            - gammaln(width / 2.0)
        )
    )


def make_stream(total: int, seed: int) -> np.ndarray:
    base = total // 2
    exponent = int(math.log2(base))
    engine = qmc.Sobol(d=WIDTH, scramble=True, seed=seed)
    if 2**exponent == base:
        uniform = engine.random_base2(exponent)
    else:
        # A non-power-of-two stream is intentionally used only for the
        # predeclared budget-edge 30k ablation.
        uniform = engine.random(base)
    eps = np.finfo(np.float64).eps
    gaussian = ndtri(np.clip(uniform, eps, 1.0 - eps))
    gaussian /= np.linalg.norm(gaussian, axis=1, keepdims=True)
    # Interleave antipodes so every power-of-two prefix is antithetic.
    paired = np.stack([gaussian, -gaussian], axis=1).reshape(
        total, WIDTH
    )
    paired *= expected_chi(WIDTH)
    return paired.astype(np.float32)


def iter_rows():
    index = 0
    for path in sorted(DATA.glob("*.parquet")):
        table = pq.read_table(
            path,
            columns=["mlp_name", "weights", "all_layer_means"],
        )
        for local in range(len(table)):
            yield (
                index,
                table["mlp_name"][local].as_py(),
                np.asarray(
                    table["weights"][local].as_py(), dtype=np.float32
                ),
                np.asarray(
                    table["all_layer_means"][local].as_py(),
                    dtype=np.float64,
                )[-1],
            )
            index += 1


def stream_predictions(
    directions: np.ndarray,
    weights: np.ndarray,
    checkpoints: list[int],
) -> dict[int, np.ndarray]:
    activations = directions.copy()
    for weight in weights:
        activations = activations @ weight
        np.maximum(activations, 0.0, out=activations)
    cumulative = np.cumsum(activations, axis=0, dtype=np.float64)
    return {
        checkpoint: cumulative[checkpoint - 1] / checkpoint
        for checkpoint in checkpoints
    }


def collect_records(ids: set[int], streams: dict[str, np.ndarray]) -> list[dict]:
    records = []
    for mlp_id, name, weights, target in iter_rows():
        if mlp_id not in ids:
            continue
        start = time.perf_counter()
        estimates = {
            stream: stream_predictions(
                directions, weights, CHECKPOINTS[stream]
            )
            for stream, directions in streams.items()
        }
        records.append(
            {
                "mlp_id": mlp_id,
                "mlp_name": name,
                "target": target,
                "full": {
                    stream: estimates[stream][STREAM_TOTALS[stream]]
                    for stream in streams
                },
                "half": {
                    stream: estimates[stream][
                        CHECKPOINTS[stream][0]
                    ]
                    for stream in streams
                },
                "seconds": time.perf_counter() - start,
            }
        )
    records.sort(key=lambda record: record["mlp_id"])
    return records


def fit_simplex_weights(records: list[dict], streams: list[str]) -> np.ndarray:
    errors = np.stack(
        [
            np.stack(
                [record["full"][stream] - record["target"] for stream in streams],
                axis=1,
            )
            for record in records
        ]
    ).reshape(-1, len(streams))
    gram = errors.T @ errors / len(errors)

    def objective(weight):
        return float(weight @ gram @ weight)

    result = minimize(
        objective,
        np.full(len(streams), 1.0 / len(streams)),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(streams),
        constraints={"type": "eq", "fun": lambda weight: weight.sum() - 1.0},
        options={"ftol": 1e-18, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def blend(record: dict, streams: list[str], weights: np.ndarray) -> np.ndarray:
    return sum(
        weight * record["full"][stream]
        for stream, weight in zip(streams, weights, strict=True)
    )


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))


def flops(total_directions: int) -> int:
    return total_directions * DEPTH * 2 * WIDTH * WIDTH


def adjusted(raw_mse: float, total_directions: int) -> float:
    return raw_mse * max(0.1, flops(total_directions) / BUDGET)


def observable_features(record: dict) -> dict[str, float]:
    a = record["full"]["A"]
    b = record["full"]["B"]
    scale = float(np.mean(a * a)) + 1e-20
    return {
        "nested_A": float(
            np.mean((a - record["half"]["A"]) ** 2) / scale
        ),
        "disagree_AB": float(np.mean((a - b) ** 2) / scale),
    }


def evaluate_fixed(
    records: list[dict],
    streams: list[str],
    weights: np.ndarray,
) -> dict:
    total = sum(STREAM_TOTALS[stream] for stream in streams)
    rows = []
    for record in records:
        raw = mse(blend(record, streams, weights), record["target"])
        rows.append(
            {
                "mlp_id": record["mlp_id"],
                "raw_mse": raw,
                "adjusted_score": adjusted(raw, total),
            }
        )
    return {
        "streams": streams,
        "weights": weights.tolist(),
        "total_directions": total,
        "flops": flops(total),
        "compute_factor": flops(total) / BUDGET,
        "mean_raw_mse": statistics.fmean(row["raw_mse"] for row in rows),
        "mean_adjusted_score": statistics.fmean(
            row["adjusted_score"] for row in rows
        ),
        "rows": rows,
    }


def adaptive_candidates(
    records: list[dict],
    weights_ab: np.ndarray,
    weights_abc: np.ndarray,
) -> list[dict]:
    features = {
        record["mlp_id"]: observable_features(record)
        for record in records
    }
    families = [
        {
            "name": "A_then_ABC",
            "feature": "nested_A",
            "low_streams": ["A"],
            "low_weights": np.array([1.0]),
            "high_streams": ["A", "B", "C"],
            "high_weights": weights_abc,
        },
        {
            "name": "A_then_AB",
            "feature": "nested_A",
            "low_streams": ["A"],
            "low_weights": np.array([1.0]),
            "high_streams": ["A", "B"],
            "high_weights": weights_ab,
        },
        {
            "name": "AB_then_ABC",
            "feature": "disagree_AB",
            "low_streams": ["A", "B"],
            "low_weights": weights_ab,
            "high_streams": ["A", "B", "C"],
            "high_weights": weights_abc,
        },
    ]
    candidates = []
    quantiles = np.linspace(0.0, 1.0, 21)
    for family in families:
        feature_values = np.array(
            [
                features[record["mlp_id"]][family["feature"]]
                for record in records
            ]
        )
        thresholds = sorted(
            set(
                [-math.inf, math.inf]
                + [float(np.quantile(feature_values, q)) for q in quantiles]
            )
        )
        for threshold in thresholds:
            rows = []
            for record in records:
                feature = features[record["mlp_id"]][family["feature"]]
                high = feature > threshold
                streams = (
                    family["high_streams"]
                    if high
                    else family["low_streams"]
                )
                weights = (
                    family["high_weights"]
                    if high
                    else family["low_weights"]
                )
                total = sum(STREAM_TOTALS[stream] for stream in streams)
                raw = mse(blend(record, streams, weights), record["target"])
                rows.append(
                    {
                        "mlp_id": record["mlp_id"],
                        "feature": feature,
                        "high_allocation": high,
                        "total_directions": total,
                        "raw_mse": raw,
                        "adjusted_score": adjusted(raw, total),
                    }
                )
            candidates.append(
                {
                    "name": family["name"],
                    "feature": family["feature"],
                    "threshold": threshold,
                    "low_streams": family["low_streams"],
                    "low_weights": family["low_weights"].tolist(),
                    "high_streams": family["high_streams"],
                    "high_weights": family["high_weights"].tolist(),
                    "mean_adjusted_score": statistics.fmean(
                        row["adjusted_score"] for row in rows
                    ),
                    "mean_raw_mse": statistics.fmean(
                        row["raw_mse"] for row in rows
                    ),
                    "high_fraction": statistics.fmean(
                        row["high_allocation"] for row in rows
                    ),
                    "rows": rows,
                }
            )
    return candidates


def evaluate_frozen_adaptive(records: list[dict], rule: dict) -> dict:
    rows = []
    for record in records:
        feature = observable_features(record)[rule["feature"]]
        high = feature > rule["threshold"]
        streams = rule["high_streams"] if high else rule["low_streams"]
        weights = np.asarray(
            rule["high_weights"] if high else rule["low_weights"]
        )
        total = sum(STREAM_TOTALS[stream] for stream in streams)
        raw = mse(blend(record, streams, weights), record["target"])
        rows.append(
            {
                "mlp_id": record["mlp_id"],
                "feature": feature,
                "high_allocation": high,
                "total_directions": total,
                "flops": flops(total),
                "raw_mse": raw,
                "adjusted_score": adjusted(raw, total),
            }
        )
    return {
        **{key: rule[key] for key in (
            "name",
            "feature",
            "threshold",
            "low_streams",
            "low_weights",
            "high_streams",
            "high_weights",
        )},
        "mean_raw_mse": statistics.fmean(row["raw_mse"] for row in rows),
        "mean_adjusted_score": statistics.fmean(
            row["adjusted_score"] for row in rows
        ),
        "high_fraction": statistics.fmean(
            row["high_allocation"] for row in rows
        ),
        "mean_directions": statistics.fmean(
            row["total_directions"] for row in rows
        ),
        "rows": rows,
    }


def strip_arrays(records: list[dict]) -> list[dict]:
    return [
        {
            "mlp_id": record["mlp_id"],
            "mlp_name": record["mlp_name"],
            "seconds": record["seconds"],
            "features": observable_features(record),
        }
        for record in records
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "multistream_rqmc.json",
    )
    args = parser.parse_args()

    streams = {
        name: make_stream(STREAM_TOTALS[name], STREAM_SEEDS[name])
        for name in ("A", "B", "C")
    }
    # Selection is fully completed before test rows are loaded/evaluated.
    selection = collect_records(set(range(50)), streams)
    weights_ab = fit_simplex_weights(selection, ["A", "B"])
    weights_abc = fit_simplex_weights(selection, ["A", "B", "C"])
    selection_fixed = {
        "A": evaluate_fixed(selection, ["A"], np.array([1.0])),
        "AB": evaluate_fixed(selection, ["A", "B"], weights_ab),
        "ABC": evaluate_fixed(
            selection, ["A", "B", "C"], weights_abc
        ),
    }
    candidates = adaptive_candidates(selection, weights_ab, weights_abc)
    best_rule = min(candidates, key=lambda rule: rule["mean_adjusted_score"])

    # Freeze, then open IDs 50--99 exactly once.
    test = collect_records(set(range(50, 100)), streams)
    test_fixed = {
        "A": evaluate_fixed(test, ["A"], np.array([1.0])),
        "AB": evaluate_fixed(test, ["A", "B"], weights_ab),
        "ABC": evaluate_fixed(test, ["A", "B", "C"], weights_abc),
    }
    test_adaptive = evaluate_frozen_adaptive(test, best_rule)

    def outlier(method: dict, mlp_id: int) -> dict:
        return next(row for row in method["rows"] if row["mlp_id"] == mlp_id)

    artifact = {
        "configuration": {
            "selection_ids": [0, 49],
            "test_ids": [50, 99],
            "stream_totals": STREAM_TOTALS,
            "stream_seeds": STREAM_SEEDS,
            "checkpoints": CHECKPOINTS,
            "flop_formula": "N * 32 * 2 * 256 * 256",
            "budget": BUDGET,
        },
        "selection": {
            "weights_ab": weights_ab.tolist(),
            "weights_abc": weights_abc.tolist(),
            "fixed": selection_fixed,
            "chosen_adaptive_rule": {
                key: best_rule[key]
                for key in best_rule
                if key != "rows"
            },
            "records": strip_arrays(selection),
        },
        "test": {
            "fixed": test_fixed,
            "adaptive": test_adaptive,
            "records": strip_arrays(test),
            "outliers": {
                str(mlp_id): {
                    "A": outlier(test_fixed["A"], mlp_id),
                    "AB": outlier(test_fixed["AB"], mlp_id),
                    "ABC": outlier(test_fixed["ABC"], mlp_id),
                    "adaptive": outlier(test_adaptive, mlp_id),
                }
                for mlp_id in (80, 95)
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {
                    "mean_raw_mse": result["mean_raw_mse"],
                    "mean_adjusted_score": result["mean_adjusted_score"],
                    "weights": result["weights"],
                    "compute_factor": result["compute_factor"],
                }
                for name, result in test_fixed.items()
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                key: test_adaptive[key]
                for key in (
                    "name",
                    "feature",
                    "threshold",
                    "mean_raw_mse",
                    "mean_adjusted_score",
                    "high_fraction",
                    "mean_directions",
                )
            },
            indent=2,
        )
    )
    print(json.dumps(artifact["test"]["outliers"], indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
