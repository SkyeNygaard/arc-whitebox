"""Selection-gated 32,768 + 30,000 direction RQMC allocation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import warnings
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))

import eval_multistream_rqmc as rq  # noqa: E402


rq.STREAM_TOTALS["D"] = 30_000
rq.STREAM_SEEDS["D"] = 404
rq.CHECKPOINTS["D"] = [15_000, 30_000]


def nested_a(record):
    a = record["full"]["A"]
    scale = float(np.mean(a * a)) + 1e-20
    return float(np.mean((a - record["half"]["A"]) ** 2) / scale)


def adaptive_candidates(records, weights_ad):
    feature_values = np.array(
        [nested_a(record) for record in records]
    )
    thresholds = sorted(
        set(
            [-math.inf, math.inf]
            + [
                float(np.quantile(feature_values, q))
                for q in np.linspace(0, 1, 21)
            ]
        )
    )
    candidates = []
    for threshold in thresholds:
        rows = []
        for record in records:
            feature = nested_a(record)
            high = feature > threshold
            streams = ["A", "D"] if high else ["A"]
            weights = weights_ad if high else np.array([1.0])
            total = sum(rq.STREAM_TOTALS[s] for s in streams)
            raw = rq.mse(rq.blend(record, streams, weights), record["target"])
            rows.append(
                {
                    "mlp_id": record["mlp_id"],
                    "feature": feature,
                    "high_allocation": high,
                    "total_directions": total,
                    "raw_mse": raw,
                    "adjusted_score": rq.adjusted(raw, total),
                }
            )
        candidates.append(
            {
                "name": "A_then_AD",
                "feature": "nested_A",
                "threshold": threshold,
                "weights_ad": weights_ad.tolist(),
                "mean_raw_mse": statistics.fmean(r["raw_mse"] for r in rows),
                "mean_adjusted_score": statistics.fmean(
                    r["adjusted_score"] for r in rows
                ),
                "high_fraction": statistics.fmean(
                    r["high_allocation"] for r in rows
                ),
                "rows": rows,
            }
        )
    return candidates


def evaluate_rule(records, rule):
    rows = []
    weights_ad = np.asarray(rule["weights_ad"])
    for record in records:
        feature = nested_a(record)
        high = feature > rule["threshold"]
        streams = ["A", "D"] if high else ["A"]
        weights = weights_ad if high else np.array([1.0])
        total = sum(rq.STREAM_TOTALS[s] for s in streams)
        raw = rq.mse(rq.blend(record, streams, weights), record["target"])
        rows.append(
            {
                "mlp_id": record["mlp_id"],
                "feature": feature,
                "high_allocation": high,
                "total_directions": total,
                "raw_mse": raw,
                "adjusted_score": rq.adjusted(raw, total),
            }
        )
    return {
        **{k: rule[k] for k in (
            "name",
            "feature",
            "threshold",
            "weights_ad",
        )},
        "mean_raw_mse": statistics.fmean(r["raw_mse"] for r in rows),
        "mean_adjusted_score": statistics.fmean(
            r["adjusted_score"] for r in rows
        ),
        "high_fraction": statistics.fmean(
            r["high_allocation"] for r in rows
        ),
        "mean_directions": statistics.fmean(
            r["total_directions"] for r in rows
        ),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "two_nearfull_rqmc.json",
    )
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        streams = {
            "A": rq.make_stream(rq.STREAM_TOTALS["A"], rq.STREAM_SEEDS["A"]),
            "D": rq.make_stream(rq.STREAM_TOTALS["D"], rq.STREAM_SEEDS["D"]),
        }
    selection = rq.collect_records(set(range(50)), streams)
    weights_ad = rq.fit_simplex_weights(selection, ["A", "D"])
    fixed_a_selection = rq.evaluate_fixed(
        selection, ["A"], np.array([1.0])
    )
    fixed_ad_selection = rq.evaluate_fixed(
        selection, ["A", "D"], weights_ad
    )
    adaptive_selection = min(
        adaptive_candidates(selection, weights_ad),
        key=lambda candidate: candidate["mean_adjusted_score"],
    )
    choices = {
        "A": fixed_a_selection["mean_adjusted_score"],
        "AD": fixed_ad_selection["mean_adjusted_score"],
        "adaptive": adaptive_selection["mean_adjusted_score"],
    }
    chosen = min(choices, key=choices.get)

    # D reaches test only if selection chose a D-using candidate.
    test_streams = {"A": streams["A"]} if chosen == "A" else streams
    test = rq.collect_records(set(range(50, 100)), test_streams)
    fixed_a_test = rq.evaluate_fixed(test, ["A"], np.array([1.0]))
    if chosen == "A":
        fixed_ad_test = None
        adaptive_test = None
    else:
        fixed_ad_test = rq.evaluate_fixed(test, ["A", "D"], weights_ad)
        adaptive_test = evaluate_rule(test, adaptive_selection)
    artifact = {
        "configuration": {
            "selection_ids": [0, 49],
            "test_ids": [50, 99],
            "stream_totals": {"A": 32768, "D": 30000},
            "stream_seeds": {"A": 101, "D": 404},
            "flop_formula": "N * 32 * 2 * 256 * 256",
        },
        "selection": {
            "weights_ad": weights_ad.tolist(),
            "A": fixed_a_selection,
            "AD": fixed_ad_selection,
            "adaptive": {
                k: v for k, v in adaptive_selection.items() if k != "rows"
            },
            "chosen": chosen,
        },
        "test": {
            "A": fixed_a_test,
            "AD": fixed_ad_test,
            "adaptive": adaptive_test,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({
        "selection_chosen": chosen,
        "weights_ad": weights_ad.tolist(),
        "selection_scores": choices,
        "test_A": fixed_a_test["mean_adjusted_score"],
        "test_AD": (
            fixed_ad_test["mean_adjusted_score"]
            if fixed_ad_test is not None else None
        ),
        "test_adaptive": (
            adaptive_test["mean_adjusted_score"]
            if adaptive_test is not None else None
        ),
        "test_adaptive_high_fraction": (
            adaptive_test["high_fraction"]
            if adaptive_test is not None else None
        ),
    }, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
