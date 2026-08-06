"""Evaluate scalar connected-C21 anchors from compact K3 experiments.

The input result files need only contain, per network, two contractions of a
candidate connected C21 with the saved Kerdock-derived directions.  No target
mean/covariance is required.  The pointwise control uses the radially
homogenized connected-cubic identity, so its exact spherical expectation is
the supplied contraction divided by ``width + 1``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from eval_connected_cubic_control import contracted_pointwise  # noqa: E402
from eval_crossfit_cumulant_control import forward_layer_and_final  # noqa: E402
from eval_exact_anchor_residual import FULL_DATA, ROWS_PER_BASIS  # noqa: E402
from eval_kerdock_design import (  # noqa: E402
    N_BASES,
    WIDTH,
    make_kerdock_design,
    random_rotation,
)
from eval_sampling_official import _load_rows  # noqa: E402
from exact_moments import sphere_radius_mean  # noqa: E402


def load_records(paths: list[Path]) -> dict[int, dict]:
    records = {}
    for path in paths:
        payload = json.loads(path.read_text())
        for record in payload["records"]:
            index = int(record["index"])
            if index in records:
                raise ValueError(f"duplicate index {index} in {path}")
            records[index] = record
    return records


def summarize(records: list[dict]) -> dict:
    baseline = np.asarray([record["baseline_mse"] for record in records])
    labels = list(records[0]["method_mses"])
    return {
        label: {
            "ratio": float(
                np.mean([record["method_mses"][label] for record in records])
                / np.mean(baseline)
            ),
            "wins": int(
                np.sum(
                    [
                        record["method_mses"][label] < record["baseline_mse"]
                        for record in records
                    ]
                )
            ),
            "worst": float(
                np.max(
                    [
                        record["method_mses"][label] / record["baseline_mse"]
                        for record in records
                    ]
                )
            ),
        }
        for label in labels
    }


def sufficient_stats(
    values: np.ndarray,
    outputs: np.ndarray,
    folds: int,
) -> dict:
    """Cache all O(samples * output-width) products for many anchors."""
    values = np.asarray(values, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)
    block_ids = np.repeat(np.arange(N_BASES), ROWS_PER_BASIS)
    fold_ids = block_ids % folds

    def stats(mask: np.ndarray) -> dict:
        x = values[mask]
        y = outputs[mask]
        return {
            "n": int(np.sum(mask)),
            "sum_x": np.sum(x, axis=0),
            "sum_y": np.sum(y, axis=0),
            "gram": x.T @ x,
            "cross": x.T @ y,
        }

    all_mask = np.ones(len(values), dtype=bool)
    return {
        "total": stats(all_mask),
        "folds": [stats(fold_ids == fold) for fold in range(folds)],
    }


def crossfit_anchors(
    cached: dict,
    anchor: np.ndarray,
    ridges: list[float],
) -> dict[float, np.ndarray]:
    """Exactly reproduce ``crossfit_grid(values-anchor, ...)`` from stats."""
    total = cached["total"]
    n = total["n"]
    anchor = np.asarray(anchor, dtype=np.float64)
    variance = (
        np.diag(total["gram"])
        - 2.0 * anchor * total["sum_x"]
        + n * np.square(anchor)
    ) / n
    scale = np.sqrt(np.maximum(variance, 0.0))
    keep = scale > 1e-12
    if not np.any(keep):
        mean_y = total["sum_y"] / n
        return {ridge: mean_y for ridge in ridges}
    a = anchor[keep]
    s = scale[keep]
    sum_x = total["sum_x"][keep]
    gram = total["gram"][np.ix_(keep, keep)]
    cross = total["cross"][keep]
    centered_gram = (
        gram
        - np.outer(sum_x, a)
        - np.outer(a, sum_x)
        + n * np.outer(a, a)
    )
    centered_cross = cross - a[:, None] * total["sum_y"][None, :]
    gram_total = centered_gram / (s[:, None] * s[None, :])
    cross_total = centered_cross / s[:, None]

    estimates = {ridge: [] for ridge in ridges}
    sizes = []
    for fold in cached["folds"]:
        nf = fold["n"]
        fold_sum_x = fold["sum_x"][keep]
        fold_gram = fold["gram"][np.ix_(keep, keep)]
        fold_cross = fold["cross"][keep]
        centered_fold_gram = (
            fold_gram
            - np.outer(fold_sum_x, a)
            - np.outer(a, fold_sum_x)
            + nf * np.outer(a, a)
        ) / (s[:, None] * s[None, :])
        centered_fold_cross = (
            fold_cross - a[:, None] * fold["sum_y"][None, :]
        ) / s[:, None]
        gram_train = gram_total - centered_fold_gram
        cross_train = cross_total - centered_fold_cross
        mean_x = (fold_sum_x / nf - a) / s
        mean_y = fold["sum_y"] / nf
        n_train = n - nf
        sizes.append(nf)
        for ridge in ridges:
            coefficient = np.linalg.solve(
                gram_train
                + ridge * n_train * np.eye(len(s)),
                cross_train,
            )
            estimates[ridge].append(mean_y - mean_x @ coefficient)
    return {
        ridge: np.average(predictions, axis=0, weights=sizes)
        for ridge, predictions in estimates.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(160, 168)))
    parser.add_argument("--result-files", type=Path, nargs="+", required=True)
    parser.add_argument("--directions-dir", type=Path, required=True)
    parser.add_argument("--factorized-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=29)
    parser.add_argument("--rotation-seed", type=int, default=3)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--born-result", type=Path)
    parser.add_argument("--born-label", default="q32_h24")
    parser.add_argument("--ensemble-base-label", default="cap512")
    parser.add_argument(
        "--ensemble-weights",
        type=float,
        nargs=2,
        metavar=("BASE", "BORN"),
    )
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0])
    parser.add_argument(
        "--anchor-deltas",
        type=float,
        nargs="+",
        default=[],
        help=(
            "Also anchor at sample_mean + delta*(analytic-sample_mean); "
            "this shrinks the predicted Kerdock integration error directly."
        ),
    )
    parser.add_argument(
        "--centers",
        choices=["sample", "factorized", "self"],
        nargs="+",
        default=["factorized"],
    )
    parser.add_argument("--control-ranks", type=int, nargs="+", default=[2])
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--ridges", type=float, nargs="+", default=[0.1])
    parser.add_argument(
        "--correction-multipliers",
        type=float,
        nargs="+",
        default=[1.0],
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source_records = load_records(args.result_files)
    born_records = (
        {
            int(record["index"]): record
            for record in json.loads(args.born_result.read_text())["records"]
        }
        if args.born_result is not None
        else {}
    )
    missing = set(args.indices) - set(source_records)
    if missing:
        raise KeyError(sorted(missing))
    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    radius = sphere_radius_mean(WIDTH)
    rows = _load_rows(FULL_DATA, args.indices)
    records = []

    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        started = time.perf_counter()
        _, activation, final = forward_layer_and_final(
            weights,
            points,
            rotation,
            args.layer,
        )
        with np.load(args.directions_dir / f"mlp_{index:05d}.npz") as directions:
            left = np.asarray(directions["sample_left"], dtype=np.float64)
            right = np.asarray(directions["sample_right"], dtype=np.float64)
        with np.load(args.factorized_dir / f"mlp_{index:05d}.npz") as factorized:
            factorized_mean = np.asarray(factorized["mean"], dtype=np.float64)
        shared_centers = {
            "sample": np.mean(activation, axis=0, dtype=np.float64),
            "factorized": factorized_mean,
        }
        source = source_records[index]
        if args.ensemble_weights is not None:
            if index not in born_records:
                raise KeyError(f"missing born record {index}")
            base = source["variants"][args.ensemble_base_label]
            base_estimate = np.asarray(base["estimate"], dtype=np.float64)
            born_estimate = np.asarray(
                born_records[index]["estimates"][args.born_label],
                dtype=np.float64,
            )
            combined = (
                args.ensemble_weights[0] * base_estimate
                + args.ensemble_weights[1] * born_estimate
            )
            source["variants"]["ensemble"] = {
                **base,
                "estimate": combined.tolist(),
                "sample_center_contraction": (
                    np.asarray(
                        base["sample_center_contraction"],
                        dtype=np.float64,
                    )
                    - base_estimate
                    + combined
                ).tolist(),
                "factorized_center_contraction": (
                    np.asarray(
                        base["factorized_center_contraction"],
                        dtype=np.float64,
                    )
                    - base_estimate
                    + combined
                ).tolist(),
            }
        labels = ["full", "oracle", *args.labels]

        baseline = np.mean(final, axis=0, dtype=np.float64)
        baseline_mse = float(np.mean(np.square(baseline - targets[-1])))
        method_mses = {}
        stats_cache = {}
        for label in labels:
            label_scales = [1.0] if label == "oracle" else args.scales
            for center_label in args.centers:
                if center_label == "self" and label == "oracle":
                    continue
                if label == "oracle":
                    center = shared_centers[center_label]
                    contraction = np.asarray(
                        source[
                            f"oracle_{center_label}_center_contraction"
                        ],
                        dtype=np.float64,
                    )
                elif label == "full":
                    if center_label in ("factorized", "self"):
                        center = factorized_mean
                        contraction = np.asarray(
                            source["full_truth"],
                            dtype=np.float64,
                        )
                    else:
                        center = shared_centers["sample"]
                        contraction = np.asarray(
                            source["full_sample_center_contraction"],
                            dtype=np.float64,
                        )
                else:
                    variant = source["variants"][label]
                    if center_label == "self":
                        center = np.asarray(
                            variant["predicted_mean"],
                            dtype=np.float64,
                        )
                        contraction = np.asarray(
                            variant["estimate"],
                            dtype=np.float64,
                        )
                    else:
                        center = shared_centers[center_label]
                        contraction = np.asarray(
                            variant[
                                f"{center_label}_center_contraction"
                            ],
                            dtype=np.float64,
                        )
                for scale in label_scales:
                    predicted_anchor = (
                        scale * contraction / (WIDTH + 1.0)
                    )
                    values = contracted_pointwise(
                        activation,
                        left,
                        right,
                        center,
                        radius,
                    )
                    center_key = (
                        center_label
                        if center_label != "self"
                        else f"self:{label}"
                    )
                    if center_key not in stats_cache:
                        stats_cache[center_key] = (
                            values,
                            sufficient_stats(values, final, args.folds),
                        )
                    else:
                        values = stats_cache[center_key][0]
                    anchors = [("direct", predicted_anchor)]
                    if label != "oracle":
                        sample_anchor = np.mean(
                            values,
                            axis=0,
                            dtype=np.float64,
                        )
                        anchors.extend(
                            (
                                f"delta{delta:g}",
                                sample_anchor
                                + delta
                                * (predicted_anchor - sample_anchor),
                            )
                            for delta in args.anchor_deltas
                        )
                    for anchor_label, anchor in anchors:
                        for control_rank in args.control_ranks:
                            full_stats = stats_cache[center_key][1]
                            reduced_stats = {
                                "total": {
                                    key: (
                                        value[:control_rank, :control_rank]
                                        if key == "gram"
                                        else value[:control_rank]
                                        if key in ("sum_x", "cross")
                                        else value
                                    )
                                    for key, value in full_stats["total"].items()
                                },
                                "folds": [
                                    {
                                        key: (
                                            value[:control_rank, :control_rank]
                                            if key == "gram"
                                            else value[:control_rank]
                                            if key in ("sum_x", "cross")
                                            else value
                                        )
                                        for key, value in fold.items()
                                    }
                                    for fold in full_stats["folds"]
                                ],
                            }
                            predictions = crossfit_anchors(
                                reduced_stats,
                                anchor[:control_rank],
                                args.ridges,
                            )
                            for ridge, control_prediction in predictions.items():
                                for multiplier in args.correction_multipliers:
                                    prediction = (
                                        baseline
                                        + multiplier
                                        * (control_prediction - baseline)
                                    )
                                    method_label = (
                                        f"{label}_scale{scale:g}"
                                        f"_{anchor_label}_{center_label}"
                                        f"_r{control_rank}"
                                        f"_ridge{ridge:g}"
                                        f"_corr{multiplier:g}"
                                    )
                                    method_mses[method_label] = float(
                                        np.mean(
                                            np.square(
                                                prediction - targets[-1]
                                            )
                                        )
                                    )
        record = {
            "index": index,
            "name": name,
            "baseline_mse": baseline_mse,
            "method_mses": method_mses,
            "seconds": time.perf_counter() - started,
        }
        records.append(record)
        best = min(method_mses, key=method_mses.get)
        print(
            f"[{index:>4}] base={baseline_mse:.4e} "
            f"best={best} {method_mses[best] / baseline_mse:.4f}x",
            flush=True,
        )

    summary = summarize(records)
    output = {
        "protocol": {
            "indices": args.indices,
            "result_files": [str(path) for path in args.result_files],
            "layer": args.layer,
            "rotation_seed": args.rotation_seed,
            "labels": args.labels,
            "born_result": (
                str(args.born_result)
                if args.born_result is not None
                else None
            ),
            "born_label": args.born_label,
            "ensemble_base_label": args.ensemble_base_label,
            "ensemble_weights": args.ensemble_weights,
            "scales": args.scales,
            "anchor_deltas": args.anchor_deltas,
            "centers": args.centers,
            "control_ranks": args.control_ranks,
            "folds": args.folds,
            "ridges": args.ridges,
            "correction_multipliers": args.correction_multipliers,
            "target_leakage": False,
            "oracle_anchor_is_ceiling_only": True,
        },
        "summary": summary,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    for label in sorted(summary, key=lambda key: summary[key]["ratio"])[:30]:
        item = summary[label]
        print(
            f"{label:<62} ratio={item['ratio']:.5f} "
            f"wins={item['wins']}/{len(records)} worst={item['worst']:.2f}x",
            flush=True,
        )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
