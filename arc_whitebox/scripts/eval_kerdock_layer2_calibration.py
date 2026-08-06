"""Calibrate Kerdock layer-2 preactivations to exact known moments.

At the fixed Kerdock radius E[chi_256], the complete mean and covariance of
the first ReLU layer are available from the arc-cosine kernel.  Because the
second preactivation is linear in that layer, its mean and covariance are
also exact:

    mu_h2 = mu_a1 W2
    Cov(h2) = W2^T Cov(a1) W2.

This selection-only experiment standardizes each sampled h2 column to those
exact moments before applying the second ReLU.  It is a deterministic
ensemble-transform correction at the deepest point where exact moments are
available without an approximation closure.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from eval_kerdock_design import (
    WIDTH,
    make_kerdock_design,
    random_rotation,
)
from eval_kerdock_layer1_ensemble_transform import (
    exact_angular_relu_moments,
)
from eval_sampling_official import DEFAULT_DATA, _load_rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "kerdock_layer2_calibration.json"


def exact_layer2_preactivation_moments(
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean_a1, covariance_a1 = exact_angular_relu_moments(weights[0])
    second_weight = weights[1].astype(np.float64)
    mean_h2 = mean_a1 @ second_weight
    covariance_times_weight = covariance_a1 @ second_weight
    variance_h2 = np.sum(
        second_weight * covariance_times_weight,
        axis=0,
    )
    return mean_h2, np.maximum(variance_h2, 0.0)


def calibrated_variants(
    preactivation: np.ndarray,
    target_mean: np.ndarray,
    target_variance: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    sample_mean = preactivation.mean(axis=0, dtype=np.float64)
    centered = preactivation.astype(np.float64) - sample_mean
    sample_variance = np.mean(np.square(centered), axis=0)
    target_std = np.sqrt(target_variance)
    sample_std = np.sqrt(np.maximum(sample_variance, 1e-30))
    std_ratio = target_std / sample_std
    delta = target_mean - sample_mean

    variants = {"baseline": np.maximum(preactivation, 0.0)}
    for strength in (
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
    ):
        shifted = preactivation.astype(np.float64) + strength * delta
        variants[f"mean_{strength:g}"] = np.maximum(
            shifted.astype(np.float32),
            0.0,
        )
        # Exponential interpolation keeps the standard-deviation ratio
        # positive even for modest extrapolation beyond strength one.
        ratio = np.power(std_ratio, strength)
        corrected = centered * ratio + (
            sample_mean + strength * delta
        )
        variants[f"mean_variance_{strength:g}"] = np.maximum(
            corrected.astype(np.float32),
            0.0,
        )
    metadata = {
        "sample_mean_mse": float(np.mean(np.square(delta))),
        "sample_variance_relative_rms": float(
            np.sqrt(
                np.mean(
                    np.square(
                        (sample_variance - target_variance)
                        / np.maximum(target_variance, 1e-30)
                    )
                )
            )
        ),
        "std_ratio_min": float(np.min(std_ratio)),
        "std_ratio_max": float(np.max(std_ratio)),
    }
    return variants, metadata


def propagate_from_layer2(
    activation: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    for weight in weights[2:]:
        activation = np.maximum(activation @ weight, 0.0)
    return activation.mean(axis=0, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=list(range(10)),
    )
    parser.add_argument("--rotation-seed", type=int, default=3)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.indices or min(args.indices) < 0 or max(args.indices) >= 50:
        raise ValueError("layer-2 calibration is restricted to IDs 0--49")

    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    rows = _load_rows(args.data, args.indices)
    records: list[dict[str, object]] = []
    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        start = time.perf_counter()
        first = np.maximum(points @ (rotation @ weights[0]), 0.0)
        second_preactivation = first @ weights[1]
        target_mean, target_variance = (
            exact_layer2_preactivation_moments(weights)
        )
        variants, metadata = calibrated_variants(
            second_preactivation,
            target_mean,
            target_variance,
        )
        if args.variants:
            missing = set(args.variants) - set(variants)
            if missing:
                raise ValueError(f"unknown variants: {sorted(missing)}")
            variants = {
                name: values
                for name, values in variants.items()
                if name in args.variants
            }
        scores = {}
        predictions = {}
        for variant_name, activation in variants.items():
            prediction = propagate_from_layer2(activation, weights)
            scores[variant_name] = float(
                np.mean(np.square(prediction - targets[-1]))
            )
            predictions[variant_name] = prediction.tolist()
        record = {
            "index": index,
            "name": name,
            "rotation_seed": args.rotation_seed,
            "seconds": time.perf_counter() - start,
            "metadata": metadata,
            "final_mse": scores,
            "predictions": predictions,
        }
        records.append(record)
        print(
            {
                "index": index,
                "seconds": record["seconds"],
                "final_mse": scores,
            },
            flush=True,
        )

    variants = sorted(records[0]["final_mse"])
    summary = {
        variant: {
            "mean_final_mse": float(
                np.mean(
                    [
                        float(record["final_mse"][variant])
                        for record in records
                    ]
                )
            ),
            "median_final_mse": float(
                np.median(
                    [
                        float(record["final_mse"][variant])
                        for record in records
                    ]
                )
            ),
        }
        for variant in variants
    }
    payload = {
        "protocol": {
            "selection_indices": args.indices,
            "holdout_loaded": False,
            "rotation_seed": args.rotation_seed,
            "target_moments": (
                "exact fixed-radius layer-2 preactivation mean/variance"
            ),
        },
        "summary": summary,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print({"out": str(args.out), "summary": summary}, flush=True)


if __name__ == "__main__":
    main()
