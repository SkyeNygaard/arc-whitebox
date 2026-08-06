"""Test moment-matching deformations of the Kerdock layer-1 ensemble.

This is a selection-only experiment.  The Kerdock points at radius E[chi_d]
have an analytically known layer-1 ReLU mean and covariance.  The finite
ensemble can therefore be shifted or affinely transformed to match those
moments before propagation through layers 2--32.

The transformation is not claimed to remain an input-space cubature rule:
some centered affine variants can create small negative layer-1 values.  It is
an ensemble-transform / sigma-point experiment, evaluated empirically before
any consideration of deployment.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from eval_kerdock_design import (
    WIDTH,
    make_kerdock_design,
    random_rotation,
)
from eval_sampling_official import DEFAULT_DATA, _load_rows
from eval_spherical_stein_cv import sphere_radius_mean


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "kerdock_layer1_ensemble_transform.json"
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def exact_angular_relu_moments(
    first_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and centered covariance at fixed radius E[chi_d]."""
    weight = first_weight.astype(np.float64)
    gram = weight.T @ weight
    sigma = np.sqrt(np.maximum(np.diag(gram), 0.0))
    scale = np.outer(sigma, sigma)
    rho = np.divide(
        gram,
        scale,
        out=np.zeros_like(gram),
        where=scale > 0.0,
    )
    rho = np.clip(rho, -1.0, 1.0)
    theta = np.arccos(rho)
    gaussian_raw_second = (
        scale
        * (
            np.sin(theta)
            + (math.pi - theta) * np.cos(theta)
        )
        / (2.0 * math.pi)
    )
    radius = sphere_radius_mean(WIDTH)
    angular_raw_second = (
        (radius * radius / WIDTH) * gaussian_raw_second
    )
    mean = sigma * INV_SQRT_2PI
    covariance = angular_raw_second - np.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    return mean, covariance


def symmetric_root(
    matrix: np.ndarray,
    inverse: bool,
    floor_relative: float = 1e-10,
) -> tuple[np.ndarray, dict[str, float]]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    largest = float(max(np.max(eigenvalues), 0.0))
    floor = max(largest * floor_relative, np.finfo(np.float64).tiny)
    clipped = np.maximum(eigenvalues, floor)
    power = -0.5 if inverse else 0.5
    root = (eigenvectors * np.power(clipped, power)[None, :]) @ eigenvectors.T
    return root, {
        "min_eigenvalue": float(np.min(eigenvalues)),
        "max_eigenvalue": float(np.max(eigenvalues)),
        "floor": floor,
        "condition_after_floor": float(np.max(clipped) / np.min(clipped)),
    }


def propagate(
    activation: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    for weight in weights[1:]:
        activation = np.maximum(activation @ weight, 0.0)
    return activation.mean(axis=0, dtype=np.float64)


def build_variants(
    activation: np.ndarray,
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
    include_full: bool,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    sample_mean = activation.mean(axis=0, dtype=np.float64)
    delta = target_mean - sample_mean
    centered = activation.astype(np.float64) - sample_mean
    sample_variance = np.mean(np.square(centered), axis=0)
    target_variance = np.diag(target_covariance)
    ratio = np.sqrt(
        np.divide(
            target_variance,
            sample_variance,
            out=np.ones_like(target_variance),
            where=sample_variance > 0.0,
        )
    )

    variants: dict[str, np.ndarray] = {"baseline": activation}
    for strength in (0.25, 0.5, 0.75, 1.0, 1.25):
        variants[f"mean_shift_{strength:g}"] = (
            activation.astype(np.float64) + strength * delta
        )
    positive_ratio = np.divide(
        target_mean,
        sample_mean,
        out=np.ones_like(target_mean),
        where=sample_mean != 0.0,
    )
    variants["positive_mean_scale"] = (
        activation.astype(np.float64) * positive_ratio
    )
    variants["diagonal_covariance"] = centered * ratio + target_mean

    metadata: dict[str, object] = {
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
        "minimum_by_variant": {
            name: float(np.min(values))
            for name, values in variants.items()
        },
    }
    if include_full:
        sample_covariance = centered.T @ centered / len(centered)
        sample_inverse_root, sample_spectrum = symmetric_root(
            sample_covariance,
            inverse=True,
        )
        target_root, target_spectrum = symmetric_root(
            target_covariance,
            inverse=False,
        )
        transform = sample_inverse_root @ target_root
        full = centered @ transform + target_mean
        variants["full_covariance"] = full
        metadata["sample_covariance_spectrum"] = sample_spectrum
        metadata["target_covariance_spectrum"] = target_spectrum
        metadata["minimum_by_variant"]["full_covariance"] = float(
            np.min(full)
        )
        achieved = np.cov(full, rowvar=False, bias=True)
        metadata["full_covariance_max_abs_error"] = float(
            np.max(np.abs(achieved - target_covariance))
        )
    return variants, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=list(range(10)),
    )
    parser.add_argument("--rotation-seed", type=int, default=3)
    parser.add_argument(
        "--include-full",
        action="store_true",
        help="also compute the full 256x256 covariance transport",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        help=(
            "optional subset to propagate; construction diagnostics are still "
            "computed for all inexpensive variants"
        ),
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.indices or min(args.indices) < 0 or max(args.indices) >= 50:
        raise ValueError("ensemble-transform study is restricted to IDs 0--49")

    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    rows = _load_rows(args.data, args.indices)
    records: list[dict[str, object]] = []
    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        start = time.perf_counter()
        first = np.maximum(points @ (rotation @ weights[0]), 0.0)
        target_mean, target_covariance = exact_angular_relu_moments(weights[0])
        variants, metadata = build_variants(
            first,
            target_mean,
            target_covariance,
            args.include_full,
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
            prediction = propagate(activation, weights)
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
            "fixed_radius": "E[chi_256]",
            "covariance_radial_factor": (
                sphere_radius_mean(WIDTH) ** 2 / WIDTH
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
