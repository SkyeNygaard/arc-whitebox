"""Evaluate the frozen Kerdock multifidelity rule with layer-2 transport.

Selection protocol
------------------
This script is hard-limited to official IDs 0--49.  It evaluates the frozen
90,624-row rule

    F3 + (P0,S + P1,S - 2 P3,S) / 16,

where F3 is the full seed-3 Kerdock 5-design and P_r,S is the mean on the
frozen set of 24 antipodal Kerdock bases under rotation r.

Before the second ReLU, an optional global affine transport moves the signed
quadrature mean and variance toward analytic fixed-radius layer-2 moments.
The final prediction always uses the same quadrature weights.  At the frozen
coefficient every nominally signed row weight is in fact strictly positive.

This is a dense research harness.  A deployable estimator must use the
separately audited structured first layer and rectangular Winograd schedule.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_kerdock_design import (  # noqa: E402
    N_BASES,
    N_POINTS,
    WIDTH,
    make_kerdock_design,
    random_rotation,
)
from eval_sampling_official import DEFAULT_DATA, _load_rows  # noqa: E402


DEFAULT_OUT = ROOT / "results" / "kerdock_multifidelity_h2_selection.json"
PILOT_BASES = np.asarray(
    [
        1,
        3,
        4,
        5,
        6,
        13,
        15,
        16,
        29,
        35,
        57,
        59,
        66,
        72,
        84,
        85,
        87,
        95,
        96,
        101,
        108,
        118,
        120,
        124,
    ],
    dtype=np.int64,
)
PILOT_ROWS = len(PILOT_BASES) * 2 * WIDTH
TOTAL_ROWS = N_POINTS + 2 * PILOT_ROWS
ALPHA = 1.0 / 8.0
SQRT_2PI = math.sqrt(2.0 * math.pi)


def make_points_and_weights() -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    canonical = make_kerdock_design()
    by_basis = canonical.reshape((N_BASES, 2 * WIDTH, WIDTH))
    pilot = by_basis[PILOT_BASES].reshape((-1, WIDTH))
    full3 = canonical @ random_rotation(WIDTH, 3)
    pilot0 = pilot @ random_rotation(WIDTH, 0)
    pilot1 = pilot @ random_rotation(WIDTH, 1)
    points = np.concatenate((full3, pilot0, pilot1), axis=0).astype(
        np.float32,
        copy=False,
    )

    # F3 + alpha/2 * (P0 + P1 - 2 P3).
    weights = np.full(TOTAL_ROWS, 1.0 / N_POINTS, dtype=np.float64)
    full_by_basis = weights[:N_POINTS].reshape((N_BASES, 2 * WIDTH))
    full_by_basis[PILOT_BASES] -= ALPHA / PILOT_ROWS
    weights[N_POINTS : N_POINTS + PILOT_ROWS] = ALPHA / (2.0 * PILOT_ROWS)
    weights[N_POINTS + PILOT_ROWS :] = ALPHA / (2.0 * PILOT_ROWS)
    if not np.isclose(weights.sum(), 1.0, atol=1e-14):
        raise AssertionError(weights.sum())
    if np.min(weights) <= 0.0:
        raise AssertionError("frozen quadrature unexpectedly has signed rows")
    return points, weights, {
        "sum": float(weights.sum()),
        "minimum": float(weights.min()),
        "maximum": float(weights.max()),
        "selected_full_weight": float(full_by_basis[PILOT_BASES[0], 0]),
        "unselected_full_weight": float(full_by_basis[0, 0]),
        "alternate_pilot_weight": float(weights[N_POINTS]),
        "negative_weights": float(np.count_nonzero(weights < 0.0)),
    }


def analytic_h2_moments(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-radius mean and marginal variance before the second ReLU."""
    first = weights[0].astype(np.float64)
    gram = first.T @ first
    sigma = np.sqrt(np.maximum(np.diag(gram), 0.0))
    denominator = sigma[:, None] * sigma[None, :]
    rho = np.divide(
        gram,
        denominator,
        out=np.zeros_like(gram),
        where=denominator > 0.0,
    )
    rho = np.clip(rho, -1.0, 1.0)
    kernel = denominator / (2.0 * math.pi) * (
        np.sqrt(np.maximum(1.0 - np.square(rho), 0.0))
        + (math.pi - np.arccos(rho)) * rho
    )
    mean_a1 = sigma / SQRT_2PI
    mean_radius = math.sqrt(2.0) * math.exp(
        math.lgamma((WIDTH + 1.0) / 2.0)
        - math.lgamma(WIDTH / 2.0)
    )
    covariance_a1 = (
        (mean_radius * mean_radius / WIDTH) * kernel
        - mean_a1[:, None] * mean_a1[None, :]
    )
    second = weights[1].astype(np.float64)
    mean_h2 = mean_a1 @ second
    covariance_times_second = covariance_a1 @ second
    variance_h2 = np.sum(second * covariance_times_second, axis=0)
    return mean_h2, np.maximum(variance_h2, 0.0)


def predict_strengths(
    network_weights: np.ndarray,
    points: np.ndarray,
    quadrature: np.ndarray,
    strengths: list[float],
) -> tuple[dict[float, np.ndarray], dict[str, float]]:
    activation1 = np.maximum(points @ network_weights[0], 0.0)
    h2 = activation1 @ network_weights[1]
    sample_mean = quadrature @ h2.astype(np.float64)
    centered = h2.astype(np.float64) - sample_mean
    sample_variance = quadrature @ np.square(centered)
    target_mean, target_variance = analytic_h2_moments(network_weights)
    variance_ratio = np.divide(
        target_variance,
        sample_variance,
        out=np.ones_like(target_variance),
        where=sample_variance > 0.0,
    )
    base_scale = np.sqrt(np.maximum(variance_ratio, 0.0))

    predictions = {}
    for strength in strengths:
        scale = np.power(base_scale, strength)
        corrected_h2 = (
            centered * scale
            + sample_mean
            + strength * (target_mean - sample_mean)
        )
        activation = np.maximum(corrected_h2, 0.0).astype(np.float32)
        for weight in network_weights[2:]:
            activation = np.maximum(activation @ weight, 0.0)
        predictions[strength] = quadrature @ activation.astype(np.float64)
    return predictions, {
        "minimum_sample_variance": float(np.min(sample_variance)),
        "nonpositive_sample_variances": float(
            np.count_nonzero(sample_variance <= 0.0)
        ),
        "mean_abs_h2_mean_discrepancy": float(
            np.mean(np.abs(sample_mean - target_mean))
        ),
        "mean_abs_log_variance_ratio": float(
            np.mean(np.abs(np.log(np.maximum(variance_ratio, 1e-30))))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(50)))
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 0.75, 1.0, 1.25, 1.5],
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.indices or min(args.indices) < 0 or max(args.indices) > 49:
        raise ValueError("this selection-only harness accepts IDs 0--49")

    points, quadrature, weight_audit = make_points_and_weights()
    if points.shape != (TOTAL_ROWS, WIDTH):
        raise AssertionError(points.shape)
    rows = _load_rows(args.data, args.indices)
    records = []
    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        started = time.perf_counter()
        predictions, diagnostics = predict_strengths(
            weights,
            points,
            quadrature,
            args.strengths,
        )
        for strength, prediction in predictions.items():
            records.append(
                {
                    "index": index,
                    "name": name,
                    "strength": strength,
                    "final_mse": float(
                        np.mean(np.square(prediction - targets[-1]))
                    ),
                }
            )
        print(
            {
                "index": index,
                "seconds": time.perf_counter() - started,
                "mse_by_strength": {
                    strength: records[-len(args.strengths) + offset][
                        "final_mse"
                    ]
                    for offset, strength in enumerate(args.strengths)
                },
                **diagnostics,
            },
            flush=True,
        )

    summaries = []
    for strength in args.strengths:
        chosen = [
            record for record in records if record["strength"] == strength
        ]
        summaries.append(
            {
                "strength": strength,
                "networks": len(chosen),
                "mean_final_mse": float(
                    np.mean([record["final_mse"] for record in chosen])
                ),
                "median_final_mse": float(
                    np.median([record["final_mse"] for record in chosen])
                ),
            }
        )
    result = {
        "protocol": {
            "split": "official IDs 0--49 only",
            "full_rotation_seed": 3,
            "pilot_rotation_seeds": [0, 1],
            "pilot_bases": PILOT_BASES.tolist(),
            "alpha": ALPHA,
            "formula": "F3 + (P0_S + P1_S - 2*P3_S)/16",
            "full_rows": N_POINTS,
            "pilot_rows_per_alternate_rotation": PILOT_ROWS,
            "total_rows": TOTAL_ROWS,
        },
        "quadrature_weight_audit": weight_audit,
        "summaries": summaries,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print({"summaries": summaries}, flush=True)


if __name__ == "__main__":
    main()
