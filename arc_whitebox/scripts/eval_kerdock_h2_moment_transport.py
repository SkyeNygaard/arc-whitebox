"""Exact layer-2 moment transport for the Kerdock/MUB cubature rule.

The first hidden activation is a ReLU of a jointly Gaussian vector.  Its
mean and full second moment are therefore available from the arc-cosine
kernel.  Positive homogeneity lets us convert those Gaussian moments to the
fixed radius ``E[chi_d]`` used by the Kerdock angular rule.

Consequently, the mean and marginal variance of the layer-2 preactivation
are exact.  This selection-only harness tests two ways to use that unusually
deep exact information:

* affinely transport each sampled layer-2 marginal to its exact mean/variance;
* use ``h`` and ``h^2`` as control variates with exact expectations, then
  shift the layer-2 activation cloud to the corrected mean.

Only official IDs 0--49 are accepted.  The frozen holdout is never opened.
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

from audit_kerdock_fwht import structured_first_numpy  # noqa: E402
from eval_kerdock_design import FIELD_SIZE, WIDTH, kerdock_chirp  # noqa: E402
from eval_sampling_official import DEFAULT_DATA, _load_rows  # noqa: E402
from eval_spherical_stein_cv import sphere_radius_mean  # noqa: E402


DEFAULT_ASSET = (
    ROOT
    / "submissions"
    / "kerdock_mub5"
    / "kerdock_mub5_seed3.npz"
)
DEFAULT_OUT = ROOT / "results" / "kerdock_h2_moment_transport_selection.json"
SQRT_2PI = math.sqrt(2.0 * math.pi)


def fixed_radius_layer1_moments(
    first_weight: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact mean and raw second moment at fixed radius ``radius``."""
    gram = first_weight.astype(np.float64).T @ first_weight.astype(np.float64)
    sigma = np.sqrt(np.maximum(np.diag(gram), 1e-30))
    rho = gram / np.maximum(sigma[:, None] * sigma[None, :], 1e-30)
    rho = np.clip(rho, -1.0, 1.0)
    root = np.sqrt(np.maximum(1.0 - np.square(rho), 0.0))
    gaussian_raw_second = (
        sigma[:, None]
        * sigma[None, :]
        * (root + (math.pi - np.arccos(rho)) * rho)
        / (2.0 * math.pi)
    )
    mean = sigma / SQRT_2PI
    # For X=R U, E[ReLU(Xw_i)ReLU(Xw_j)] = E[R^2] times
    # the angular second moment.  Here E[R^2]=d and radius=E[R].
    fixed_raw_second = (radius * radius / WIDTH) * gaussian_raw_second
    return mean, fixed_raw_second


def exact_h2_marginals(
    first_weight: np.ndarray,
    second_weight: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_a1, raw_a1 = fixed_radius_layer1_moments(first_weight, radius)
    mean_h2 = mean_a1 @ second_weight.astype(np.float64)
    raw_h2 = np.sum(
        second_weight.astype(np.float64)
        * (raw_a1 @ second_weight.astype(np.float64)),
        axis=0,
    )
    variance_h2 = np.maximum(raw_h2 - np.square(mean_h2), 1e-20)
    return mean_h2, raw_h2, variance_h2


def relu_quadratic_control_mean(
    preactivation: np.ndarray,
    exact_mean: np.ndarray,
    exact_raw_second: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Correct mean(ReLU(h)) using exact E[h] and E[h^2], per neuron."""
    h = preactivation.astype(np.float64)
    y = np.maximum(h, 0.0)
    sample_h1 = h.mean(axis=0)
    sample_h2 = np.square(h).mean(axis=0)
    sample_h3 = np.power(h, 3).mean(axis=0)
    sample_h4 = np.power(h, 4).mean(axis=0)
    sample_y = y.mean(axis=0)
    sample_yh = (y * h).mean(axis=0)
    sample_yh2 = (y * np.square(h)).mean(axis=0)

    c11 = np.maximum(sample_h2 - np.square(sample_h1), 0.0) + ridge
    c12 = sample_h3 - sample_h1 * sample_h2
    c22 = np.maximum(sample_h4 - np.square(sample_h2), 0.0) + ridge
    cy1 = sample_yh - sample_y * sample_h1
    cy2 = sample_yh2 - sample_y * sample_h2
    determinant = np.maximum(c11 * c22 - np.square(c12), ridge * ridge)
    beta1 = (cy1 * c22 - cy2 * c12) / determinant
    beta2 = (cy2 * c11 - cy1 * c12) / determinant
    return (
        sample_y
        + beta1 * (exact_mean - sample_h1)
        + beta2 * (exact_raw_second - sample_h2)
    )


def continue_forward(
    activation: np.ndarray,
    later_weights: np.ndarray,
) -> np.ndarray:
    current = activation.astype(np.float32)
    for weight in later_weights:
        current = np.maximum(current @ weight, 0.0)
    return current.mean(axis=0, dtype=np.float64)


def evaluate_network(
    weights: np.ndarray,
    target: np.ndarray,
    rotation: np.ndarray,
    chirps: np.ndarray,
    radius: float,
    ridge: float,
) -> dict[str, object]:
    first_pre, _ = structured_first_numpy(
        weights[0],
        rotation,
        chirps,
        radius,
    )
    first_activation = np.maximum(first_pre, 0.0)
    h2 = first_activation @ weights[1]
    exact_mean, exact_raw_second, exact_variance = exact_h2_marginals(
        weights[0],
        weights[1],
        radius,
    )
    sample_mean = h2.mean(axis=0, dtype=np.float64)
    sample_raw_second = np.square(h2.astype(np.float64)).mean(axis=0)
    sample_variance = np.maximum(
        sample_raw_second - np.square(sample_mean),
        1e-20,
    )

    corrected_relu_mean = relu_quadratic_control_mean(
        h2,
        exact_mean,
        exact_raw_second,
        ridge,
    )
    raw_activation = np.maximum(h2, 0.0)
    raw_activation_mean = raw_activation.mean(axis=0, dtype=np.float64)

    # Evaluate variants sequentially: retaining seven 66,048-by-256 clouds at
    # once needlessly approaches the evaluator's memory cap.
    baseline_prediction = continue_forward(raw_activation, weights[2:])
    mses: dict[str, float] = {
        "baseline": float(
            np.mean(np.square(baseline_prediction - target))
        )
    }
    maximum_prediction_pair_difference = 0.0

    transported_h2 = h2 + (exact_mean - sample_mean)[None, :]
    prediction = continue_forward(
        np.maximum(transported_h2, 0.0),
        weights[2:],
    )
    mses["h2_mean"] = float(np.mean(np.square(prediction - target)))
    maximum_prediction_pair_difference = max(
        maximum_prediction_pair_difference,
        float(np.max(np.abs(prediction - baseline_prediction))),
    )
    del transported_h2

    transported_h2 = (
        exact_mean[None, :]
        + (h2 - sample_mean[None, :])
        * np.sqrt(exact_variance / sample_variance)[None, :]
    )
    prediction = continue_forward(
        np.maximum(transported_h2, 0.0),
        weights[2:],
    )
    mses["h2_mean_variance"] = float(
        np.mean(np.square(prediction - target))
    )
    maximum_prediction_pair_difference = max(
        maximum_prediction_pair_difference,
        float(np.max(np.abs(prediction - baseline_prediction))),
    )
    del transported_h2

    for strength in (0.25, 0.5, 0.75, 1.0):
        activation = (
            raw_activation
            + strength
            * (corrected_relu_mean - raw_activation_mean)[None, :]
        )
        prediction = continue_forward(activation, weights[2:])
        mses[f"quadratic_cv_shift_{strength:g}"] = float(
            np.mean(np.square(prediction - target))
        )
        maximum_prediction_pair_difference = max(
            maximum_prediction_pair_difference,
            float(np.max(np.abs(prediction - baseline_prediction))),
        )
        del activation
    return {
        "mse": mses,
        "diagnostics": {
            "h2_mean_mse_before": float(
                np.mean(np.square(sample_mean - exact_mean))
            ),
            "h2_variance_relative_rmse_before": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            (sample_variance - exact_variance)
                            / np.maximum(exact_variance, 1e-20)
                        )
                    )
                )
            ),
            "quadratic_cv_mean_shift_rms": float(
                np.sqrt(
                    np.mean(
                        np.square(corrected_relu_mean - raw_activation_mean)
                    )
                )
            ),
            "maximum_prediction_pair_difference": (
                maximum_prediction_pair_difference
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=list(range(10)),
    )
    parser.add_argument("--ridge", type=float, default=1e-10)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if (
        not args.indices
        or min(args.indices) < 0
        or max(args.indices) >= 50
    ):
        raise ValueError("selection-only harness accepts IDs 0--49")

    asset = np.load(args.asset)
    rotation = np.asarray(asset["rotation"], dtype=np.float32)
    chirps = np.asarray(asset["chirps"], dtype=np.float32)
    expected_chirps = np.stack(
        [kerdock_chirp(index) for index in range(FIELD_SIZE)]
    )
    if not np.array_equal(chirps, expected_chirps):
        raise AssertionError("asset chirps do not match construction")
    radius = sphere_radius_mean(WIDTH)

    records = []
    rows = _load_rows(args.data, args.indices)
    for index, (name, weights, targets) in zip(
        args.indices,
        rows,
        strict=True,
    ):
        started = time.perf_counter()
        result = evaluate_network(
            weights,
            targets[-1],
            rotation,
            chirps,
            radius,
            args.ridge,
        )
        record = {
            "index": index,
            "name": name,
            "seconds": time.perf_counter() - started,
            **result,
        }
        records.append(record)
        print(record, flush=True)

    names = sorted(records[0]["mse"])
    summary = {
        name: {
            "mean_final_mse": float(
                np.mean([record["mse"][name] for record in records])
            ),
            "median_final_mse": float(
                np.median([record["mse"][name] for record in records])
            ),
            "gain_over_baseline": float(
                np.mean(
                    [record["mse"]["baseline"] for record in records]
                )
                / np.mean([record["mse"][name] for record in records])
            ),
        }
        for name in names
    }
    output = {
        "protocol": {
            "indices": args.indices,
            "holdout_loaded": False,
            "rotation_seed": 3,
            "ridge": args.ridge,
            "fixed_radius": radius,
        },
        "summary": summary,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
