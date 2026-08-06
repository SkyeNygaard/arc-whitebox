"""Exact-anchor-free Stein controls from late ReLU activations.

For ``X ~ N(0, I)`` and a regular vector field ``phi``,

    E[div(phi)(X) - X . phi(X)] = 0.

Two cheap specializations are tested for scalar, degree-one homogeneous
features ``s`` of a bias-free ReLU network.

Radial (forward quantities only):

    phi(x) = x s(x)
    T phi = (d + 1 - ||x||^2) s(x).

Tangential (one forward-mode JVP):

    phi(x) = A x s(x),  A^T = -A
    T phi = (A x) . grad s(x).

The tangential identity follows because ``tr(A)=0`` and ``x^T A x=0``.  It
does not require an expectation anchor or a model of late moments.
Coefficients are fitted only from already-sampled pointwise outputs, holding
out whole Kerdock bases through ``crossfit_grid``.
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

from eval_crossfit_cumulant_control import (  # noqa: E402
    crossfit_grid,
    empirical_c21_state,
)
from eval_exact_anchor_residual import FULL_DATA, ROWS_PER_BASIS  # noqa: E402
from eval_kerdock_design import (  # noqa: E402
    N_BASES,
    WIDTH,
    make_kerdock_design,
    random_rotation,
)
from eval_sampling_official import _load_rows  # noqa: E402
from exact_moments import sphere_radius_mean  # noqa: E402


def skew_direction(
    inputs: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Return ``A x`` rowwise for a fixed rank-two skew matrix ``A``."""
    rng = np.random.default_rng(seed)
    left = rng.standard_normal(WIDTH)
    left /= np.linalg.norm(left)
    right = rng.standard_normal(WIDTH)
    right -= left * (left @ right)
    right /= np.linalg.norm(right)
    x = np.asarray(inputs, dtype=np.float32)
    return (
        (x @ right.astype(np.float32))[:, None] * left[None, :]
        - (x @ left.astype(np.float32))[:, None] * right[None, :]
    ).astype(np.float32)


def forward_with_jvp_captures(
    weights: np.ndarray,
    points: np.ndarray,
    rotation: np.ndarray,
    layers: list[int],
    skew_seed: int,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    """Run the primal network and one tangential JVP to ``max(layers)``."""
    layer_set = set(layers)
    maximum_layer = max(layers)
    inputs = points @ rotation
    tangent = skew_direction(inputs, skew_seed)
    activation = inputs
    derivative = tangent
    captured = {}
    captured_derivative = {}

    for layer, weight in enumerate(weights):
        preactivation = activation @ weight
        if layer <= maximum_layer:
            preactivation_derivative = derivative @ weight
        gate = preactivation > 0.0
        activation = np.maximum(preactivation, 0.0)
        if layer <= maximum_layer:
            derivative = preactivation_derivative * gate
        if layer in layer_set:
            captured[layer] = activation.copy()
            captured_derivative[layer] = derivative.copy()

    return captured, captured_derivative, activation, inputs


def stein_features(
    activation: np.ndarray,
    derivative: np.ndarray,
    inputs: np.ndarray,
    rank: int,
    radius: float,
) -> dict[str, np.ndarray]:
    """Build radial and tangential linear/cubic Stein features."""
    left, right, _ = empirical_c21_state(activation, rank)
    h = np.asarray(activation, dtype=np.float64)
    dh = np.asarray(derivative, dtype=np.float64)

    linear_left = h @ left
    linear_right = h @ right
    tangent_linear_left = dh @ left
    tangent_linear_right = dh @ right

    squared_left = np.square(h) @ left
    linear_cubic_right = h @ right
    raw_cubic = squared_left * linear_cubic_right / np.square(radius)
    tangent_squared_left = (2.0 * h * dh) @ left
    tangent_cubic_right = dh @ right
    tangent_cubic = (
        tangent_squared_left * linear_cubic_right
        + squared_left * tangent_cubic_right
    ) / np.square(radius)

    radial_factor = (
        WIDTH + 1.0
        - np.sum(np.square(np.asarray(inputs, dtype=np.float64)), axis=1)
    )
    radial = radial_factor[:, None]
    return {
        "radial_linear_left": radial * linear_left,
        "radial_linear_both": radial
        * np.concatenate((linear_left, linear_right), axis=1),
        "radial_cubic": radial * raw_cubic,
        "tangent_linear_left": tangent_linear_left,
        "tangent_linear_both": np.concatenate(
            (tangent_linear_left, tangent_linear_right),
            axis=1,
        ),
        "tangent_cubic": tangent_cubic,
        "tangent_joint": np.concatenate(
            (
                tangent_linear_left,
                tangent_linear_right,
                tangent_cubic,
            ),
            axis=1,
        ),
    }


def summarize(records: list[dict]) -> dict:
    baseline = np.asarray([record["baseline_mse"] for record in records])
    labels = list(records[0]["method_mses"])
    result = {}
    for label in labels:
        values = np.asarray(
            [record["method_mses"][label] for record in records]
        )
        result[label] = {
            "ratio": float(np.mean(values) / np.mean(baseline)),
            "wins": int(np.sum(values < baseline)),
            "worst": float(np.max(values / baseline)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=list(range(160, 168)),
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[15, 29])
    parser.add_argument("--rotation-seed", type=int, default=3)
    parser.add_argument("--skew-seeds", type=int, nargs="+", default=[20260729])
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--ridges", type=float, nargs="+", default=[0.1])
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "results" / "stein_late_control_holdout8.json",
    )
    args = parser.parse_args()
    if len(args.skew_seeds) != 1:
        raise ValueError(
            "The decisive test intentionally permits one JVP only; "
            "run separate frozen seeds instead of silently adding cost."
        )

    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    radius = sphere_radius_mean(WIDTH)
    rows = _load_rows(FULL_DATA, args.indices)
    records = []

    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        started = time.perf_counter()
        captures, derivatives, final, inputs = forward_with_jvp_captures(
            weights,
            points,
            rotation,
            args.layers,
            args.skew_seeds[0],
        )
        baseline_prediction = final.mean(axis=0, dtype=np.float64)
        baseline_mse = float(
            np.mean(np.square(baseline_prediction - targets[-1]))
        )
        method_mses = {}
        diagnostics = {}
        for layer in args.layers:
            features = stein_features(
                captures[layer],
                derivatives[layer],
                inputs,
                args.rank,
                radius,
            )
            for feature_label, values in features.items():
                predictions, fit = crossfit_grid(
                    values,
                    final,
                    args.folds,
                    args.ridges,
                )
                diagnostics[f"layer{layer}_{feature_label}"] = {
                    **fit,
                    "sample_mean_norm": float(
                        np.linalg.norm(np.mean(values, axis=0))
                    ),
                    "sample_rms": float(np.sqrt(np.mean(np.square(values)))),
                }
                for ridge, prediction in predictions.items():
                    label = (
                        f"layer{layer}_{feature_label}:ridge={ridge:g}"
                    )
                    method_mses[label] = float(
                        np.mean(np.square(prediction - targets[-1]))
                    )

        record = {
            "index": index,
            "name": name,
            "baseline_mse": baseline_mse,
            "method_mses": method_mses,
            "diagnostics": diagnostics,
            "seconds": time.perf_counter() - started,
        }
        records.append(record)
        best = min(method_mses, key=method_mses.get)
        print(
            f"[{index}] base={baseline_mse:.4e} best={best} "
            f"{method_mses[best] / baseline_mse:.4f}x "
            f"({record['seconds']:.1f}s)",
            flush=True,
        )

    summary = summarize(records)
    output = {
        "protocol": {
            "indices": args.indices,
            "layers": args.layers,
            "rotation_seed": args.rotation_seed,
            "skew_seeds": args.skew_seeds,
            "rank": args.rank,
            "folds": args.folds,
            "ridges": args.ridges,
            "target_leakage": False,
            "stein_sign": "div(phi) - x dot phi",
            "radial_identity": "(d + 1 - norm(x)^2) s(x)",
            "tangential_identity": "(A x) dot grad s(x), A^T=-A",
            "rows": N_BASES * ROWS_PER_BASIS,
        },
        "summary": summary,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
