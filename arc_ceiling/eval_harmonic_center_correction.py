"""Estimate sparse cubic center corrections with exact ridge harmonics.

The sparse connected-cubic ceiling showed that the arbitrary-center correction

    A_i,v(m) - C21_i,v / (d + 1)

is as large as the entire Kerdock quadrature error even though the Kerdock
sample mean is very accurate.  This experiment estimates only the lower-order
state needed by that correction:

* the layer-29 Gaussian mean ``mu``;
* selected marginal second moments ``M2_i``;
* selected projections ``(M11 v_k)_i``.

Their fixed-radius pointwise values are regressed on degree 6/8/10/12 zonal
Gegenbauer harmonics, whose spherical expectations are exactly zero.  Held-
basis residual means therefore give target-free moment estimates.  The first
pass combines the estimated correction with oracle connected C21, isolating
lower-state accuracy.  A conditional second pass may substitute the
diagonal-corrected factorized C21.
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from eval_connected_cubic_control import (  # noqa: E402
    contract,
    contracted_pointwise,
    exact_anchor_matrix,
)
from eval_crossfit_cumulant_control import (  # noqa: E402
    crossfit_grid,
    empirical_c21_state,
)
from eval_exact_anchor_residual import FULL_DATA, ROWS_PER_BASIS  # noqa: E402
from eval_exact_ridge_harmonic_control import (  # noqa: E402
    normalized_ridge_harmonic,
)
from eval_kerdock_design import (  # noqa: E402
    N_BASES,
    WIDTH,
    make_kerdock_design,
    random_rotation,
)
from eval_oracle_cumulant_bridge import connected_m21, moment_path  # noqa: E402
from eval_sampling_official import _load_rows  # noqa: E402
from eval_sparse_connected_cubic_probes import (  # noqa: E402
    feature_cost_estimate,
    sparse_directions,
)
from exact_moments import sphere_radius_mean  # noqa: E402


def forward_target_with_gates(
    weights: np.ndarray,
    points: np.ndarray,
    rotation: np.ndarray,
    target_layer: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Return target activation, final output, and empirical gate rates."""
    pre = points @ (rotation @ weights[0].astype(np.float32))
    gates = [np.mean(pre > 0.0, axis=0, dtype=np.float64)]
    activation = np.maximum(pre, 0.0)
    target = activation.copy() if target_layer == 0 else None
    for layer, weight in enumerate(weights[1:], start=1):
        pre = activation @ weight
        gates.append(np.mean(pre > 0.0, axis=0, dtype=np.float64))
        activation = np.maximum(pre, 0.0)
        if layer == target_layer:
            target = activation.copy()
    if target is None:
        raise ValueError((target_layer, len(weights)))
    return target, activation, gates


def input_to_layer_map(
    weights: np.ndarray,
    rotation: np.ndarray,
    gates: list[np.ndarray],
    target_layer: int,
) -> np.ndarray:
    """Expected-gate input-to-post-target map."""
    result = (
        np.asarray(rotation, dtype=np.float64)
        @ np.asarray(weights[0], dtype=np.float64)
    )
    result *= gates[0][None, :]
    for layer in range(1, target_layer + 1):
        result = result @ np.asarray(weights[layer], dtype=np.float64)
        result *= gates[layer][None, :]
    return result


def moment_output_layout(
    activation: np.ndarray,
    right: np.ndarray,
    metadata: list[dict[str, int | float]],
) -> tuple[np.ndarray, dict]:
    """Pack fixed-radius moment integrands and their output layout."""
    coordinates = sorted(
        {
            int(item["coordinate"])
            for item in metadata
        }
    )
    coordinate_column = {
        coordinate: column
        for column, coordinate in enumerate(coordinates)
    }
    pair_column = {
        (int(item["mode"]), int(item["coordinate"])): column
        for column, item in enumerate(metadata)
    }
    projections = np.asarray(activation, dtype=np.float64) @ right
    outputs = [
        np.asarray(activation, dtype=np.float64),
        np.square(
            np.asarray(activation[:, coordinates], dtype=np.float64)
        ),
    ]
    pair_values = np.empty(
        (len(activation), len(metadata)),
        dtype=np.float64,
    )
    for column, item in enumerate(metadata):
        mode = int(item["mode"])
        coordinate = int(item["coordinate"])
        pair_values[:, column] = (
            activation[:, coordinate] * projections[:, mode]
        )
    outputs.append(pair_values)
    packed = np.concatenate(outputs, axis=1)
    layout = {
        "mu": [0, WIDTH],
        "m2": [WIDTH, WIDTH + len(coordinates)],
        "pair": [
            WIDTH + len(coordinates),
            WIDTH + len(coordinates) + len(metadata),
        ],
        "coordinates": coordinates,
        "coordinate_column": coordinate_column,
        "pair_column": {
            f"{mode}:{coordinate}": column
            for (mode, coordinate), column in pair_column.items()
        },
    }
    return packed, layout


def unpack_moment_estimate(
    estimate: np.ndarray,
    layout: dict,
    radius: float,
) -> tuple[np.ndarray, dict[int, float], dict[tuple[int, int], float]]:
    """Convert fixed-radius estimates to Gaussian first/second moments."""
    mu_start, mu_stop = layout["mu"]
    m2_start, m2_stop = layout["m2"]
    pair_start, pair_stop = layout["pair"]
    mu = np.asarray(estimate[mu_start:mu_stop], dtype=np.float64)
    radial_second_scale = WIDTH / np.square(radius)
    m2_values = (
        radial_second_scale
        * np.asarray(estimate[m2_start:m2_stop], dtype=np.float64)
    )
    pair_values = (
        radial_second_scale
        * np.asarray(estimate[pair_start:pair_stop], dtype=np.float64)
    )
    m2 = {
        int(coordinate): float(m2_values[column])
        for column, coordinate in enumerate(layout["coordinates"])
    }
    pair = {}
    for key, column in layout["pair_column"].items():
        mode, coordinate = map(int, key.split(":"))
        pair[(mode, coordinate)] = float(pair_values[column])
    return mu, m2, pair


def estimated_center_correction(
    sample_mean: np.ndarray,
    sample_right: np.ndarray,
    metadata: list[dict[str, int | float]],
    mu: np.ndarray,
    m2: dict[int, float],
    pair: dict[tuple[int, int], float],
) -> np.ndarray:
    """Return ``A(m)-C21/(d+1)`` for each sparse ``(i,v_k)`` probe."""
    correction = np.empty(len(metadata), dtype=np.float64)
    sample_projection = sample_mean @ sample_right
    mean_projection = mu @ sample_right
    for column, item in enumerate(metadata):
        mode = int(item["mode"])
        coordinate = int(item["coordinate"])
        center_i = sample_mean[coordinate]
        mean_i = mu[coordinate]
        delta_i = center_i - mean_i
        delta_projection = (
            sample_projection[mode] - mean_projection[mode]
        )
        correction[column] = (
            -2.0 * delta_i * pair[(mode, coordinate)]
            - m2[coordinate] * delta_projection
            + 2.0
            * (np.square(center_i) - np.square(mean_i))
            * mean_projection[mode]
        ) / (WIDTH + 1.0)
    return correction


def cost_estimate(
    *,
    harmonic_directions: int,
    degrees: list[int],
    moment_targets: int,
    folds: int,
    q: int,
) -> dict[str, int]:
    points = N_BASES * ROWS_PER_BASIS
    harmonic_features = harmonic_directions * len(degrees)
    gate_map = 2 * 30 * WIDTH**3
    map_svd = 4 * WIDTH**3
    ridge_projection = 2 * points * WIDTH * harmonic_directions
    gegenbauer = (
        6 * points * harmonic_directions * sum(degrees)
    )
    moment_products = (
        4 * points * harmonic_features**2
        + 4 * points * harmonic_features * moment_targets
    )
    moment_solves = int(
        folds
        * (
            (2.0 / 3.0) * harmonic_features**3
            + 2.0
            * harmonic_features**2
            * moment_targets
        )
    )
    sparse_cost = feature_cost_estimate(q, folds=folds)
    harmonic_total = (
        gate_map
        + map_svd
        + ridge_projection
        + gegenbauer
        + moment_products
        + moment_solves
    )
    return {
        "harmonic_features": harmonic_features,
        "moment_targets": moment_targets,
        "gate_map": int(gate_map),
        "map_svd": int(map_svd),
        "ridge_projection": int(ridge_projection),
        "gegenbauer_evaluation": int(gegenbauer),
        "moment_crossfit_products": int(moment_products),
        "moment_crossfit_solves": int(moment_solves),
        "harmonic_moment_total": int(harmonic_total),
        "sparse_control_including_direction_discovery": int(
            sparse_cost["including_direction_discovery"]
        ),
        "combined_upper_bound": int(
            harmonic_total
            + sparse_cost["including_direction_discovery"]
        ),
    }


def summarize(records: list[dict]) -> dict:
    baseline = np.asarray([record["baseline_mse"] for record in records])
    labels = list(records[0]["method_mses"])
    result = {}
    for label in labels:
        mse = np.asarray([record["method_mses"][label] for record in records])
        result[label] = {
            "mse_ratio": float(np.mean(mse) / np.mean(baseline)),
            "wins": int(np.sum(mse < baseline)),
            "worst": float(np.max(mse / baseline)),
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
    parser.add_argument("--layer", type=int, default=29)
    parser.add_argument("--rotation-seed", type=int, default=3)
    parser.add_argument(
        "--q-values",
        type=int,
        nargs="+",
        default=[16, 32, 64],
    )
    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=[6, 8, 10, 12],
    )
    parser.add_argument("--harmonic-directions", type=int, default=32)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument(
        "--factorized-dir",
        type=Path,
        help=(
            "Optional diagonal-corrected factorized c21 directory. Omit for "
            "the oracle-C21 lower-state isolation pass."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            HERE
            / "results"
            / "harmonic_center_correction_oraclec21_selection8.json"
        ),
    )
    args = parser.parse_args()
    if sorted(set(args.q_values)) != [16, 32, 64]:
        raise ValueError("The bounded protocol fixes q={16,32,64}")
    if args.degrees != [6, 8, 10, 12]:
        raise ValueError("The bounded protocol fixes degrees 6,8,10,12")

    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    radius = sphere_radius_mean(WIDTH)
    rows = _load_rows(FULL_DATA, args.indices)
    records = []

    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        started = time.perf_counter()
        activation, final, gates = forward_target_with_gates(
            weights,
            points,
            rotation,
            args.layer,
        )
        sample_mean = np.mean(activation, axis=0, dtype=np.float64)
        sample_left, sample_right, _ = empirical_c21_state(
            activation,
            rank=2,
        )
        _, _, max_metadata = sparse_directions(
            sample_left,
            sample_right,
            max(args.q_values),
        )

        linearized_map = input_to_layer_map(
            weights,
            rotation,
            gates,
            args.layer,
        )
        input_directions, singular_values, _ = np.linalg.svd(
            linearized_map,
            full_matrices=False,
        )
        input_directions = input_directions[
            :,
            : args.harmonic_directions,
        ]
        harmonic_features = np.concatenate(
            [
                normalized_ridge_harmonic(
                    points,
                    input_directions,
                    radius,
                    degree,
                )
                for degree in args.degrees
            ],
            axis=1,
        )
        moment_outputs, moment_layout = moment_output_layout(
            activation,
            sample_right,
            max_metadata,
        )
        moment_predictions, moment_fit = crossfit_grid(
            harmonic_features,
            moment_outputs,
            args.folds,
            [args.ridge],
        )
        mu_estimate, m2_estimate, pair_estimate = (
            unpack_moment_estimate(
                moment_predictions[args.ridge],
                moment_layout,
                radius,
            )
        )

        with np.load(moment_path(index)) as oracle:
            true_mean = np.asarray(
                oracle["mean"][args.layer],
                dtype=np.float64,
            )
            true_second = np.asarray(
                oracle["M11"][args.layer],
                dtype=np.float64,
            )
            true_raw_m21 = np.asarray(
                oracle["M21"][args.layer],
                dtype=np.float64,
            )
            true_marginal_second = np.asarray(
                oracle["m2"][args.layer],
                dtype=np.float64,
            )
            true_c21 = connected_m21(
                true_mean,
                true_second,
                true_raw_m21,
                true_marginal_second,
            )

        factorized_c21 = None
        if args.factorized_dir is not None:
            with np.load(
                args.factorized_dir / f"mlp_{index:05d}.npz"
            ) as factorized:
                factorized_c21 = np.asarray(
                    factorized["c21"],
                    dtype=np.float64,
                )

        baseline_prediction = np.mean(final, axis=0, dtype=np.float64)
        baseline_mse = float(
            np.mean(np.square(baseline_prediction - targets[-1]))
        )
        method_mses = {}
        anchor_diagnostics = {}
        for q in args.q_values:
            left, right, metadata = sparse_directions(
                sample_left,
                sample_right,
                q,
            )
            values = contracted_pointwise(
                activation,
                left,
                right,
                sample_mean,
                radius,
            )
            exact_anchor = contract(
                left,
                exact_anchor_matrix(
                    true_mean,
                    true_second,
                    true_raw_m21,
                    true_marginal_second,
                    sample_mean,
                ),
                right,
            )
            oracle_c21_anchor = (
                contract(left, true_c21, right) / (WIDTH + 1.0)
            )
            oracle_m2 = {
                int(item["coordinate"]): float(
                    true_marginal_second[int(item["coordinate"])]
                )
                for item in metadata
            }
            oracle_pair = {
                (int(item["mode"]), int(item["coordinate"])): float(
                    true_second[int(item["coordinate"])]
                    @ sample_right[:, int(item["mode"])]
                )
                for item in metadata
            }
            oracle_correction = estimated_center_correction(
                sample_mean,
                sample_right,
                metadata,
                true_mean,
                oracle_m2,
                oracle_pair,
            )
            if not np.allclose(
                oracle_c21_anchor + oracle_correction,
                exact_anchor,
                rtol=2e-7,
                atol=2e-10,
            ):
                raise AssertionError(
                    (
                        oracle_c21_anchor + oracle_correction,
                        exact_anchor,
                    )
                )
            estimated_correction = estimated_center_correction(
                sample_mean,
                sample_right,
                metadata,
                mu_estimate,
                m2_estimate,
                pair_estimate,
            )
            anchors = {
                f"q{q}_exact": exact_anchor,
                f"q{q}_oracle_c21_only": oracle_c21_anchor,
                f"q{q}_oracle_c21_harmonic_lower": (
                    oracle_c21_anchor + estimated_correction
                ),
            }
            if factorized_c21 is not None:
                anchors[f"q{q}_factor_c21_harmonic_lower"] = (
                    contract(left, factorized_c21, right)
                    / (WIDTH + 1.0)
                    + estimated_correction
                )

            same_cloud_anchor = np.mean(values, axis=0, dtype=np.float64)
            same_cloud_error = max(
                float(np.linalg.norm(same_cloud_anchor - exact_anchor)),
                1e-30,
            )
            for label, anchor in anchors.items():
                predictions, _ = crossfit_grid(
                    values - anchor,
                    final,
                    args.folds,
                    [args.ridge],
                )
                method_mses[label] = float(
                    np.mean(
                        np.square(
                            predictions[args.ridge] - targets[-1]
                        )
                    )
                )
                anchor_diagnostics[label] = {
                    "relative_to_same_cloud": float(
                        np.linalg.norm(anchor - exact_anchor)
                        / same_cloud_error
                    ),
                    "absolute_error_norm": float(
                        np.linalg.norm(anchor - exact_anchor)
                    ),
                }

        moment_diagnostics = {
            "mu_sample_relative_error": float(
                np.linalg.norm(sample_mean - true_mean)
                / max(np.linalg.norm(true_mean), 1e-30)
            ),
            "mu_harmonic_relative_error": float(
                np.linalg.norm(mu_estimate - true_mean)
                / max(np.linalg.norm(true_mean), 1e-30)
            ),
            "moment_fit": moment_fit,
            "linearized_singular_values": singular_values[:16].tolist(),
        }
        record = {
            "index": index,
            "name": name,
            "baseline_mse": baseline_mse,
            "method_mses": method_mses,
            "anchor_diagnostics": anchor_diagnostics,
            "moment_diagnostics": moment_diagnostics,
            "moment_targets": int(moment_outputs.shape[1]),
            "seconds": time.perf_counter() - started,
        }
        records.append(record)
        best = min(method_mses, key=method_mses.get)
        print(
            f"[{index}] base={baseline_mse:.4e} "
            f"best={best}:{method_mses[best] / baseline_mse:.4f}x "
            f"q64harmonic="
            f"{method_mses['q64_oracle_c21_harmonic_lower'] / baseline_mse:.3f}x "
            f"mu(sample/harm)="
            f"{moment_diagnostics['mu_sample_relative_error']:.4g}/"
            f"{moment_diagnostics['mu_harmonic_relative_error']:.4g} "
            f"({record['seconds']:.1f}s)",
            flush=True,
        )

    summary = summarize(records)
    max_targets = max(record["moment_targets"] for record in records)
    cost = {
        str(q): cost_estimate(
            harmonic_directions=args.harmonic_directions,
            degrees=args.degrees,
            moment_targets=max_targets,
            folds=args.folds,
            q=q,
        )
        for q in args.q_values
    }
    output = {
        "protocol": {
            "indices": args.indices,
            "layer": args.layer,
            "rotation_seed": args.rotation_seed,
            "q_values": args.q_values,
            "degrees": args.degrees,
            "harmonic_directions": args.harmonic_directions,
            "direction_family": "leading_input_singular_of_expected_gate_map",
            "folds": args.folds,
            "ridge": args.ridge,
            "factorized_dir": (
                None
                if args.factorized_dir is None
                else str(args.factorized_dir)
            ),
            "target_leakage": False,
            "radial_second_scale": WIDTH / np.square(radius),
        },
        "cost_estimate": cost,
        "summary": summary,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(json.dumps({"cost_estimate": cost}, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
