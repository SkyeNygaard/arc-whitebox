"""Coordinate-sparse connected-cubic probes at target layer 29.

For each of the first two sample-c21 SVD modes ``(u_k, v_k)``, select the top
``q`` coordinates by ``|u_ik|`` and expose the individual controls

    S_{i,v_k}(x),  i in top_q(|u_k|)

instead of summing them into the usual rank-one aggregate.  Whole Kerdock
bases are held out by ``crossfit_grid``; no final expectation target is used
to select directions or fit output coefficients.

The first pass compares exact arbitrary-center anchors with the connected-c21
anchor that omits the small sample-center correction.  A conditional second
pass can load the diagonal-corrected factorized c21 and test fixed correction
shrinks toward the baseline estimator.
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
    forward_layer_and_final,
)
from eval_exact_anchor_residual import FULL_DATA, ROWS_PER_BASIS  # noqa: E402
from eval_kerdock_design import (  # noqa: E402
    N_BASES,
    WIDTH,
    make_kerdock_design,
    random_rotation,
)
from eval_oracle_cumulant_bridge import connected_m21, moment_path  # noqa: E402
from eval_sampling_official import _load_rows  # noqa: E402
from exact_moments import sphere_radius_mean  # noqa: E402


def sparse_directions(
    sample_left: np.ndarray,
    sample_right: np.ndarray,
    q: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int | float]]]:
    """Return separate ``(e_i, v_k)`` columns for two nested supports."""
    if sample_left.shape[1] != 2 or sample_right.shape[1] != 2:
        raise ValueError((sample_left.shape, sample_right.shape))
    left = np.zeros((WIDTH, 2 * q), dtype=np.float64)
    right = np.empty((WIDTH, 2 * q), dtype=np.float64)
    metadata = []
    column = 0
    for mode in range(2):
        # Stable sorting makes all q supports exactly nested.
        coordinates = np.argsort(
            -np.abs(sample_left[:, mode]),
            kind="stable",
        )[:q]
        for coordinate in coordinates:
            left[coordinate, column] = 1.0
            right[:, column] = sample_right[:, mode]
            metadata.append(
                {
                    "mode": mode,
                    "coordinate": int(coordinate),
                    "sample_left_weight": float(
                        sample_left[coordinate, mode]
                    ),
                }
            )
            column += 1
    return left, right, metadata


def feature_cost_estimate(q: int, *, folds: int) -> dict[str, int]:
    """Conservative optimized FLOP estimate beyond the network forward."""
    points = N_BASES * ROWS_PER_BASIS
    modes = 2
    features = modes * q
    outputs = WIDTH

    # Common sample-c21 direction discovery: H.T@H and H^2.T@H.
    direction_moments = 4 * points * WIDTH**2
    direction_svd = 4 * WIDTH**3

    # Optimized sparse feature construction computes H@V once, gathers H_i,
    # and performs a conservative 20 scalar operations per exposed feature.
    mode_projections = 2 * points * WIDTH * modes
    sparse_pointwise = 20 * points * features

    # crossfit_grid forms total and held-out Gram/cross-products.  Across all
    # folds, held-out rows cover the dataset once, hence one additional copy.
    crossfit_products = (
        4 * points * features**2
        + 4 * points * features * outputs
    )
    solve = int(
        folds
        * (
            (2.0 / 3.0) * features**3
            + 2.0 * features**2 * outputs
        )
    )
    feature_only = (
        mode_projections
        + sparse_pointwise
        + crossfit_products
        + solve
    )
    return {
        "features": features,
        "direction_moments": int(direction_moments),
        "direction_svd": int(direction_svd),
        "mode_projections": int(mode_projections),
        "sparse_pointwise": int(sparse_pointwise),
        "crossfit_products": int(crossfit_products),
        "crossfit_solves": int(solve),
        "feature_only_total": int(feature_only),
        "including_direction_discovery": int(
            feature_only + direction_moments + direction_svd
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
        default=[2, 4, 8, 16, 32, 64],
    )
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument(
        "--factorized-dir",
        type=Path,
        help=(
            "Optional diagonal-corrected factorized c21 directory. Omit for "
            "the exact/oracle-connected ceiling pass."
        ),
    )
    parser.add_argument(
        "--factor-correction-shrinks",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            HERE
            / "results"
            / "sparse_connected_cubic_selection8_oracle.json"
        ),
    )
    args = parser.parse_args()
    if sorted(set(args.q_values)) != [2, 4, 8, 16, 32, 64]:
        raise ValueError("The bounded protocol fixes q={2,4,8,16,32,64}")

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
        sample_mean = np.mean(activation, axis=0, dtype=np.float64)
        sample_left, sample_right, _ = empirical_c21_state(
            activation,
            rank=2,
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
            factor_path = (
                args.factorized_dir / f"mlp_{index:05d}.npz"
            )
            with np.load(factor_path) as factorized:
                factorized_c21 = np.asarray(
                    factorized["c21"],
                    dtype=np.float64,
                )

        baseline_prediction = final.mean(axis=0, dtype=np.float64)
        baseline_mse = float(
            np.mean(np.square(baseline_prediction - targets[-1]))
        )
        method_mses = {}
        anchor_diagnostics = {}
        coordinate_metadata = {}

        for q in args.q_values:
            left, right, metadata = sparse_directions(
                sample_left,
                sample_right,
                q,
            )
            coordinate_metadata[str(q)] = metadata
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
            connected_only_anchor = (
                contract(left, true_c21, right) / (WIDTH + 1.0)
            )
            same_cloud_anchor = np.mean(values, axis=0, dtype=np.float64)
            same_cloud_error = max(
                float(np.linalg.norm(same_cloud_anchor - exact_anchor)),
                1e-30,
            )
            anchors = {
                f"q{q}_exact": exact_anchor,
                f"q{q}_oracle_connected_only": connected_only_anchor,
                f"q{q}_same_cloud": same_cloud_anchor,
            }
            if factorized_c21 is not None:
                anchors[f"q{q}_factorized"] = (
                    contract(left, factorized_c21, right)
                    / (WIDTH + 1.0)
                )

            for label, anchor in anchors.items():
                predictions, fit = crossfit_grid(
                    values - anchor,
                    final,
                    args.folds,
                    [args.ridge],
                )
                prediction = predictions[args.ridge]
                method_mses[label] = float(
                    np.mean(np.square(prediction - targets[-1]))
                )
                if label.endswith("_factorized"):
                    for shrink in args.factor_correction_shrinks:
                        if shrink == 1.0:
                            continue
                        blended = baseline_prediction + shrink * (
                            prediction - baseline_prediction
                        )
                        method_mses[
                            f"{label}_correction_shrink{shrink:g}"
                        ] = float(
                            np.mean(
                                np.square(blended - targets[-1])
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
                    "same_cloud_error_norm": same_cloud_error,
                    "fit": fit,
                }

            anchor_diagnostics[
                f"q{q}_center_correction"
            ] = {
                "relative_to_same_cloud": float(
                    np.linalg.norm(
                        connected_only_anchor - exact_anchor
                    )
                    / same_cloud_error
                ),
                "absolute_error_norm": float(
                    np.linalg.norm(
                        connected_only_anchor - exact_anchor
                    )
                ),
            }

        record = {
            "index": index,
            "name": name,
            "baseline_mse": baseline_mse,
            "method_mses": method_mses,
            "anchor_diagnostics": anchor_diagnostics,
            "coordinates": coordinate_metadata,
            "sample_mean_relative_error": float(
                np.linalg.norm(sample_mean - true_mean)
                / max(np.linalg.norm(true_mean), 1e-30)
            ),
            "seconds": time.perf_counter() - started,
        }
        records.append(record)
        best = min(method_mses, key=method_mses.get)
        print(
            f"[{index}] base={baseline_mse:.4e} "
            f"best={best}:{method_mses[best] / baseline_mse:.4f}x "
            f"q64exact={method_mses['q64_exact'] / baseline_mse:.3f}x "
            f"({record['seconds']:.1f}s)",
            flush=True,
        )

    summary = summarize(records)
    cost = {
        str(q): feature_cost_estimate(q, folds=args.folds)
        for q in args.q_values
    }
    output = {
        "protocol": {
            "indices": args.indices,
            "layer": args.layer,
            "rotation_seed": args.rotation_seed,
            "q_values": args.q_values,
            "modes": 2,
            "center": "sample_mean",
            "folds": args.folds,
            "ridge": args.ridge,
            "factorized_dir": (
                None
                if args.factorized_dir is None
                else str(args.factorized_dir)
            ),
            "factor_correction_shrinks": (
                args.factor_correction_shrinks
            ),
            "uses_final_targets_for_construction": False,
        },
        "feature_cost_estimate": cost,
        "summary": summary,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(json.dumps({"feature_cost_estimate": cost}, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
