"""Estimate the layer-29 mean with exact input harmonics, then re-center.

The degree-6/8/10/12 zonal input harmonics have exactly zero spherical
expectation.  Regressing the evaluated layer-29 activations on those features
and averaging held-basis residuals therefore gives a target-free estimate of
the population activation mean:

    mu_hat = crossfit Q[H_29 - beta^T g],   E[g] = 0.

This experiment asks the sharper question identified by the arbitrary-center
decomposition: is ``mu_hat`` accurate enough that a connected-cubic control
centered at ``mu_hat`` can use the oracle connected-C21 anchor directly?

Only selection IDs are used.  Oracle moments enter only the diagnostic C21
anchor, exact-center ceiling, and error measurements.
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
    sample_direction_families,
)
from eval_crossfit_cumulant_control import crossfit_grid  # noqa: E402
from eval_exact_anchor_residual import FULL_DATA  # noqa: E402
from eval_exact_ridge_harmonic_control import (  # noqa: E402
    expected_gate_maps,
    normalized_ridge_harmonic,
)
from eval_harmonic_center_correction import forward_target_with_gates  # noqa: E402
from eval_kerdock_design import (  # noqa: E402
    WIDTH,
    make_kerdock_design,
    random_rotation,
)
from eval_oracle_cumulant_bridge import connected_m21, moment_path  # noqa: E402
from eval_sampling_official import _load_rows  # noqa: E402
from eval_sparse_connected_cubic_probes import sparse_directions  # noqa: E402
from exact_moments import sphere_radius_mean  # noqa: E402


def paired_summary(records: list[dict]) -> dict[str, dict]:
    baseline = np.asarray(
        [record["baseline_mse"] for record in records],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20260729)
    boot = rng.integers(0, len(records), size=(20_000, len(records)))
    result = {}
    for label in records[0]["method_mses"]:
        values = np.asarray(
            [record["method_mses"][label] for record in records],
            dtype=np.float64,
        )
        ratios = (
            np.mean(values[boot], axis=1)
            / np.mean(baseline[boot], axis=1)
        )
        result[label] = {
            "method_mean_mse": float(np.mean(values)),
            "raw_mse_ratio": float(np.mean(values) / np.mean(baseline)),
            "ci95": [
                float(value)
                for value in np.percentile(ratios, [2.5, 97.5])
            ],
            "wins": int(np.sum(values < baseline)),
            "worst_per_network_ratio": float(np.max(values / baseline)),
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
        "--degrees",
        type=int,
        nargs="+",
        default=[6, 8, 10, 12],
    )
    parser.add_argument("--harmonic-directions", type=int, default=8)
    parser.add_argument("--mean-folds", type=int, default=6)
    parser.add_argument("--mean-ridge", type=float, default=1.0)
    parser.add_argument("--control-folds", type=int, default=6)
    parser.add_argument("--control-ridge", type=float, default=0.1)
    parser.add_argument(
        "--center-blends",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Optional frozen alpha values for sample + alpha*(full harmonic "
            "- sample). Alpha=1 is already the ordinary harmonic estimate."
        ),
    )
    parser.add_argument(
        "--sparse-q",
        type=int,
        nargs="*",
        default=[16, 64],
    )
    parser.add_argument(
        "--factorized-dir",
        type=Path,
        default=HERE / "results" / "factorized_k3_layer29",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            HERE / "results" / "harmonic_mean_center_selection8.json"
        ),
    )
    args = parser.parse_args()
    if args.degrees != [6, 8, 10, 12]:
        raise ValueError("bounded protocol fixes degrees 6,8,10,12")
    if args.harmonic_directions != 8 or args.mean_ridge != 1.0:
        raise ValueError("bounded protocol fixes qdir=8 and mean ridge=1")
    if any(q not in (16, 64) for q in args.sparse_q):
        raise ValueError(args.sparse_q)

    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    radius = sphere_radius_mean(WIDTH)
    rows = _load_rows(FULL_DATA, args.indices)
    records = []

    for index, (name, weights, targets) in zip(
        args.indices,
        rows,
        strict=True,
    ):
        started = time.perf_counter()
        activation, final, gates = forward_target_with_gates(
            weights,
            points,
            rotation,
            args.layer,
        )
        sample_mean = np.mean(activation, axis=0, dtype=np.float64)

        # Match the validated final-output ridge dictionary exactly: leading
        # input left singular vectors of the expected-gate end-to-end map.
        end_to_end, _ = expected_gate_maps(weights, rotation, gates)
        input_directions, singular_values, _ = np.linalg.svd(
            end_to_end,
            full_matrices=False,
        )
        input_directions = input_directions[
            :,
            : args.harmonic_directions,
        ]
        harmonic_by_degree = {
            degree: normalized_ridge_harmonic(
                points,
                input_directions,
                radius,
                degree,
            )
            for degree in args.degrees
        }
        harmonic_means = {}
        harmonic_fit = {}
        for count in range(1, len(args.degrees) + 1):
            degrees = args.degrees[:count]
            label = (
                f"degree{degrees[0]}"
                if len(degrees) == 1
                else "degrees" + "_".join(map(str, degrees))
            )
            features = np.concatenate(
                [harmonic_by_degree[degree] for degree in degrees],
                axis=1,
            )
            predictions, fit = crossfit_grid(
                features,
                activation,
                args.mean_folds,
                [args.mean_ridge],
            )
            harmonic_means[label] = predictions[args.mean_ridge]
            harmonic_fit[label] = {
                **fit,
                "feature_mean_norm": float(
                    np.linalg.norm(np.mean(features, axis=0))
                ),
            }

        with np.load(
            args.factorized_dir / f"mlp_{index:05d}.npz"
        ) as factorized:
            factorized_mean = np.asarray(
                factorized["mean"],
                dtype=np.float64,
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

        centers = {
            "sample": sample_mean,
            **{
                f"harmonic_{label}": estimate
                for label, estimate in harmonic_means.items()
            },
            "factorized": factorized_mean,
            "true": true_mean,
        }
        full_harmonic_mean = harmonic_means["degrees6_8_10_12"]
        for blend in args.center_blends:
            centers[f"harmonic_full_blend{blend:g}"] = (
                sample_mean
                + blend * (full_harmonic_mean - sample_mean)
            )
        sample_left, sample_right = sample_direction_families(
            activation,
            2,
            radius,
        )["radial_corrected_dirs"]
        probe_sets = {
            "rank2": (sample_left, sample_right),
        }
        for q in args.sparse_q:
            sparse_left, sparse_right, _ = sparse_directions(
                sample_left,
                sample_right,
                q,
            )
            probe_sets[f"sparse_q{q}"] = (
                sparse_left,
                sparse_right,
            )

        target = targets[-1]
        baseline_prediction = np.mean(final, axis=0, dtype=np.float64)
        baseline_mse = float(
            np.mean(np.square(baseline_prediction - target))
        )
        method_mses = {}
        anchor_diagnostics = {}
        for probe_label, (left, right) in probe_sets.items():
            connected_anchor = (
                contract(left, true_c21, right) / (WIDTH + 1.0)
            )
            for center_label, center in centers.items():
                values = contracted_pointwise(
                    activation,
                    left,
                    right,
                    center,
                    radius,
                )
                features = values - connected_anchor
                predictions, _ = crossfit_grid(
                    features,
                    final,
                    args.control_folds,
                    [args.control_ridge],
                )
                method_label = (
                    f"{probe_label}_{center_label}_center_oracle_c21"
                )
                method_mses[method_label] = float(
                    np.mean(
                        np.square(
                            predictions[args.control_ridge] - target
                        )
                    )
                )

                exact_center_anchor = contract(
                    left,
                    exact_anchor_matrix(
                        true_mean,
                        true_second,
                        true_raw_m21,
                        true_marginal_second,
                        center,
                    ),
                    right,
                )
                quadrature_discrepancy = (
                    np.mean(values, axis=0, dtype=np.float64)
                    - exact_center_anchor
                )
                anchor_diagnostics[method_label] = {
                    "connected_anchor_error_over_q_minus_e": float(
                        np.linalg.norm(
                            connected_anchor - exact_center_anchor
                        )
                        / max(
                            np.linalg.norm(quadrature_discrepancy),
                            1e-30,
                        )
                    ),
                    "connected_anchor_error_norm": float(
                        np.linalg.norm(
                            connected_anchor - exact_center_anchor
                        )
                    ),
                }

            # Existing exact arbitrary-center rank-2 ceiling for a direct
            # protocol identity check.
            if probe_label == "rank2":
                sample_values = contracted_pointwise(
                    activation,
                    left,
                    right,
                    sample_mean,
                    radius,
                )
                sample_exact_anchor = contract(
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
                predictions, _ = crossfit_grid(
                    sample_values - sample_exact_anchor,
                    final,
                    args.control_folds,
                    [args.control_ridge],
                )
                method_mses[
                    "rank2_sample_center_oracle_exact_anchor"
                ] = float(
                    np.mean(
                        np.square(
                            predictions[args.control_ridge] - target
                        )
                    )
                )

        sample_error = np.linalg.norm(sample_mean - true_mean)
        mean_diagnostics = {
            "sample": {
                "absolute_error_norm": float(sample_error),
                "error_over_sample": 1.0,
            },
            "factorized": {
                "absolute_error_norm": float(
                    np.linalg.norm(factorized_mean - true_mean)
                ),
                "error_over_sample": float(
                    np.linalg.norm(factorized_mean - true_mean)
                    / max(sample_error, 1e-30)
                ),
            },
            **{
                f"harmonic_{label}": {
                    "absolute_error_norm": float(
                        np.linalg.norm(estimate - true_mean)
                    ),
                    "error_over_sample": float(
                        np.linalg.norm(estimate - true_mean)
                        / max(sample_error, 1e-30)
                    ),
                }
                for label, estimate in harmonic_means.items()
            },
        }
        for blend in args.center_blends:
            label = f"harmonic_full_blend{blend:g}"
            estimate = centers[label]
            mean_diagnostics[label] = {
                "absolute_error_norm": float(
                    np.linalg.norm(estimate - true_mean)
                ),
                "error_over_sample": float(
                    np.linalg.norm(estimate - true_mean)
                    / max(sample_error, 1e-30)
                ),
            }
        record = {
            "index": index,
            "name": name,
            "baseline_mse": baseline_mse,
            "method_mses": method_mses,
            "mean_diagnostics": mean_diagnostics,
            "harmonic_fit": harmonic_fit,
            "linearized_singular_values": singular_values[:16].tolist(),
            "anchor_diagnostics": anchor_diagnostics,
            "seconds": time.perf_counter() - started,
        }
        records.append(record)
        full_label = "harmonic_degrees6_8_10_12"
        print(
            f"[{index}] meanerr(harm/sample)="
            f"{mean_diagnostics[full_label]['error_over_sample']:.3f} "
            f"rank2(sample/harm/true)="
            f"{method_mses['rank2_sample_center_oracle_c21'] / baseline_mse:.3f}/"
            f"{method_mses[f'rank2_{full_label}_center_oracle_c21'] / baseline_mse:.3f}/"
            f"{method_mses['rank2_true_center_oracle_c21'] / baseline_mse:.3f} "
            f"({record['seconds']:.1f}s)",
            flush=True,
        )

    summary = paired_summary(records)
    mean_labels = list(records[0]["mean_diagnostics"])
    pooled_mean_diagnostics = {}
    for label in mean_labels:
        squared_error = np.asarray(
            [
                np.square(
                    record["mean_diagnostics"][label][
                        "absolute_error_norm"
                    ]
                )
                for record in records
            ],
            dtype=np.float64,
        )
        sample_squared_error = np.asarray(
            [
                np.square(
                    record["mean_diagnostics"]["sample"][
                        "absolute_error_norm"
                    ]
                )
                for record in records
            ],
            dtype=np.float64,
        )
        pooled_mean_diagnostics[label] = {
            "pooled_l2_error_over_sample": float(
                np.sqrt(np.sum(squared_error))
                / max(np.sqrt(np.sum(sample_squared_error)), 1e-30)
            ),
            "mean_per_network_error_over_sample": float(
                np.mean(
                    [
                        record["mean_diagnostics"][label][
                            "error_over_sample"
                        ]
                        for record in records
                    ]
                )
            ),
            "wins_vs_sample": int(
                np.sum(squared_error < sample_squared_error)
            ),
        }

    output = {
        "protocol": {
            "indices": args.indices,
            "layer": args.layer,
            "rotation_seed": args.rotation_seed,
            "degrees": args.degrees,
            "harmonic_directions": args.harmonic_directions,
            "direction_family": (
                "leading input singular directions of empirical-gate "
                "end-to-end map"
            ),
            "mean_folds": args.mean_folds,
            "mean_ridge": args.mean_ridge,
            "control_folds": args.control_folds,
            "control_ridge": args.control_ridge,
            "center_blends": args.center_blends,
            "sparse_q": args.sparse_q,
            "factorized_dir": str(args.factorized_dir),
            "harmonic_anchor": "exact zero spherical mean",
            "control_anchor": "oracle true connected C21",
            "target_leakage_in_coefficients": False,
            "scope": "selection IDs only; no new holdout",
        },
        "pooled_mean_diagnostics": pooled_mean_diagnostics,
        "summary": summary,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(pooled_mean_diagnostics, indent=2), flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
