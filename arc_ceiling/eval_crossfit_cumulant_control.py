"""Can oracle deep-cumulant directions become a target-free control variate?

`eval_oracle_cumulant_bridge.py` showed that exactly anchored layer-29 cubic
features predict Kerdock integration error across held-out rotations when their
coefficients are fit with forbidden target means.  This script removes that
leakage.  It learns coefficients only by regressing the already-evaluated
pointwise network output on the pointwise control features, holding out whole
Kerdock bases, and averages held-basis residual means:

    mu_hat = average_folds Q_holdout[f - beta_train^T phi],
    E[phi] = 0.

The oracle higher-moment file is still used for the feature directions and
Gaussian anchor.  Thus this is the second-stage ceiling:

    oracle state + deployable coefficient estimation.

If it succeeds, the remaining problem is sharply isolated to estimating the
late-layer x1a/c21 state within the runtime budget.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from eval_exact_anchor_residual import FULL_DATA, ROWS_PER_BASIS  # noqa: E402
from eval_kerdock_design import N_BASES, WIDTH, make_kerdock_design, random_rotation  # noqa: E402
from eval_oracle_cumulant_bridge import (  # noqa: E402
    connected_m21,
    direction_families,
    moment_path,
    truncated_svd,
)
from eval_sampling_official import _load_rows  # noqa: E402
from exact_moments import sphere_radius_mean  # noqa: E402


def forward_layer_and_final(
    weights: np.ndarray,
    points: np.ndarray,
    rotation: np.ndarray,
    capture_layer: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preactivation = points @ (rotation @ weights[0].astype(np.float32))
    activation = np.maximum(preactivation, 0.0)
    captured_pre = preactivation.copy() if capture_layer == 0 else None
    captured = activation if capture_layer == 0 else None
    for layer in range(1, len(weights)):
        preactivation = activation @ weights[layer]
        activation = np.maximum(preactivation, 0.0)
        if layer == capture_layer:
            captured_pre = preactivation.copy()
            captured = activation.copy()
    if captured is None or captured_pre is None:
        raise ValueError((capture_layer, len(weights)))
    return captured_pre, captured, activation


def forward_to_layer(
    weights: np.ndarray,
    points: np.ndarray,
    rotation: np.ndarray,
    layer: int,
) -> np.ndarray:
    activation = np.maximum(
        points @ (rotation @ weights[0].astype(np.float32)),
        0.0,
    )
    for current_layer in range(1, layer + 1):
        activation = np.maximum(
            activation @ weights[current_layer],
            0.0,
        )
    return activation


def pointwise_features(
    activation: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    anchor: np.ndarray,
    radius: float,
) -> np.ndarray:
    h = activation.astype(np.float64, copy=False)
    return (
        (np.square(h) @ left) * (h @ right) / radius**2
        - anchor
    )


def empirical_c21_state(
    activation: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sample-SVD directions and the fixed-radius anchor estimate."""
    h = activation.astype(np.float64, copy=False)
    n = len(h)
    mean = np.mean(h, axis=0)
    second = (h.T @ h) / n
    raw_m21 = (np.square(h).T @ h) / n
    cumulant = connected_m21(mean, second, raw_m21, np.diag(second))
    left, right = truncated_svd(cumulant, rank)
    return left, right, raw_m21


def sobol_normals(n_points: int) -> np.ndarray:
    if n_points <= 0 or n_points & (n_points - 1):
        raise ValueError("Gaussian anchor point count must be a positive power of two")
    unit = qmc.Sobol(WIDTH, scramble=True, seed=20260729).random_base2(
        int(np.log2(n_points))
    )
    # Avoid infinities if a future Sobol implementation emits exact endpoints.
    return ndtri(np.clip(unit, 2.0**-53, 1.0 - 2.0**-53))


def gaussian_closure_anchor(
    pre_mean: np.ndarray,
    pre_second: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    normal_points: np.ndarray,
) -> np.ndarray:
    covariance = np.asarray(pre_second, dtype=np.float64) - np.outer(
        pre_mean,
        pre_mean,
    )
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(np.max(eigenvalues)) * 1e-12, 0.0)
    root = eigenvectors * np.sqrt(np.maximum(eigenvalues, floor))
    h = np.maximum(
        normal_points @ root.T + np.asarray(pre_mean, dtype=np.float64),
        0.0,
    )
    raw_contraction = np.mean(
        (np.square(h) @ left) * (h @ right),
        axis=0,
    )
    return raw_contraction / (WIDTH + 1)


def raw_m21_from_cumulants(
    mean: np.ndarray,
    covariance: np.ndarray,
    c21: np.ndarray,
) -> np.ndarray:
    second = covariance + np.outer(mean, mean)
    marginal_second = np.diag(second)
    return (
        c21
        + 2.0 * mean[:, None] * second
        + marginal_second[:, None] * mean[None, :]
        - 2.0 * np.square(mean[:, None]) * mean[None, :]
    )


def crossfit_grid(
    features: np.ndarray,
    outputs: np.ndarray,
    folds: int,
    ridges: list[float],
) -> tuple[dict[float, np.ndarray], dict[str, float]]:
    """Fit on basis blocks and compute residual means from sufficient stats."""
    if len(features) != N_BASES * ROWS_PER_BASIS:
        raise ValueError((features.shape, N_BASES, ROWS_PER_BASIS))
    scale = np.sqrt(np.mean(np.square(features), axis=0))
    keep = scale > 1e-12
    if not np.any(keep):
        baseline = outputs.mean(axis=0, dtype=np.float64)
        return {ridge: baseline for ridge in ridges}, {
            "features": float(features.shape[1]),
            "kept": 0.0,
            "feature_rms": 0.0,
        }

    x = features[:, keep] / scale[keep]
    y = outputs.astype(np.float64, copy=False)
    block_ids = np.repeat(np.arange(N_BASES), ROWS_PER_BASIS)
    fold_ids = block_ids % folds
    gram_total = x.T @ x
    cross_total = x.T @ y

    estimates = {ridge: [] for ridge in ridges}
    sizes = []
    max_condition = {ridge: 0.0 for ridge in ridges}
    for fold in range(folds):
        test = fold_ids == fold
        x_test = x[test]
        y_test = y[test]
        gram_train = gram_total - x_test.T @ x_test
        cross_train = cross_total - x_test.T @ y_test
        n_train = len(x) - int(np.sum(test))
        sizes.append(int(np.sum(test)))
        mean_x = np.mean(x_test, axis=0)
        mean_y = np.mean(y_test, axis=0)
        for ridge in ridges:
            system = gram_train + ridge * n_train * np.eye(x.shape[1])
            coefficient = np.linalg.solve(system, cross_train)
            estimates[ridge].append(mean_y - mean_x @ coefficient)
            max_condition[ridge] = max(
                max_condition[ridge],
                float(np.linalg.cond(system)),
            )

    predictions = {
        ridge: np.average(values, axis=0, weights=sizes)
        for ridge, values in estimates.items()
    }
    diagnostics = {
        "features": float(features.shape[1]),
        "kept": float(np.sum(keep)),
        "feature_rms": float(np.sqrt(np.mean(np.square(features)))),
        **{
            f"condition_ridge_{ridge:g}": condition
            for ridge, condition in max_condition.items()
        },
    }
    return predictions, diagnostics


def paired_summary(records: list[dict], labels: list[str]) -> dict:
    baseline = np.asarray([record["baseline_mse"] for record in records])
    rng = np.random.default_rng(20260729)
    boot = rng.integers(0, len(records), size=(20000, len(records)))
    result = {}
    for label in labels:
        values = np.asarray([record["method_mses"][label] for record in records])
        ratios = values[boot].mean(axis=1) / baseline[boot].mean(axis=1)
        result[label] = {
            "ratio": float(values.mean() / baseline.mean()),
            "ci95": [float(x) for x in np.percentile(ratios, [2.5, 97.5])],
            "wins": int(np.sum(values < baseline)),
            "worst": float(np.max(values / baseline)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(100, 108)))
    parser.add_argument("--layer", type=int, default=29)
    parser.add_argument("--rotation-seed", type=int, default=3)
    parser.add_argument("--folds", type=int, nargs="+", default=[3, 6])
    parser.add_argument("--ridges", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    parser.add_argument("--gaussian-anchor-points", type=int, default=0)
    parser.add_argument("--factorized-k3-dir", type=Path)
    parser.add_argument("--pilot-rotation-seed", type=int)
    parser.add_argument(
        "--pilot-bases",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 129],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "results" / "crossfit_cumulant_control_selection8.json",
    )
    args = parser.parse_args()

    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    radius = sphere_radius_mean(WIDTH)
    normal_points = (
        sobol_normals(args.gaussian_anchor_points)
        if args.gaussian_anchor_points
        else None
    )
    rows = _load_rows(FULL_DATA, args.indices)
    records = []

    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        started = time.perf_counter()
        with np.load(moment_path(index)) as moment_data:
            families = direction_families(moment_data, args.layer, 16)
            oracle_raw_m21 = np.asarray(
                moment_data["M21"][args.layer],
                dtype=np.float64,
            )
            oracle_post_mean = np.asarray(
                moment_data["mean"][args.layer],
                dtype=np.float64,
            )
            oracle_post_second = np.asarray(
                moment_data["M11"][args.layer],
                dtype=np.float64,
            )
            oracle_post_c21 = connected_m21(
                oracle_post_mean,
                oracle_post_second,
                oracle_raw_m21,
                np.asarray(moment_data["m2"][args.layer], dtype=np.float64),
            )
            oracle_pre_mean = np.asarray(
                moment_data["pre_mean"][args.layer],
                dtype=np.float64,
            )
            oracle_pre_second = np.asarray(
                moment_data["pre_M11"][args.layer],
                dtype=np.float64,
            )
        captured_pre, captured, final = forward_layer_and_final(
            weights,
            points,
            rotation,
            args.layer,
        )
        baseline_prediction = final.mean(axis=0, dtype=np.float64)
        baseline_mse = float(
            np.mean(np.square(baseline_prediction - targets[-1]))
        )

        configs = {
            "x1a_rank16": pointwise_features(
                captured,
                families["x1a"].left[:, :16],
                families["x1a"].right[:, :16],
                families["x1a"].anchor[:16],
                radius,
            ),
            "c21_rank4": pointwise_features(
                captured,
                families["c21"].left[:, :4],
                families["c21"].right[:, :4],
                families["c21"].anchor[:4],
                radius,
            ),
        }

        # Direction/anchor ablation.  Directions can potentially be recovered
        # for free from the activation cloud already evaluated by Kerdock.  The
        # anchor cannot: centering on the same cloud erases exactly the
        # quadrature discrepancy we need.  Blend tests quantify how accurate a
        # transported/analytic anchor must be.
        sample_left, sample_right, sample_raw_m21 = empirical_c21_state(
            captured,
            rank=4,
        )
        sample_direction_oracle_anchor = np.einsum(
            "ik,ij,jk->k",
            sample_left,
            oracle_raw_m21,
            sample_right,
        ) / (WIDTH + 1)
        sample_direction_sample_anchor = np.einsum(
            "ik,ij,jk->k",
            sample_left,
            sample_raw_m21,
            sample_right,
        ) / radius**2
        sample_centered_features = pointwise_features(
            captured,
            sample_left,
            sample_right,
            sample_direction_sample_anchor,
            radius,
        )
        sample_block_means = sample_centered_features.reshape(
            N_BASES,
            ROWS_PER_BASIS,
            4,
        ).mean(axis=1)
        sample_block_se = np.std(
            sample_block_means,
            axis=0,
            ddof=1,
        ) / np.sqrt(N_BASES)
        anchor_vectors = {
            "oracle": sample_direction_oracle_anchor.tolist(),
            "sample": sample_direction_sample_anchor.tolist(),
            "sample_block_se": sample_block_se.tolist(),
        }
        for blend in (0.0, 0.1, 0.25, 0.5, 1.0):
            anchor = (
                (1.0 - blend) * sample_direction_oracle_anchor
                + blend * sample_direction_sample_anchor
            )
            configs[f"sampledir_c21_rank4_anchorblend{blend:g}"] = (
                pointwise_features(
                    captured,
                    sample_left,
                    sample_right,
                    anchor,
                    radius,
                )
            )

        oracle_left = families["c21"].left[:, :4]
        oracle_right = families["c21"].right[:, :4]
        oracle_direction_sample_anchor = np.einsum(
            "ik,ij,jk->k",
            oracle_left,
            sample_raw_m21,
            oracle_right,
        ) / radius**2
        configs["oracledir_c21_rank4_sampleanchor"] = pointwise_features(
            captured,
            oracle_left,
            oracle_right,
            oracle_direction_sample_anchor,
            radius,
        )

        if normal_points is not None:
            oracle_pre_gaussian_anchor = gaussian_closure_anchor(
                oracle_pre_mean,
                oracle_pre_second,
                sample_left,
                sample_right,
                normal_points,
            )
            sample_pre_mean = np.mean(captured_pre, axis=0, dtype=np.float64)
            sample_pre_second = (
                captured_pre.astype(np.float64).T
                @ captured_pre.astype(np.float64)
            ) / len(captured_pre)
            sample_pre_gaussian_anchor = gaussian_closure_anchor(
                sample_pre_mean,
                sample_pre_second,
                sample_left,
                sample_right,
                normal_points,
            )
            configs["sampledir_c21_rank4_oraclepre_gaussiananchor"] = (
                pointwise_features(
                    captured,
                    sample_left,
                    sample_right,
                    oracle_pre_gaussian_anchor,
                    radius,
                )
            )
            configs["sampledir_c21_rank4_samplepre_gaussiananchor"] = (
                pointwise_features(
                    captured,
                    sample_left,
                    sample_right,
                    sample_pre_gaussian_anchor,
                    radius,
                )
            )
            anchor_vectors["oracle_pre_gaussian"] = (
                oracle_pre_gaussian_anchor.tolist()
            )
            anchor_vectors["sample_pre_gaussian"] = (
                sample_pre_gaussian_anchor.tolist()
            )
            diagnostics_anchor = {
                "oracle_pre_gaussian_vs_true": float(
                    np.linalg.norm(
                        oracle_pre_gaussian_anchor
                        - sample_direction_oracle_anchor
                    )
                    / max(
                        np.linalg.norm(
                            sample_direction_sample_anchor
                            - sample_direction_oracle_anchor
                        ),
                        1e-30,
                    )
                ),
                "sample_pre_gaussian_vs_true": float(
                    np.linalg.norm(
                        sample_pre_gaussian_anchor
                        - sample_direction_oracle_anchor
                    )
                    / max(
                        np.linalg.norm(
                            sample_direction_sample_anchor
                            - sample_direction_oracle_anchor
                        ),
                        1e-30,
                    )
                ),
            }
        else:
            diagnostics_anchor = {}

        if args.factorized_k3_dir is not None:
            factorized_path = (
                args.factorized_k3_dir / f"mlp_{index:05d}.npz"
            )
            with np.load(factorized_path) as factorized:
                predicted_post_mean = np.asarray(
                    factorized["mean"],
                    dtype=np.float64,
                )
                predicted_post_covariance = np.asarray(
                    factorized["covariance"],
                    dtype=np.float64,
                )
                predicted_post_c21 = np.asarray(
                    factorized["c21"],
                    dtype=np.float64,
                )
            oracle_post_covariance = (
                oracle_post_second
                - np.outer(oracle_post_mean, oracle_post_mean)
            )
            raw_variants = {
                "factorized": raw_m21_from_cumulants(
                    predicted_post_mean,
                    predicted_post_covariance,
                    predicted_post_c21,
                ),
                "factorized_c21_oracle_marginals": raw_m21_from_cumulants(
                    oracle_post_mean,
                    oracle_post_covariance,
                    predicted_post_c21,
                ),
                "oracle_c21_factorized_marginals": raw_m21_from_cumulants(
                    predicted_post_mean,
                    predicted_post_covariance,
                    oracle_post_c21,
                ),
            }
            for variant_name, predicted_raw_m21 in raw_variants.items():
                predicted_anchor = np.einsum(
                    "ik,ij,jk->k",
                    sample_left,
                    predicted_raw_m21,
                    sample_right,
                ) / (WIDTH + 1)
                configs[f"sampledir_c21_rank4_{variant_name}_anchor"] = (
                    pointwise_features(
                        captured,
                        sample_left,
                        sample_right,
                        predicted_anchor,
                        radius,
                    )
                )
                diagnostics_anchor[f"{variant_name}_vs_true"] = float(
                    np.linalg.norm(
                        predicted_anchor - sample_direction_oracle_anchor
                    )
                    / max(
                        np.linalg.norm(
                            sample_direction_sample_anchor
                            - sample_direction_oracle_anchor
                        ),
                        1e-30,
                    )
                )
                anchor_vectors[variant_name] = predicted_anchor.tolist()
                if variant_name == "factorized":
                    for correction_scale in (0.1, 0.2, 0.27, 0.35, 0.5):
                        calibrated_anchor = (
                            sample_direction_sample_anchor
                            + correction_scale
                            * (
                                predicted_anchor
                                - sample_direction_sample_anchor
                            )
                        )
                        config_name = (
                            "sampledir_c21_rank4_factorized_delta"
                            f"{correction_scale:g}"
                        )
                        configs[config_name] = pointwise_features(
                            captured,
                            sample_left,
                            sample_right,
                            calibrated_anchor,
                            radius,
                        )
                    for se_cap in (0.01, 0.02, 0.05, 0.1):
                        correction = np.clip(
                            0.27
                            * (
                                predicted_anchor
                                - sample_direction_sample_anchor
                            ),
                            -se_cap * sample_block_se,
                            se_cap * sample_block_se,
                        )
                        config_name = (
                            "sampledir_c21_rank4_factorized_delta0.27"
                            f"_secap{se_cap:g}"
                        )
                        configs[config_name] = pointwise_features(
                            captured,
                            sample_left,
                            sample_right,
                            sample_direction_sample_anchor + correction,
                            radius,
                        )

        if args.pilot_rotation_seed is not None:
            pilot_rotation = random_rotation(
                WIDTH,
                args.pilot_rotation_seed,
            )
            pilot_activation = forward_to_layer(
                weights,
                points,
                pilot_rotation,
                args.layer,
            )
            pilot_raw_features = (
                (np.square(pilot_activation.astype(np.float64)) @ sample_left)
                * (pilot_activation.astype(np.float64) @ sample_right)
                / radius**2
            ).reshape(N_BASES, ROWS_PER_BASIS, 4)
            pilot_block_means = np.mean(pilot_raw_features, axis=1)
            # Do not privilege the axes basis or early Kerdock labels in a
            # nested budget sweep.
            basis_order = np.random.default_rng(
                10_000 + args.pilot_rotation_seed
            ).permutation(N_BASES)
            for pilot_bases in args.pilot_bases:
                if not 1 <= pilot_bases <= N_BASES:
                    raise ValueError(pilot_bases)
                pilot_anchor = np.mean(
                    pilot_block_means[basis_order[:pilot_bases]],
                    axis=0,
                )
                config_name = (
                    f"sampledir_c21_rank4_pilot{pilot_bases}bases"
                )
                configs[config_name] = pointwise_features(
                    captured,
                    sample_left,
                    sample_right,
                    pilot_anchor,
                    radius,
                )
                anchor_vectors[f"pilot_{pilot_bases}_bases"] = (
                    pilot_anchor.tolist()
                )
        method_mses = {}
        diagnostics = {}
        for config_name, features in configs.items():
            for folds in args.folds:
                predictions, fit_diagnostics = crossfit_grid(
                    features,
                    final,
                    folds,
                    args.ridges,
                )
                diagnostics[f"{config_name}:folds={folds}"] = fit_diagnostics
                for ridge, prediction in predictions.items():
                    label = f"{config_name}:folds={folds}:ridge={ridge:g}"
                    method_mses[label] = float(
                        np.mean(np.square(prediction - targets[-1]))
                    )

        best_label = min(method_mses, key=method_mses.get)
        record = {
            "index": index,
            "name": name,
            "baseline_mse": baseline_mse,
            "method_mses": method_mses,
            "diagnostics": diagnostics,
            "anchor_diagnostics": diagnostics_anchor,
            "anchor_vectors": anchor_vectors,
            "seconds": time.perf_counter() - started,
        }
        records.append(record)
        print(
            f"[{index:>4}] {name[:20]:<20} base={baseline_mse:.4e} "
            f"best={best_label} "
            f"{method_mses[best_label] / baseline_mse:.4f}x "
            f"({record['seconds']:.1f}s)",
            flush=True,
        )

    labels = list(records[0]["method_mses"])
    summary = paired_summary(records, labels)
    print("\nOracle-state, target-free coefficient test", flush=True)
    for label in sorted(summary, key=lambda key: summary[key]["ratio"]):
        item = summary[label]
        print(
            f"{label:<42} ratio={item['ratio']:.5f} "
            f"CI=[{item['ci95'][0]:.5f},{item['ci95'][1]:.5f}] "
            f"wins={item['wins']}/{len(records)} worst={item['worst']:.2f}x",
            flush=True,
        )

    output = {
        "protocol": {
            "indices": args.indices,
            "layer": args.layer,
            "rotation_seed": args.rotation_seed,
            "folds": args.folds,
            "ridges": args.ridges,
            "gaussian_anchor_points": args.gaussian_anchor_points,
            "factorized_k3_dir": (
                str(args.factorized_k3_dir)
                if args.factorized_k3_dir is not None
                else None
            ),
            "pilot_rotation_seed": args.pilot_rotation_seed,
            "pilot_bases": args.pilot_bases,
            "target_leakage": False,
            "oracle_state_warning": (
                "Directions and anchors use 100M-sample higher moments. "
                "Coefficient estimation uses only pointwise sampled outputs."
            ),
        },
        "summary": summary,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
