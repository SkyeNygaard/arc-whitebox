"""Decompose and estimate the arbitrary-center connected-cubic anchor.

For center ``m``, target mean ``mu`` and raw second moment ``M11``,

    A(m) = C21 / (d + 1) + R(m; mu, M11) / (d + 1),

where

    R_ij = -2 (m_i-mu_i) M11_ij
           - M2_i (m_j-mu_j)
           + 2 (m_i^2-mu_i^2) mu_j.

Thus the lower-order correction requires only K1/K2, not raw M21.  This
selection-only experiment tests factorized post K1/K2 and radially corrected
sample K1/K2, including an outer held-basis version so that setting the sample
center equal to the same sample K1 estimate cannot silently become an
algebraic no-op.
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

from eval_connected_cubic_control import (  # noqa: E402
    contract,
    contracted_pointwise,
    exact_anchor_matrix,
    sample_direction_families,
)
from eval_crossfit_cumulant_control import forward_layer_and_final  # noqa: E402
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


def center_correction_matrix(
    center: np.ndarray,
    mean: np.ndarray,
    second: np.ndarray,
    marginal_second: np.ndarray | None = None,
) -> np.ndarray:
    """Return the lower-order part of the arbitrary-center anchor."""
    center = np.asarray(center, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    marginal = (
        np.diag(second)
        if marginal_second is None
        else np.asarray(marginal_second, dtype=np.float64)
    )
    delta = center - mean
    return (
        -2.0 * delta[:, None] * second
        - marginal[:, None] * delta[None, :]
        + 2.0
        * (np.square(center) - np.square(mean))[:, None]
        * mean[None, :]
    ) / (WIDTH + 1.0)


def radial_sample_k1_k2(
    activation: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate Gaussian K1/raw-K2 from fixed-radius sample activations."""
    activation = np.asarray(activation, dtype=np.float64)
    mean = np.mean(activation, axis=0, dtype=np.float64)
    fixed_second = activation.T @ activation / len(activation)
    gaussian_second = WIDTH * fixed_second / np.square(radius)
    return mean, gaussian_second


def fold_ids(folds: int) -> np.ndarray:
    blocks = np.repeat(np.arange(N_BASES), ROWS_PER_BASIS)
    return blocks % folds


def crossfit_fold_features(
    features_by_fold: np.ndarray,
    outputs: np.ndarray,
    folds: int,
    ridge: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Cross-fit when the feature/anchor estimate differs by held fold."""
    features_by_fold = np.asarray(features_by_fold, dtype=np.float64)
    if features_by_fold.ndim != 3:
        raise ValueError(features_by_fold.shape)
    if features_by_fold.shape[:2] != (folds, len(outputs)):
        raise ValueError((features_by_fold.shape, outputs.shape, folds))
    ids = fold_ids(folds)
    estimates = []
    sizes = []
    kept_counts = []
    conditions = []
    feature_rms = []
    for fold in range(folds):
        test = ids == fold
        features = features_by_fold[fold]
        # Scaling uses no output information and matches the transductive
        # convention already used by crossfit_grid.
        scale = np.sqrt(np.mean(np.square(features), axis=0))
        keep = scale > 1e-12
        kept_counts.append(int(np.sum(keep)))
        feature_rms.append(float(np.sqrt(np.mean(np.square(features)))))
        if not np.any(keep):
            estimates.append(np.mean(outputs[test], axis=0, dtype=np.float64))
            sizes.append(int(np.sum(test)))
            conditions.append(1.0)
            continue
        x = features[:, keep] / scale[keep]
        x_train = x[~test]
        y_train = outputs[~test].astype(np.float64, copy=False)
        system = (
            x_train.T @ x_train
            + ridge * len(x_train) * np.eye(x_train.shape[1])
        )
        coefficient = np.linalg.solve(system, x_train.T @ y_train)
        estimates.append(
            np.mean(outputs[test], axis=0, dtype=np.float64)
            - np.mean(x[test], axis=0) @ coefficient
        )
        sizes.append(int(np.sum(test)))
        conditions.append(float(np.linalg.cond(system)))
    return np.average(estimates, axis=0, weights=sizes), {
        "features": float(features_by_fold.shape[2]),
        "kept_min": float(min(kept_counts)),
        "kept_max": float(max(kept_counts)),
        "feature_rms_mean": float(np.mean(feature_rms)),
        "condition_max": float(max(conditions)),
    }


def repeated_fold_features(features: np.ndarray, folds: int) -> np.ndarray:
    return np.repeat(np.asarray(features)[None, :, :], folds, axis=0)


def anchor_metric(
    predicted: np.ndarray,
    exact: np.ndarray,
    quadrature_discrepancy: np.ndarray,
) -> dict[str, float]:
    """Support either one anchor or a different anchor for every fold."""
    predicted = np.asarray(predicted, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    discrepancy = np.asarray(quadrature_discrepancy, dtype=np.float64)
    while exact.ndim < predicted.ndim:
        exact = np.expand_dims(exact, axis=-2)
        discrepancy = np.expand_dims(discrepancy, axis=-2)
    exact = np.broadcast_to(exact, predicted.shape)
    discrepancy = np.broadcast_to(discrepancy, predicted.shape)
    error = predicted - exact
    return {
        "relative_to_q_minus_e": float(
            np.linalg.norm(error) / max(np.linalg.norm(discrepancy), 1e-30)
        ),
        "relative_to_exact": float(
            np.linalg.norm(error) / max(np.linalg.norm(exact), 1e-30)
        ),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_absolute_error": float(np.mean(np.abs(error))),
    }


def paired_summary(
    records: list[dict],
    labels: list[str],
) -> dict[str, dict]:
    baseline = np.asarray(
        [record["baseline_mse"] for record in records],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20260729)
    boot = rng.integers(0, len(records), size=(20_000, len(records)))
    result = {}
    for label in labels:
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
    parser.add_argument("--direction-rank", type=int, default=2)
    parser.add_argument("--probe-rank", type=int, default=32)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[1.7, 1.8, 1.9, 1.95, 2.0, 2.1, 2.2],
    )
    parser.add_argument(
        "--factorized-correction-scales",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.15, 0.2],
        help="Frozen shrinkages of the biased factorized K1/K2 correction.",
    )
    parser.add_argument(
        "--factorized-dir",
        type=Path,
        default=HERE / "results" / "factorized_k3_layer29",
    )
    parser.add_argument(
        "--dual-results",
        type=Path,
        default=(
            HERE
            / "results"
            / "dual_connected_control_rank2_selection8.json"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            HERE
            / "results"
            / "arbitrary_center_correction_rank2_selection8.json"
        ),
    )
    args = parser.parse_args()

    with args.dual_results.open() as handle:
        dual_results = json.load(handle)
    dual_by_index = {
        int(record["index"]): record
        for record in dual_results["records"]
    }
    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    radius = sphere_radius_mean(WIDTH)
    rows = _load_rows(FULL_DATA, args.indices)
    ids = fold_ids(args.folds)
    records = []

    for index, (name, weights, targets) in zip(
        args.indices,
        rows,
        strict=True,
    ):
        started = time.perf_counter()
        with np.load(moment_path(index)) as moments:
            true_mean = np.asarray(
                moments["mean"][args.layer],
                dtype=np.float64,
            )
            true_second = np.asarray(
                moments["M11"][args.layer],
                dtype=np.float64,
            )
            true_raw_m21 = np.asarray(
                moments["M21"][args.layer],
                dtype=np.float64,
            )
            true_marginal_second = np.asarray(
                moments["m2"][args.layer],
                dtype=np.float64,
            )
        true_c21 = connected_m21(
            true_mean,
            true_second,
            true_raw_m21,
            true_marginal_second,
        )
        _, captured, final = forward_layer_and_final(
            weights,
            points,
            rotation,
            args.layer,
        )
        sample_mean, sample_second = radial_sample_k1_k2(
            captured,
            radius,
        )
        left, right = sample_direction_families(
            captured,
            args.direction_rank,
            radius,
        )["radial_corrected_dirs"]
        values_sample_center = contracted_pointwise(
            captured,
            left,
            right,
            sample_mean,
            radius,
        )
        values_true_center = contracted_pointwise(
            captured,
            left,
            right,
            true_mean,
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
        true_connected_anchor = (
            contract(left, true_c21, right) / (WIDTH + 1.0)
        )
        true_correction = contract(
            left,
            center_correction_matrix(
                sample_mean,
                true_mean,
                true_second,
                true_marginal_second,
            ),
            right,
        )
        if not np.allclose(
            exact_anchor,
            true_connected_anchor + true_correction,
            rtol=2e-8,
            atol=2e-11,
        ):
            raise AssertionError(
                (
                    index,
                    exact_anchor,
                    true_connected_anchor + true_correction,
                )
            )

        with np.load(
            args.factorized_dir / f"mlp_{index:05d}.npz"
        ) as factorized:
            factorized_mean = np.asarray(
                factorized["mean"],
                dtype=np.float64,
            )
            factorized_covariance = np.asarray(
                factorized["covariance"],
                dtype=np.float64,
            )
        factorized_second = (
            factorized_covariance
            + np.outer(factorized_mean, factorized_mean)
        )
        factorized_correction = contract(
            left,
            center_correction_matrix(
                sample_mean,
                factorized_mean,
                factorized_second,
            ),
            right,
        )
        values_factorized_center = contracted_pointwise(
            captured,
            left,
            right,
            factorized_mean,
            radius,
        )

        # With global sample K1 equal to the pointwise center this is exactly
        # zero; retain it as an explicit diagnostic rather than silently
        # calling the same-cloud construction a successful estimator.
        sample_global_correction = contract(
            left,
            center_correction_matrix(
                sample_mean,
                sample_mean,
                sample_second,
            ),
            right,
        )
        if not np.allclose(sample_global_correction, 0.0, atol=1e-14):
            raise AssertionError(sample_global_correction)

        sample_fold_corrections = []
        values_sample_train_centers = []
        for fold in range(args.folds):
            train = ids != fold
            train_mean, train_second = radial_sample_k1_k2(
                captured[train],
                radius,
            )
            sample_fold_corrections.append(
                contract(
                    left,
                    center_correction_matrix(
                        sample_mean,
                        train_mean,
                        train_second,
                    ),
                    right,
                )
            )
            values_sample_train_centers.append(
                contracted_pointwise(
                    captured,
                    left,
                    right,
                    train_mean,
                    radius,
                )
            )
        sample_fold_corrections = np.asarray(
            sample_fold_corrections,
            dtype=np.float64,
        )
        values_sample_train_centers = np.asarray(
            values_sample_train_centers,
            dtype=np.float64,
        )

        q_record = dual_by_index[index]
        stored_exact = np.asarray(
            q_record["exact_arbitrary_anchor"],
            dtype=np.float64,
        )
        if not np.allclose(
            exact_anchor,
            stored_exact,
            rtol=2e-8,
            atol=2e-11,
        ):
            raise AssertionError((index, exact_anchor, stored_exact))
        cheap_anchor = (
            np.asarray(q_record["cheap_q"], dtype=np.float64)
            / (WIDTH + 1.0)
        )
        probe_anchor = (
            np.asarray(
                q_record[f"cheap_probe{args.probe_rank}_q"],
                dtype=np.float64,
            )
            / (WIDTH + 1.0)
        )

        global_features = {
            "oracle_full_exact": (
                values_sample_center - exact_anchor
            ),
            "exact_k3_plus_factorized_k12_correction": (
                values_sample_center
                - true_connected_anchor
                - factorized_correction
            ),
            "exact_k3_plus_sample_global_k12_correction_noop": (
                values_sample_center
                - true_connected_anchor
                - sample_global_correction
            ),
            "oracle_k3_only_true_center": (
                values_true_center - true_connected_anchor
            ),
            "oracle_correction_only": (
                values_sample_center
                - values_true_center
                - true_correction
            ),
            "oracle_components_joint": np.concatenate(
                (
                    values_true_center - true_connected_anchor,
                    values_sample_center
                    - values_true_center
                    - true_correction,
                ),
                axis=1,
            ),
            "factorized_k12_correction_only": (
                values_sample_center
                - values_factorized_center
                - factorized_correction
            ),
        }
        global_anchors = {
            "oracle_full_exact": exact_anchor,
            "exact_k3_plus_factorized_k12_correction": (
                true_connected_anchor + factorized_correction
            ),
            "exact_k3_plus_sample_global_k12_correction_noop": (
                true_connected_anchor + sample_global_correction
            ),
        }
        for correction_scale in args.factorized_correction_scales:
            label = (
                "exact_k3_plus_factorized_k12_correction_"
                f"scale{correction_scale:g}"
            )
            anchor = (
                true_connected_anchor
                + correction_scale * factorized_correction
            )
            global_features[label] = values_sample_center - anchor
            global_anchors[label] = anchor
        for scale in args.scales:
            for dual_label, dual_anchor in (
                ("cheap", cheap_anchor),
                (f"probe{args.probe_rank}", probe_anchor),
            ):
                for correction_label, correction in (
                    ("factorized", factorized_correction),
                    ("oracle", true_correction),
                ):
                    label = (
                        f"{dual_label}_scale{scale:g}_plus_"
                        f"{correction_label}_correction"
                    )
                    anchor = scale * dual_anchor + correction
                    global_features[label] = values_sample_center - anchor
                    global_anchors[label] = anchor
                for correction_scale in args.factorized_correction_scales:
                    label = (
                        f"{dual_label}_scale{scale:g}_plus_"
                        "factorized_correction_"
                        f"scale{correction_scale:g}"
                    )
                    anchor = (
                        scale * dual_anchor
                        + correction_scale * factorized_correction
                    )
                    global_features[label] = values_sample_center - anchor
                    global_anchors[label] = anchor

        fold_features = {
            "exact_k3_plus_sample_fold_k12_correction": (
                values_sample_center[None, :, :]
                - true_connected_anchor[None, None, :]
                - sample_fold_corrections[:, None, :]
            ),
            "sample_fold_k12_correction_only": (
                values_sample_center[None, :, :]
                - values_sample_train_centers
                - sample_fold_corrections[:, None, :]
            ),
        }
        fold_anchors = {
            "exact_k3_plus_sample_fold_k12_correction": (
                true_connected_anchor[None, :]
                + sample_fold_corrections
            ),
        }
        for scale in args.scales:
            for dual_label, dual_anchor in (
                ("cheap", cheap_anchor),
                (f"probe{args.probe_rank}", probe_anchor),
            ):
                label = (
                    f"{dual_label}_scale{scale:g}_plus_"
                    "sample_fold_correction"
                )
                fold_features[label] = (
                    values_sample_center[None, :, :]
                    - scale * dual_anchor[None, None, :]
                    - sample_fold_corrections[:, None, :]
                )
                fold_anchors[label] = (
                    scale * dual_anchor[None, :]
                    + sample_fold_corrections
                )

        target = targets[-1]
        baseline_prediction = np.mean(final, axis=0, dtype=np.float64)
        baseline_mse = float(
            np.mean(np.square(baseline_prediction - target))
        )
        method_mses = {}
        fit_diagnostics = {}
        for label, features in global_features.items():
            prediction, fit = crossfit_fold_features(
                repeated_fold_features(features, args.folds),
                final,
                args.folds,
                args.ridge,
            )
            method_mses[label] = float(
                np.mean(np.square(prediction - target))
            )
            fit_diagnostics[label] = fit
        for label, features in fold_features.items():
            prediction, fit = crossfit_fold_features(
                features,
                final,
                args.folds,
                args.ridge,
            )
            method_mses[label] = float(
                np.mean(np.square(prediction - target))
            )
            fit_diagnostics[label] = fit

        quadrature_discrepancy = (
            np.mean(values_sample_center, axis=0) - exact_anchor
        )
        anchor_diagnostics = {
            label: anchor_metric(
                anchor,
                exact_anchor,
                quadrature_discrepancy,
            )
            for label, anchor in global_anchors.items()
        }
        anchor_diagnostics.update(
            {
                label: anchor_metric(
                    anchor,
                    exact_anchor,
                    quadrature_discrepancy,
                )
                for label, anchor in fold_anchors.items()
            }
        )
        correction_diagnostics = {
            "factorized": anchor_metric(
                factorized_correction,
                true_correction,
                quadrature_discrepancy,
            ),
            "sample_global_noop": anchor_metric(
                sample_global_correction,
                true_correction,
                quadrature_discrepancy,
            ),
            "sample_fold": anchor_metric(
                sample_fold_corrections,
                true_correction,
                quadrature_discrepancy,
            ),
        }
        record = {
            "index": index,
            "name": name,
            "baseline_mse": baseline_mse,
            "method_mses": method_mses,
            "fit_diagnostics": fit_diagnostics,
            "anchor_diagnostics": anchor_diagnostics,
            "correction_diagnostics": correction_diagnostics,
            "exact_anchor": exact_anchor.tolist(),
            "true_connected_anchor": true_connected_anchor.tolist(),
            "true_correction": true_correction.tolist(),
            "factorized_correction": factorized_correction.tolist(),
            "sample_global_correction": sample_global_correction.tolist(),
            "sample_fold_corrections": sample_fold_corrections.tolist(),
            "cheap_connected_anchor_unscaled": cheap_anchor.tolist(),
            f"probe{args.probe_rank}_connected_anchor_unscaled": (
                probe_anchor.tolist()
            ),
            "quadrature_discrepancy": quadrature_discrepancy.tolist(),
            "seconds": time.perf_counter() - started,
        }
        records.append(record)
        print(
            f"[{index}] base={baseline_mse:.4e} "
            f"exact={method_mses['oracle_full_exact'] / baseline_mse:.3f}x "
            f"exact+factorcorr="
            f"{method_mses['exact_k3_plus_factorized_k12_correction'] / baseline_mse:.3f}x "
            f"exact+samplefold="
            f"{method_mses['exact_k3_plus_sample_fold_k12_correction'] / baseline_mse:.3f}x "
            f"corronly="
            f"{method_mses['oracle_correction_only'] / baseline_mse:.3f}x "
            f"({record['seconds']:.1f}s)",
            flush=True,
        )

    labels = list(records[0]["method_mses"])
    summary = paired_summary(records, labels)
    pooled_exact = np.asarray(
        [record["exact_anchor"] for record in records],
        dtype=np.float64,
    )
    pooled_connected = np.asarray(
        [record["true_connected_anchor"] for record in records],
        dtype=np.float64,
    )
    pooled_true_correction = np.asarray(
        [record["true_correction"] for record in records],
        dtype=np.float64,
    )
    pooled_factorized_correction = np.asarray(
        [record["factorized_correction"] for record in records],
        dtype=np.float64,
    )
    pooled_sample_global_correction = np.asarray(
        [record["sample_global_correction"] for record in records],
        dtype=np.float64,
    )
    pooled_sample_fold_correction = np.asarray(
        [record["sample_fold_corrections"] for record in records],
        dtype=np.float64,
    )
    pooled_cheap = np.asarray(
        [
            record["cheap_connected_anchor_unscaled"]
            for record in records
        ],
        dtype=np.float64,
    )
    pooled_probe = np.asarray(
        [
            record[
                f"probe{args.probe_rank}_connected_anchor_unscaled"
            ]
            for record in records
        ],
        dtype=np.float64,
    )
    pooled_discrepancy = np.asarray(
        [record["quadrature_discrepancy"] for record in records],
        dtype=np.float64,
    )
    pooled_anchor_diagnostics = {
        "exact_k3_plus_factorized_k12_correction": anchor_metric(
            pooled_connected + pooled_factorized_correction,
            pooled_exact,
            pooled_discrepancy,
        ),
        "exact_k3_plus_sample_global_k12_correction_noop": anchor_metric(
            pooled_connected + pooled_sample_global_correction,
            pooled_exact,
            pooled_discrepancy,
        ),
        "exact_k3_plus_sample_fold_k12_correction": anchor_metric(
            pooled_connected[:, None, :]
            + pooled_sample_fold_correction,
            pooled_exact,
            pooled_discrepancy,
        ),
    }
    for correction_scale in args.factorized_correction_scales:
        label = (
            "exact_k3_plus_factorized_k12_correction_"
            f"scale{correction_scale:g}"
        )
        pooled_anchor_diagnostics[label] = anchor_metric(
            pooled_connected
            + correction_scale * pooled_factorized_correction,
            pooled_exact,
            pooled_discrepancy,
        )
    pooled_correction_diagnostics = {
        "factorized": anchor_metric(
            pooled_factorized_correction,
            pooled_true_correction,
            pooled_discrepancy,
        ),
        "sample_global_noop": anchor_metric(
            pooled_sample_global_correction,
            pooled_true_correction,
            pooled_discrepancy,
        ),
        "sample_fold": anchor_metric(
            pooled_sample_fold_correction,
            pooled_true_correction,
            pooled_discrepancy,
        ),
    }
    for scale in args.scales:
        for dual_label, dual_anchor in (
            ("cheap", pooled_cheap),
            (f"probe{args.probe_rank}", pooled_probe),
        ):
            for correction_label, correction in (
                ("factorized", pooled_factorized_correction),
                ("oracle", pooled_true_correction),
            ):
                label = (
                    f"{dual_label}_scale{scale:g}_plus_"
                    f"{correction_label}_correction"
                )
                pooled_anchor_diagnostics[label] = anchor_metric(
                    scale * dual_anchor + correction,
                    pooled_exact,
                    pooled_discrepancy,
                )
            for correction_scale in args.factorized_correction_scales:
                label = (
                    f"{dual_label}_scale{scale:g}_plus_"
                    "factorized_correction_"
                    f"scale{correction_scale:g}"
                )
                pooled_anchor_diagnostics[label] = anchor_metric(
                    scale * dual_anchor
                    + correction_scale * pooled_factorized_correction,
                    pooled_exact,
                    pooled_discrepancy,
                )
            label = (
                f"{dual_label}_scale{scale:g}_plus_"
                "sample_fold_correction"
            )
            pooled_anchor_diagnostics[label] = anchor_metric(
                scale * dual_anchor[:, None, :]
                + pooled_sample_fold_correction,
                pooled_exact,
                pooled_discrepancy,
            )
    output = {
        "protocol": {
            "indices": args.indices,
            "layer": args.layer,
            "rotation_seed": args.rotation_seed,
            "direction_rank": args.direction_rank,
            "directions": "sample radial-corrected connected-c21 SVD",
            "pointwise_center": "global sample activation mean",
            "folds": args.folds,
            "ridge": args.ridge,
            "scales": args.scales,
            "factorized_correction_scales": (
                args.factorized_correction_scales
            ),
            "probe_rank": args.probe_rank,
            "factorized_dir": str(args.factorized_dir),
            "dual_results": str(args.dual_results),
            "raw_m21_reconstructed_by_deployable_methods": False,
            "target_leakage_in_coefficients": False,
            "scope": "selection IDs only; no new holdout",
        },
        "summary": summary,
        "pooled_anchor_diagnostics": pooled_anchor_diagnostics,
        "pooled_correction_diagnostics": pooled_correction_diagnostics,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
