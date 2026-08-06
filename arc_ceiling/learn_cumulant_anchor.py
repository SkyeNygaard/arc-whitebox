"""Learn the four missing late-cumulant anchor corrections from basis traces.

For each network, the main Kerdock cloud cheaply supplies:

* rank-4 directions from the sample connected-M21 matrix;
* 129 per-basis estimates of each cubic contraction.

The unknown target is only the small difference between the uniform Kerdock
average and the high-precision Gaussian anchor.  This script asks whether that
difference is statistically identifiable from the *shape* of the 129 basis
estimates.  It uses honest network splits and ridge models; a zero prediction
is the deployable "use the sample anchor" baseline.

This is deliberately a narrow learned-cubature test.  It does not use final
network targets or output errors.
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

from eval_exact_anchor_residual import FULL_DATA, ROWS_PER_BASIS  # noqa: E402
from eval_kerdock_design import N_BASES, WIDTH, make_kerdock_design, random_rotation  # noqa: E402
from eval_oracle_cumulant_bridge import connected_m21, moment_path  # noqa: E402
from eval_sampling_official import _load_rows  # noqa: E402
from exact_moments import sphere_radius_mean  # noqa: E402


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
    for current in range(1, layer + 1):
        activation = np.maximum(activation @ weights[current], 0.0)
    return activation


def standardized_moments(values: np.ndarray) -> list[float]:
    centered = values - np.mean(values)
    scale = max(float(np.sqrt(np.mean(np.square(centered)))), 1e-30)
    z = centered / scale
    return [
        float(np.mean(z**3)),
        float(np.mean(z**4) - 3.0),
        float(np.mean(z**5)),
        float(np.mean(z**6) - 15.0),
        float(np.min(z)),
        float(np.max(z)),
        float(np.mean(np.abs(z))),
    ]


def make_rows(
    index: int,
    weights: np.ndarray,
    points: np.ndarray,
    rotation: np.ndarray,
    layer: int,
    radius: float,
) -> tuple[list[np.ndarray], list[float], list[float], list[int]]:
    activation = forward_to_layer(weights, points, rotation, layer)
    h = activation.astype(np.float64, copy=False)
    mean = np.mean(h, axis=0)
    second = (h.T @ h) / len(h)
    raw_m21 = (np.square(h).T @ h) / len(h)
    c21 = connected_m21(mean, second, raw_m21, np.diag(second))
    left, singular_values, right_t = np.linalg.svd(c21, full_matrices=False)
    left = left[:, :4]
    right = right_t.T[:, :4]

    pointwise = (
        (np.square(h) @ left) * (h @ right) / radius**2
    ).reshape(N_BASES, ROWS_PER_BASIS, 4)
    block_means = np.mean(pointwise, axis=1)
    sample_anchor = np.mean(block_means, axis=0)

    with np.load(moment_path(index)) as moment_data:
        oracle_raw_m21 = np.asarray(
            moment_data["M21"][layer],
            dtype=np.float64,
        )
    oracle_anchor = np.einsum(
        "ik,ij,jk->k",
        left,
        oracle_raw_m21,
        right,
    ) / (WIDTH + 1)

    block_covariance = np.cov(block_means, rowvar=False)
    covariance_features = block_covariance[np.triu_indices(4)]
    normalized_singular_values = singular_values[:12] / max(
        float(singular_values[0]),
        1e-30,
    )
    feature_rows = []
    targets = []
    scales = []
    components = []
    quantile_grid = np.linspace(0.0, 1.0, 25)

    for component in range(4):
        values = block_means[:, component]
        block_sd = max(float(np.std(values, ddof=1)), 1e-30)
        standard_error = block_sd / np.sqrt(N_BASES)
        centered = (values - np.mean(values)) / block_sd
        ordered = centered
        sorted_quantiles = np.quantile(centered, quantile_grid)
        spectrum = np.fft.rfft(centered) / np.sqrt(N_BASES)
        spectral_features = np.concatenate(
            (spectrum.real[1:17], spectrum.imag[1:17])
        )
        direction_features = np.asarray(
            [
                np.sum(np.abs(left[:, component])),
                np.max(np.abs(left[:, component])),
                np.sum(np.abs(right[:, component])),
                np.max(np.abs(right[:, component])),
                np.dot(left[:, component], right[:, component]),
                np.mean(np.abs(left[:, component] * right[:, component])),
                sample_anchor[component] / block_sd,
                np.log(max(abs(sample_anchor[component]), 1e-30)),
                np.log(block_sd),
                np.mean(h == 0.0),
            ],
            dtype=np.float64,
        )
        one_hot = np.eye(4, dtype=np.float64)[component]
        invariant = np.concatenate(
            (
                sorted_quantiles,
                standardized_moments(values),
                normalized_singular_values,
                covariance_features
                / max(float(np.trace(block_covariance)), 1e-30),
                direction_features,
                one_hot,
            )
        )
        # Group layout: invariant first, then fixed-label and Fourier traces.
        feature_rows.append(
            np.concatenate((invariant, ordered, spectral_features))
        )
        targets.append(
            float((oracle_anchor[component] - sample_anchor[component]) / standard_error)
        )
        scales.append(standard_error)
        components.append(component)
    return feature_rows, targets, scales, components


def build_dataset(
    indices: list[int],
    layer: int,
    rotation_seed: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, rotation_seed)
    radius = sphere_radius_mean(WIDTH)
    features = []
    targets = []
    scales = []
    network_ids = []
    components = []

    for start in range(0, len(indices), batch_size):
        batch_ids = indices[start : start + batch_size]
        rows = _load_rows(FULL_DATA, batch_ids)
        for index, (_, weights, _) in zip(batch_ids, rows, strict=True):
            began = time.perf_counter()
            x_rows, y_rows, row_scales, row_components = make_rows(
                index,
                weights,
                points,
                rotation,
                layer,
                radius,
            )
            features.extend(x_rows)
            targets.extend(y_rows)
            scales.extend(row_scales)
            network_ids.extend([index] * 4)
            components.extend(row_components)
            print(
                f"[{index:>4}] built 4 anchor rows "
                f"({time.perf_counter() - began:.2f}s)",
                flush=True,
            )

    return {
        "features": np.asarray(features, dtype=np.float64),
        "targets": np.asarray(targets, dtype=np.float64),
        "scales": np.asarray(scales, dtype=np.float64),
        "network_ids": np.asarray(network_ids, dtype=np.int64),
        "components": np.asarray(components, dtype=np.int64),
    }


def ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    ridge: float,
) -> np.ndarray:
    system = train_x.T @ train_x + ridge * len(train_x) * np.eye(train_x.shape[1])
    coefficient = np.linalg.solve(system, train_x.T @ train_y)
    return test_x @ coefficient


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    denominator = max(float(np.mean(np.square(target))), 1e-30)
    return {
        "mse_ratio": float(np.mean(np.square(target - prediction)) / denominator),
        "correlation": float(np.corrcoef(target, prediction)[0, 1])
        if np.std(prediction) > 0
        else 0.0,
        "prediction_rms": float(np.sqrt(np.mean(np.square(prediction)))),
        "target_rms": float(np.sqrt(np.mean(np.square(target)))),
    }


def train_and_test(
    data: dict[str, np.ndarray],
    invariant_width: int,
) -> dict:
    network_ids = data["network_ids"]
    train = network_ids < 120
    valid = (network_ids >= 120) & (network_ids < 160)
    test = network_ids >= 160
    feature_sets = {
        "invariant": data["features"][:, :invariant_width],
        "all": data["features"],
    }
    ridges = np.logspace(-5, 3, 17)
    clips = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    output = {}

    for feature_name, raw_x in feature_sets.items():
        center = np.mean(raw_x[train], axis=0)
        scale = np.std(raw_x[train], axis=0)
        keep = scale > 1e-10
        x = (raw_x[:, keep] - center[keep]) / scale[keep]
        candidates = []
        for ridge in ridges:
            raw_prediction = ridge_predict(
                x[train],
                data["targets"][train],
                x[valid],
                float(ridge),
            )
            for clip in clips:
                prediction = np.clip(raw_prediction, -clip, clip)
                candidates.append(
                    {
                        "ridge": float(ridge),
                        "clip": clip,
                        **metrics(data["targets"][valid], prediction),
                    }
                )
        selected = min(candidates, key=lambda row: row["mse_ratio"])
        combined = train | valid
        final_prediction = ridge_predict(
            x[combined],
            data["targets"][combined],
            x[test],
            selected["ridge"],
        )
        final_prediction = np.clip(
            final_prediction,
            -selected["clip"],
            selected["clip"],
        )
        output[feature_name] = {
            "kept_features": int(np.sum(keep)),
            "selected": selected,
            "test": metrics(data["targets"][test], final_prediction),
            "test_by_component": {
                str(component): metrics(
                    data["targets"][test & (data["components"] == component)],
                    final_prediction[
                        data["components"][test] == component
                    ],
                )
                for component in range(4)
            },
        }
    output["split"] = {
        "train_networks": [0, 119],
        "valid_networks": [120, 159],
        "test_networks": [160, 199],
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(200)))
    parser.add_argument("--layer", type=int, default=29)
    parser.add_argument("--rotation-seed", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--dataset-out",
        type=Path,
        default=HERE / "results" / "cumulant_anchor_learning_dataset.npz",
    )
    parser.add_argument(
        "--results-out",
        type=Path,
        default=HERE / "results" / "cumulant_anchor_learning.json",
    )
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    if args.reuse:
        with np.load(args.dataset_out) as loaded:
            data = {key: np.asarray(loaded[key]) for key in loaded.files}
    else:
        data = build_dataset(
            args.indices,
            args.layer,
            args.rotation_seed,
            args.batch_size,
        )
        args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.dataset_out, **data)

    # 25 quantiles + 7 moments + 12 singular values + 10 covariance entries
    # + 10 direction/global features + 4 component indicators.
    invariant_width = 68
    result = {
        "protocol": {
            "indices": args.indices,
            "layer": args.layer,
            "rotation_seed": args.rotation_seed,
            "target": "(oracle_anchor - Kerdock_anchor) / block_standard_error",
            "no_final_targets": True,
        },
        "rows": int(len(data["targets"])),
        "target_rms": float(np.sqrt(np.mean(np.square(data["targets"])))),
        "models": train_and_test(data, invariant_width),
    }
    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    args.results_out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
