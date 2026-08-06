"""Oracle ceiling for an exactly anchored pointwise residual surrogate.

For each fixed network, independent rotations of the complete Kerdock design
provide repeated quadrature errors.  A valid pointwise surrogate ``g_W`` has
network-specific but rotation-independent coefficients.  We therefore:

1. compute the true Kerdock error and exactly centered first-layer feature
   means under several rotations;
2. fit one coefficient map per network on training rotations only;
3. test whether it predicts/cancels error on held-out rotations.

The fit is deliberately oracle: it sees high-precision target means for the
training rotations.  It is not deployable.  Its purpose is to answer whether
the proposed shallow feature family has enough stable error geometry to merit
building a transported-state model that predicts its coefficients.
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

from eval_exact_anchor_residual import (  # noqa: E402
    FULL_DATA,
    FeatureConfig,
    forward_cloud,
    gaussian_relu_moments,
    mean_field_jacobian,
    projection_matrix,
)
from eval_kerdock_design import WIDTH, make_kerdock_design, random_rotation  # noqa: E402
from eval_sampling_official import _load_rows  # noqa: E402
from exact_moments import sphere_radius_mean  # noqa: E402


CONFIGS = (
    FeatureConfig("had4_linear_pair", 4, "pair"),
    FeatureConfig("had8_linear_pair", 8, "pair"),
    FeatureConfig("had16_linear_diag", 16, "diag"),
    FeatureConfig("had32_linear_diag", 32, "diag"),
    FeatureConfig("linear_full", WIDTH, "none"),
)


def feature_quadrature_mean(
    h1: np.ndarray,
    mean: np.ndarray,
    second: np.ndarray,
    config: FeatureConfig,
    radius: float,
) -> np.ndarray:
    """Mean of the exactly centered features without materialising N x p."""
    if config.rank == WIDTH and config.quadratic == "none":
        return h1.mean(axis=0, dtype=np.float64) - mean
    u = projection_matrix(config.rank)
    raw_z = h1.astype(np.float64) @ u
    linear = raw_z.mean(axis=0) - mean @ u
    if config.quadratic == "none":
        return linear
    sample_second = (raw_z.T @ raw_z) / (len(raw_z) * radius)
    anchor = (radius / WIDTH) * (u.T @ second @ u)
    difference = sample_second - anchor
    if config.quadratic == "diag":
        quadratic = np.diag(difference)
    elif config.quadratic == "pair":
        ii, jj = np.triu_indices(config.rank)
        quadratic = difference[ii, jj]
    else:
        raise ValueError(config.quadratic)
    return np.concatenate((linear, quadratic))


def oracle_fit(
    q_train: np.ndarray,
    error_train: np.ndarray,
    q_test: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit rotation-independent output coefficients using training rotations."""
    scale = np.sqrt(np.mean(np.square(q_train), axis=0))
    keep = scale > 1e-12
    if not np.any(keep):
        return np.zeros((len(q_test), error_train.shape[1]), dtype=np.float64), {
            "kept": 0.0,
            "condition": 1.0,
        }
    train = q_train[:, keep] / scale[keep]
    test = q_test[:, keep] / scale[keep]
    # The number of rotations is much smaller than the feature count.  Solve in
    # the rotation-space dual; ridge is relative to the mean row norm.
    row_scale = float(np.mean(np.sum(np.square(train), axis=1)))
    if not np.isfinite(row_scale) or row_scale <= 0.0:
        return np.zeros((len(q_test), error_train.shape[1]), dtype=np.float64), {
            "kept": float(np.sum(keep)),
            "condition": 1.0,
        }
    system = train @ train.T + ridge * row_scale * np.eye(len(train))
    dual = np.linalg.solve(system, error_train)
    coef_prediction = test @ train.T @ dual
    return coef_prediction, {
        "kept": float(np.sum(keep)),
        "condition": float(np.linalg.cond(system)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(32, 40)))
    parser.add_argument("--train-rotations", type=int, nargs="+", default=list(range(12)))
    parser.add_argument(
        "--test-rotations",
        type=int,
        nargs="+",
        default=[12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    )
    parser.add_argument("--ridges", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "results" / "oracle_pointwise_bridge_full8.json",
    )
    args = parser.parse_args()
    if set(args.train_rotations) & set(args.test_rotations):
        raise ValueError("rotation split overlaps")

    points = make_kerdock_design()
    radius = sphere_radius_mean(WIDTH)
    rotations = {
        seed: random_rotation(WIDTH, seed)
        for seed in args.train_rotations + args.test_rotations
    }
    rows = _load_rows(FULL_DATA, args.indices)
    records = []

    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        start = time.perf_counter()
        mean, second = gaussian_relu_moments(weights[0])
        jacobian = mean_field_jacobian(weights, mean, second)
        target = targets[-1]
        errors = {}
        q_by_config = {config.name: {} for config in CONFIGS}
        mean_field_errors = {}

        for seed, rotation in rotations.items():
            h1, final = forward_cloud(weights, points, rotation)
            baseline_prediction = final.mean(axis=0, dtype=np.float64)
            error = baseline_prediction - target
            errors[seed] = error
            linear_q = h1.mean(axis=0, dtype=np.float64) - mean
            mean_field_errors[seed] = error - linear_q @ jacobian
            for config in CONFIGS:
                feature_mean = feature_quadrature_mean(
                    h1, mean, second, config, radius
                )
                q_by_config[config.name][seed] = feature_mean

        train_seeds = args.train_rotations
        test_seeds = args.test_rotations
        error_train = np.stack([errors[s] for s in train_seeds])
        error_test = np.stack([errors[s] for s in test_seeds])
        baseline_mse = float(np.mean(np.square(error_test)))
        method_mses = {
            "mean_field_no_fit": float(
                np.mean(np.square(np.stack([mean_field_errors[s] for s in test_seeds])))
            )
        }
        diagnostics = {}

        # Oracle scalar calibration of the analytic mean-field direction on
        # training rotations, then frozen evaluation on test rotations.
        delta_train = np.stack(
            [mean_field_errors[s] - errors[s] for s in train_seeds]
        )
        scalar = -float(np.sum(error_train * delta_train)) / max(
            float(np.sum(np.square(delta_train))), 1e-30
        )
        delta_test = np.stack(
            [mean_field_errors[s] - errors[s] for s in test_seeds]
        )
        method_mses["mean_field_oracle_scalar"] = float(
            np.mean(np.square(error_test + scalar * delta_test))
        )
        diagnostics["mean_field_oracle_scalar"] = scalar

        for config in CONFIGS:
            q_train = np.stack([q_by_config[config.name][s] for s in train_seeds])
            q_test = np.stack([q_by_config[config.name][s] for s in test_seeds])
            for ridge in args.ridges:
                prediction, diag = oracle_fit(
                    q_train, error_train, q_test, ridge
                )
                label = f"{config.name}:ridge={ridge:g}"
                method_mses[label] = float(
                    np.mean(np.square(error_test - prediction))
                )
                diagnostics[label] = diag

        best_label = min(method_mses, key=method_mses.get)
        record = {
            "index": index,
            "name": name,
            "baseline_mse": baseline_mse,
            "method_mses": method_mses,
            "diagnostics": diagnostics,
            "seconds": time.perf_counter() - start,
        }
        records.append(record)
        print(
            f"[{index:>4}] {name[:20]:<20} base={baseline_mse:.4e} "
            f"best={best_label} {method_mses[best_label] / baseline_mse:.4f}x "
            f"({record['seconds']:.1f}s)",
            flush=True,
        )

    labels = list(records[0]["method_mses"])
    baseline = np.asarray([r["baseline_mse"] for r in records])
    rng = np.random.default_rng(20260729)
    boot = rng.integers(0, len(records), size=(20000, len(records)))
    summary = {}
    print("\nHeld-rotation oracle ceiling", flush=True)
    for label in labels:
        values = np.asarray([r["method_mses"][label] for r in records])
        ratios = values[boot].mean(axis=1) / baseline[boot].mean(axis=1)
        summary[label] = {
            "ratio": float(values.mean() / baseline.mean()),
            "ci95": [float(v) for v in np.percentile(ratios, [2.5, 97.5])],
            "wins": int(np.sum(values < baseline)),
            "worst": float(np.max(values / baseline)),
        }
        item = summary[label]
        print(
            f"{label:<40} ratio={item['ratio']:.5f} "
            f"CI=[{item['ci95'][0]:.5f},{item['ci95'][1]:.5f}] "
            f"wins={item['wins']}/{len(records)} worst={item['worst']:.2f}x",
            flush=True,
        )

    output = {
        "protocol": {
            "indices": args.indices,
            "train_rotations": args.train_rotations,
            "test_rotations": args.test_rotations,
            "ridges": args.ridges,
            "oracle_warning": (
                "Coefficients use high-precision targets on training rotations. "
                "This is an expressivity ceiling only, not a deployable estimator."
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
