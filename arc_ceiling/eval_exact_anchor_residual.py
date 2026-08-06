"""Exactly anchored first-layer residual controls for complete Kerdock.

The estimator is

    mu_hat = Q_K(f - g) + E[g].

Here ``g`` is built from the *actual pointwise first-layer activations*

    h(x) = ReLU(x @ W1)

and has an analytic Gaussian expectation.  Coefficients are fitted only from
the already-computed pointwise final outputs, with whole Kerdock basis blocks
held out.  This makes the experiment a direct test of the missing bridge:

    tractable state/features -> pointwise g(x) -> residual Kerdock correction.

Linear features use ``h - E[h]``.  Quadratic features are radialized to retain
degree-one homogeneity:

    (u^T h)(v^T h) / ||x|| - E[(u^T h)(v^T h) / ||X||].

For X ~ N(0, I_d), radius and direction are independent, E||X|| = r, and
E||X||^2 = d, so the exact quadratic anchor is

    (r / d) u^T E[h h^T] v.

The Kerdock points all have norm r.  ReLU makes these angular features
non-polynomial despite their shallow algebraic form, so the spherical
5-design does not annihilate their degree-6+ content.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from eval_kerdock_design import (  # noqa: E402
    N_BASES,
    WIDTH,
    make_kerdock_design,
    random_rotation,
    walsh_hadamard,
)
from eval_sampling_official import _load_rows  # noqa: E402
from exact_moments import INV_SQRT_2PI, sphere_radius_mean  # noqa: E402


MINI_DATA = ROOT / "data" / "official_phase1_mini" / "data"
FULL_DATA = ROOT / "data" / "official_phase1_full" / "data"
ROWS_PER_BASIS = 2 * WIDTH


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    rank: int
    quadratic: str


CONFIGS = (
    FeatureConfig("linear_full", WIDTH, "none"),
    FeatureConfig("had16_linear_pair", 16, "pair"),
    FeatureConfig("had32_linear_diag", 32, "diag"),
    FeatureConfig("had64_linear_diag", 64, "diag"),
)


def gaussian_relu_moments(first_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact E[h] and E[h h^T] for h=ReLU(X @ W), X~N(0,I)."""
    weight = np.asarray(first_weight, dtype=np.float64)
    gram = weight.T @ weight
    sigma = np.sqrt(np.maximum(np.diag(gram), 0.0))
    scale = np.outer(sigma, sigma)
    rho = np.divide(gram, scale, out=np.zeros_like(gram), where=scale > 0.0)
    theta = np.arccos(np.clip(rho, -1.0, 1.0))
    second = (
        scale
        * (np.sin(theta) + (math.pi - theta) * np.cos(theta))
        / (2.0 * math.pi)
    )
    return sigma * INV_SQRT_2PI, 0.5 * (second + second.T)


def mean_field_jacobian(
    weights: np.ndarray, first_mean: np.ndarray, first_second: np.ndarray
) -> np.ndarray:
    """Expected-gate linear response from layer-1 activation to final output."""
    mean = first_mean.copy()
    var = np.maximum(np.diag(first_second) - np.square(mean), 1e-20)
    jacobian = np.eye(WIDTH, dtype=np.float64)
    jacobian_scale = 1.0
    inv_sqrt_2pi = INV_SQRT_2PI
    for weight32 in weights[1:]:
        weight = weight32.astype(np.float64)
        pre_mean = mean @ weight
        pre_var = var @ np.square(weight)
        pre_sd = np.sqrt(np.maximum(pre_var, 1e-20))
        t = pre_mean / pre_sd
        gate = ndtr(t)
        phi = np.exp(-0.5 * np.square(t)) * inv_sqrt_2pi
        second = (
            (np.square(pre_mean) + pre_var) * gate
            + pre_mean * pre_sd * phi
        )
        mean = pre_mean * gate + pre_sd * phi
        var = np.maximum(second - np.square(mean), 1e-20)
        jacobian = jacobian @ (weight * gate[None, :])
        norm = np.linalg.norm(jacobian)
        if norm > 0.0:
            # The correction below needs the physical scale.  Rescale the
            # running state and carry the scalar separately to avoid underflow.
            jacobian /= norm
            jacobian_scale *= norm
    return jacobian * jacobian_scale


def projection_matrix(rank: int) -> np.ndarray:
    if rank == WIDTH:
        return np.eye(WIDTH, dtype=np.float64)
    # Fixed, network-independent orthonormal directions.  A deployed version
    # can use an FWHT rather than a dense multiply.
    return (walsh_hadamard().astype(np.float64) / math.sqrt(WIDTH))[:, :rank]


def make_features(
    h1: np.ndarray,
    mean: np.ndarray,
    second: np.ndarray,
    config: FeatureConfig,
    radius: float,
) -> np.ndarray:
    """Create centered features with exact Gaussian expectation zero."""
    u = projection_matrix(config.rank)
    if config.rank == WIDTH and config.quadratic == "none":
        return h1.astype(np.float64) - mean

    raw_z = h1.astype(np.float64) @ u
    linear = raw_z - mean @ u
    if config.quadratic == "none":
        return linear

    projected_anchor = (radius / WIDTH) * (u.T @ second @ u)
    if config.quadratic == "diag":
        quadratic = np.square(raw_z) / radius - np.diag(projected_anchor)
    elif config.quadratic == "pair":
        ii, jj = np.triu_indices(config.rank)
        quadratic = (raw_z[:, ii] * raw_z[:, jj]) / radius - projected_anchor[ii, jj]
    else:
        raise ValueError(config.quadratic)
    return np.concatenate((linear, quadratic), axis=1)


def crossfit_control(
    features: np.ndarray,
    outputs: np.ndarray,
    n_folds: int,
    ridge: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Whole-basis-block cross-fitted residual estimate."""
    n, p = features.shape
    if n != N_BASES * ROWS_PER_BASIS:
        raise ValueError((n, N_BASES, ROWS_PER_BASIS))

    # Scaling uses only x/features, never pointwise outputs or target means.
    scale = np.sqrt(np.mean(np.square(features), axis=0))
    keep = scale > 1e-12
    x = features[:, keep] / scale[keep]
    p_kept = x.shape[1]

    block_ids = np.repeat(np.arange(N_BASES), ROWS_PER_BASIS)
    fold_ids = block_ids % n_folds
    gram_total = x.T @ x
    cross_total = x.T @ outputs.astype(np.float64)

    estimates = []
    test_sizes = []
    condition_numbers = []
    for fold in range(n_folds):
        test = fold_ids == fold
        xt = x[test]
        yt = outputs[test].astype(np.float64)
        gram_train = gram_total - xt.T @ xt
        cross_train = cross_total - xt.T @ yt
        n_train = n - int(test.sum())
        penalty = ridge * n_train
        system = gram_train + penalty * np.eye(p_kept)
        coef = np.linalg.solve(system, cross_train)
        estimates.append(np.mean(yt - xt @ coef, axis=0))
        test_sizes.append(int(test.sum()))
        condition_numbers.append(float(np.linalg.cond(system)))

    return np.average(estimates, axis=0, weights=test_sizes), {
        "features": float(p),
        "features_kept": float(p_kept),
        "max_condition": float(max(condition_numbers)),
    }


def forward_cloud(
    weights: np.ndarray, points: np.ndarray, rotation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pre = points @ (rotation @ weights[0].astype(np.float32))
    h1 = np.maximum(pre, 0.0).astype(np.float32)
    activation = h1
    for weight in weights[1:]:
        activation = np.maximum(activation @ weight, 0.0)
    return h1, activation


def paired_summary(records: list[dict], labels: list[str]) -> dict:
    baseline = np.asarray([r["baseline_mse"] for r in records])
    rng = np.random.default_rng(20260729)
    boot = rng.integers(0, len(records), size=(20000, len(records)))
    result = {}
    for label in labels:
        values = np.asarray([r["mse"][label] for r in records])
        ratios = values[boot].mean(axis=1) / baseline[boot].mean(axis=1)
        result[label] = {
            "mean_mse": float(values.mean()),
            "ratio": float(values.mean() / baseline.mean()),
            "ci95": [float(v) for v in np.percentile(ratios, [2.5, 97.5])],
            "wins": int(np.sum(values < baseline)),
            "worst_ratio": float(np.max(values / baseline)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("mini", "full"), default="full")
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--rotation-seed", type=int, default=3)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument(
        "--ridges", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1]
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=[config.name for config in CONFIGS],
        default=[config.name for config in CONFIGS],
    )
    parser.add_argument("--skip-crossfit", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "results" / "exact_anchor_residual_full8.json",
    )
    args = parser.parse_args()

    data = MINI_DATA if args.split == "mini" else FULL_DATA
    rows = _load_rows(data, args.indices)
    points = make_kerdock_design()
    radius = sphere_radius_mean(WIDTH)
    rotation = random_rotation(WIDTH, args.rotation_seed)
    selected_configs = (
        [] if args.skip_crossfit else [c for c in CONFIGS if c.name in args.configs]
    )
    labels = ["mean_field_linear"] + [
        f"{config.name}:ridge={ridge:g}"
        for config in selected_configs
        for ridge in args.ridges
    ]

    records = []
    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        started = time.perf_counter()
        h1, final = forward_cloud(weights, points, rotation)
        baseline_prediction = final.mean(axis=0, dtype=np.float64)
        target = targets[-1]
        mean, second = gaussian_relu_moments(weights[0])

        jacobian = mean_field_jacobian(weights, mean, second)
        mean_field_prediction = (
            baseline_prediction
            + (mean - h1.mean(axis=0, dtype=np.float64)) @ jacobian
        )
        predictions = {"mean_field_linear": mean_field_prediction}
        diagnostics = {}

        for config in selected_configs:
            features = make_features(h1, mean, second, config, radius)
            for ridge in args.ridges:
                label = f"{config.name}:ridge={ridge:g}"
                prediction, diag = crossfit_control(
                    features, final, args.folds, ridge
                )
                predictions[label] = prediction
                diagnostics[label] = diag
            del features

        baseline_mse = float(np.mean(np.square(baseline_prediction - target)))
        baseline_error = baseline_prediction - target
        mean_field_delta = mean_field_prediction - baseline_prediction
        record = {
            "index": index,
            "name": name,
            "baseline_mse": baseline_mse,
            "mse": {
                label: float(np.mean(np.square(predictions[label] - target)))
                for label in labels
            },
            "mean_field_quadratic": {
                "error_delta": float(np.mean(baseline_error * mean_field_delta)),
                "delta2": float(np.mean(np.square(mean_field_delta))),
            },
            "diagnostics": diagnostics,
            "seconds": time.perf_counter() - started,
        }
        records.append(record)
        best_label = min(labels, key=lambda label: record["mse"][label])
        print(
            f"[{index:>4}] {name[:20]:<20} base={baseline_mse:.4e} "
            f"best={best_label} {record['mse'][best_label] / baseline_mse:.4f}x "
            f"({record['seconds']:.1f}s)",
            flush=True,
        )

    summary = paired_summary(records, labels)
    print("\nAggregate paired results", flush=True)
    for label in labels:
        item = summary[label]
        print(
            f"{label:<38} ratio={item['ratio']:.5f} "
            f"CI=[{item['ci95'][0]:.5f},{item['ci95'][1]:.5f}] "
            f"wins={item['wins']}/{len(records)} worst={item['worst_ratio']:.2f}x",
            flush=True,
        )

    output = {
        "protocol": {
            "split": args.split,
            "indices": args.indices,
            "rotation_seed": args.rotation_seed,
            "folds": args.folds,
            "ridges": args.ridges,
            "configs": [c.name for c in selected_configs],
            "anti_leakage": (
                "Coefficients use pointwise final outputs from training basis blocks "
                "only; each correction is evaluated on held-out whole basis blocks. "
                "No ground-truth means enter fitting."
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
