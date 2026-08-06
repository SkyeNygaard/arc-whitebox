"""Evaluate honest train/holdout postprocessing of nested QMC estimates.

The estimators in this file all use the same number of network evaluations.
Quarter- and half-sample estimates are checkpoints of the full stream, rather
than additional forward passes.  Scalar combination weights are fit on whole
MLPs 0--49 and are then frozen before evaluating MLPs 50--99.

This script intentionally measures forward time only.  Input designs are
precomputed and reused for all MLPs, which is both a realistic submission
optimization and prevents design-generation noise from obscuring comparisons.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from eval_sampling_official import DEFAULT_DATA, Design, _load_rows


def make_design(
    kind: str,
    width: int,
    samples: int,
    seed: int,
    chunk: int,
) -> list[np.ndarray]:
    design = Design(
        kind=kind,
        n=width,
        total=samples,
        seed=seed,
        antithetic=True,
        sphere=True,
    )
    blocks: list[np.ndarray] = []
    used = 0
    while used < samples:
        block = design.next(min(chunk, samples - used))
        if not len(block):
            raise RuntimeError(f"{kind} design ended after {used}/{samples} samples")
        blocks.append(block)
        used += len(block)
    return blocks


def make_rotated_tight_frame(
    width: int,
    samples: int,
    seed: int,
    chunk: int,
    iterations: int = 8,
) -> list[np.ndarray]:
    """Build an unbiased, approximately tight spherical frame.

    Alternating row normalization and covariance whitening makes a reused
    Sobol point set nearly satisfy both equal-radius and exact-covariance
    constraints.  A final independent Haar rotation is essential: conditional
    on the optimized frame, every rotated row is marginally uniform on the
    sphere.  The resulting sample average is therefore unbiased even though
    its rows are deliberately dependent.
    """
    blocks = make_design("sobol", width, samples, seed, chunk)
    sizes = [len(block) for block in blocks]
    x = np.concatenate(blocks).astype(np.float64)
    radius = float(np.linalg.norm(x[0]))
    for _ in range(iterations):
        x *= radius / np.linalg.norm(x, axis=1, keepdims=True)
        covariance = (x.T @ x) / len(x)
        chol = np.linalg.cholesky(covariance)
        x = np.linalg.solve(chol, x.T).T
    x *= radius / np.linalg.norm(x, axis=1, keepdims=True)

    rng = np.random.default_rng(seed + 10_000_019)
    gaussian = rng.standard_normal((width, width))
    rotation, r = np.linalg.qr(gaussian)
    rotation *= np.where(np.diag(r) < 0.0, -1.0, 1.0)[None, :]
    x = x @ rotation
    offsets = np.cumsum([0, *sizes])
    return [
        x[offsets[i] : offsets[i + 1]].astype(np.float32)
        for i in range(len(sizes))
    ]


def final_checkpoints(
    weights: np.ndarray,
    blocks: list[np.ndarray],
    checkpoints: tuple[int, ...],
) -> tuple[np.ndarray, float]:
    """Return final-layer means at cumulative sample checkpoints in one pass."""
    wanted = set(checkpoints)
    sums = np.zeros(weights.shape[-1], dtype=np.float64)
    estimates: dict[int, np.ndarray] = {}
    used = 0
    start = time.perf_counter()
    for original in blocks:
        offset = 0
        while offset < len(original):
            next_cp = min((cp for cp in checkpoints if cp > used), default=checkpoints[-1])
            take = min(len(original) - offset, next_cp - used)
            x = original[offset : offset + take]
            a = x
            for weight in weights:
                a = np.maximum(a @ weight, 0.0)
            sums += a.sum(axis=0, dtype=np.float64)
            used += take
            offset += take
            if used in wanted:
                estimates[used] = sums.copy() / used
    elapsed = time.perf_counter() - start
    if used != checkpoints[-1] or len(estimates) != len(checkpoints):
        raise RuntimeError(f"only reached checkpoints {sorted(estimates)}")
    return np.stack([estimates[cp] for cp in checkpoints]), elapsed


def fit_scalar_features(
    base: np.ndarray,
    features: np.ndarray,
    target: np.ndarray,
    ridge: float = 0.0,
) -> np.ndarray:
    """Fit shared scalar coefficients over all training MLP/neuron residuals."""
    x = features.reshape(-1, features.shape[-1]).astype(np.float64)
    y = (target - base).reshape(-1).astype(np.float64)
    gram = x.T @ x
    scale = float(np.trace(gram) / max(len(gram), 1))
    return np.linalg.solve(
        gram + np.eye(len(gram)) * ridge * max(scale, 1e-30),
        x.T @ y,
    )


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(prediction - target)))


def summarize(
    label: str,
    prediction: np.ndarray,
    target: np.ndarray,
    seconds: np.ndarray,
) -> dict[str, float | str]:
    per_mlp = np.mean(np.square(prediction - target), axis=1)
    return {
        "method": label,
        "mse": float(np.mean(per_mlp)),
        "median_mlp_mse": float(np.median(per_mlp)),
        "mean_forward_seconds": float(np.mean(seconds)),
        "mse_seconds": float(np.mean(per_mlp * seconds)),
        "per_mlp_mse": per_mlp.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--samples", type=int, default=32768)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--train-stop", type=int, default=50)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.samples % 4:
        raise ValueError("--samples must be divisible by four")
    indices = list(range(args.count))
    rows = _load_rows(args.data, indices)
    width = rows[0][1].shape[-1]
    checkpoints = (args.samples // 4, args.samples // 2, args.samples)

    designs = {
        ("sobol", seed): make_design("sobol", width, args.samples, seed, args.chunk)
        for seed in args.seeds
    }
    # A randomized orthogonal frame is an exactly unbiased spherical design:
    # every row is marginally uniform on the sphere and each 256-row block has
    # exact empirical covariance.
    designs[("orthogonal", args.seeds[0])] = make_design(
        "orthogonal", width, args.samples, args.seeds[0], args.chunk
    )
    designs[("rotated_tight_frame", args.seeds[0])] = make_rotated_tight_frame(
        width, args.samples, args.seeds[0], args.chunk
    )

    targets = np.stack([target[-1] for _, _, target in rows])
    estimates: dict[tuple[str, int], list[np.ndarray]] = {
        key: [] for key in designs
    }
    timings: dict[tuple[str, int], list[float]] = {key: [] for key in designs}
    for row_number, (_, weights, _) in enumerate(rows):
        for key, blocks in designs.items():
            estimate, elapsed = final_checkpoints(weights, blocks, checkpoints)
            estimates[key].append(estimate)
            timings[key].append(elapsed)
        print({"completed": row_number + 1, "of": len(rows)}, flush=True)

    estimate_arrays = {key: np.stack(value) for key, value in estimates.items()}
    timing_arrays = {key: np.asarray(value) for key, value in timings.items()}
    train = slice(0, args.train_stop)
    test = slice(args.train_stop, args.count)

    primary_key = ("sobol", args.seeds[0])
    primary = estimate_arrays[primary_key]
    quarter, half, full = (primary[:, i] for i in range(3))
    nested_features = np.stack((full - half, half - quarter), axis=-1)
    nested_coef = fit_scalar_features(
        full[train], nested_features[train], targets[train], args.ridge
    )
    nested_prediction = full + nested_features @ nested_coef

    one_feature = (full - half)[..., None]
    one_coef = fit_scalar_features(
        full[train], one_feature[train], targets[train], args.ridge
    )
    richardson_prediction = full + one_feature[..., 0] * one_coef[0]

    # Equal-cost split-scramble estimators.  A k-way estimate uses the first
    # N/k checkpoint from each of k independent scrambles.
    split_predictions: dict[int, np.ndarray] = {}
    split_seconds: dict[int, np.ndarray] = {}
    for count in (2, 4):
        if len(args.seeds) < count:
            continue
        checkpoint_index = {2: 1, 4: 0}[count]
        keys = [("sobol", seed) for seed in args.seeds[:count]]
        split_predictions[count] = np.mean(
            [estimate_arrays[key][:, checkpoint_index] for key in keys], axis=0
        )
        split_seconds[count] = np.sum(
            [timing_arrays[key] * (1.0 / count) for key in keys], axis=0
        )

    # Full-scramble ensembles cost k times as much, so MSE*time determines
    # whether they are an efficiency gain rather than merely a lower-error run.
    full_ensembles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    full_seed_estimates = np.stack(
        [estimate_arrays[("sobol", seed)][:, 2] for seed in args.seeds],
        axis=-1,
    )
    for count in range(2, len(args.seeds) + 1):
        keys = [("sobol", seed) for seed in args.seeds[:count]]
        seconds = np.sum([timing_arrays[key] for key in keys], axis=0)
        full_ensembles[f"sobol_{count}_full_scrambles"] = (
            np.mean(full_seed_estimates[..., :count], axis=-1),
            seconds,
        )
        if count >= 3:
            full_ensembles[f"sobol_{count}_full_coordinate_median"] = (
                np.median(full_seed_estimates[..., :count], axis=-1),
                seconds,
            )

    # A constrained stacker can detect persistent seed/design quality without
    # learning a target offset: base + sum_i c_i(other_i-base) always has
    # estimator weights summing to one.  Coefficients are scalar and frozen on
    # the disjoint training MLPs.
    stack_coefficients = None
    stacked_full = None
    stacked_seconds = None
    if len(args.seeds) >= 2:
        stack_features = full_seed_estimates[..., 1:] - full_seed_estimates[..., :1]
        stack_coefficients = fit_scalar_features(
            full[train], stack_features[train], targets[train], args.ridge
        )
        stacked_full = full + stack_features @ stack_coefficients
        stacked_seconds = np.sum(
            [timing_arrays[("sobol", seed)] for seed in args.seeds], axis=0
        )

    orth_key = ("orthogonal", args.seeds[0])
    orthogonal = estimate_arrays[orth_key][:, 2]
    tight_key = ("rotated_tight_frame", args.seeds[0])
    tight_frame = estimate_arrays[tight_key][:, 2]

    train_results = [
        summarize("sobol_full", full[train], targets[train], timing_arrays[primary_key][train]),
        summarize(
            "nested_richardson_1",
            richardson_prediction[train],
            targets[train],
            timing_arrays[primary_key][train],
        ),
        summarize(
            "nested_richardson_2",
            nested_prediction[train],
            targets[train],
            timing_arrays[primary_key][train],
        ),
        summarize(
            "orthogonal_frames",
            orthogonal[train],
            targets[train],
            timing_arrays[orth_key][train],
        ),
        summarize(
            "rotated_tight_frame",
            tight_frame[train],
            targets[train],
            timing_arrays[tight_key][train],
        ),
    ]
    test_results = [
        summarize("sobol_full", full[test], targets[test], timing_arrays[primary_key][test]),
        summarize(
            "nested_richardson_1",
            richardson_prediction[test],
            targets[test],
            timing_arrays[primary_key][test],
        ),
        summarize(
            "nested_richardson_2",
            nested_prediction[test],
            targets[test],
            timing_arrays[primary_key][test],
        ),
        summarize(
            "orthogonal_frames",
            orthogonal[test],
            targets[test],
            timing_arrays[orth_key][test],
        ),
        summarize(
            "rotated_tight_frame",
            tight_frame[test],
            targets[test],
            timing_arrays[tight_key][test],
        ),
    ]
    for count, prediction in split_predictions.items():
        train_results.append(
            summarize(
                f"sobol_{count}_split_scrambles",
                prediction[train],
                targets[train],
                split_seconds[count][train],
            )
        )
        test_results.append(
            summarize(
                f"sobol_{count}_split_scrambles",
                prediction[test],
                targets[test],
                split_seconds[count][test],
            )
        )
    for label, (prediction, seconds) in full_ensembles.items():
        train_results.append(
            summarize(label, prediction[train], targets[train], seconds[train])
        )
        test_results.append(
            summarize(label, prediction[test], targets[test], seconds[test])
        )
    if stacked_full is not None and stacked_seconds is not None:
        train_results.append(
            summarize(
                "sobol_full_constrained_stack",
                stacked_full[train],
                targets[train],
                stacked_seconds[train],
            )
        )
        test_results.append(
            summarize(
                "sobol_full_constrained_stack",
                stacked_full[test],
                targets[test],
                stacked_seconds[test],
            )
        )

    report = {
        "samples": args.samples,
        "train_indices": [0, args.train_stop - 1],
        "test_indices": [args.train_stop, args.count - 1],
        "nested_coefficients": {
            "one_feature": one_coef.tolist(),
            "two_feature": nested_coef.tolist(),
            "full_scramble_stack": (
                None if stack_coefficients is None else stack_coefficients.tolist()
            ),
        },
        "train": train_results,
        "test": test_results,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
