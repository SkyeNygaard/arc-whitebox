#!/usr/bin/env python3
"""Path 6 implementation experiments for WHestBench.

This is a self-contained NumPy development harness.  It ports the current
production depth-5 partial-output-tree Winograd kernel and tests exact ways to
reuse its final preactivation for a uniform penultimate-layer translation.
It also profiles chunked direct accumulation and K32/all-basis feature
reductions on the exact 66,048-row layout.

Authoritative FlopScope runs must be made in the local WHestBench environment;
this harness emits algebraic flop counts using the operation costs observed in
the archived production FlopScope profiles.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

WIDTH = 256
N_BASES = 129
ROWS_PER_BASIS = 512
N_POINTS = N_BASES * ROWS_PER_BASIS
# Authoritative archived FlopScope cost of one production partial-output-tree
# multiplication for (66,048 x 256) @ (256 x 256).
PARTIAL_TREE_MATMUL_FLOPS = 5_481_223_424
ELEMENTS = N_POINTS * WIDTH


def _encode(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half_rows = left.shape[-2] // 2
    half_inner = left.shape[-1] // 2
    half_output = right.shape[-1] // 2
    a11 = left[..., :half_rows, :half_inner]
    a12 = left[..., :half_rows, half_inner:]
    a21 = left[..., half_rows:, :half_inner]
    a22 = left[..., half_rows:, half_inner:]
    b11 = right[..., :half_inner, :half_output]
    b12 = right[..., :half_inner, half_output:]
    b21 = right[..., half_inner:, :half_output]
    b22 = right[..., half_inner:, half_output:]
    s1 = a21 + a22
    s2 = s1 - a11
    s3 = a11 - a21
    s4 = a12 - s2
    t1 = b12 - b11
    t2 = b22 - t1
    t3 = b22 - b12
    t4 = t2 - b21
    return (
        np.stack((a11, a12, s4, a22, s1, s2, s3), axis=-3),
        np.stack((b11, b21, b22, t4, t1, t2, t3), axis=-3),
    )


def _decode(products: np.ndarray) -> np.ndarray:
    p1, p2, p3, p4, p5, p6, p7 = (
        products[..., i, :, :] for i in range(7)
    )
    u1 = p1 + p2
    u2 = p1 + p6
    u3 = u2 + p7
    u4 = u2 + p5
    return np.block([[u1, u4 + p3], [u3 - p4, u3 + p5]])


def _level_one_quadrants(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half_rows = left.shape[-2] // 2
    half_inner = left.shape[-1] // 2
    half_output = right.shape[-1] // 2
    a11 = left[..., :half_rows, :half_inner]
    a12 = left[..., :half_rows, half_inner:]
    a21 = left[..., half_rows:, :half_inner]
    a22 = left[..., half_rows:, half_inner:]
    b11 = right[..., :half_inner, :half_output]
    b12 = right[..., :half_inner, half_output:]
    b21 = right[..., half_inner:, :half_output]
    b22 = right[..., half_inner:, half_output:]
    s1 = a21 + a22
    s2 = s1 - a11
    s3 = a11 - a21
    s4 = a12 - s2
    t1 = b12 - b11
    t2 = b22 - t1
    t3 = b22 - b12
    t4 = t2 - b21
    p1 = a11 @ b11
    p2 = a12 @ b21
    p3 = s4 @ b22
    p4 = a22 @ t4
    p5 = s1 @ t1
    p6 = s2 @ t2
    p7 = s3 @ t3
    u1 = p1 + p2
    u2 = p1 + p6
    u3 = u2 + p7
    u4 = u2 + p5
    return u1, u4 + p3, u3 - p4, u3 + p5


def _depth_two_output_tree(
    left: np.ndarray, right: np.ndarray
) -> tuple[tuple[np.ndarray, ...], ...]:
    half_rows = left.shape[-2] // 2
    half_inner = left.shape[-1] // 2
    half_output = right.shape[-1] // 2
    a11 = left[..., :half_rows, :half_inner]
    a12 = left[..., :half_rows, half_inner:]
    a21 = left[..., half_rows:, :half_inner]
    a22 = left[..., half_rows:, half_inner:]
    b11 = right[..., :half_inner, :half_output]
    b12 = right[..., :half_inner, half_output:]
    b21 = right[..., half_inner:, :half_output]
    b22 = right[..., half_inner:, half_output:]
    s1 = a21 + a22
    s2 = s1 - a11
    s3 = a11 - a21
    s4 = a12 - s2
    t1 = b12 - b11
    t2 = b22 - t1
    t3 = b22 - b12
    t4 = t2 - b21
    products = (
        _level_one_quadrants(a11, b11),
        _level_one_quadrants(a12, b21),
        _level_one_quadrants(s4, b22),
        _level_one_quadrants(a22, t4),
        _level_one_quadrants(s1, t1),
        _level_one_quadrants(s2, t2),
        _level_one_quadrants(s3, t3),
    )
    decoded: list[tuple[np.ndarray, ...]] = []
    for quadrant in range(4):
        p1, p2, p3, p4, p5, p6, p7 = (
            product[quadrant] for product in products
        )
        u1 = p1 + p2
        u2 = p1 + p6
        u3 = u2 + p7
        u4 = u2 + p5
        decoded.append((u1, u4 + p3, u3 - p4, u3 + p5))
    return tuple(
        tuple(decoded[leaf][root] for leaf in range(4))
        for root in range(4)
    )


def partial_tree_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    encoded_left = left
    encoded_right = right
    for _ in range(3):
        encoded_left, encoded_right = _encode(encoded_left, encoded_right)
    tree = _depth_two_output_tree(encoded_left, encoded_right)
    products = np.block(
        [
            [tree[0][0], tree[0][1], tree[1][0], tree[1][1]],
            [tree[0][2], tree[0][3], tree[1][2], tree[1][3]],
            [tree[2][0], tree[2][1], tree[3][0], tree[3][1]],
            [tree[2][2], tree[2][3], tree[3][2], tree[3][3]],
        ]
    )
    for _ in range(3):
        products = _decode(products)
    return products


def vector_mean_relu(pre: np.ndarray, shift: np.ndarray) -> np.ndarray:
    corrected = np.maximum(pre + shift[None, :], 0.0)
    return corrected.astype(np.float64).mean(axis=0)


def chunked_mean_relu(
    pre: np.ndarray, shift: np.ndarray, chunk_rows: int
) -> np.ndarray:
    total = np.zeros(WIDTH, dtype=np.float64)
    for start in range(0, len(pre), chunk_rows):
        block = pre[start : start + chunk_rows]
        total += np.maximum(block + shift[None, :], 0.0).astype(
            np.float64
        ).sum(axis=0)
    return total / len(pre)


def dense_shift(delta: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return delta @ weight


def sparse_shift(
    indices: np.ndarray, values: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    return values @ weight[indices]


def winograd_compatible_shift(
    delta: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    # Minimum legal row count for five row halvings is 32.  Repeated rows stay
    # repeated under the exact linear map, so one output row is the shift.
    repeated = np.broadcast_to(delta, (32, WIDTH)).copy()
    return partial_tree_matmul(repeated, weight)[0]


def replay_mean(
    activation: np.ndarray, delta: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    corrected_pre = partial_tree_matmul(
        activation + delta[None, :], weight
    )
    return np.maximum(corrected_pre, 0.0).astype(np.float64).mean(axis=0)


def k32_basis_means(activation: np.ndarray) -> np.ndarray:
    return activation[: 32 * ROWS_PER_BASIS].reshape(
        32, ROWS_PER_BASIS, WIDTH
    ).astype(np.float64).mean(axis=1)


def all_basis_means(activation: np.ndarray) -> np.ndarray:
    return activation.reshape(N_BASES, ROWS_PER_BASIS, WIDTH).astype(
        np.float64
    ).mean(axis=1)


def k32_mean_gram8(activation: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = activation[: 32 * ROWS_PER_BASIS, coords].astype(np.float64)
    return h.mean(axis=0), h.T @ h / len(h)


@dataclass
class Timing:
    label: str
    seconds: list[float]
    median_s: float
    min_s: float
    max_rss_kib: int


def timed(label: str, fn: Callable[[], object], repeats: int) -> tuple[Timing, object]:
    values: list[float] = []
    result = None
    # One untimed warmup for cheap kernels.  Expensive partial-tree replay is
    # already warmed by the baseline preactivation in the driver.
    if label not in {"replay_partial_tree"}:
        result = fn()
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        values.append(time.perf_counter() - start)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (
        Timing(
            label=label,
            seconds=values,
            median_s=float(statistics.median(values)),
            min_s=float(min(values)),
            max_rss_kib=int(rss),
        ),
        result,
    )


def flop_accounting(q_sparse: int = 8) -> dict[str, int | float]:
    # Costs follow observed FlopScope conventions in the archived row-0
    # profile: maximum=N, astype=N, mean=2N, add=N, matmul=m*n*(2k-1).
    replay = (
        PARTIAL_TREE_MATMUL_FLOPS
        + ELEMENTS  # maximum
        + ELEMENTS  # astype float64
        + 2 * ELEMENTS  # mean
    )
    dense_shift_cost = WIDTH * (2 * WIDTH - 1)
    sparse_shift_cost = WIDTH * (2 * q_sparse - 1)
    reuse_dense = (
        dense_shift_cost
        + ELEMENTS  # broadcast add
        + ELEMENTS  # maximum
        + ELEMENTS  # astype
        + 2 * ELEMENTS  # mean
    )
    reuse_sparse = reuse_dense - dense_shift_cost + sparse_shift_cost
    return {
        "partial_tree_matmul": PARTIAL_TREE_MATMUL_FLOPS,
        "full_replay_total": replay,
        "reuse_dense_shift_total": reuse_dense,
        "reuse_sparse_q8_total": reuse_sparse,
        "saved_vs_replay_dense": replay - reuse_dense,
        "saved_vs_replay_sparse_q8": replay - reuse_sparse,
        "reuse_dense_fraction_of_replay": reuse_dense / replay,
        "reuse_sparse_q8_fraction_of_replay": reuse_sparse / replay,
        "k32_basis_mean_upper_bound": 3 * 32 * ROWS_PER_BASIS * WIDTH,
        "all_basis_mean_upper_bound": 3 * ELEMENTS,
        "k32_gram8_matmul": 8 * (2 * 32 * ROWS_PER_BASIS - 1) * 8,
    }


def build_inputs(seed: int, delta_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # ReLU-like penultimate activations with a nonzero atom at zero.
    activation = np.maximum(
        rng.normal(0.15, 0.70, size=(N_POINTS, WIDTH)), 0.0
    ).astype(np.float32)
    weight = (
        rng.standard_normal((WIDTH, WIDTH), dtype=np.float32)
        / math.sqrt(WIDTH)
    ).astype(np.float32)
    raw_delta = rng.standard_normal(WIDTH).astype(np.float32)
    raw_delta /= np.linalg.norm(raw_delta)
    rms_h = float(np.sqrt(np.mean(np.square(activation), dtype=np.float64)))
    delta = raw_delta * (delta_scale * rms_h * math.sqrt(WIDTH))
    indices = np.sort(rng.choice(WIDTH, size=8, replace=False))
    sparse_values = delta[indices].copy()
    sparse_delta = np.zeros(WIDTH, dtype=np.float32)
    sparse_delta[indices] = sparse_values
    return activation, weight, delta, indices, sparse_delta


def run_full(args: argparse.Namespace) -> dict:
    activation, weight, delta, indices, sparse_delta = build_inputs(
        args.seed, args.delta_scale
    )
    # Baseline final preactivation already exists in the integrated estimator.
    baseline_timing, pre = timed(
        "baseline_partial_tree_preactivation",
        lambda: partial_tree_matmul(activation, weight),
        max(1, args.expensive_repeats),
    )

    replay_timing, replay = timed(
        "replay_partial_tree",
        lambda: replay_mean(activation, delta, weight),
        max(1, args.expensive_repeats),
    )

    dense_shift_value = dense_shift(delta, weight)
    sparse_shift_value = sparse_shift(indices, sparse_delta[indices], weight)
    winograd_shift_value = winograd_compatible_shift(delta, weight)

    timings: list[Timing] = [baseline_timing, replay_timing]
    outputs: dict[str, np.ndarray] = {"replay": replay}

    t, out = timed(
        "reuse_vector_dense_shift",
        lambda: vector_mean_relu(pre, dense_shift_value),
        args.cheap_repeats,
    )
    timings.append(t)
    outputs["reuse_vector_dense_shift"] = np.asarray(out)

    t, out = timed(
        "reuse_vector_winograd_shift",
        lambda: vector_mean_relu(pre, winograd_shift_value),
        args.cheap_repeats,
    )
    timings.append(t)
    outputs["reuse_vector_winograd_shift"] = np.asarray(out)

    for chunk in args.chunks:
        t, out = timed(
            f"reuse_chunk_{chunk}",
            lambda chunk=chunk: chunked_mean_relu(
                pre, dense_shift_value, chunk
            ),
            args.cheap_repeats,
        )
        timings.append(t)
        outputs[f"reuse_chunk_{chunk}"] = np.asarray(out)

    # Sparse-delta accuracy path.
    sparse_replay = replay_mean(activation, sparse_delta, weight)
    sparse_reuse = vector_mean_relu(pre, sparse_shift_value)

    feature_timings = []
    for label, fn in (
        ("k32_basis_means", lambda: k32_basis_means(activation)),
        ("all_129_basis_means", lambda: all_basis_means(activation)),
        (
            "k32_mean_gram8",
            lambda: k32_mean_gram8(activation, np.arange(8)),
        ),
    ):
        t, _ = timed(label, fn, args.feature_repeats)
        feature_timings.append(t)

    comparisons = {}
    for label, out in outputs.items():
        diff = out.astype(np.float64) - replay.astype(np.float64)
        comparisons[label] = {
            "max_abs_mean_difference": float(np.max(np.abs(diff))),
            "rms_mean_difference": float(np.sqrt(np.mean(np.square(diff)))),
        }
    sparse_diff = sparse_reuse - sparse_replay

    return {
        "protocol": {
            "seed": args.seed,
            "n_points": N_POINTS,
            "width": WIDTH,
            "delta_scale_relative_to_activation_rms": args.delta_scale,
            "expensive_repeats": args.expensive_repeats,
            "cheap_repeats": args.cheap_repeats,
            "feature_repeats": args.feature_repeats,
            "chunks": args.chunks,
            "numpy_version": np.__version__,
            "threads": {
                key: os.environ.get(key)
                for key in (
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_NUM_THREADS",
                )
            },
        },
        "flop_accounting": flop_accounting(),
        "timings": [asdict(item) for item in timings],
        "feature_timings": [asdict(item) for item in feature_timings],
        "mean_comparisons_to_full_replay": comparisons,
        "shift_comparisons": {
            "dense_vs_winograd_max_abs": float(
                np.max(np.abs(dense_shift_value - winograd_shift_value))
            ),
            "dense_vs_winograd_rms": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            dense_shift_value.astype(np.float64)
                            - winograd_shift_value.astype(np.float64)
                        )
                    )
                )
            ),
            "sparse_reuse_vs_replay_max_abs_mean": float(
                np.max(np.abs(sparse_diff))
            ),
            "sparse_reuse_vs_replay_rms_mean": float(
                np.sqrt(np.mean(np.square(sparse_diff)))
            ),
        },
    }


def run_accuracy_panel(args: argparse.Namespace) -> dict:
    rows = []
    for seed in args.panel_seeds:
        # H and W are invariant across the scale sweep for a fixed seed.
        activation, weight, unit_delta, _, _ = build_inputs(seed, 1.0)
        pre = partial_tree_matmul(activation, weight)
        for scale in args.panel_scales:
            delta = unit_delta * scale
            replay = replay_mean(activation, delta, weight)
            dense = vector_mean_relu(pre, dense_shift(delta, weight))
            winograd = vector_mean_relu(
                pre, winograd_compatible_shift(delta, weight)
            )
            dense_diff = dense - replay
            winograd_diff = winograd - replay
            rows.append(
                {
                    "seed": seed,
                    "delta_scale": scale,
                    "dense_max_abs_mean_diff": float(
                        np.max(np.abs(dense_diff))
                    ),
                    "dense_rms_mean_diff": float(
                        np.sqrt(np.mean(np.square(dense_diff)))
                    ),
                    "winograd_max_abs_mean_diff": float(
                        np.max(np.abs(winograd_diff))
                    ),
                    "winograd_rms_mean_diff": float(
                        np.sqrt(np.mean(np.square(winograd_diff)))
                    ),
                }
            )
    def agg(key: str) -> dict[str, float]:
        vals = np.asarray([row[key] for row in rows])
        return {
            "max": float(vals.max()),
            "median": float(np.median(vals)),
            "mean": float(vals.mean()),
        }
    return {
        "protocol": {
            "seeds": args.panel_seeds,
            "scales": args.panel_scales,
            "cases": len(rows),
            "n_points": N_POINTS,
            "width": WIDTH,
        },
        "summary": {
            key: agg(key)
            for key in (
                "dense_max_abs_mean_diff",
                "dense_rms_mean_diff",
                "winograd_max_abs_mean_diff",
                "winograd_rms_mean_diff",
            )
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full", "accuracy-panel"), default="full")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--delta-scale", type=float, default=0.003)
    parser.add_argument("--expensive-repeats", type=int, default=3)
    parser.add_argument("--cheap-repeats", type=int, default=9)
    parser.add_argument("--feature-repeats", type=int, default=15)
    parser.add_argument("--chunks", type=int, nargs="+", default=[512, 2048, 8192, 16384])
    parser.add_argument("--panel-seeds", type=int, nargs="+", default=[101, 202, 303, 404])
    parser.add_argument("--panel-scales", type=float, nargs="+", default=[0.0001, 0.0003, 0.001, 0.003])
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_full(args) if args.mode == "full" else run_accuracy_panel(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"wrote": str(args.out), "mode": args.mode}, indent=2))


if __name__ == "__main__":
    main()
