"""Audit fast matrix multiplication for the Kerdock/MUB estimator.

The Kerdock design has exactly ``66_048 = 258 * 256`` rows.  Consequently,
every post-first-layer multiplication can be viewed as a batch of 258 square
``256 x 256`` products against one shared weight matrix.  This script tests
whether Strassen-style bilinear algorithms reduce *authoritative*
``flopscope==0.9.1`` cost without damaging the quadrature estimate.

Two seven-product algorithms are implemented:

* ``strassen``: the classical 18-addition schedule;
* ``winograd``: the 15-addition Strassen--Winograd schedule.

The recursion is breadth-first and packed.  All ``7**depth`` leaf products are
executed by one broadcast batched ``matmul``.  This avoids thousands of Python
and flopscope wrapper calls at depth 3--4, at the honest cost of tracked
``stack`` operations for the packed representations.  Depths 1--4 are the
primary audit; deeper levels remain available to locate the arithmetic/copy
minimum and to establish no-go bounds for larger row ensembles.

The script is intentionally restricted to selection IDs 0--49.  It never opens
the frozen holdout IDs.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import flopscope
import flopscope.numpy as fnp
import numpy as np

from eval_sampling_official import DEFAULT_DATA, _load_rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET = (
    ROOT
    / "submissions"
    / "kerdock_mub5"
    / "kerdock_mub5_seed3.npz"
)
DEFAULT_OUT = ROOT / "results" / "strassen_audit_selection.json"

WIDTH = 256
DEPTH = 32
N_POINTS = 66_048
N_BLOCKS = N_POINTS // WIDTH
KERDOCK_BASES = 128
BUDGET = 272_000_000_000
LAMBDA_FLOPS_PER_SECOND = 100_000_000_000.0
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def mean_gaussian_radius(width: int = WIDTH) -> float:
    return math.sqrt(2.0) * math.exp(
        math.lgamma((width + 1.0) / 2.0)
        - math.lgamma(width / 2.0)
    )


def fwht_axis_one(values: Any, xp: Any) -> Any:
    """Unnormalised Walsh--Hadamard transform along the point axis."""
    span = 1
    while span < WIDTH:
        grouped = values.reshape(
            (KERDOCK_BASES, WIDTH // (2 * span), 2, span, WIDTH)
        )
        left = grouped[:, :, 0, :, :]
        right = grouped[:, :, 1, :, :]
        values = xp.stack(
            (left + right, left - right),
            axis=2,
        ).reshape((KERDOCK_BASES, WIDTH, WIDTH))
        span *= 2
    return values


def first_layer_design(
    first_weight: Any,
    rotation: Any,
    chirps: Any,
    xp: Any,
) -> Any:
    """Evaluate the rotated 66,048-point design's first layer."""
    effective_weight = rotation @ first_weight
    radius = mean_gaussian_radius()
    weighted = chirps[:, :, None] * effective_weight[None, :, :]
    preactivation = fwht_axis_one(weighted, xp) * (
        radius / math.sqrt(WIDTH)
    )
    kerdock_rows = xp.stack(
        (preactivation, -preactivation),
        axis=2,
    ).reshape((-1, WIDTH))
    coordinate_rows = xp.stack(
        (radius * effective_weight, -radius * effective_weight),
        axis=1,
    ).reshape((-1, WIDTH))
    return xp.maximum(
        xp.concatenate((kerdock_rows, coordinate_rows), axis=0),
        0.0,
    )


def first_layer_blocks(
    first_weight: Any,
    rotation: Any,
    chirps: Any,
    xp: Any,
) -> Any:
    """First layer directly in ``(258,256,256)`` block layout.

    Row order has no effect on an equal-weight cubature rule.  Grouping each
    signed basis as a square block avoids the otherwise billed 66,048-row
    reshape before every fast multiplication.
    """
    effective_weight = rotation @ first_weight
    radius = mean_gaussian_radius()
    weighted = chirps[:, :, None] * effective_weight[None, :, :]
    preactivation = fwht_axis_one(weighted, xp) * (
        radius / math.sqrt(WIDTH)
    )
    kerdock_blocks = xp.concatenate(
        (preactivation, -preactivation),
        axis=0,
    )
    coordinate_blocks = xp.stack(
        (radius * effective_weight, -radius * effective_weight),
        axis=0,
    )
    return xp.maximum(
        xp.concatenate(
            (kerdock_blocks, coordinate_blocks),
            axis=0,
        ),
        0.0,
    )


def _encode_strassen(a: Any, b: Any, xp: Any) -> tuple[Any, Any]:
    """One packed level of the classical Strassen input transform."""
    half_rows = a.shape[-2] // 2
    half_inner = a.shape[-1] // 2
    half_output = b.shape[-1] // 2
    a11 = a[..., :half_rows, :half_inner]
    a12 = a[..., :half_rows, half_inner:]
    a21 = a[..., half_rows:, :half_inner]
    a22 = a[..., half_rows:, half_inner:]
    b11 = b[..., :half_inner, :half_output]
    b12 = b[..., :half_inner, half_output:]
    b21 = b[..., half_inner:, :half_output]
    b22 = b[..., half_inner:, half_output:]

    # Six batched A additions and four shared-B additions.
    encoded_a = xp.stack(
        (
            a11 + a22,
            a21 + a22,
            a11,
            a22,
            a11 + a12,
            a21 - a11,
            a12 - a22,
        ),
        axis=-3,
    )
    encoded_b = xp.stack(
        (
            b11 + b22,
            b11,
            b12 - b22,
            b21 - b11,
            b22,
            b11 + b12,
            b21 + b22,
        ),
        axis=-3,
    )
    return encoded_a, encoded_b


def _decode_strassen(products: Any, xp: Any) -> Any:
    """One packed level of the classical Strassen output transform."""
    p1 = products[..., 0, :, :]
    p2 = products[..., 1, :, :]
    p3 = products[..., 2, :, :]
    p4 = products[..., 3, :, :]
    p5 = products[..., 4, :, :]
    p6 = products[..., 5, :, :]
    p7 = products[..., 6, :, :]
    c11 = p1 + p4 - p5 + p7
    c12 = p3 + p5
    c21 = p2 + p4
    c22 = p1 - p2 + p3 + p6
    return xp.block([[c11, c12], [c21, c22]])


def _encode_winograd(a: Any, b: Any, xp: Any) -> tuple[Any, Any]:
    """One packed level of Winograd's 15-addition Strassen schedule."""
    half_rows = a.shape[-2] // 2
    half_inner = a.shape[-1] // 2
    half_output = b.shape[-1] // 2
    a11 = a[..., :half_rows, :half_inner]
    a12 = a[..., :half_rows, half_inner:]
    a21 = a[..., half_rows:, :half_inner]
    a22 = a[..., half_rows:, half_inner:]
    b11 = b[..., :half_inner, :half_output]
    b12 = b[..., :half_inner, half_output:]
    b21 = b[..., half_inner:, :half_output]
    b22 = b[..., half_inner:, half_output:]

    # Four batched A additions and four shared-B additions.
    s1 = a21 + a22
    s2 = s1 - a11
    s3 = a11 - a21
    s4 = a12 - s2
    t1 = b12 - b11
    t2 = b22 - t1
    t3 = b22 - b12
    t4 = t2 - b21
    encoded_a = xp.stack(
        (a11, a12, s4, a22, s1, s2, s3),
        axis=-3,
    )
    encoded_b = xp.stack(
        (b11, b21, b22, t4, t1, t2, t3),
        axis=-3,
    )
    return encoded_a, encoded_b


def _decode_winograd(products: Any, xp: Any) -> Any:
    """One packed level of Winograd's 15-addition output schedule."""
    p1 = products[..., 0, :, :]
    p2 = products[..., 1, :, :]
    p3 = products[..., 2, :, :]
    p4 = products[..., 3, :, :]
    p5 = products[..., 4, :, :]
    p6 = products[..., 5, :, :]
    p7 = products[..., 6, :, :]
    u1 = p1 + p2
    u2 = p1 + p6
    u3 = u2 + p7
    u4 = u2 + p5
    u5 = u4 + p3
    u6 = u3 - p4
    u7 = u3 + p5
    return xp.block([[u1, u5], [u6, u7]])


def packed_fast_matmul(
    left: Any,
    weight: Any,
    recursion_depth: int,
    variant: str,
    xp: Any,
) -> Any:
    """Multiply a square/tall left operand using packed breadth-first leaves."""
    if recursion_depth == 0:
        return left @ weight
    if recursion_depth < 0 or recursion_depth > 8:
        raise ValueError("recursion depth must be between 0 and 8")
    if WIDTH % (2**recursion_depth):
        raise ValueError("matrix width is not divisible by recursion leaves")
    if variant == "strassen":
        encoder = _encode_strassen
        decoder = _decode_strassen
    elif variant == "winograd":
        encoder = _encode_winograd
        decoder = _decode_winograd
    else:
        raise ValueError(f"unknown fast-matmul variant: {variant}")

    # Every recursion level remains an explicit tensor axis:
    #
    #   A: (sample_blocks, 7, 7, ..., leaf, leaf)
    #   B: (               7, 7, ..., leaf, leaf)
    #
    # NumPy's batched matmul broadcasts B over the sample-block axis.  Keeping
    # the recursion tree instead of flattening it is important because
    # flopscope 0.9.1 deliberately bills reshape by numel even for views.
    a = left
    b = weight
    for _ in range(recursion_depth):
        encoded_a, encoded_b = encoder(a, b, xp)
        a = encoded_a
        b = encoded_b
    products = a @ b
    for _ in range(recursion_depth):
        products = decoder(products, xp)
    return products


def depth_first_fast_matmul(
    left: Any,
    weight: Any,
    recursion_depth: int,
    variant: str,
    xp: Any,
) -> Any:
    """Depth-first seven-product recursion with no packed stack copies."""
    if recursion_depth == 0:
        return left @ weight
    half_rows = left.shape[-2] // 2
    half_inner = left.shape[-1] // 2
    half_output = weight.shape[-1] // 2
    a11 = left[..., :half_rows, :half_inner]
    a12 = left[..., :half_rows, half_inner:]
    a21 = left[..., half_rows:, :half_inner]
    a22 = left[..., half_rows:, half_inner:]
    b11 = weight[..., :half_inner, :half_output]
    b12 = weight[..., :half_inner, half_output:]
    b21 = weight[..., half_inner:, :half_output]
    b22 = weight[..., half_inner:, half_output:]

    recurse = depth_first_fast_matmul
    next_depth = recursion_depth - 1
    if variant == "strassen":
        p1 = recurse(
            a11 + a22,
            b11 + b22,
            next_depth,
            variant,
            xp,
        )
        p2 = recurse(
            a21 + a22,
            b11,
            next_depth,
            variant,
            xp,
        )
        p3 = recurse(
            a11,
            b12 - b22,
            next_depth,
            variant,
            xp,
        )
        p4 = recurse(
            a22,
            b21 - b11,
            next_depth,
            variant,
            xp,
        )
        p5 = recurse(
            a11 + a12,
            b22,
            next_depth,
            variant,
            xp,
        )
        p6 = recurse(
            a21 - a11,
            b11 + b12,
            next_depth,
            variant,
            xp,
        )
        p7 = recurse(
            a12 - a22,
            b21 + b22,
            next_depth,
            variant,
            xp,
        )
        c11 = p1 + p4 - p5 + p7
        c12 = p3 + p5
        c21 = p2 + p4
        c22 = p1 - p2 + p3 + p6
    elif variant == "winograd":
        s1 = a21 + a22
        s2 = s1 - a11
        s3 = a11 - a21
        s4 = a12 - s2
        t1 = b12 - b11
        t2 = b22 - t1
        t3 = b22 - b12
        t4 = t2 - b21
        p1 = recurse(a11, b11, next_depth, variant, xp)
        p2 = recurse(a12, b21, next_depth, variant, xp)
        p3 = recurse(s4, b22, next_depth, variant, xp)
        p4 = recurse(a22, t4, next_depth, variant, xp)
        p5 = recurse(s1, t1, next_depth, variant, xp)
        p6 = recurse(s2, t2, next_depth, variant, xp)
        p7 = recurse(s3, t3, next_depth, variant, xp)
        u1 = p1 + p2
        u2 = p1 + p6
        u3 = u2 + p7
        u4 = u2 + p5
        c11 = u1
        c12 = u4 + p3
        c21 = u3 - p4
        c22 = u3 + p5
    else:
        raise ValueError(f"unknown fast-matmul variant: {variant}")
    return xp.block([[c11, c12], [c21, c22]])


def fast_matmul(
    left: Any,
    weight: Any,
    recursion_depth: int,
    variant: str,
    schedule: str,
    xp: Any,
) -> Any:
    if schedule == "packed":
        return packed_fast_matmul(
            left,
            weight,
            recursion_depth,
            variant,
            xp,
        )
    if schedule == "depth_first":
        return depth_first_fast_matmul(
            left,
            weight,
            recursion_depth,
            variant,
            xp,
        )
    if schedule.startswith("hybrid_p"):
        packed_levels = int(schedule.removeprefix("hybrid_p"))
        if not 0 < packed_levels < recursion_depth:
            raise ValueError(
                "hybrid packed levels must be between zero and total depth"
            )
        if variant == "strassen":
            encoder = _encode_strassen
            decoder = _decode_strassen
        elif variant == "winograd":
            encoder = _encode_winograd
            decoder = _decode_winograd
        else:
            raise ValueError(
                f"unknown fast-matmul variant: {variant}"
            )
        a = left
        b = weight
        for _ in range(packed_levels):
            a, b = encoder(a, b, xp)
        products = depth_first_fast_matmul(
            a,
            b,
            recursion_depth - packed_levels,
            variant,
            xp,
        )
        for _ in range(packed_levels):
            products = decoder(products, xp)
        return products
    raise ValueError(f"unknown execution schedule: {schedule}")


def complete_prediction(
    weights: list[Any],
    rotation: Any,
    chirps: Any,
    variant: str,
    recursion_depth: int,
    schedule: str,
    layout: str,
    xp: Any,
) -> Any:
    """Submission-shaped prediction, including all currently billed ops."""
    if layout == "rectangular":
        activation = first_layer_design(
            weights[0],
            rotation,
            chirps,
            xp,
        )
        mean_axis: int | tuple[int, int] = 0
    elif layout == "square_batch":
        activation = first_layer_blocks(
            weights[0],
            rotation,
            chirps,
            xp,
        )
        mean_axis = (0, 1)
    else:
        raise ValueError(f"unknown activation layout: {layout}")
    for weight in weights[1:]:
        activation = fast_matmul(
            activation,
            weight,
            recursion_depth,
            variant,
            schedule,
            xp,
        )
        activation = xp.maximum(activation, 0.0)
    final_mean = xp.mean(
        activation.astype(xp.float64),
        axis=mean_axis,
    )
    first_mean = (
        xp.sqrt(xp.sum(weights[0] * weights[0], axis=0))
        * INV_SQRT_2PI
    )
    rows = [xp.zeros(WIDTH) for _ in range(DEPTH)]
    rows[0] = first_mean
    rows[-1] = final_mean
    return xp.stack(rows, axis=0)


def numpy_drift_trace(
    weights: np.ndarray,
    rotation: np.ndarray,
    chirps: np.ndarray,
    variant: str,
    recursion_depth: int,
    schedule: str,
    layout: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compare a fast forward with standard BLAS layer by layer."""
    if layout == "rectangular":
        baseline = first_layer_design(
            weights[0],
            rotation,
            chirps,
            np,
        )
        mean_axis: int | tuple[int, int] = 0
    elif layout == "square_batch":
        baseline = first_layer_blocks(
            weights[0],
            rotation,
            chirps,
            np,
        )
        mean_axis = (0, 1)
    else:
        raise ValueError(f"unknown activation layout: {layout}")
    fast = baseline.copy()
    per_layer: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for layer, weight in enumerate(weights[1:], start=1):
        baseline_pre = baseline @ weight
        fast_pre = fast_matmul(
            fast,
            weight,
            recursion_depth,
            variant,
            schedule,
            np,
        )
        mismatch = np.count_nonzero(
            (baseline_pre > 0.0) != (fast_pre > 0.0)
        )
        difference = fast_pre.astype(np.float64) - baseline_pre.astype(
            np.float64
        )
        baseline = np.maximum(baseline_pre, 0.0)
        fast = np.maximum(fast_pre, 0.0)
        per_layer.append(
            {
                "layer": layer,
                "preactivation_max_abs_difference": float(
                    np.max(np.abs(difference))
                ),
                "preactivation_rms_difference": float(
                    np.sqrt(np.mean(np.square(difference)))
                ),
                "gate_mismatches": int(mismatch),
                "gate_mismatch_fraction": float(mismatch / difference.size),
            }
        )
    elapsed = time.perf_counter() - started
    baseline_mean = baseline.astype(np.float64).mean(axis=mean_axis)
    fast_mean = fast.astype(np.float64).mean(axis=mean_axis)
    mean_difference = fast_mean - baseline_mean
    return fast_mean, {
        "elapsed_s": elapsed,
        "final_mean_max_abs_difference": float(
            np.max(np.abs(mean_difference))
        ),
        "final_mean_rms_difference": float(
            np.sqrt(np.mean(np.square(mean_difference)))
        ),
        "total_gate_mismatches": int(
            sum(int(row["gate_mismatches"]) for row in per_layer)
        ),
        "maximum_layer_gate_mismatch_fraction": float(
            max(float(row["gate_mismatch_fraction"]) for row in per_layer)
        ),
        "per_layer": per_layer,
    }


def flopscope_prediction(
    weights_np: np.ndarray,
    rotation_np: np.ndarray,
    chirps_np: np.ndarray,
    variant: str,
    recursion_depth: int,
    schedule: str,
    layout: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Execute one complete prediction under authoritative instrumentation."""
    with flopscope.BudgetContext(
        flop_budget=BUDGET,
        quiet=True,
    ) as context:
        weights = [
            fnp.asarray(weight).astype(fnp.float32)
            for weight in weights_np
        ]
        rotation = fnp.asarray(rotation_np)
        chirps = fnp.asarray(chirps_np)
        prediction = complete_prediction(
            weights,
            rotation,
            chirps,
            variant,
            recursion_depth,
            schedule,
            layout,
            fnp,
        )
    return np.asarray(prediction), context.summary_dict()


def microbenchmark(
    depths: list[int],
    variants: list[str],
    schedules: list[str],
    layouts: list[str],
    repeats: int,
    rows_count: int,
) -> list[dict[str, Any]]:
    """Benchmark one batch product independently of the MLP."""
    rng = np.random.default_rng(20260728)
    rectangular_np = rng.standard_normal(
        (rows_count, WIDTH),
        dtype=np.float32,
    )
    weight_np = (
        rng.standard_normal((WIDTH, WIDTH), dtype=np.float32)
        / math.sqrt(WIDTH)
    ).astype(np.float32)
    rows: list[dict[str, Any]] = []
    for layout in layouts:
        if layout == "rectangular":
            left_np = rectangular_np
        elif layout == "square_batch":
            if rows_count % WIDTH:
                raise ValueError(
                    "square_batch micro rows must be divisible by width"
                )
            left_np = rectangular_np.reshape(
                (rows_count // WIDTH, WIDTH, WIDTH)
            )
        else:
            raise ValueError(f"unknown activation layout: {layout}")
        reference = left_np @ weight_np
        for schedule in schedules:
            for variant in variants:
                for recursion_depth in depths:
                    samples: list[dict[str, Any]] = []
                    candidate_np = None
                    for _ in range(repeats):
                        with flopscope.BudgetContext(
                            flop_budget=10**15,
                            quiet=True,
                        ) as context:
                            candidate = fast_matmul(
                                fnp.asarray(left_np),
                                fnp.asarray(weight_np),
                                recursion_depth,
                                variant,
                                schedule,
                                fnp,
                            )
                        candidate_np = np.asarray(candidate)
                        samples.append(context.summary_dict())
                    if candidate_np is None:
                        raise AssertionError("microbenchmark ran no samples")
                    difference = (
                        candidate_np.astype(np.float64)
                        - reference.astype(np.float64)
                    )
                    flops = [
                        int(sample["flops_used"]) for sample in samples
                    ]
                    rows.append(
                        {
                            "layout": layout,
                            "schedule": schedule,
                            "variant": variant,
                            "recursion_depth": recursion_depth,
                            "leaf_size": WIDTH // (2**recursion_depth),
                            "leaf_products": 7**recursion_depth,
                            "tracked_flops": flops[0],
                            "tracked_flop_ratio_to_blas": float(
                                flops[0]
                                / (
                                    rows_count
                                    * WIDTH
                                    * (2 * WIDTH - 1)
                                )
                            ),
                            "wall_time_s_median": float(
                                np.median(
                                    [
                                        float(sample["wall_time_s"])
                                        for sample in samples
                                    ]
                                )
                            ),
                            "residual_wall_time_s_median": float(
                                np.median(
                                    [
                                        float(
                                            sample[
                                                "residual_wall_time_s"
                                            ]
                                        )
                                        for sample in samples
                                    ]
                                )
                            ),
                            "max_abs_difference": float(
                                np.max(np.abs(difference))
                            ),
                            "rms_difference": float(
                                np.sqrt(np.mean(np.square(difference)))
                            ),
                            "operation_breakdown": samples[-1][
                                "operations"
                            ],
                        }
                    )
                    print({"micro": rows[-1]}, flush=True)
    return rows


def evaluate_selection(
    indices: list[int],
    depths: list[int],
    variants: list[str],
    schedules: list[str],
    layouts: list[str],
    data: Path,
    rotation: np.ndarray,
    chirps: np.ndarray,
    profile_all: bool,
) -> list[dict[str, Any]]:
    """Run full 32-layer selection networks and compute adjusted scores."""
    if not indices or min(indices) < 0 or max(indices) >= 50:
        raise ValueError("selection audit is restricted to IDs 0--49")
    rows = _load_rows(data, indices)
    output: list[dict[str, Any]] = []
    for global_index, (name, weights, targets) in zip(indices, rows):
        target = targets[-1]
        for layout in layouts:
            baseline_prediction = complete_prediction(
                [weight.astype(np.float32) for weight in weights],
                rotation,
                chirps,
                "strassen",
                0,
                "packed",
                layout,
                np,
            )
            baseline_final = np.asarray(baseline_prediction[-1])
            baseline_mse = float(
                np.mean(np.square(baseline_final - target))
            )
            for schedule in schedules:
                for variant in variants:
                    for recursion_depth in depths:
                        fast_final, drift = numpy_drift_trace(
                            weights,
                            rotation,
                            chirps,
                            variant,
                            recursion_depth,
                            schedule,
                            layout,
                        )
                        fast_mse = float(
                            np.mean(np.square(fast_final - target))
                        )
                        profile: dict[str, Any] | None = None
                        if profile_all or global_index == indices[0]:
                            instrumented_prediction, summary = (
                                flopscope_prediction(
                                    weights,
                                    rotation,
                                    chirps,
                                    variant,
                                    recursion_depth,
                                    schedule,
                                    layout,
                                )
                            )
                            instrumented_final = instrumented_prediction[-1]
                            profile_mse = float(
                                np.mean(
                                    np.square(
                                        instrumented_final - target
                                    )
                                )
                            )
                            effective_compute = float(
                                int(summary["flops_used"])
                                + LAMBDA_FLOPS_PER_SECOND
                                * float(
                                    summary["residual_wall_time_s"]
                                )
                            )
                            multiplier = max(
                                0.1,
                                effective_compute / BUDGET,
                            )
                            profile = {
                                "tracked_flops": int(
                                    summary["flops_used"]
                                ),
                                "wall_time_s": float(
                                    summary["wall_time_s"]
                                ),
                                "backend_time_s": float(
                                    summary["flopscope_backend_time_s"]
                                ),
                                "overhead_time_s": float(
                                    summary[
                                        "flopscope_overhead_time_s"
                                    ]
                                ),
                                "residual_wall_time_s": float(
                                    summary["residual_wall_time_s"]
                                ),
                                "effective_compute": effective_compute,
                                "score_multiplier": multiplier,
                                "combined_budget_exhausted": bool(
                                    effective_compute > BUDGET
                                ),
                                "instrumented_raw_final_mse": profile_mse,
                                "instrumented_adjusted_score": (
                                    profile_mse * multiplier
                                ),
                                "instrumented_vs_numpy_max_abs_difference": float(
                                    np.max(
                                        np.abs(
                                            instrumented_final.astype(
                                                np.float64
                                            )
                                            - fast_final
                                        )
                                    )
                                ),
                                "operation_breakdown": summary[
                                    "operations"
                                ],
                            }
                        result = {
                            "index": global_index,
                            "name": name,
                            "layout": layout,
                            "schedule": schedule,
                            "variant": variant,
                            "recursion_depth": recursion_depth,
                            "baseline_raw_final_mse": baseline_mse,
                            "fast_raw_final_mse": fast_mse,
                            "raw_mse_delta": fast_mse - baseline_mse,
                            "drift": drift,
                            "flopscope_profile": profile,
                        }
                        output.append(result)
                        print(
                            {
                                "selection": {
                                    "index": global_index,
                                    "layout": layout,
                                    "schedule": schedule,
                                    "variant": variant,
                                    "depth": recursion_depth,
                                    "baseline_mse": baseline_mse,
                                    "fast_mse": fast_mse,
                                    "max_abs_drift": drift[
                                        "final_mean_max_abs_difference"
                                    ],
                                    "tracked_flops": (
                                        None
                                        if profile is None
                                        else profile["tracked_flops"]
                                    ),
                                    "multiplier": (
                                        None
                                        if profile is None
                                        else profile["score_multiplier"]
                                    ),
                                }
                            },
                            flush=True,
                        )
    return output


def summarise_selection(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str, str, int],
        list[dict[str, Any]],
    ] = {}
    for row in rows:
        key = (
            str(row["layout"]),
            str(row["schedule"]),
            str(row["variant"]),
            int(row["recursion_depth"]),
        )
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (
        layout,
        schedule,
        variant,
        recursion_depth,
    ), group in sorted(groups.items()):
        profiles = [
            row["flopscope_profile"]
            for row in group
            if row["flopscope_profile"] is not None
        ]
        # When only the first selection ID is instrumented, its exact FLOP
        # count is deterministic and its timing is the best available score
        # projection.  --profile-all produces the exact per-ID mean instead.
        projected_multiplier = float(
            np.mean(
                [
                    float(profile["score_multiplier"])
                    for profile in profiles
                ]
            )
        )
        raw_mses = [float(row["fast_raw_final_mse"]) for row in group]
        baseline_mses = [
            float(row["baseline_raw_final_mse"]) for row in group
        ]
        summaries.append(
            {
                "layout": layout,
                "schedule": schedule,
                "variant": variant,
                "recursion_depth": recursion_depth,
                "networks": len(group),
                "mean_baseline_raw_final_mse": float(
                    np.mean(baseline_mses)
                ),
                "mean_fast_raw_final_mse": float(np.mean(raw_mses)),
                "mean_raw_mse_delta": float(
                    np.mean(
                        [
                            float(row["raw_mse_delta"])
                            for row in group
                        ]
                    )
                ),
                "max_final_mean_abs_drift": float(
                    max(
                        float(
                            row["drift"][
                                "final_mean_max_abs_difference"
                            ]
                        )
                        for row in group
                    )
                ),
                "total_gate_mismatches": int(
                    sum(
                        int(row["drift"]["total_gate_mismatches"])
                        for row in group
                    )
                ),
                "profiles": len(profiles),
                "mean_tracked_flops": float(
                    np.mean(
                        [
                            int(profile["tracked_flops"])
                            for profile in profiles
                        ]
                    )
                ),
                "mean_residual_wall_time_s": float(
                    np.mean(
                        [
                            float(profile["residual_wall_time_s"])
                            for profile in profiles
                        ]
                    )
                ),
                "projected_score_multiplier": projected_multiplier,
                "projected_adjusted_score": float(
                    np.mean(raw_mses) * projected_multiplier
                ),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=list(range(10)),
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["strassen", "winograd"],
        default=["strassen", "winograd"],
    )
    parser.add_argument(
        "--schedules",
        nargs="+",
        choices=[
            "packed",
            "depth_first",
            "hybrid_p1",
            "hybrid_p2",
            "hybrid_p3",
            "hybrid_p4",
            "hybrid_p5",
            "hybrid_p6",
            "hybrid_p7",
        ],
        default=["packed"],
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=["rectangular", "square_batch"],
        default=["rectangular"],
    )
    parser.add_argument("--micro-repeats", type=int, default=2)
    parser.add_argument(
        "--micro-rows",
        type=int,
        default=N_POINTS,
        help="left-operand rows for the standalone kernel audit",
    )
    parser.add_argument(
        "--micro-only",
        action="store_true",
    )
    parser.add_argument(
        "--profile-all",
        action="store_true",
        help="Instrument every selection network, not only the first.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if (
        not args.indices
        or min(args.indices) < 0
        or max(args.indices) >= 50
    ):
        raise ValueError("this script is restricted to selection IDs 0--49")
    if any(depth < 1 or depth > 8 for depth in args.depths):
        raise ValueError("the audit accepts recursion depths 1--8")
    if args.micro_rows <= 0 or args.micro_rows % (2 ** max(args.depths)):
        raise ValueError(
            "micro rows must be positive and divisible by the deepest split"
        )

    asset = np.load(args.asset)
    rotation = np.asarray(asset["rotation"], dtype=np.float32)
    chirps = np.asarray(asset["chirps"], dtype=np.float32)

    micro = microbenchmark(
        args.depths,
        args.variants,
        args.schedules,
        args.layouts,
        args.micro_repeats,
        args.micro_rows,
    )
    selection: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    if not args.micro_only:
        selection = evaluate_selection(
            args.indices,
            args.depths,
            args.variants,
            args.schedules,
            args.layouts,
            args.data,
            rotation,
            chirps,
            args.profile_all,
        )
        summaries = summarise_selection(selection)
        print({"summaries": summaries}, flush=True)

    payload = {
        "method": (
            "Strassen/Winograd on rectangular or 258-square-block "
            "Kerdock activations"
        ),
        "selection_only": True,
        "indices": args.indices,
        "width": WIDTH,
        "depth": DEPTH,
        "points": N_POINTS,
        "square_blocks": N_BLOCKS,
        "micro_rows": args.micro_rows,
        "budget": BUDGET,
        "lambda_flops_per_second": LAMBDA_FLOPS_PER_SECOND,
        "microbenchmark": micro,
        "selection": selection,
        "summaries": summaries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print({"wrote": str(args.out)}, flush=True)


if __name__ == "__main__":
    main()
