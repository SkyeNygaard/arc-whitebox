"""Assembly-free quadtree audit for Winograd and alternative-basis Strassen.

Ordinary recursive implementations materialise every decoded 2x2 block with
``np.block``.  Flopscope correctly bills each assembly by the output size.
That movement is unnecessary: the decoded quadrants can remain a symbolic
quadtree until the complete product is available, then be assembled once.

For the Schwartz--Vaknin alternative basis, keeping the recursive phi/nu
transforms as trees also removes every intermediate basis-change assembly
while retaining their exact arithmetic.  This file compares both tensor
decompositions under identical packed/depth-first execution schedules.

Only selection ID 0 may be used for the optional numerical drift check.  No
holdout IDs are opened.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import flopscope
import flopscope.numpy as fnp
import numpy as np

from eval_sampling_official import DEFAULT_DATA, _load_rows
from eval_strassen_audit import (
    BUDGET,
    DEFAULT_ASSET,
    DEPTH,
    INV_SQRT_2PI,
    LAMBDA_FLOPS_PER_SECOND,
    N_POINTS,
    WIDTH,
    _decode_winograd,
    _encode_winograd,
    first_layer_design,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "tree_bilinear_audit.json"
Tree = Any


def tree_add(a: Tree, b: Tree, depth: int, xp: Any) -> Tree:
    if depth == 0:
        return a + b
    return tuple(
        tree_add(a[index], b[index], depth - 1, xp)
        for index in range(4)
    )


def tree_subtract(a: Tree, b: Tree, depth: int, xp: Any) -> Tree:
    if depth == 0:
        return a - b
    return tuple(
        tree_subtract(a[index], b[index], depth - 1, xp)
        for index in range(4)
    )


def tree_four_term(
    a: Tree,
    b: Tree,
    c: Tree,
    d: Tree,
    depth: int,
    signs: tuple[int, int, int, int],
    xp: Any,
) -> Tree:
    """A four-term expression evaluated with exactly three binary ops."""
    if depth == 0:
        result = a + b if signs[1] == 1 else a - b
        result = result + c if signs[2] == 1 else result - c
        result = result + d if signs[3] == 1 else result - d
        return result
    return tuple(
        tree_four_term(
            a[index],
            b[index],
            c[index],
            d[index],
            depth - 1,
            signs,
            xp,
        )
        for index in range(4)
    )


def raw_tree(values: Any, depth: int) -> Tree:
    if depth == 0:
        return values
    half_rows = values.shape[-2] // 2
    half_cols = values.shape[-1] // 2
    return (
        raw_tree(values[..., :half_rows, :half_cols], depth - 1),
        raw_tree(values[..., :half_rows, half_cols:], depth - 1),
        raw_tree(values[..., half_rows:, :half_cols], depth - 1),
        raw_tree(values[..., half_rows:, half_cols:], depth - 1),
    )


def phi_tree(values: Any, depth: int, xp: Any) -> Tree:
    """Recursive phi_opt without constructing intermediate matrices."""
    if depth == 0:
        return values
    half_rows = values.shape[-2] // 2
    half_cols = values.shape[-1] // 2
    q11 = phi_tree(
        values[..., :half_rows, :half_cols], depth - 1, xp
    )
    q12 = phi_tree(
        values[..., :half_rows, half_cols:], depth - 1, xp
    )
    q21 = phi_tree(
        values[..., half_rows:, :half_cols], depth - 1, xp
    )
    q22 = phi_tree(
        values[..., half_rows:, half_cols:], depth - 1, xp
    )
    transformed_22 = tree_add(
        tree_subtract(q12, q21, depth - 1, xp),
        q22,
        depth - 1,
        xp,
    )
    return q11, q12, q21, transformed_22


def nu_inverse_tree(values: Tree, depth: int, xp: Any) -> Tree:
    """Recursive nu_opt^-1 without constructing intermediate matrices."""
    if depth == 0:
        return values
    q11 = nu_inverse_tree(values[0], depth - 1, xp)
    q12 = nu_inverse_tree(values[1], depth - 1, xp)
    q21 = nu_inverse_tree(values[2], depth - 1, xp)
    q22 = nu_inverse_tree(values[3], depth - 1, xp)
    return (
        q11,
        tree_subtract(q12, q22, depth - 1, xp),
        tree_subtract(q22, q21, depth - 1, xp),
        q22,
    )


def encode_alternative_tree(
    left: Tree,
    right: Tree,
    depth: int,
    xp: Any,
) -> tuple[tuple[Tree, ...], tuple[Tree, ...]]:
    child_depth = depth - 1
    a11, a12, a21, a22 = left
    b11, b12, b21, b22 = right
    return (
        (
            a11,
            a12,
            a21,
            a22,
            tree_add(a21, a22, child_depth, xp),
            tree_subtract(a22, a12, child_depth, xp),
            tree_subtract(a22, a11, child_depth, xp),
        ),
        (
            b11,
            b21,
            tree_subtract(b22, b11, child_depth, xp),
            b22,
            tree_add(b21, b22, child_depth, xp),
            tree_subtract(b22, b12, child_depth, xp),
            b12,
        ),
    )


def decode_alternative_tree(
    products: Sequence[Tree],
    child_depth: int,
    xp: Any,
) -> Tree:
    m1, m2, m3, m4, m5, m6, m7 = products
    return (
        tree_add(m1, m2, child_depth, xp),
        tree_subtract(m5, m7, child_depth, xp),
        tree_add(m3, m6, child_depth, xp),
        tree_four_term(
            m5,
            m2,
            m4,
            m6,
            child_depth,
            (1, -1, -1, 1),
            xp,
        ),
    )


def encode_winograd_tree(
    left: Tree,
    right: Tree,
    depth: int,
    xp: Any,
) -> tuple[tuple[Tree, ...], tuple[Tree, ...]]:
    child_depth = depth - 1
    a11, a12, a21, a22 = left
    b11, b12, b21, b22 = right
    s1 = tree_add(a21, a22, child_depth, xp)
    s2 = tree_subtract(s1, a11, child_depth, xp)
    s3 = tree_subtract(a11, a21, child_depth, xp)
    s4 = tree_subtract(a12, s2, child_depth, xp)
    t1 = tree_subtract(b12, b11, child_depth, xp)
    t2 = tree_subtract(b22, t1, child_depth, xp)
    t3 = tree_subtract(b22, b12, child_depth, xp)
    t4 = tree_subtract(t2, b21, child_depth, xp)
    return (
        (a11, a12, s4, a22, s1, s2, s3),
        (b11, b21, b22, t4, t1, t2, t3),
    )


def decode_winograd_tree(
    products: Sequence[Tree],
    child_depth: int,
    xp: Any,
) -> Tree:
    p1, p2, p3, p4, p5, p6, p7 = products
    u1 = tree_add(p1, p2, child_depth, xp)
    u2 = tree_add(p1, p6, child_depth, xp)
    u3 = tree_add(u2, p7, child_depth, xp)
    u4 = tree_add(u2, p5, child_depth, xp)
    return (
        u1,
        tree_add(u4, p3, child_depth, xp),
        tree_subtract(u3, p4, child_depth, xp),
        tree_add(u3, p5, child_depth, xp),
    )


def stack_trees(
    trees: Sequence[Tree],
    depth: int,
    xp: Any,
) -> Tree:
    if depth == 0:
        return xp.stack(tuple(trees), axis=-3)
    return tuple(
        stack_trees(
            tuple(tree[index] for tree in trees),
            depth - 1,
            xp,
        )
        for index in range(4)
    )


def take_packed_branch(
    tree: Tree,
    depth: int,
    index: int,
) -> Tree:
    if depth == 0:
        return tree[..., index, :, :]
    return tuple(
        take_packed_branch(child, depth - 1, index)
        for child in tree
    )


def tree_depth_first(
    left: Tree,
    right: Tree,
    depth: int,
    variant: str,
    xp: Any,
) -> Tree:
    if depth == 0:
        return left @ right
    if variant == "alternative":
        encoded_left, encoded_right = encode_alternative_tree(
            left, right, depth, xp
        )
        decoder = decode_alternative_tree
    elif variant == "winograd":
        encoded_left, encoded_right = encode_winograd_tree(
            left, right, depth, xp
        )
        decoder = decode_winograd_tree
    else:
        raise ValueError(f"unknown variant {variant!r}")
    products = tuple(
        tree_depth_first(
            encoded_left[index],
            encoded_right[index],
            depth - 1,
            variant,
            xp,
        )
        for index in range(7)
    )
    return decoder(products, depth - 1, xp)


def winograd_depth_first_output_tree(
    left: Any,
    right: Any,
    depth: int,
    xp: Any,
) -> Tree:
    """Standard full-matrix encodes, but no intermediate output assemblies."""
    if depth == 0:
        return left @ right
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
    descend = winograd_depth_first_output_tree
    rest = depth - 1
    products = (
        descend(a11, b11, rest, xp),
        descend(a12, b21, rest, xp),
        descend(s4, b22, rest, xp),
        descend(a22, t4, rest, xp),
        descend(s1, t1, rest, xp),
        descend(s2, t2, rest, xp),
        descend(s3, t3, rest, xp),
    )
    return decode_winograd_tree(products, rest, xp)


def winograd_level_one_quadrants(
    left: Any,
    right: Any,
    xp: Any,
) -> tuple[Any, Any, Any, Any]:
    """Unrolled depth-one Winograd returning four views/arrays, not a block."""
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


def decode_winograd_quadrant_tuples(
    products: Sequence[tuple[Any, Any, Any, Any]],
) -> tuple[
    tuple[Any, Any, Any, Any],
    tuple[Any, Any, Any, Any],
    tuple[Any, Any, Any, Any],
    tuple[Any, Any, Any, Any],
]:
    """Unrolled tree decode for four leaf positions."""
    decoded = []
    for quadrant in range(4):
        p1, p2, p3, p4, p5, p6, p7 = (
            product[quadrant] for product in products
        )
        u1 = p1 + p2
        u2 = p1 + p6
        u3 = u2 + p7
        u4 = u2 + p5
        decoded.append((u1, u4 + p3, u3 - p4, u3 + p5))
    # Transpose from leaf-position-major to root-quadrant-major.
    return tuple(
        tuple(decoded[leaf][root] for leaf in range(4))
        for root in range(4)
    )  # type: ignore[return-value]


def winograd_depth_two_output_tree_unrolled(
    left: Any,
    right: Any,
    xp: Any,
) -> Tree:
    """Unrolled depth-two suffix used by the best p3/t2 schedule."""
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
        winograd_level_one_quadrants(a11, b11, xp),
        winograd_level_one_quadrants(a12, b21, xp),
        winograd_level_one_quadrants(s4, b22, xp),
        winograd_level_one_quadrants(a22, t4, xp),
        winograd_level_one_quadrants(s1, t1, xp),
        winograd_level_one_quadrants(s2, t2, xp),
        winograd_level_one_quadrants(s3, t3, xp),
    )
    return decode_winograd_quadrant_tuples(products)


def winograd_partial_output_tree_matmul(
    left: Any,
    right: Any,
    depth: int,
    packed_levels: int,
    output_tree_levels: int,
    xp: Any,
) -> Any:
    """Hybrid Winograd with the deepest decoded levels kept as a tree.

    ``output_tree_levels`` must cover the depth-first suffix.  Additional
    levels consume packed branch axes before the one and only tree assembly;
    any remaining outer levels use the conventional vectorised decoder.
    """
    suffix = depth - packed_levels
    if not 0 <= packed_levels <= depth:
        raise ValueError("invalid packed depth")
    if not suffix <= output_tree_levels <= depth:
        raise ValueError(
            "tree levels must cover the depth-first suffix and not exceed depth"
        )
    a = left
    b = right
    for _ in range(packed_levels):
        a, b = _encode_winograd(a, b, xp)
    if suffix == 2 and output_tree_levels == 2:
        product = winograd_depth_two_output_tree_unrolled(a, b, xp)
    else:
        product = winograd_depth_first_output_tree(a, b, suffix, xp)
    tree_depth = suffix
    packed_tree_levels = output_tree_levels - suffix
    for _ in range(packed_tree_levels):
        products = tuple(
            take_packed_branch(product, tree_depth, index)
            for index in range(7)
        )
        product = decode_winograd_tree(products, tree_depth, xp)
        tree_depth += 1
    product = xp.block(flatten_tree_grid(product, tree_depth))
    for _ in range(packed_levels - packed_tree_levels):
        product = _decode_winograd(product, xp)
    return product


def flatten_tree_grid(tree: Tree, depth: int) -> list[list[Any]]:
    """Return a 2**depth square grid for one final ``np.block`` call."""
    if depth == 0:
        return [[tree]]
    grids = [
        flatten_tree_grid(child, depth - 1)
        for child in tree
    ]
    half_rows = len(grids[0])
    top = [
        grids[0][row] + grids[1][row]
        for row in range(half_rows)
    ]
    bottom = [
        grids[2][row] + grids[3][row]
        for row in range(half_rows)
    ]
    return top + bottom


def tree_bilinear_matmul(
    left: Any,
    right: Any,
    depth: int,
    variant: str,
    packed_levels: int,
    xp: Any,
) -> Any:
    if not 0 <= packed_levels <= depth:
        raise ValueError("packed_levels must lie in [0, depth]")
    if variant == "alternative":
        a = phi_tree(left, depth, xp)
        b = phi_tree(right, depth, xp)
        encoder = encode_alternative_tree
        decoder = decode_alternative_tree
    elif variant == "winograd":
        a = raw_tree(left, depth)
        b = raw_tree(right, depth)
        encoder = encode_winograd_tree
        decoder = decode_winograd_tree
    else:
        raise ValueError(f"unknown variant {variant!r}")

    remaining_depth = depth
    for _ in range(packed_levels):
        encoded_a, encoded_b = encoder(
            a, b, remaining_depth, xp
        )
        remaining_depth -= 1
        a = stack_trees(encoded_a, remaining_depth, xp)
        b = stack_trees(encoded_b, remaining_depth, xp)
    product = tree_depth_first(
        a, b, remaining_depth, variant, xp
    )
    for _ in range(packed_levels):
        products = tuple(
            take_packed_branch(product, remaining_depth, index)
            for index in range(7)
        )
        product = decoder(products, remaining_depth, xp)
        remaining_depth += 1
    if variant == "alternative":
        product = nu_inverse_tree(product, depth, xp)
    return xp.block(flatten_tree_grid(product, depth))


def profile_micro(
    depth: int,
    variant: str,
    packed_levels: int,
    repeats: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(20260728)
    left = rng.standard_normal((N_POINTS, WIDTH), dtype=np.float32)
    right = (
        rng.standard_normal((WIDTH, WIDTH), dtype=np.float32)
        / math.sqrt(WIDTH)
    ).astype(np.float32)
    reference = left @ right
    summaries = []
    result_np = None
    for _ in range(repeats):
        with flopscope.BudgetContext(flop_budget=10**15, quiet=True) as ctx:
            result = tree_bilinear_matmul(
                fnp.asarray(left),
                fnp.asarray(right),
                depth,
                variant,
                packed_levels,
                fnp,
            )
        result_np = np.asarray(result)
        summaries.append(ctx.summary_dict())
    assert result_np is not None
    difference = result_np.astype(np.float64) - reference.astype(np.float64)
    summary = summaries[-1]
    residual = float(
        np.median(
            [
                float(item["residual_wall_time_s"])
                for item in summaries
            ]
        )
    )
    flops = int(summary["flops_used"])
    return {
        "depth": depth,
        "variant": variant,
        "packed_levels": packed_levels,
        "tracked_flops": flops,
        "ratio_to_dense": float(
            flops / (N_POINTS * WIDTH * (2 * WIDTH - 1))
        ),
        "residual_wall_time_s_median": residual,
        "effective_compute": float(
            flops + LAMBDA_FLOPS_PER_SECOND * residual
        ),
        "max_abs_difference": float(np.max(np.abs(difference))),
        "rms_difference": float(
            np.sqrt(np.mean(np.square(difference)))
        ),
        "operation_breakdown": summary["operations"],
    }


def profile_partial_micro(
    depth: int,
    packed_levels: int,
    output_tree_levels: int,
    repeats: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(20260728)
    left = rng.standard_normal((N_POINTS, WIDTH), dtype=np.float32)
    right = (
        rng.standard_normal((WIDTH, WIDTH), dtype=np.float32)
        / math.sqrt(WIDTH)
    ).astype(np.float32)
    reference = left @ right
    summaries = []
    result_np = None
    for _ in range(repeats):
        with flopscope.BudgetContext(flop_budget=10**15, quiet=True) as ctx:
            result = winograd_partial_output_tree_matmul(
                fnp.asarray(left),
                fnp.asarray(right),
                depth,
                packed_levels,
                output_tree_levels,
                fnp,
            )
        result_np = np.asarray(result)
        summaries.append(ctx.summary_dict())
    assert result_np is not None
    difference = result_np.astype(np.float64) - reference.astype(np.float64)
    summary = summaries[-1]
    residual = float(
        np.median(
            [
                float(item["residual_wall_time_s"])
                for item in summaries
            ]
        )
    )
    flops = int(summary["flops_used"])
    return {
        "depth": depth,
        "variant": "winograd_partial_output_tree",
        "packed_levels": packed_levels,
        "output_tree_levels": output_tree_levels,
        "tracked_flops": flops,
        "ratio_to_dense": float(
            flops / (N_POINTS * WIDTH * (2 * WIDTH - 1))
        ),
        "residual_wall_time_s_median": residual,
        "effective_compute": float(
            flops + LAMBDA_FLOPS_PER_SECOND * residual
        ),
        "max_abs_difference": float(np.max(np.abs(difference))),
        "rms_difference": float(
            np.sqrt(np.mean(np.square(difference)))
        ),
        "operation_breakdown": summary["operations"],
    }


def id0_drift(
    depth: int,
    variant: str,
    packed_levels: int,
    data: Path,
    rotation: np.ndarray,
    chirps: np.ndarray,
) -> dict[str, Any]:
    name, weights, targets = _load_rows(data, [0])[0]
    activation = first_layer_design(
        weights[0].astype(np.float32),
        rotation,
        chirps,
        np,
    )
    baseline = activation.copy()
    candidate = activation.copy()
    mismatches = 0
    started = time.perf_counter()
    for weight in weights[1:]:
        baseline_pre = baseline @ weight
        candidate_pre = tree_bilinear_matmul(
            candidate,
            weight,
            depth,
            variant,
            packed_levels,
            np,
        )
        mismatches += int(
            np.count_nonzero(
                (baseline_pre > 0) != (candidate_pre > 0)
            )
        )
        baseline = np.maximum(baseline_pre, 0)
        candidate = np.maximum(candidate_pre, 0)
    elapsed = time.perf_counter() - started
    baseline_mean = baseline.mean(axis=0, dtype=np.float64)
    candidate_mean = candidate.mean(axis=0, dtype=np.float64)
    difference = candidate_mean - baseline_mean
    target = targets[-1]
    return {
        "name": name,
        "depth": depth,
        "variant": variant,
        "packed_levels": packed_levels,
        "elapsed_s": elapsed,
        "baseline_mse": float(
            np.mean(np.square(baseline_mean - target))
        ),
        "candidate_mse": float(
            np.mean(np.square(candidate_mean - target))
        ),
        "mean_max_abs_difference": float(
            np.max(np.abs(difference))
        ),
        "mean_rms_difference": float(
            np.sqrt(np.mean(np.square(difference)))
        ),
        "total_gate_mismatches": mismatches,
    }


def id0_partial_drift_and_profile(
    depth: int,
    packed_levels: int,
    output_tree_levels: int,
    data: Path,
    rotation: np.ndarray,
    chirps: np.ndarray,
) -> dict[str, Any]:
    """Full-network NumPy drift plus authoritative flopscope prediction."""
    name, weights, targets = _load_rows(data, [0])[0]
    activation = first_layer_design(
        weights[0].astype(np.float32),
        rotation,
        chirps,
        np,
    )
    baseline = activation.copy()
    candidate = activation.copy()
    mismatches = 0
    started = time.perf_counter()
    for weight in weights[1:]:
        baseline_pre = baseline @ weight
        candidate_pre = winograd_partial_output_tree_matmul(
            candidate,
            weight,
            depth,
            packed_levels,
            output_tree_levels,
            np,
        )
        mismatches += int(
            np.count_nonzero(
                (baseline_pre > 0) != (candidate_pre > 0)
            )
        )
        baseline = np.maximum(baseline_pre, 0)
        candidate = np.maximum(candidate_pre, 0)
    elapsed = time.perf_counter() - started
    baseline_mean = baseline.mean(axis=0, dtype=np.float64)
    candidate_mean = candidate.mean(axis=0, dtype=np.float64)
    difference = candidate_mean - baseline_mean
    target = targets[-1]

    with flopscope.BudgetContext(
        flop_budget=BUDGET,
        quiet=True,
    ) as context:
        tracked_weights = [
            fnp.asarray(weight).astype(fnp.float32)
            for weight in weights
        ]
        tracked_rotation = fnp.asarray(rotation)
        tracked_chirps = fnp.asarray(chirps)
        tracked_activation = first_layer_design(
            tracked_weights[0],
            tracked_rotation,
            tracked_chirps,
            fnp,
        )
        for weight in tracked_weights[1:]:
            tracked_activation = winograd_partial_output_tree_matmul(
                tracked_activation,
                weight,
                depth,
                packed_levels,
                output_tree_levels,
                fnp,
            )
            tracked_activation = fnp.maximum(tracked_activation, 0.0)
        final_mean = fnp.mean(
            tracked_activation.astype(fnp.float64),
            axis=0,
        )
        first_mean = (
            fnp.sqrt(
                fnp.sum(
                    tracked_weights[0] * tracked_weights[0],
                    axis=0,
                )
            )
            * INV_SQRT_2PI
        )
        prediction_rows = [fnp.zeros(WIDTH) for _ in range(DEPTH)]
        prediction_rows[0] = first_mean
        prediction_rows[-1] = final_mean
        prediction = fnp.stack(prediction_rows, axis=0)
    summary = context.summary_dict()
    tracked_final = np.asarray(prediction)[-1]
    tracked_mse = float(np.mean(np.square(tracked_final - target)))
    effective = float(
        int(summary["flops_used"])
        + LAMBDA_FLOPS_PER_SECOND
        * float(summary["residual_wall_time_s"])
    )
    multiplier = max(0.1, effective / BUDGET)
    return {
        "name": name,
        "depth": depth,
        "variant": "winograd_partial_output_tree",
        "packed_levels": packed_levels,
        "output_tree_levels": output_tree_levels,
        "numpy_elapsed_s": elapsed,
        "baseline_mse": float(
            np.mean(np.square(baseline_mean - target))
        ),
        "candidate_mse": float(
            np.mean(np.square(candidate_mean - target))
        ),
        "mean_max_abs_difference": float(
            np.max(np.abs(difference))
        ),
        "mean_rms_difference": float(
            np.sqrt(np.mean(np.square(difference)))
        ),
        "total_gate_mismatches": mismatches,
        "flopscope": {
            "tracked_flops": int(summary["flops_used"]),
            "residual_wall_time_s": float(
                summary["residual_wall_time_s"]
            ),
            "effective_compute": effective,
            "score_multiplier": multiplier,
            "tracked_mse": tracked_mse,
            "adjusted_score": tracked_mse * multiplier,
            "instrumented_vs_numpy_max_abs_difference": float(
                np.max(
                    np.abs(
                        tracked_final.astype(np.float64)
                        - candidate_mean
                    )
                )
            ),
            "operation_breakdown": summary["operations"],
        },
    }


def parse_config(text: str) -> tuple[int, str, int]:
    depth_text, variant, packed_text = text.split(",")
    return int(depth_text), variant, int(packed_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "5,winograd,2",
            "5,winograd,3",
            "5,alternative,2",
            "5,alternative,3",
            "6,winograd,3",
            "6,alternative,3",
        ],
    )
    parser.add_argument(
        "--partial-configs",
        nargs="+",
        default=[],
        help="depth,packed_levels,output_tree_levels",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--drift-best", type=int, default=0)
    parser.add_argument(
        "--full-partial",
        default=None,
        help="profile ID0 for depth,packed_levels,output_tree_levels",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    configs = [parse_config(text) for text in args.configs]
    partial_configs = [
        tuple(int(part) for part in text.split(","))
        for text in args.partial_configs
    ]
    for depth, variant, packed in configs:
        if depth < 1 or depth > 8:
            raise ValueError(f"invalid depth {depth}")
        if variant not in {"winograd", "alternative"}:
            raise ValueError(f"invalid variant {variant}")
        if packed < 0 or packed > depth:
            raise ValueError(f"invalid packed depth {packed}")
    for config in partial_configs:
        if len(config) != 3:
            raise ValueError(f"invalid partial config {config}")
        depth, packed, tree_levels = config
        if not 0 <= packed <= depth:
            raise ValueError(f"invalid packed depth {config}")
        if not depth - packed <= tree_levels <= depth:
            raise ValueError(f"invalid output tree depth {config}")

    micro = []
    for config in configs:
        row = profile_micro(*config, args.repeats)
        micro.append(row)
        print({"micro": row}, flush=True)
    for config in partial_configs:
        row = profile_partial_micro(*config, args.repeats)
        micro.append(row)
        print({"micro": row}, flush=True)
    micro.sort(key=lambda row: float(row["effective_compute"]))

    drift = []
    if args.drift_best:
        asset = np.load(args.asset)
        rotation = np.asarray(asset["rotation"], dtype=np.float32)
        chirps = np.asarray(asset["chirps"], dtype=np.float32)
        for row in micro[: args.drift_best]:
            result = id0_drift(
                int(row["depth"]),
                str(row["variant"]),
                int(row["packed_levels"]),
                args.data,
                rotation,
                chirps,
            )
            drift.append(result)
            print({"drift": result}, flush=True)
    full_partial = None
    if args.full_partial is not None:
        config = tuple(
            int(part) for part in args.full_partial.split(",")
        )
        if len(config) != 3:
            raise ValueError("--full-partial needs depth,packed,tree")
        asset = np.load(args.asset)
        rotation = np.asarray(asset["rotation"], dtype=np.float32)
        chirps = np.asarray(asset["chirps"], dtype=np.float32)
        full_partial = id0_partial_drift_and_profile(
            *config,
            args.data,
            rotation,
            chirps,
        )
        print({"full_partial": full_partial}, flush=True)

    payload = {
        "method": "assembly-free quadtree bilinear recursion",
        "selection_only": True,
        "selection_id_limit": 50,
        "points": N_POINTS,
        "width": WIDTH,
        "micro": micro,
        "id0_drift": drift,
        "full_partial": full_partial,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print({"wrote": str(args.out)}, flush=True)


if __name__ == "__main__":
    main()
