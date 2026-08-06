"""Generalized sparse rank-7 decomposition with a five-coordinate basis.

This is the r=2 sparse-decomposition/reuse construction suggested by the
Beniamini--Schwartz and Cenk-style alternative-basis line of work.  Winograd's
four canonical quadrant coordinates are first embedded in a five-coordinate
dictionary.  The dictionary-to-rank maps need only two additions per operand,
and the rank-to-output-dictionary map needs four.  Crucially, recursive
canonical/dictionary transforms live on 4/5-ary trees rather than being paid
independently at every node of the 7-ary bilinear tree.

The production-shaped path keeps activations as 258 independent 256x256 tiles,
shares the encoded right operand, and chunks only the tall tile batch.  Every
instrumented operation uses flopscope.numpy.  Official data access is hard
restricted to selection IDs 0--49.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

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
    WIDTH,
    fast_matmul,
    first_layer_blocks,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "sparse_rank7_reuse_audit.json"
N_TILES = 258


def _take_axis(values: Any, axis: int, index: int) -> Any:
    selection = [slice(None)] * values.ndim
    selection[axis] = index
    return values[tuple(selection)]


def _quadrants(values: Any) -> tuple[Any, Any, Any, Any]:
    half_rows = values.shape[-2] // 2
    half_columns = values.shape[-1] // 2
    return (
        values[..., :half_rows, :half_columns],
        values[..., :half_rows, half_columns:],
        values[..., half_rows:, :half_columns],
        values[..., half_rows:, half_columns:],
    )


def _canonical_to_dictionary(
    values: Any,
    levels: int,
    side: str,
    xp: Any,
) -> Any:
    """Apply Phi_U^tensor-levels or Phi_V^tensor-levels breadth-first."""
    encoded = values
    for _ in range(levels):
        x11, x12, x21, x22 = _quadrants(encoded)
        if side == "left":
            # [u1,u2,u4,u5,u6] =
            # [a11,a12,a22,a21+a22,a21+a22-a11].
            shared = x21 + x22
            fifth = shared - x11
            forms = (x11, x12, x22, shared, fifth)
        elif side == "right":
            # [v1,v2,v3,v5,v6] =
            # [b11,b21,b22,b12-b11,b22-(b12-b11)].
            shared = x12 - x11
            fifth = x22 - shared
            forms = (x11, x21, x22, shared, fifth)
        else:
            raise ValueError(f"unknown dictionary side {side!r}")
        encoded = xp.stack(forms, axis=-3)
    return encoded


def _dictionary_rank_forms(
    values: Any,
    axis: int,
    side: str,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    x1 = _take_axis(values, axis, 0)
    x2 = _take_axis(values, axis, 1)
    x3 = _take_axis(values, axis, 2)
    x4 = _take_axis(values, axis, 3)
    x5 = _take_axis(values, axis, 4)
    if side == "left":
        # Recover old Winograd rows u3=u2-u6 and u7=u4-u6.
        return x1, x2, x2 - x5, x3, x4, x5, x3 - x5
    if side == "right":
        # Recover old Winograd rows v4=v6-v2 and v7=v6-v1.
        return x1, x2, x3, x5 - x2, x4, x5, x5 - x1
    raise ValueError(f"unknown bilinear side {side!r}")


def _pack_dictionary_axis(
    values: Any,
    axis: int,
    side: str,
    xp: Any,
) -> Any:
    return xp.stack(
        _dictionary_rank_forms(values, axis, side),
        axis=axis,
    )


def _rank_to_output_dictionary_forms(
    products: Any,
    axis: int | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    if axis is None:
        p1, p2, p3, p4, p5, p6, p7 = products
    else:
        p1 = _take_axis(products, axis, 0)
        p2 = _take_axis(products, axis, 1)
        p3 = _take_axis(products, axis, 2)
        p4 = _take_axis(products, axis, 3)
        p5 = _take_axis(products, axis, 4)
        p6 = _take_axis(products, axis, 5)
        p7 = _take_axis(products, axis, 6)
    y1 = p1 + p2
    y2 = p3 - p7
    y3 = p4
    y4 = p5
    y5 = (p1 + p6) + p7
    return y1, y2, y3, y4, y5


def _unpack_rank_axis(
    products: Any,
    axis: int,
    xp: Any,
) -> Any:
    return xp.stack(
        _rank_to_output_dictionary_forms(products, axis),
        axis=axis,
    )


def _bilinear_depth_first(
    left: Any,
    right: Any,
    remaining_levels: int,
    left_axis: int,
    right_axis: int,
    xp: Any,
) -> Any:
    if remaining_levels == 0:
        return left @ right
    left_forms = _dictionary_rank_forms(left, left_axis, "left")
    right_forms = _dictionary_rank_forms(right, right_axis, "right")
    products = tuple(
        _bilinear_depth_first(
            left_form,
            right_form,
            remaining_levels - 1,
            left_axis,
            right_axis,
            xp,
        )
        for left_form, right_form in zip(left_forms, right_forms)
    )
    return xp.stack(
        _rank_to_output_dictionary_forms(products),
        axis=left_axis,
    )


def _dictionary_to_canonical_block(values: Any, xp: Any) -> Any:
    """Apply Tau^T on the innermost basis leg and assemble quadrants."""
    y1 = values[..., 0, :, :]
    y2 = values[..., 1, :, :]
    y3 = values[..., 2, :, :]
    y4 = values[..., 3, :, :]
    y5 = values[..., 4, :, :]
    shared = y4 + y5
    c11 = y1
    c12 = y2 + shared
    c21 = y5 - y3
    c22 = shared
    return xp.block([[c11, c12], [c21, c22]])


def encode_right(
    right: Any,
    levels: int,
    packed_levels: int,
    xp: Any,
) -> Any:
    encoded = _canonical_to_dictionary(
        right,
        levels,
        "right",
        xp,
    )
    for axis in range(packed_levels):
        encoded = _pack_dictionary_axis(
            encoded,
            axis,
            "right",
            xp,
        )
    return encoded


def multiply_tile_chunk(
    left: Any,
    encoded_right: Any,
    levels: int,
    packed_levels: int,
    xp: Any,
) -> Any:
    encoded_left = _canonical_to_dictionary(
        left,
        levels,
        "left",
        xp,
    )
    for packed_axis in range(packed_levels):
        encoded_left = _pack_dictionary_axis(
            encoded_left,
            1 + packed_axis,
            "left",
            xp,
        )
    remaining = levels - packed_levels
    products = _bilinear_depth_first(
        encoded_left,
        encoded_right,
        remaining,
        1 + packed_levels,
        packed_levels,
        xp,
    )
    for packed_axis in range(packed_levels - 1, -1, -1):
        products = _unpack_rank_axis(
            products,
            1 + packed_axis,
            xp,
        )
    for _ in range(levels):
        products = _dictionary_to_canonical_block(products, xp)
    return products


def sparse_rank7_matmul(
    left_tiles: Any,
    right: Any,
    levels: int,
    packed_levels: int,
    tile_chunk: int,
    xp: Any,
) -> Any:
    if left_tiles.shape[-2:] != (WIDTH, WIDTH):
        raise ValueError("left operand must use (...,256,256) tiles")
    if not 0 <= packed_levels <= levels:
        raise ValueError("packed_levels must lie in 0..levels")
    encoded_right = encode_right(
        right,
        levels,
        packed_levels,
        xp,
    )
    outputs = [
        multiply_tile_chunk(
            left_tiles[offset : offset + tile_chunk],
            encoded_right,
            levels,
            packed_levels,
            xp,
        )
        for offset in range(0, left_tiles.shape[0], tile_chunk)
    ]
    if len(outputs) == 1:
        return outputs[0]
    return xp.concatenate(outputs, axis=0)


def _tree_add(left: Any, right: Any) -> Any:
    if isinstance(left, tuple):
        return tuple(
            _tree_add(left_child, right_child)
            for left_child, right_child in zip(left, right)
        )
    return left + right


def _tree_subtract(left: Any, right: Any) -> Any:
    if isinstance(left, tuple):
        return tuple(
            _tree_subtract(left_child, right_child)
            for left_child, right_child in zip(left, right)
        )
    return left - right


def _tree_map(function: Any, tree: Any) -> Any:
    if isinstance(tree, tuple):
        return tuple(_tree_map(function, child) for child in tree)
    return function(tree)


def _canonical_to_dictionary_tree(
    values: Any,
    levels: int,
    side: str,
) -> Any:
    """Tuple-tree Phi transform with arithmetic but no materializing stacks."""
    if levels == 0:
        return values
    x11, x12, x21, x22 = _quadrants(values)
    if side == "left":
        shared = x21 + x22
        fifth = shared - x11
        forms = (x11, x12, x22, shared, fifth)
    elif side == "right":
        shared = x12 - x11
        fifth = x22 - shared
        forms = (x11, x21, x22, shared, fifth)
    else:
        raise ValueError(f"unknown dictionary side {side!r}")
    return tuple(
        _canonical_to_dictionary_tree(form, levels - 1, side)
        for form in forms
    )


def _dictionary_to_rank_tree(
    dictionary_tree: Any,
    packed_levels: int,
    side: str,
) -> Any:
    """Convert prefix dictionary legs to a nested seven-branch tree."""
    if packed_levels == 0:
        return dictionary_tree
    x1, x2, x3, x4, x5 = dictionary_tree
    if side == "left":
        forms = (
            x1,
            x2,
            _tree_subtract(x2, x5),
            x3,
            x4,
            x5,
            _tree_subtract(x3, x5),
        )
    elif side == "right":
        forms = (
            x1,
            x2,
            x3,
            _tree_subtract(x5, x2),
            x4,
            x5,
            _tree_subtract(x5, x1),
        )
    else:
        raise ValueError(f"unknown bilinear side {side!r}")
    return tuple(
        _dictionary_to_rank_tree(
            form,
            packed_levels - 1,
            side,
        )
        for form in forms
    )


def _follow_path(tree: Any, path: tuple[int, ...]) -> Any:
    node = tree
    for index in path:
        node = node[index]
    return node


def _stack_rank_prefix_by_suffix(
    rank_tree: Any,
    packed_levels: int,
    suffix_levels: int,
    stack_axis: int,
    xp: Any,
) -> Any:
    """Transpose a rank^p-of-dictionary^q tree and stack rank only once."""
    rank_paths = tuple(
        itertools.product(range(7), repeat=packed_levels)
    )

    def build(suffix_path: tuple[int, ...], remaining: int) -> Any:
        if remaining:
            return tuple(
                build(suffix_path + (index,), remaining - 1)
                for index in range(5)
            )
        arrays = tuple(
            _follow_path(
                _follow_path(rank_tree, rank_path),
                suffix_path,
            )
            for rank_path in rank_paths
        )
        if len(arrays) == 1:
            return arrays[0]
        return xp.stack(arrays, axis=stack_axis)

    return build((), suffix_levels)


def _encode_tuple_operand(
    values: Any,
    levels: int,
    packed_levels: int,
    side: str,
    stack_axis: int,
    xp: Any,
) -> Any:
    dictionary = _canonical_to_dictionary_tree(
        values,
        levels,
        side,
    )
    rank_tree = _dictionary_to_rank_tree(
        dictionary,
        packed_levels,
        side,
    )
    return _stack_rank_prefix_by_suffix(
        rank_tree,
        packed_levels,
        levels - packed_levels,
        stack_axis,
        xp,
    )


def _suffix_bilinear(
    left_tree: Any,
    right_tree: Any,
    remaining_levels: int,
) -> Any:
    if remaining_levels == 0:
        return left_tree @ right_tree
    l1, l2, l3, l4, l5 = left_tree
    r1, r2, r3, r4, r5 = right_tree
    left_forms = (
        l1,
        l2,
        _tree_subtract(l2, l5),
        l3,
        l4,
        l5,
        _tree_subtract(l3, l5),
    )
    right_forms = (
        r1,
        r2,
        r3,
        _tree_subtract(r5, r2),
        r4,
        r5,
        _tree_subtract(r5, r1),
    )
    p1, p2, p3, p4, p5, p6, p7 = tuple(
        _suffix_bilinear(
            left_form,
            right_form,
            remaining_levels - 1,
        )
        for left_form, right_form in zip(left_forms, right_forms)
    )
    y1 = _tree_add(p1, p2)
    y2 = _tree_subtract(p3, p7)
    y3 = p4
    y4 = p5
    y5 = _tree_add(_tree_add(p1, p6), p7)
    return y1, y2, y3, y4, y5


def _rank_stride(
    values: Any,
    axis: int,
    rank_index: int,
    final_axis: bool,
) -> Any:
    selection = [slice(None)] * values.ndim
    selection[axis] = (
        rank_index
        if final_axis
        else slice(rank_index, None, 7)
    )
    return values[tuple(selection)]


def _decode_flat_rank_prefix(
    output_tree: Any,
    packed_levels: int,
    rank_axis: int,
) -> Any:
    """Decode a flattened 7^p batch into a nested 5^p tuple tree."""
    tree = output_tree
    for level in range(packed_levels):
        final_axis = level == packed_levels - 1
        p1, p2, p3, p4, p5, p6, p7 = tuple(
            _tree_map(
                lambda values, rank_index=rank_index: _rank_stride(
                    values,
                    rank_axis,
                    rank_index,
                    final_axis,
                ),
                tree,
            )
            for rank_index in range(7)
        )
        y1 = _tree_add(p1, p2)
        y2 = _tree_subtract(p3, p7)
        y3 = p4
        y4 = p5
        y5 = _tree_add(_tree_add(p1, p6), p7)
        tree = (y1, y2, y3, y4, y5)
    return tree


def _dictionary_tree_to_canonical(
    tree: Any,
    levels: int,
    xp: Any,
) -> Any:
    if levels == 0:
        return tree
    y1, y2, y3, y4, y5 = tree
    shared = _tree_add(y4, y5)
    c11_tree = y1
    c12_tree = _tree_add(y2, shared)
    c21_tree = _tree_subtract(y5, y3)
    c22_tree = shared
    c11 = _dictionary_tree_to_canonical(
        c11_tree,
        levels - 1,
        xp,
    )
    c12 = _dictionary_tree_to_canonical(
        c12_tree,
        levels - 1,
        xp,
    )
    c21 = _dictionary_tree_to_canonical(
        c21_tree,
        levels - 1,
        xp,
    )
    c22 = _dictionary_tree_to_canonical(
        c22_tree,
        levels - 1,
        xp,
    )
    return xp.block([[c11, c12], [c21, c22]])


def multiply_tile_chunk_tuple(
    left: Any,
    encoded_right_tree: Any,
    levels: int,
    packed_levels: int,
    xp: Any,
) -> Any:
    encoded_left_tree = _encode_tuple_operand(
        left,
        levels,
        packed_levels,
        "left",
        1,
        xp,
    )
    output_suffix_tree = _suffix_bilinear(
        encoded_left_tree,
        encoded_right_tree,
        levels - packed_levels,
    )
    output_dictionary_tree = _decode_flat_rank_prefix(
        output_suffix_tree,
        packed_levels,
        1,
    )
    return _dictionary_tree_to_canonical(
        output_dictionary_tree,
        levels,
        xp,
    )


def sparse_rank7_matmul_tuple(
    left_tiles: Any,
    right: Any,
    levels: int,
    packed_levels: int,
    tile_chunk: int,
    xp: Any,
) -> Any:
    encoded_right_tree = _encode_tuple_operand(
        right,
        levels,
        packed_levels,
        "right",
        0,
        xp,
    )
    outputs = [
        multiply_tile_chunk_tuple(
            left_tiles[offset : offset + tile_chunk],
            encoded_right_tree,
            levels,
            packed_levels,
            xp,
        )
        for offset in range(0, left_tiles.shape[0], tile_chunk)
    ]
    if len(outputs) == 1:
        return outputs[0]
    return xp.concatenate(outputs, axis=0)


def sparse_outer_winograd_matmul(
    left_tiles: Any,
    right: Any,
    sparse_levels: int,
    total_levels: int,
    inner_packed_levels: int,
    tile_chunk: int,
    xp: Any,
) -> Any:
    """Use sparse reuse on outer levels and ordinary Winograd on leaves."""
    if not 1 <= sparse_levels <= total_levels:
        raise ValueError("sparse levels must lie in 1..total levels")
    inner_levels = total_levels - sparse_levels
    if not 0 <= inner_packed_levels <= inner_levels:
        raise ValueError("invalid inner packed prefix")
    encoded_right = _encode_tuple_operand(
        right,
        sparse_levels,
        sparse_levels,
        "right",
        0,
        xp,
    )
    outputs = []
    for offset in range(0, left_tiles.shape[0], tile_chunk):
        encoded_left = _encode_tuple_operand(
            left_tiles[offset : offset + tile_chunk],
            sparse_levels,
            sparse_levels,
            "left",
            1,
            xp,
        )
        if inner_levels == 0:
            products = encoded_left @ encoded_right
        else:
            if inner_packed_levels == 0:
                schedule = "depth_first"
            elif inner_packed_levels == inner_levels:
                schedule = "packed"
            else:
                schedule = f"hybrid_p{inner_packed_levels}"
            products = fast_matmul(
                encoded_left,
                encoded_right,
                inner_levels,
                "winograd",
                schedule,
                xp,
            )
        output_tree = _decode_flat_rank_prefix(
            products,
            sparse_levels,
            1,
        )
        outputs.append(
            _dictionary_tree_to_canonical(
                output_tree,
                sparse_levels,
                xp,
            )
        )
    if len(outputs) == 1:
        return outputs[0]
    return xp.concatenate(outputs, axis=0)


def sparse_outer_winograd_rectangular(
    left: Any,
    right: Any,
    sparse_levels: int,
    total_levels: int,
    inner_packed_levels: int,
    xp: Any,
) -> Any:
    """Direct tall variant with one shared flattened outer-rank axis."""
    inner_levels = total_levels - sparse_levels
    encoded_left = _encode_tuple_operand(
        left,
        sparse_levels,
        sparse_levels,
        "left",
        0,
        xp,
    )
    encoded_right = _encode_tuple_operand(
        right,
        sparse_levels,
        sparse_levels,
        "right",
        0,
        xp,
    )
    if inner_levels == 0:
        products = encoded_left @ encoded_right
    else:
        if inner_packed_levels == 0:
            schedule = "depth_first"
        elif inner_packed_levels == inner_levels:
            schedule = "packed"
        else:
            schedule = f"hybrid_p{inner_packed_levels}"
        products = fast_matmul(
            encoded_left,
            encoded_right,
            inner_levels,
            "winograd",
            schedule,
            xp,
        )
    output_tree = _decode_flat_rank_prefix(
        products,
        sparse_levels,
        0,
    )
    return _dictionary_tree_to_canonical(
        output_tree,
        sparse_levels,
        xp,
    )


def small_correctness() -> list[dict[str, float | int]]:
    rng = np.random.default_rng(726_551)
    records: list[dict[str, float | int]] = []
    for levels in (1, 2, 3):
        size = 2**levels
        left = rng.standard_normal((3, size, size))
        right = rng.standard_normal((size, size))
        dense = left @ right
        for packed_levels in range(levels + 1):
            encoded_right = encode_right(
                right,
                levels,
                packed_levels,
                np,
            )
            result = multiply_tile_chunk(
                left,
                encoded_right,
                levels,
                packed_levels,
                np,
            )
            difference = result - dense
            records.append(
                {
                    "levels": levels,
                    "packed_levels": packed_levels,
                    "max_abs_error": float(
                        np.max(np.abs(difference))
                    ),
                    "rms_error": float(
                        np.sqrt(np.mean(np.square(difference)))
                    ),
                }
            )
    return records


def profile_layer(
    activation_np: np.ndarray,
    weight_np: np.ndarray,
    levels: int,
    packed_levels: int,
    tile_chunk: int,
    implementation: str = "stacked",
) -> dict[str, object]:
    dense = activation_np @ weight_np
    kernel = (
        sparse_rank7_matmul_tuple
        if implementation == "tuple"
        else sparse_rank7_matmul
    )
    with flopscope.BudgetContext(
        flop_budget=BUDGET,
        quiet=True,
    ) as context:
        started = time.perf_counter()
        result = kernel(
            fnp.asarray(activation_np),
            fnp.asarray(weight_np),
            levels,
            packed_levels,
            tile_chunk,
            fnp,
        )
        elapsed = time.perf_counter() - started
        summary = context.summary_dict()
    difference = np.asarray(result) - dense
    residual = float(summary["residual_wall_time_s"])
    effective = int(summary["flops_used"]) + (
        LAMBDA_FLOPS_PER_SECOND * residual
    )
    return {
        "implementation": implementation,
        "levels": levels,
        "packed_levels": packed_levels,
        "tile_chunk": tile_chunk,
        "tracked_flops": int(summary["flops_used"]),
        "wall_time_s": elapsed,
        "backend_time_s": float(summary["flopscope_backend_time_s"]),
        "overhead_time_s": float(summary["flopscope_overhead_time_s"]),
        "residual_wall_time_s": residual,
        "effective_compute": effective,
        "max_abs_error": float(np.max(np.abs(difference))),
        "rms_error": float(np.sqrt(np.mean(np.square(difference)))),
        "operations": summary["operations"],
    }


def profile_layer_mixed(
    activation_np: np.ndarray,
    weight_np: np.ndarray,
    sparse_levels: int,
    total_levels: int,
    inner_packed_levels: int,
    tile_chunk: int,
) -> dict[str, object]:
    dense = activation_np @ weight_np
    with flopscope.BudgetContext(
        flop_budget=BUDGET,
        quiet=True,
    ) as context:
        started = time.perf_counter()
        result = sparse_outer_winograd_matmul(
            fnp.asarray(activation_np),
            fnp.asarray(weight_np),
            sparse_levels,
            total_levels,
            inner_packed_levels,
            tile_chunk,
            fnp,
        )
        elapsed = time.perf_counter() - started
        summary = context.summary_dict()
    difference = np.asarray(result) - dense
    residual = float(summary["residual_wall_time_s"])
    effective = int(summary["flops_used"]) + (
        LAMBDA_FLOPS_PER_SECOND * residual
    )
    return {
        "implementation": "sparse_outer_winograd_inner",
        "sparse_levels": sparse_levels,
        "total_levels": total_levels,
        "inner_packed_levels": inner_packed_levels,
        "tile_chunk": tile_chunk,
        "tracked_flops": int(summary["flops_used"]),
        "wall_time_s": elapsed,
        "backend_time_s": float(summary["flopscope_backend_time_s"]),
        "overhead_time_s": float(summary["flopscope_overhead_time_s"]),
        "residual_wall_time_s": residual,
        "effective_compute": effective,
        "max_abs_error": float(np.max(np.abs(difference))),
        "rms_error": float(np.sqrt(np.mean(np.square(difference)))),
        "operations": summary["operations"],
    }


def full_network_profile(
    weights_np: np.ndarray,
    target: np.ndarray,
    rotation_np: np.ndarray,
    chirps_np: np.ndarray,
    levels: int,
    packed_levels: int,
    tile_chunk: int,
    implementation: str = "stacked",
) -> dict[str, object]:
    kernel = (
        sparse_rank7_matmul_tuple
        if implementation == "tuple"
        else sparse_rank7_matmul
    )
    with flopscope.BudgetContext(
        flop_budget=BUDGET,
        quiet=True,
    ) as context:
        weights = [
            fnp.asarray(weight).astype(fnp.float32)
            for weight in weights_np
        ]
        activation = first_layer_blocks(
            weights[0],
            fnp.asarray(rotation_np),
            fnp.asarray(chirps_np),
            fnp,
        )
        for weight in weights[1:]:
            activation = fnp.maximum(
                kernel(
                    activation,
                    weight,
                    levels,
                    packed_levels,
                    tile_chunk,
                    fnp,
                ),
                0.0,
            )
        final_mean = fnp.mean(
            activation.astype(fnp.float64),
            axis=(0, 1),
        )
        first_mean = (
            fnp.sqrt(
                fnp.sum(weights[0] * weights[0], axis=0)
            )
            * INV_SQRT_2PI
        )
        rows = [fnp.zeros(WIDTH) for _ in range(DEPTH)]
        rows[0] = first_mean
        rows[-1] = final_mean
        prediction = fnp.stack(rows, axis=0)
        summary = context.summary_dict()
    raw_mse = float(
        np.mean(np.square(np.asarray(prediction[-1]) - target))
    )
    residual = float(summary["residual_wall_time_s"])
    effective = int(summary["flops_used"]) + (
        LAMBDA_FLOPS_PER_SECOND * residual
    )
    multiplier = min(1.0, effective / BUDGET)
    return {
        "implementation": implementation,
        "levels": levels,
        "packed_levels": packed_levels,
        "tile_chunk": tile_chunk,
        "raw_final_mse": raw_mse,
        "tracked_flops": int(summary["flops_used"]),
        "wall_time_s": float(summary["wall_time_s"]),
        "backend_time_s": float(summary["flopscope_backend_time_s"]),
        "overhead_time_s": float(summary["flopscope_overhead_time_s"]),
        "residual_wall_time_s": residual,
        "effective_compute": effective,
        "score_multiplier": multiplier,
        "adjusted_score": raw_mse * multiplier,
        "combined_budget_exhausted": effective > BUDGET,
        "operations": summary["operations"],
    }


def _parse_schedules(
    values: list[str],
    implementation: str,
) -> list[tuple[int, int, int, str]]:
    schedules = []
    for value in values:
        levels_text, packed_text, chunk_text = value.split(":")
        levels = int(levels_text)
        packed = int(packed_text)
        chunk = int(chunk_text)
        if not 1 <= levels <= 8:
            raise ValueError("levels must lie in 1..8")
        if not 0 <= packed <= levels:
            raise ValueError("packed levels must lie in 0..levels")
        if not 1 <= chunk <= N_TILES:
            raise ValueError("tile chunk must lie in 1..258")
        schedules.append((levels, packed, chunk, implementation))
    return schedules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--schedule",
        action="append",
        default=[],
        help="levels:packed_levels:tile_chunk; repeat to sweep",
    )
    parser.add_argument(
        "--tuple-schedule",
        action="append",
        default=[],
        help="copy-minimized tuple-tree levels:packed:chunk schedule",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--profile-network", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.index < 50:
        raise ValueError("audit is restricted to selection IDs 0--49")
    default_schedules = (
        [
            "5:2:16",
            "5:3:16",
            "5:4:16",
            "6:3:8",
            "6:4:8",
            "6:5:8",
        ]
        if not args.schedule and not args.tuple_schedule
        else []
    )
    schedules = (
        _parse_schedules(
            args.schedule or default_schedules,
            "stacked",
        )
        + _parse_schedules(args.tuple_schedule, "tuple")
    )
    correctness = small_correctness()
    if max(row["max_abs_error"] for row in correctness) > 1e-9:
        raise AssertionError("small sparse-decomposition audit failed")

    name, weights, targets = _load_rows(args.data, [args.index])[0]
    asset = np.load(args.asset)
    rotation = asset["rotation"].astype(np.float32)
    chirps = asset["chirps"].astype(np.float32)
    activation = first_layer_blocks(
        weights[0].astype(np.float32),
        rotation,
        chirps,
        np,
    )
    profiles = [
        profile_layer(
            activation,
            weights[1].astype(np.float32),
            levels,
            packed,
            chunk,
            implementation,
        )
        for levels, packed, chunk, implementation in schedules
    ]
    feasible = [
        profile
        for profile in profiles
        if 31 * float(profile["wall_time_s"]) < 29.5
    ]
    candidates = feasible or profiles
    best = min(
        candidates,
        key=lambda profile: (
            float(profile["tracked_flops"])
            + 31
            * LAMBDA_FLOPS_PER_SECOND
            * float(profile["residual_wall_time_s"])
        ),
    )
    network = None
    if args.profile_network:
        network = full_network_profile(
            weights,
            targets[-1],
            rotation,
            chirps,
            int(best["levels"]),
            int(best["packed_levels"]),
            int(best["tile_chunk"]),
            str(best["implementation"]),
        )
    result = {
        "protocol": {
            "selection_index": args.index,
            "selection_name": name,
            "holdout_loaded": False,
            "flopscope_version": flopscope.__version__,
            "schedule_format": "levels:packed_levels:tile_chunk",
        },
        "small_float64_correctness": correctness,
        "layer_profiles": profiles,
        "best_schedule": {
            "levels": best["levels"],
            "packed_levels": best["packed_levels"],
            "tile_chunk": best["tile_chunk"],
            "implementation": best["implementation"],
        },
        "full_network_profile": network,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
