"""Persistent canonical-block Winograd for the Kerdock MLP.

The 66,048 design rows are kept as 258 square 256x256 tiles.  Across hidden
layers each tile is represented by a canonical Morton-ordered tensor of 64
32x32 blocks (three recursive 2x2 levels).  The outer three Winograd levels
consume and produce that representation directly:

* no 256x256 output assembly after a layer;
* no re-slicing that dense output before the next layer;
* one tensorwise ReLU on the canonical blocks;
* the final mean is reduced directly from blocks.

Only the two 32x32 suffix levels are assembled, once, while their packed
outer branch axes are still present.  The result is then decoded back to the
64-block canonical tensor with no intermediate block materialisation.

The optional full-network check is hard-restricted to selection ID 0.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import flopscope
import flopscope.numpy as fnp
import numpy as np

from eval_sampling_official import DEFAULT_DATA, _load_rows
from eval_strassen_audit import (
    BUDGET,
    DEFAULT_ASSET,
    DEPTH,
    INV_SQRT_2PI,
    KERDOCK_BASES,
    LAMBDA_FLOPS_PER_SECOND,
    N_POINTS,
    WIDTH,
    first_layer_design,
)
from eval_tree_bilinear import (
    decode_winograd_tree,
    flatten_tree_grid,
    raw_tree,
    take_packed_branch,
    winograd_depth_two_output_tree_unrolled,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "persistent_block_winograd.json"
CANONICAL_LEVELS = 3
BLOCKS_PER_AXIS = 2**CANONICAL_LEVELS
CANONICAL_BLOCKS = 4**CANONICAL_LEVELS
BLOCK_SIZE = WIDTH // BLOCKS_PER_AXIS


def flatten_tree_leaves(tree: Any, depth: int) -> list[Any]:
    if depth == 0:
        return [tree]
    leaves: list[Any] = []
    for child in tree:
        leaves.extend(flatten_tree_leaves(child, depth - 1))
    return leaves


def dense_to_canonical(
    values: Any,
    levels: int,
    xp: Any,
) -> Any:
    """Stack Morton-ordered block views into one canonical tensor."""
    tree = raw_tree(values, levels)
    leaves = flatten_tree_leaves(tree, levels)
    return xp.stack(tuple(leaves), axis=-3)


def canonical_encode_winograd(
    left: Any,
    right: Any,
    levels: int,
    xp: Any,
) -> tuple[Any, Any]:
    """Pack Winograd branches while consuming a flat spatial-block axis."""
    a = left
    b = right
    remaining = levels
    for _ in range(levels):
        leaf_count = 4**remaining
        quarter = leaf_count // 4
        a11 = a[..., :quarter, :, :]
        a12 = a[..., quarter : 2 * quarter, :, :]
        a21 = a[..., 2 * quarter : 3 * quarter, :, :]
        a22 = a[..., 3 * quarter :, :, :]
        b11 = b[..., :quarter, :, :]
        b12 = b[..., quarter : 2 * quarter, :, :]
        b21 = b[..., 2 * quarter : 3 * quarter, :, :]
        b22 = b[..., 3 * quarter :, :, :]

        s1 = a21 + a22
        s2 = s1 - a11
        s3 = a11 - a21
        s4 = a12 - s2
        t1 = b12 - b11
        t2 = b22 - t1
        t3 = b22 - b12
        t4 = t2 - b21
        # Insert a new branch axis immediately before the surviving spatial
        # block axis (the latter is always third from the end).
        a = xp.stack(
            (a11, a12, s4, a22, s1, s2, s3),
            axis=-4,
        )
        b = xp.stack(
            (b11, b21, b22, t4, t1, t2, t3),
            axis=-4,
        )
        remaining -= 1
    return a[..., 0, :, :], b[..., 0, :, :]


def decode_outer_to_canonical(
    product: Any,
    levels: int,
    mode: str,
    xp: Any,
) -> Any:
    """Decode packed branch axes to one persistent canonical-block tensor."""
    # A flat Morton tuple avoids thousands of Python calls from recursively
    # walking the same tree for every scalar-array addition.
    leaves: tuple[Any, ...] = (product,)
    flat_levels = levels if mode == "flat" else levels - 1
    for _ in range(flat_levels):
        c11s = []
        c12s = []
        c21s = []
        c22s = []
        for leaf in leaves:
            p1 = leaf[..., 0, :, :]
            p2 = leaf[..., 1, :, :]
            p3 = leaf[..., 2, :, :]
            p4 = leaf[..., 3, :, :]
            p5 = leaf[..., 4, :, :]
            p6 = leaf[..., 5, :, :]
            p7 = leaf[..., 6, :, :]
            u1 = p1 + p2
            u2 = p1 + p6
            u3 = u2 + p7
            u4 = u2 + p5
            c11s.append(u1)
            c12s.append(u4 + p3)
            c21s.append(u3 - p4)
            c22s.append(u3 + p5)
        leaves = tuple(c11s + c12s + c21s + c22s)
    if mode == "group_last":
        # Group the 16 existing spatial leaves once.  The final packed decode
        # is then seven tensor operations instead of 16*7 small operations.
        spatial = xp.stack(leaves, axis=-3)
        p1 = spatial[..., 0, :, :, :]
        p2 = spatial[..., 1, :, :, :]
        p3 = spatial[..., 2, :, :, :]
        p4 = spatial[..., 3, :, :, :]
        p5 = spatial[..., 4, :, :, :]
        p6 = spatial[..., 5, :, :, :]
        p7 = spatial[..., 6, :, :, :]
        u1 = p1 + p2
        u2 = p1 + p6
        u3 = u2 + p7
        u4 = u2 + p5
        return xp.concatenate(
            (u1, u4 + p3, u3 - p4, u3 + p5),
            axis=-3,
        )
    if mode != "flat":
        raise ValueError(f"unknown decode mode {mode!r}")
    return xp.stack(leaves, axis=-3)


def persistent_block_matmul(
    canonical_left: Any,
    dense_right: Any,
    decode_mode: str,
    xp: Any,
) -> Any:
    """One exact 256x256 product in persistent 64-block representation."""
    canonical_right = dense_to_canonical(
        dense_right,
        CANONICAL_LEVELS,
        xp,
    )
    encoded_left, encoded_right = canonical_encode_winograd(
        canonical_left,
        canonical_right,
        CANONICAL_LEVELS,
        xp,
    )
    suffix_tree = winograd_depth_two_output_tree_unrolled(
        encoded_left,
        encoded_right,
        xp,
    )
    # One assembly for the two-level 32x32 suffix, with all 7^3 outer
    # branches and 258 sample tiles vectorised in the leading dimensions.
    packed_product = xp.block(flatten_tree_grid(suffix_tree, 2))
    return decode_outer_to_canonical(
        packed_product,
        CANONICAL_LEVELS,
        decode_mode,
        xp,
    )


def canonical_final_mean(canonical: Any, xp: Any) -> Any:
    """Reduce batch/row coordinates and restore the 256 output columns."""
    # Each of the 64 Morton leaves contributes a 32-vector averaged over its
    # 8,256 rows.
    leaf_means = xp.mean(canonical.astype(xp.float64), axis=-2)

    def tree_from_flat(values: Any, depth: int, offset: int = 0) -> Any:
        if depth == 0:
            return values[offset]
        stride = 4 ** (depth - 1)
        return tuple(
            tree_from_flat(
                values,
                depth - 1,
                offset + quadrant * stride,
            )
            for quadrant in range(4)
        )

    tree = tree_from_flat(leaf_means, CANONICAL_LEVELS)

    # At a tree node, horizontally adjacent quadrants address disjoint output
    # columns while vertically adjacent quadrants address additional rows.
    def reduce_rows(node: Any, depth: int) -> list[Any]:
        if depth == 0:
            return [node]
        q11, q12, q21, q22 = node
        top_left = reduce_rows(q11, depth - 1)
        top_right = reduce_rows(q12, depth - 1)
        bottom_left = reduce_rows(q21, depth - 1)
        bottom_right = reduce_rows(q22, depth - 1)
        # Each final output column averages equally over all eight block rows.
        left = [
            (top + bottom) * 0.5
            for top, bottom in zip(top_left, bottom_left)
        ]
        right = [
            (top + bottom) * 0.5
            for top, bottom in zip(top_right, bottom_right)
        ]
        return left + right

    column_blocks = reduce_rows(tree, CANONICAL_LEVELS)
    return xp.concatenate(tuple(column_blocks), axis=0)


def micro_profile(repeats: int, decode_mode: str) -> dict[str, Any]:
    rng = np.random.default_rng(20260728)
    left_dense = rng.standard_normal(
        (N_POINTS, WIDTH),
        dtype=np.float32,
    )
    right = (
        rng.standard_normal((WIDTH, WIDTH), dtype=np.float32)
        / math.sqrt(WIDTH)
    ).astype(np.float32)
    reference = left_dense @ right
    summaries = []
    candidate_np = None
    for _ in range(repeats):
        with flopscope.BudgetContext(flop_budget=10**15, quiet=True) as ctx:
            canonical = dense_to_canonical(
                fnp.asarray(left_dense),
                CANONICAL_LEVELS,
                fnp,
            )
            candidate = persistent_block_matmul(
                canonical,
                fnp.asarray(right),
                decode_mode,
                fnp,
            )
        candidate_np = np.asarray(candidate)
        summaries.append(ctx.summary_dict())
    assert candidate_np is not None
    reference_canonical = dense_to_canonical(
        reference, CANONICAL_LEVELS, np
    )
    difference = (
        candidate_np.astype(np.float64)
        - reference_canonical.astype(np.float64)
    )
    summary = summaries[-1]
    residual = float(
        np.median(
            [
                float(row["residual_wall_time_s"])
                for row in summaries
            ]
        )
    )
    flops = int(summary["flops_used"])
    return {
        "canonical_levels": CANONICAL_LEVELS,
        "decode_mode": decode_mode,
        "canonical_blocks": CANONICAL_BLOCKS,
        "block_size": BLOCK_SIZE,
        "tracked_flops_including_initial_canonicalization": flops,
        "steady_state_tracked_flops": flops - N_POINTS * WIDTH,
        "residual_wall_time_s_median": residual,
        "effective_compute_including_initial_canonicalization": float(
            flops + LAMBDA_FLOPS_PER_SECOND * residual
        ),
        "max_abs_difference": float(np.max(np.abs(difference))),
        "rms_difference": float(
            np.sqrt(np.mean(np.square(difference)))
        ),
        "operation_breakdown": summary["operations"],
    }


def full_id0(
    data: Path,
    rotation: np.ndarray,
    chirps: np.ndarray,
    decode_mode: str,
) -> dict[str, Any]:
    name, weights, targets = _load_rows(data, [0])[0]
    first = first_layer_design(
        weights[0].astype(np.float32),
        rotation,
        chirps,
        np,
    )
    dense = first.copy()
    canonical = dense_to_canonical(first, CANONICAL_LEVELS, np)
    started = time.perf_counter()
    for weight in weights[1:]:
        dense = np.maximum(dense @ weight, 0.0)
        canonical = np.maximum(
            persistent_block_matmul(
                canonical, weight, decode_mode, np
            ),
            0.0,
        )
    numpy_elapsed = time.perf_counter() - started
    dense_mean = dense.mean(axis=0, dtype=np.float64)
    canonical_mean = canonical_final_mean(canonical, np)
    target = targets[-1]
    delta = canonical_mean - dense_mean

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
        tracked_first = first_layer_design(
            tracked_weights[0],
            tracked_rotation,
            tracked_chirps,
            fnp,
        )
        tracked_canonical = dense_to_canonical(
            tracked_first,
            CANONICAL_LEVELS,
            fnp,
        )
        for weight in tracked_weights[1:]:
            tracked_canonical = fnp.maximum(
                persistent_block_matmul(
                    tracked_canonical,
                    weight,
                    decode_mode,
                    fnp,
                ),
                0.0,
            )
        final_mean = canonical_final_mean(tracked_canonical, fnp)
        first_mean = (
            fnp.sqrt(
                fnp.sum(
                    tracked_weights[0] * tracked_weights[0],
                    axis=0,
                )
            )
            * INV_SQRT_2PI
        )
        rows = [fnp.zeros(WIDTH) for _ in range(DEPTH)]
        rows[0] = first_mean
        rows[-1] = final_mean
        prediction = fnp.stack(rows, axis=0)
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
        "decode_mode": decode_mode,
        "numpy_elapsed_s": numpy_elapsed,
        "dense_mse": float(np.mean(np.square(dense_mean - target))),
        "canonical_mse": float(
            np.mean(np.square(canonical_mean - target))
        ),
        "mean_max_abs_difference": float(np.max(np.abs(delta))),
        "mean_rms_difference": float(
            np.sqrt(np.mean(np.square(delta)))
        ),
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
                        - canonical_mean
                    )
                )
            ),
            "operation_breakdown": summary["operations"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--decode-mode",
        choices=["flat", "group_last"],
        default="group_last",
    )
    parser.add_argument("--full-id0", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    micro = micro_profile(args.repeats, args.decode_mode)
    print({"micro": micro}, flush=True)
    result = None
    if args.full_id0:
        asset = np.load(args.asset)
        rotation = np.asarray(asset["rotation"], dtype=np.float32)
        chirps = np.asarray(asset["chirps"], dtype=np.float32)
        result = full_id0(
            args.data,
            rotation,
            chirps,
            args.decode_mode,
        )
        print({"full_id0": result}, flush=True)
    payload = {
        "method": "persistent 64-block canonical Winograd",
        "selection_only": True,
        "selection_id_limit": 50,
        "points": N_POINTS,
        "width": WIDTH,
        "micro": micro,
        "full_id0": result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print({"wrote": str(args.out)}, flush=True)


if __name__ == "__main__":
    main()
