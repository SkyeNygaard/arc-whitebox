"""Audit tracked Strassen propagation for the Kerdock activation matrix.

The Kerdock design has 66,048 = 258 * 256 rows.  Consequently every deep
layer can be viewed as 258 independent 256-by-256 matrix products with the
same weight matrix.  Recursive Strassen multiplication reduces the charged
scalar arithmetic, potentially freeing budget for additional angular points.

This script measures:

* exact Flopscope 0.9.1 cost of one or more 256-row tiles;
* numerical disagreement with NumPy's dense float32 matmul;
* end-to-end drift and target MSE on selection IDs only.

It deliberately does not use hidden NumPy computation inside a tracked
prediction.  Every multiplication and addition in ``strassen_tiles`` is
performed through the array namespace passed to it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import flopscope
import flopscope.numpy as fnp
import numpy as np

from eval_kerdock_design import (
    WIDTH,
    make_kerdock_design,
    random_rotation,
)
from eval_sampling_official import DEFAULT_DATA, _load_rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "kerdock_strassen_audit.json"
POINTS = 66_048


def _strassen_parallel(
    left: Any,
    right: Any,
    levels: int,
    xp: Any,
) -> Any:
    """Multiply parallel square problems with vectorized Strassen branches.

    ``left`` has shape ``(batch, problems, n, n)`` and ``right`` has shape
    ``(problems, n, n)``.  The output has the same leading shapes as left.
    """
    if levels == 0:
        return left @ right[None, :, :, :]

    n = left.shape[-1]
    half = n // 2
    a11 = left[..., :half, :half]
    a12 = left[..., :half, half:]
    a21 = left[..., half:, :half]
    a22 = left[..., half:, half:]
    b11 = right[..., :half, :half]
    b12 = right[..., :half, half:]
    b21 = right[..., half:, :half]
    b22 = right[..., half:, half:]

    left_children = xp.stack(
        (
            a11 + a22,
            a21 + a22,
            a11,
            a22,
            a11 + a12,
            a21 - a11,
            a12 - a22,
        ),
        axis=2,
    )
    right_children = xp.stack(
        (
            b11 + b22,
            b11,
            b12 - b22,
            b21 - b11,
            b22,
            b11 + b12,
            b21 + b22,
        ),
        axis=1,
    )
    batch = left.shape[0]
    problems = left.shape[1]
    products = _strassen_parallel(
        left_children.reshape((batch, problems * 7, half, half)),
        right_children.reshape((problems * 7, half, half)),
        levels - 1,
        xp,
    ).reshape((batch, problems, 7, half, half))

    m1 = products[:, :, 0]
    m2 = products[:, :, 1]
    m3 = products[:, :, 2]
    m4 = products[:, :, 3]
    m5 = products[:, :, 4]
    m6 = products[:, :, 5]
    m7 = products[:, :, 6]
    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6
    top = xp.concatenate((c11, c12), axis=-1)
    bottom = xp.concatenate((c21, c22), axis=-1)
    return xp.concatenate((top, bottom), axis=-2)


def strassen_tiles(
    activation: Any,
    weight: Any,
    levels: int,
    tile_batch: int,
    xp: Any,
) -> Any:
    if activation.shape[0] % WIDTH != 0:
        raise ValueError("row count must be a multiple of 256")
    tiles = activation.reshape((-1, WIDTH, WIDTH))
    right = weight.reshape((1, WIDTH, WIDTH))
    outputs = []
    for offset in range(0, tiles.shape[0], tile_batch):
        left = tiles[offset : offset + tile_batch, None, :, :]
        product = _strassen_parallel(left, right, levels, xp)
        outputs.append(product[:, 0])
    return xp.concatenate(outputs, axis=0).reshape((-1, WIDTH))


def theoretical_tile_cost(levels: int, size: int = WIDTH) -> int:
    """Flopscope cost per left tile, excluding shared right-side additions."""
    if levels == 0:
        return size * size * (2 * size - 1)
    half = size // 2
    # Five left input sums and eight output sums, each charged 2 FLOPs/item.
    return 7 * theoretical_tile_cost(levels - 1, half) + (
        2 * 13 * half * half
    )


def theoretical_shared_right_cost(levels: int, size: int = WIDTH) -> int:
    if levels == 0:
        return 0
    half = size // 2
    # Five right input sums at this node, each charged 2 FLOPs/item.
    return (
        7 * theoretical_shared_right_cost(levels - 1, half)
        + 2 * 5 * half * half
    )


def profile_one_layer(
    activation: np.ndarray,
    weight: np.ndarray,
    levels: int,
    tile_batch: int,
) -> dict[str, object]:
    rows = min(
        activation.shape[0],
        tile_batch * WIDTH,
    )
    rows -= rows % WIDTH
    sample = activation[:rows]
    tracked_activation = fnp.asarray(sample)
    tracked_weight = fnp.asarray(weight)
    with flopscope.BudgetContext(flop_budget=272_000_000_000) as context:
        result = strassen_tiles(
            tracked_activation,
            tracked_weight,
            levels,
            tile_batch,
            fnp,
        )
        summary = context.summary_dict()
    dense = sample @ weight
    difference = np.asarray(result) - dense
    tiles = rows // WIDTH
    predicted = (
        tiles * theoretical_tile_cost(levels)
        + theoretical_shared_right_cost(levels)
    )
    return {
        "rows": rows,
        "tiles": tiles,
        "levels": levels,
        "flopscope_flops_used": int(summary["flops_used"]),
        "theoretical_flops": predicted,
        "flops_agree": int(summary["flops_used"]) == predicted,
        "dense_flops": rows * WIDTH * (2 * WIDTH - 1),
        "max_abs_error": float(np.max(np.abs(difference))),
        "rms_error": float(np.sqrt(np.mean(np.square(difference)))),
        "operations": summary["operations"],
    }


def end_to_end(
    weights: np.ndarray,
    target: np.ndarray,
    rotation_seed: int,
    levels: int,
    tile_batch: int,
) -> dict[str, float]:
    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, rotation_seed)
    first = np.maximum(points @ (rotation @ weights[0]), 0.0)

    start = time.perf_counter()
    dense = first
    for weight in weights[1:]:
        dense = np.maximum(dense @ weight, 0.0)
    dense_prediction = dense.mean(axis=0, dtype=np.float64)
    dense_seconds = time.perf_counter() - start

    start = time.perf_counter()
    fast = first
    for weight in weights[1:]:
        fast = np.maximum(
            strassen_tiles(
                fast,
                weight,
                levels,
                tile_batch,
                np,
            ),
            0.0,
        )
    fast_prediction = fast.mean(axis=0, dtype=np.float64)
    fast_seconds = time.perf_counter() - start
    difference = fast_prediction - dense_prediction
    return {
        "dense_seconds": dense_seconds,
        "strassen_seconds": fast_seconds,
        "dense_final_mse": float(
            np.mean(np.square(dense_prediction - target))
        ),
        "strassen_final_mse": float(
            np.mean(np.square(fast_prediction - target))
        ),
        "prediction_max_abs_error": float(np.max(np.abs(difference))),
        "prediction_rms_error": float(
            np.sqrt(np.mean(np.square(difference)))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--rotation-seed", type=int, default=3)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--tile-batch", type=int, default=16)
    parser.add_argument("--end-to-end-level", type=int, default=4)
    parser.add_argument("--skip-end-to-end", action="store_true")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not 0 <= args.index < 50:
        raise ValueError("Strassen audit is restricted to selection IDs 0--49")
    if any(level < 0 or level > 8 for level in args.levels):
        raise ValueError("levels must lie in 0--8")

    name, weights, targets = _load_rows(args.data, [args.index])[0]
    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    first = np.maximum(points @ (rotation @ weights[0]), 0.0)
    profiles = [
        profile_one_layer(
            first,
            weights[1],
            level,
            args.tile_batch,
        )
        for level in args.levels
    ]
    for profile in profiles:
        print({"profile": profile}, flush=True)
    full = None
    if not args.skip_end_to_end:
        full = end_to_end(
            weights,
            targets[-1],
            args.rotation_seed,
            args.end_to_end_level,
            args.tile_batch,
        )
        print({"end_to_end": full}, flush=True)

    level_costs = {}
    for level in range(9):
        deep_cost = (
            31 * 258 * theoretical_tile_cost(level)
            + 31 * theoretical_shared_right_cost(level)
        )
        level_costs[str(level)] = {
            "deep_31_flops": deep_cost,
            "ratio_to_classical": (
                deep_cost
                / (
                    31
                    * POINTS
                    * WIDTH
                    * (2 * WIDTH - 1)
                )
            ),
        }
    payload = {
        "protocol": {
            "selection_index": args.index,
            "name": name,
            "holdout_loaded": False,
            "rotation_seed": args.rotation_seed,
            "flopscope_version": getattr(flopscope, "__version__", "unknown"),
        },
        "profiles": profiles,
        "level_costs": level_costs,
        "end_to_end": full,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print({"out": str(args.out)}, flush=True)


if __name__ == "__main__":
    main()
