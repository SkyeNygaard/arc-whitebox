"""Exact alternative-basis Strassen audit for Kerdock deep layers.

Implements the Schwartz--Vaknin optimal-basis decomposition from equation
(3.1) of:

    O. Schwartz and N. Vaknin, "Pebbling Game and Alternative Basis for
    High Performance Matrix Multiplication", SIAM J. Sci. Comput. 2024.
    https://doi.org/10.1137/22M1502719

The decomposition has a 12-addition bilinear phase (3 left encodes, 3 right
encodes, 6 decodes), compared with 18 additions for ordinary Strassen.  It
pays recursive input/output basis transforms with two additions per 2x2
block.  This script implements partial recursion ending in classical matmul,
validates exactness against BLAS, and records authoritative Flopscope 0.9.1
costs.  It is hard-restricted to selection IDs 0--49.
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

from eval_kerdock_design import (
    FIELD_SIZE,
    WIDTH,
    kerdock_chirp,
    make_kerdock_design,
    random_rotation,
)
from eval_sampling_official import DEFAULT_DATA, _load_rows
from eval_spherical_stein_cv import sphere_radius_mean


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = (
    ROOT / "results" / "alternative_basis_strassen_audit.json"
)
N_POINTS = 66_048
TILES = N_POINTS // WIDTH


def flopscope_fwht_batch(values: fnp.ndarray) -> fnp.ndarray:
    span = 1
    while span < WIDTH:
        grouped = values.reshape(
            (FIELD_SIZE, WIDTH // (2 * span), 2, span, WIDTH)
        )
        left = grouped[:, :, 0]
        right = grouped[:, :, 1]
        values = fnp.stack(
            (left + right, left - right),
            axis=2,
        ).reshape((FIELD_SIZE, WIDTH, WIDTH))
        span *= 2
    return values


def structured_first_layer_flopscope(
    first_weight: fnp.ndarray,
    rotation: fnp.ndarray,
    chirps: fnp.ndarray,
) -> fnp.ndarray:
    radius = sphere_radius_mean(WIDTH)
    effective_weight = rotation @ first_weight
    weighted = chirps[:, :, None] * effective_weight[None, :, :]
    transformed = flopscope_fwht_batch(weighted) * (
        radius / math.sqrt(WIDTH)
    )
    kerdock_rows = fnp.stack(
        (transformed, -transformed),
        axis=2,
    ).reshape((-1, WIDTH))
    coordinate_rows = fnp.stack(
        (radius * effective_weight, -radius * effective_weight),
        axis=1,
    ).reshape((-1, WIDTH))
    return fnp.maximum(
        fnp.concatenate((kerdock_rows, coordinate_rows), axis=0),
        0.0,
    )


def recursive_basis_transform(
    values: Any,
    levels: int,
    transform: str,
    xp: Any,
) -> Any:
    """Apply recursive phi_opt or nu_opt^{-1}.

    ``values`` has shape ``(batch, problems, n, n)``.  Row-order 2x2 block
    vectorization is used, matching equation (3.1).
    """
    if levels == 0:
        return values
    n = values.shape[-1]
    half = n // 2
    a11 = values[..., :half, :half]
    a12 = values[..., :half, half:]
    a21 = values[..., half:, :half]
    a22 = values[..., half:, half:]
    batch = values.shape[0]
    problems = values.shape[1]
    children = xp.stack((a11, a12, a21, a22), axis=2).reshape(
        (batch, problems * 4, half, half)
    )
    transformed = recursive_basis_transform(
        children,
        levels - 1,
        transform,
        xp,
    ).reshape((batch, problems, 4, half, half))
    q11 = transformed[:, :, 0]
    q12 = transformed[:, :, 1]
    q21 = transformed[:, :, 2]
    q22 = transformed[:, :, 3]
    if transform == "phi":
        # phi_opt [a11,a12,a21,a22]^T =
        # [a11,a12,a21,a12-a21+a22]^T.
        o11 = q11
        o12 = q12
        o21 = q21
        o22 = q12 - q21 + q22
    elif transform == "nu_inverse":
        # nu_opt^{-1} [c11,c12,c21,c22]^T =
        # [c11,c12-c22,c22-c21,c22]^T.
        o11 = q11
        o12 = q12 - q22
        o21 = q22 - q21
        o22 = q22
    else:
        raise ValueError(f"unknown basis transform {transform!r}")
    top = xp.concatenate((o11, o12), axis=-1)
    bottom = xp.concatenate((o21, o22), axis=-1)
    return xp.concatenate((top, bottom), axis=-2)


def alternative_bilinear(
    left: Any,
    right: Any,
    levels: int,
    xp: Any,
) -> Any:
    """Recursive U_opt/V_opt/W_opt bilinear phase.

    ``left`` has shape ``(batch, problems, n, n)`` and ``right`` has shape
    ``(problems, n, n)``.  Right encodings are shared across all left tiles.
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

    # Rows of U_opt and V_opt from equation (3.1).
    left_children = xp.stack(
        (
            a11,
            a12,
            a21,
            a22,
            a21 + a22,
            a22 - a12,
            a22 - a11,
        ),
        axis=2,
    )
    right_children = xp.stack(
        (
            b11,
            b21,
            b22 - b11,
            b22,
            b21 + b22,
            b22 - b12,
            b12,
        ),
        axis=1,
    )
    batch = left.shape[0]
    problems = left.shape[1]
    products = alternative_bilinear(
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

    # W_opt^T decode.
    c11 = m1 + m2
    c12 = m5 - m7
    c21 = m3 + m6
    c22 = m5 - m2 - m4 + m6
    top = xp.concatenate((c11, c12), axis=-1)
    bottom = xp.concatenate((c21, c22), axis=-1)
    return xp.concatenate((top, bottom), axis=-2)


def recursive_basis_transform_sequential(
    values: Any,
    levels: int,
    transform: str,
    xp: Any,
) -> Any:
    """Low-wrapper-overhead version without branch stacks/reshapes."""
    if levels == 0:
        return values
    n = values.shape[-1]
    half = n // 2
    q11 = recursive_basis_transform_sequential(
        values[..., :half, :half],
        levels - 1,
        transform,
        xp,
    )
    q12 = recursive_basis_transform_sequential(
        values[..., :half, half:],
        levels - 1,
        transform,
        xp,
    )
    q21 = recursive_basis_transform_sequential(
        values[..., half:, :half],
        levels - 1,
        transform,
        xp,
    )
    q22 = recursive_basis_transform_sequential(
        values[..., half:, half:],
        levels - 1,
        transform,
        xp,
    )
    if transform == "phi":
        o11, o12, o21 = q11, q12, q21
        o22 = q12 - q21 + q22
    elif transform == "nu_inverse":
        o11 = q11
        o12 = q12 - q22
        o21 = q22 - q21
        o22 = q22
    else:
        raise ValueError(f"unknown basis transform {transform!r}")
    return xp.concatenate(
        (
            xp.concatenate((o11, o12), axis=-1),
            xp.concatenate((o21, o22), axis=-1),
        ),
        axis=-2,
    )


def alternative_bilinear_sequential(
    left: Any,
    right: Any,
    levels: int,
    xp: Any,
) -> Any:
    """12-addition bilinear phase without branch materialization."""
    if levels == 0:
        return left @ right
    n = left.shape[-1]
    half = n // 2
    a11 = left[..., :half, :half]
    a12 = left[..., :half, half:]
    a21 = left[..., half:, :half]
    a22 = left[..., half:, half:]
    b11 = right[:half, :half]
    b12 = right[:half, half:]
    b21 = right[half:, :half]
    b22 = right[half:, half:]
    recurse = alternative_bilinear_sequential
    m1 = recurse(a11, b11, levels - 1, xp)
    m2 = recurse(a12, b21, levels - 1, xp)
    m3 = recurse(a21, b22 - b11, levels - 1, xp)
    m4 = recurse(a22, b22, levels - 1, xp)
    m5 = recurse(a21 + a22, b21 + b22, levels - 1, xp)
    m6 = recurse(a22 - a12, b22 - b12, levels - 1, xp)
    m7 = recurse(a22 - a11, b12, levels - 1, xp)
    c11 = m1 + m2
    c12 = m5 - m7
    c21 = m3 + m6
    c22 = m5 - m2 - m4 + m6
    return xp.concatenate(
        (
            xp.concatenate((c11, c12), axis=-1),
            xp.concatenate((c21, c22), axis=-1),
        ),
        axis=-2,
    )


def alternative_basis_tiles_sequential(
    activation: Any,
    weight: Any,
    levels: int,
    xp: Any,
) -> Any:
    if activation.shape[0] % WIDTH != 0:
        raise ValueError("row count must be divisible by 256")
    tiles = activation.reshape((-1, WIDTH, WIDTH))
    transformed_weight = recursive_basis_transform_sequential(
        weight,
        levels,
        "phi",
        xp,
    )
    transformed_left = recursive_basis_transform_sequential(
        tiles,
        levels,
        "phi",
        xp,
    )
    transformed_product = alternative_bilinear_sequential(
        transformed_left,
        transformed_weight,
        levels,
        xp,
    )
    product = recursive_basis_transform_sequential(
        transformed_product,
        levels,
        "nu_inverse",
        xp,
    )
    return product.reshape((-1, WIDTH))


def alternative_basis_tiles(
    activation: Any,
    weight: Any,
    levels: int,
    tile_batch: int,
    xp: Any,
) -> Any:
    if activation.shape[0] % WIDTH != 0:
        raise ValueError("row count must be divisible by 256")
    tiles = activation.reshape((-1, WIDTH, WIDTH))
    transformed_weight = recursive_basis_transform(
        weight.reshape((1, 1, WIDTH, WIDTH)),
        levels,
        "phi",
        xp,
    )[0]
    outputs = []
    for offset in range(0, tiles.shape[0], tile_batch):
        left = tiles[offset : offset + tile_batch, None]
        transformed_left = recursive_basis_transform(
            left,
            levels,
            "phi",
            xp,
        )
        transformed_product = alternative_bilinear(
            transformed_left,
            transformed_weight,
            levels,
            xp,
        )
        product = recursive_basis_transform(
            transformed_product,
            levels,
            "nu_inverse",
            xp,
        )
        outputs.append(product[:, 0])
    return xp.concatenate(outputs, axis=0).reshape((-1, WIDTH))


def paper_arithmetic_per_tile(levels: int, size: int = WIDTH) -> int:
    """Flopscope-style scalar cost, excluding array-movement wrappers."""

    def bilinear(n: int, depth: int) -> int:
        if depth == 0:
            return n * n * (2 * n - 1)
        half = n // 2
        # Nine tile-dependent bilinear additions: 3 U + 6 W.
        return 7 * bilinear(half, depth - 1) + 2 * 9 * half * half

    def basis(n: int, depth: int) -> int:
        if depth == 0:
            return 0
        half = n // 2
        # Two additions, Flopscope charges two FLOPs per output scalar.
        return 4 * basis(half, depth - 1) + 2 * 2 * half * half

    return bilinear(size, levels) + 2 * basis(size, levels)


def paper_shared_right_cost(levels: int, size: int = WIDTH) -> int:
    def encode(n: int, depth: int) -> int:
        if depth == 0:
            return 0
        half = n // 2
        return 7 * encode(half, depth - 1) + 2 * 3 * half * half

    def basis(n: int, depth: int) -> int:
        if depth == 0:
            return 0
        half = n // 2
        return 4 * basis(half, depth - 1) + 2 * 2 * half * half

    return encode(size, levels) + basis(size, levels)


def small_correctness() -> list[dict[str, float | int]]:
    rng = np.random.default_rng(7719)
    records = []
    for size in (4, 8, 16):
        left = rng.standard_normal((3, size, size)).astype(np.float64)
        right = rng.standard_normal((size, size)).astype(np.float64)
        for levels in range(1, int(math_log2(size)) + 1):
            # The production wrapper is fixed at WIDTH, so exercise the core
            # transforms directly for small matrices.
            left_phi = recursive_basis_transform(
                left[:, None],
                levels,
                "phi",
                np,
            )
            right_phi = recursive_basis_transform(
                right[None, None],
                levels,
                "phi",
                np,
            )[0]
            encoded = alternative_bilinear(
                left_phi,
                right_phi,
                levels,
                np,
            )
            result = recursive_basis_transform(
                encoded,
                levels,
                "nu_inverse",
                np,
            )[:, 0]
            difference = result - left @ right
            records.append(
                {
                    "size": size,
                    "levels": levels,
                    "max_abs_error": float(np.max(np.abs(difference))),
                    "rms_error": float(
                        np.sqrt(np.mean(np.square(difference)))
                    ),
                }
            )
    return records


def math_log2(value: int) -> int:
    return value.bit_length() - 1


def profile_level(
    activation: np.ndarray,
    weight: np.ndarray,
    level: int,
    tile_batch: int,
) -> dict[str, object]:
    rows = min(len(activation), tile_batch * WIDTH)
    rows -= rows % WIDTH
    sample = activation[:rows]
    tracked_activation = fnp.asarray(sample)
    tracked_weight = fnp.asarray(weight)
    with flopscope.BudgetContext(flop_budget=272_000_000_000) as context:
        result = alternative_basis_tiles(
            tracked_activation,
            tracked_weight,
            level,
            tile_batch,
            fnp,
        )
        summary = context.summary_dict()
    dense = sample @ weight
    difference = np.asarray(result) - dense
    tiles = rows // WIDTH
    scalar_model = (
        tiles * paper_arithmetic_per_tile(level)
        + paper_shared_right_cost(level)
    )
    return {
        "implementation": "vectorized_branches",
        "levels": level,
        "rows": rows,
        "tiles": tiles,
        "flopscope_flops_used": int(summary["flops_used"]),
        "scalar_arithmetic_model": scalar_model,
        "wrapper_overhead": int(summary["flops_used"]) - scalar_model,
        "dense_flops": rows * WIDTH * (2 * WIDTH - 1),
        "ratio_to_dense": float(
            summary["flops_used"]
            / (rows * WIDTH * (2 * WIDTH - 1))
        ),
        "max_abs_error": float(np.max(np.abs(difference))),
        "rms_error": float(np.sqrt(np.mean(np.square(difference)))),
        "operations": summary["operations"],
    }


def profile_level_sequential(
    activation: np.ndarray,
    weight: np.ndarray,
    level: int,
    tile_batch: int,
) -> dict[str, object]:
    rows = min(len(activation), tile_batch * WIDTH)
    rows -= rows % WIDTH
    sample = activation[:rows]
    tracked_activation = fnp.asarray(sample)
    tracked_weight = fnp.asarray(weight)
    with flopscope.BudgetContext(flop_budget=272_000_000_000) as context:
        result = alternative_basis_tiles_sequential(
            tracked_activation,
            tracked_weight,
            level,
            fnp,
        )
        summary = context.summary_dict()
    dense = sample @ weight
    difference = np.asarray(result) - dense
    tiles = rows // WIDTH
    scalar_model = (
        tiles * paper_arithmetic_per_tile(level)
        + paper_shared_right_cost(level)
    )
    return {
        "implementation": "sequential_branches",
        "levels": level,
        "rows": rows,
        "tiles": tiles,
        "flopscope_flops_used": int(summary["flops_used"]),
        "scalar_arithmetic_model": scalar_model,
        "wrapper_overhead": int(summary["flops_used"]) - scalar_model,
        "dense_flops": rows * WIDTH * (2 * WIDTH - 1),
        "ratio_to_dense": float(
            summary["flops_used"]
            / (rows * WIDTH * (2 * WIDTH - 1))
        ),
        "max_abs_error": float(np.max(np.abs(difference))),
        "rms_error": float(np.sqrt(np.mean(np.square(difference)))),
        "operations": summary["operations"],
    }


def end_to_end(
    weights: np.ndarray,
    target: np.ndarray,
    first: np.ndarray,
    level: int,
    tile_batch: int,
    sequential: bool,
) -> dict[str, float]:
    dense = first
    start = time.perf_counter()
    for weight in weights[1:]:
        dense = np.maximum(dense @ weight, 0.0)
    dense_prediction = dense.mean(axis=0, dtype=np.float64)
    dense_seconds = time.perf_counter() - start

    fast = first
    start = time.perf_counter()
    for weight in weights[1:]:
        if sequential:
            product = alternative_basis_tiles_sequential(
                fast,
                weight,
                level,
                np,
            )
        else:
            product = alternative_basis_tiles(
                fast,
                weight,
                level,
                tile_batch,
                np,
            )
        fast = np.maximum(
            product,
            0.0,
        )
    fast_prediction = fast.mean(axis=0, dtype=np.float64)
    fast_seconds = time.perf_counter() - start
    difference = fast_prediction - dense_prediction
    return {
        "dense_seconds": dense_seconds,
        "alternative_basis_seconds": fast_seconds,
        "dense_final_mse": float(
            np.mean(np.square(dense_prediction - target))
        ),
        "alternative_basis_final_mse": float(
            np.mean(np.square(fast_prediction - target))
        ),
        "implementation": (
            "sequential_branches" if sequential else "vectorized_branches"
        ),
        "prediction_max_abs_error": float(np.max(np.abs(difference))),
        "prediction_rms_error": float(
            np.sqrt(np.mean(np.square(difference)))
        ),
    }


def profile_full_prediction(
    weights_np: np.ndarray,
    target: np.ndarray,
    rotation_np: np.ndarray,
    level: int,
) -> tuple[dict[str, object], np.ndarray]:
    chirps_np = np.stack(
        [kerdock_chirp(index) for index in range(FIELD_SIZE)]
    ).astype(np.float32)
    weights = [
        fnp.asarray(weight, dtype=fnp.float32)
        for weight in weights_np
    ]
    rotation = fnp.asarray(rotation_np, dtype=fnp.float32)
    chirps = fnp.asarray(chirps_np, dtype=fnp.float32)
    with flopscope.BudgetContext(flop_budget=272_000_000_000) as context:
        activation = structured_first_layer_flopscope(
            weights[0],
            rotation,
            chirps,
        )
        for weight in weights[1:]:
            activation = fnp.maximum(
                alternative_basis_tiles_sequential(
                    activation,
                    weight,
                    level,
                    fnp,
                ),
                0.0,
            )
        prediction = fnp.mean(
            activation.astype(fnp.float64),
            axis=0,
        )
        summary = context.summary_dict()
    prediction_np = np.asarray(prediction)
    summary["final_mse"] = float(
        np.mean(np.square(prediction_np - target))
    )
    return summary, prediction_np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--rotation-seed", type=int, default=3)
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5],
    )
    parser.add_argument("--tile-batch", type=int, default=4)
    parser.add_argument("--end-to-end-level", type=int, default=4)
    parser.add_argument("--skip-end-to-end", action="store_true")
    parser.add_argument(
        "--end-to-end-sequential",
        action="store_true",
    )
    parser.add_argument(
        "--sequential-only",
        action="store_true",
        help="skip the higher-memory vectorized-branch profiles",
    )
    parser.add_argument(
        "--profile-full-prediction",
        action="store_true",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not 0 <= args.index < 50:
        raise ValueError("audit is restricted to selection IDs 0--49")
    if any(level < 0 or level > 8 for level in args.levels):
        raise ValueError("levels must lie in 0--8")

    small = small_correctness()
    if max(record["max_abs_error"] for record in small) > 1e-9:
        raise AssertionError("small alternative-basis correctness failed")
    name, weights, targets = _load_rows(args.data, [args.index])[0]
    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    first = np.maximum(points @ (rotation @ weights[0]), 0.0)
    profiles = (
        []
        if args.sequential_only
        else [
            profile_level(
                first,
                weights[1],
                level,
                args.tile_batch,
            )
            for level in args.levels
        ]
    )
    sequential_profiles = [
        profile_level_sequential(
            first,
            weights[1],
            level,
            args.tile_batch,
        )
        for level in args.levels
    ]
    for profile in profiles:
        print({"profile": profile}, flush=True)
    for profile in sequential_profiles:
        print({"profile": profile}, flush=True)
    full = None
    if not args.skip_end_to_end:
        full = end_to_end(
            weights,
            targets[-1],
            first,
            args.end_to_end_level,
            args.tile_batch,
            args.end_to_end_sequential,
        )
        print({"end_to_end": full}, flush=True)
    full_profile = None
    if args.profile_full_prediction:
        full_profile, tracked_prediction = profile_full_prediction(
            weights,
            targets[-1],
            rotation,
            args.end_to_end_level,
        )
        print(
            {
                "full_prediction_flops": full_profile["flops_used"],
                "full_prediction_remaining": full_profile[
                    "flops_remaining"
                ],
                "full_prediction_mse": full_profile["final_mse"],
            },
            flush=True,
        )
        if full is not None:
            # The research NumPy path and tracked path should differ only by
            # ordinary float association.
            full_profile["tracked_prediction_mean"] = float(
                np.mean(tracked_prediction)
            )

    extrapolated = {}
    for profile in profiles + sequential_profiles:
        # Conservative linear extrapolation from a full tile batch.  It
        # repeats the shared right transform for each chunk exactly as the
        # current wrapper does.
        per_tile = profile["flopscope_flops_used"] / profile["tiles"]
        deep = 31 * TILES * per_tile
        key = f"{profile['implementation']}_L{profile['levels']}"
        extrapolated[key] = {
            "deep_31_flops": deep,
            "ratio_to_classical_deep": (
                deep
                / (
                    31
                    * N_POINTS
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
            "source": "https://doi.org/10.1137/22M1502719",
        },
        "small_float64_correctness": small,
        "profiles": profiles,
        "sequential_profiles": sequential_profiles,
        "extrapolated_full_design": extrapolated,
        "end_to_end": full,
        "full_prediction_flopscope": full_profile,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print({"out": str(args.out)}, flush=True)


if __name__ == "__main__":
    main()
