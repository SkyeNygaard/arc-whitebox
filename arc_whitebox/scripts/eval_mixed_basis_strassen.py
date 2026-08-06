"""Audit mixed Winograd / alternative-basis Strassen recursion schedules.

The schedule is written ``W^outer A^alternative W^inner``.  A contiguous
alternative-basis run is important: Schwartz--Vaknin's lower-addition
bilinear phase only amortises its recursive phi/nu basis changes after
several levels.  Splitting the A run into one-level wrappers destroys that
advantage.

This script measures the real 66,048 x 256 by 256 x 256 kernel under
flopscope 0.9.1, including basis transforms, branch packing, and output
assembly.  It is hard-restricted to selection IDs 0--49.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import flopscope
import flopscope.numpy as fnp
import numpy as np

from eval_sampling_official import DEFAULT_DATA, _load_rows
from eval_strassen_audit import (
    BUDGET,
    DEFAULT_ASSET,
    INV_SQRT_2PI,
    LAMBDA_FLOPS_PER_SECOND,
    N_POINTS,
    WIDTH,
    _decode_winograd,
    _encode_winograd,
    fast_matmul,
    first_layer_design,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "mixed_basis_strassen.json"


def basis_transform_block(
    values: Any,
    levels: int,
    transform: str,
    xp: Any,
) -> Any:
    """Recursive phi or nu^-1 using one billed block assembly per node."""
    if levels == 0:
        return values
    half_rows = values.shape[-2] // 2
    half_cols = values.shape[-1] // 2
    q11 = basis_transform_block(
        values[..., :half_rows, :half_cols],
        levels - 1,
        transform,
        xp,
    )
    q12 = basis_transform_block(
        values[..., :half_rows, half_cols:],
        levels - 1,
        transform,
        xp,
    )
    q21 = basis_transform_block(
        values[..., half_rows:, :half_cols],
        levels - 1,
        transform,
        xp,
    )
    q22 = basis_transform_block(
        values[..., half_rows:, half_cols:],
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
    return xp.block([[o11, o12], [o21, o22]])


def _encode_alternative(
    left: Any,
    right: Any,
    xp: Any,
) -> tuple[Any, Any]:
    """One packed U_opt/V_opt level on phi-transformed operands."""
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
    return (
        xp.stack(
            (
                a11,
                a12,
                a21,
                a22,
                a21 + a22,
                a22 - a12,
                a22 - a11,
            ),
            axis=-3,
        ),
        xp.stack(
            (
                b11,
                b21,
                b22 - b11,
                b22,
                b21 + b22,
                b22 - b12,
                b12,
            ),
            axis=-3,
        ),
    )


def _decode_alternative(products: Any, xp: Any) -> Any:
    """One W_opt^T level, still in the alternative output basis."""
    m1 = products[..., 0, :, :]
    m2 = products[..., 1, :, :]
    m3 = products[..., 2, :, :]
    m4 = products[..., 3, :, :]
    m5 = products[..., 4, :, :]
    m6 = products[..., 5, :, :]
    m7 = products[..., 6, :, :]
    c11 = m1 + m2
    c12 = m5 - m7
    c21 = m3 + m6
    c22 = m5 - m2 - m4 + m6
    return xp.block([[c11, c12], [c21, c22]])


def alternative_depth_first_with_leaf(
    left: Any,
    right: Any,
    levels: int,
    leaf: Callable[[Any, Any], Any],
    xp: Any,
) -> Any:
    """Depth-first alternative bilinear recursion with a custom leaf."""
    if levels == 0:
        return leaf(left, right)
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
    descend = alternative_depth_first_with_leaf
    rest = levels - 1
    m1 = descend(a11, b11, rest, leaf, xp)
    m2 = descend(a12, b21, rest, leaf, xp)
    m3 = descend(a21, b22 - b11, rest, leaf, xp)
    m4 = descend(a22, b22, rest, leaf, xp)
    m5 = descend(a21 + a22, b21 + b22, rest, leaf, xp)
    m6 = descend(a22 - a12, b22 - b12, rest, leaf, xp)
    m7 = descend(a22 - a11, b12, rest, leaf, xp)
    c11 = m1 + m2
    c12 = m5 - m7
    c21 = m3 + m6
    c22 = m5 - m2 - m4 + m6
    return xp.block([[c11, c12], [c21, c22]])


def inner_winograd(
    left: Any,
    right: Any,
    levels: int,
    packed_levels: int,
    xp: Any,
) -> Any:
    if levels == 0:
        return left @ right
    if packed_levels == levels:
        schedule = "packed"
    elif packed_levels == 0:
        schedule = "depth_first"
    else:
        schedule = f"hybrid_p{packed_levels}"
    return fast_matmul(left, right, levels, "winograd", schedule, xp)


def mixed_waw_matmul(
    left: Any,
    right: Any,
    outer_w: int,
    alternative: int,
    inner_w: int,
    alternative_packed: int,
    inner_packed: int,
    xp: Any,
) -> Any:
    """Execute W^outer A^alternative W^inner on arbitrary leading axes."""
    if alternative <= 0:
        return inner_winograd(
            left,
            right,
            outer_w + inner_w,
            inner_packed,
            xp,
        )
    if not 0 <= alternative_packed <= alternative:
        raise ValueError("invalid alternative packed depth")
    if not 0 <= inner_packed <= inner_w:
        raise ValueError("invalid inner Winograd packed depth")

    # Pack the outer Winograd prefix.  With the searched schedules this prefix
    # is at most two levels and packing dominates depth-first wrapper overhead.
    a = left
    b = right
    for _ in range(outer_w):
        a, b = _encode_winograd(a, b, xp)

    # One shared basis boundary for the whole contiguous A run.
    a = basis_transform_block(a, alternative, "phi", xp)
    b = basis_transform_block(b, alternative, "phi", xp)
    for _ in range(alternative_packed):
        a, b = _encode_alternative(a, b, xp)

    def leaf(x: Any, y: Any) -> Any:
        return inner_winograd(
            x,
            y,
            inner_w,
            inner_packed,
            xp,
        )

    product = alternative_depth_first_with_leaf(
        a,
        b,
        alternative - alternative_packed,
        leaf,
        xp,
    )
    for _ in range(alternative_packed):
        product = _decode_alternative(product, xp)
    product = basis_transform_block(
        product,
        alternative,
        "nu_inverse",
        xp,
    )
    for _ in range(outer_w):
        product = _decode_winograd(product, xp)
    return product


def parse_config(text: str) -> tuple[int, int, int, int, int]:
    """Parse ``outer,alternative,inner,alt_pack,inner_pack``."""
    fields = tuple(int(part) for part in text.split(","))
    if len(fields) != 5:
        raise ValueError(
            "config must be outer,alternative,inner,alt_pack,inner_pack"
        )
    return fields  # type: ignore[return-value]


def profile_micro(
    config: tuple[int, int, int, int, int],
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
    candidate_np = None
    for _ in range(repeats):
        with flopscope.BudgetContext(flop_budget=10**15, quiet=True) as ctx:
            candidate = mixed_waw_matmul(
                fnp.asarray(left),
                fnp.asarray(right),
                *config,
                fnp,
            )
        candidate_np = np.asarray(candidate)
        summaries.append(ctx.summary_dict())
    assert candidate_np is not None
    difference = candidate_np.astype(np.float64) - reference.astype(np.float64)
    flops = int(summaries[-1]["flops_used"])
    residuals = [
        float(summary["residual_wall_time_s"]) for summary in summaries
    ]
    residual = float(np.median(residuals))
    return {
        "config": config,
        "schedule": (
            f"W{config[0]}-A{config[1]}-W{config[2]}"
            f"/ap{config[3]}/ip{config[4]}"
        ),
        "total_depth": sum(config[:3]),
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
        "operation_breakdown": summaries[-1]["operations"],
    }


def id0_drift(
    config: tuple[int, int, int, int, int],
    data: Path,
    rotation: np.ndarray,
    chirps: np.ndarray,
) -> dict[str, Any]:
    rows = _load_rows(data, [0])
    name, weights, targets = rows[0]
    activation = first_layer_design(
        weights[0].astype(np.float32),
        rotation,
        chirps,
        np,
    )
    baseline = activation.copy()
    fast = activation.copy()
    gate_mismatches = 0
    started = time.perf_counter()
    for weight in weights[1:]:
        baseline_pre = baseline @ weight
        fast_pre = mixed_waw_matmul(
            fast,
            weight,
            *config,
            np,
        )
        gate_mismatches += int(
            np.count_nonzero((baseline_pre > 0) != (fast_pre > 0))
        )
        baseline = np.maximum(baseline_pre, 0)
        fast = np.maximum(fast_pre, 0)
    elapsed = time.perf_counter() - started
    baseline_mean = baseline.mean(axis=0, dtype=np.float64)
    fast_mean = fast.mean(axis=0, dtype=np.float64)
    delta = fast_mean - baseline_mean
    target = targets[-1]
    return {
        "index": 0,
        "name": name,
        "elapsed_s": elapsed,
        "baseline_mse": float(np.mean(np.square(baseline_mean - target))),
        "fast_mse": float(np.mean(np.square(fast_mean - target))),
        "mean_max_abs_difference": float(np.max(np.abs(delta))),
        "mean_rms_difference": float(
            np.sqrt(np.mean(np.square(delta)))
        ),
        "total_gate_mismatches": gate_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "0,5,0,3,0",
            "0,4,1,3,1",
            "1,4,0,3,0",
            "0,6,0,3,0",
            "0,5,1,3,1",
            "1,5,0,3,0",
            "0,4,2,3,2",
            "1,4,1,3,1",
            "2,4,0,3,0",
        ],
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--drift-best", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    configs = [parse_config(text) for text in args.configs]
    for config in configs:
        outer, alternative, inner, apack, ipack = config
        if min(config) < 0 or alternative <= 0:
            raise ValueError(f"invalid config {config}")
        if outer + alternative + inner > 8:
            raise ValueError(f"depth exceeds 8: {config}")
        if apack > alternative or ipack > inner:
            raise ValueError(f"invalid packing: {config}")

    micro = []
    for config in configs:
        row = profile_micro(config, args.repeats)
        micro.append(row)
        print({"micro": row}, flush=True)
    micro.sort(key=lambda row: float(row["effective_compute"]))

    drift = []
    if args.drift_best:
        asset = np.load(args.asset)
        rotation = np.asarray(asset["rotation"], dtype=np.float32)
        chirps = np.asarray(asset["chirps"], dtype=np.float32)
        for row in micro[: args.drift_best]:
            config = tuple(int(x) for x in row["config"])
            record = id0_drift(
                config, args.data, rotation, chirps
            )
            drift.append({"config": config, **record})
            print({"drift": drift[-1]}, flush=True)

    payload = {
        "method": "mixed contiguous alternative-basis and Winograd recursion",
        "selection_only": True,
        "selection_id_limit": 50,
        "points": N_POINTS,
        "width": WIDTH,
        "micro": micro,
        "id0_drift": drift,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print({"wrote": str(args.out)}, flush=True)


if __name__ == "__main__":
    main()
