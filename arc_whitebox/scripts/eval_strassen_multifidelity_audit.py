"""Full-network fast-matmul audit for the 90,624-row Kerdock rule.

This evaluates the frozen selection construction

    F3 + (P0_S + P1_S - 2 P3_S) / 16,

where ``F3`` is the complete seed-3 Kerdock/MUB design and each ``Ps_S`` is
the mean over 24 selected signed Hadamard bases at rotation seed ``s``.
Only the seed-0 and seed-1 pilots add rows: 66,048 + 2*12,288 = 90,624.

All rows are propagated together.  The selected seed-3 bases are ordered
first in ``F3``, so their final activation is a free basic slice.  The audit
compares ordinary NumPy matmul with tracked depth-4 Winograd multiplication
and never opens holdout IDs.
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

from eval_kerdock_design import random_rotation
from eval_sampling_official import DEFAULT_DATA, _load_rows
from eval_strassen_audit import (
    BUDGET,
    DEPTH,
    INV_SQRT_2PI,
    LAMBDA_FLOPS_PER_SECOND,
    WIDTH,
    fast_matmul,
    mean_gaussian_radius,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET = (
    ROOT
    / "submissions"
    / "kerdock_mub5"
    / "kerdock_mub5_seed3.npz"
)
DEFAULT_OUT = (
    ROOT
    / "results"
    / "strassen_multifidelity_90624_row0.json"
)

FULL_ROWS = 66_048
PILOT_ROWS = 12_288
TOTAL_ROWS = FULL_ROWS + 2 * PILOT_ROWS
PILOT_BASES = (
    1,
    3,
    4,
    5,
    6,
    13,
    15,
    16,
    29,
    35,
    57,
    59,
    66,
    72,
    84,
    85,
    87,
    95,
    96,
    101,
    108,
    118,
    120,
    124,
)


def fwht_axis_one(values: Any, xp: Any) -> Any:
    """Unnormalised FWHT for an arbitrary number of Kerdock bases."""
    bases = values.shape[0]
    span = 1
    while span < WIDTH:
        grouped = values.reshape(
            (bases, WIDTH // (2 * span), 2, span, WIDTH)
        )
        left = grouped[:, :, 0, :, :]
        right = grouped[:, :, 1, :, :]
        values = xp.stack(
            (left + right, left - right),
            axis=2,
        ).reshape((bases, WIDTH, WIDTH))
        span *= 2
    return values


def basis_preactivations(
    first_weight: Any,
    rotation: Any,
    chirps: Any,
    xp: Any,
) -> tuple[Any, Any]:
    effective_weight = rotation @ first_weight
    weighted = chirps[:, :, None] * effective_weight[None, :, :]
    preactivation = fwht_axis_one(weighted, xp) * (
        mean_gaussian_radius() / math.sqrt(WIDTH)
    )
    return preactivation, effective_weight


def signed_basis_rows(preactivation: Any, xp: Any) -> Any:
    """Flatten each basis's +256/-256 rows into one contiguous 512 block."""
    return xp.stack(
        (preactivation, -preactivation),
        axis=1,
    ).reshape((-1, WIDTH))


def first_layer_multifidelity(
    first_weight: Any,
    rotations: dict[int, Any],
    full_chirps_reordered: Any,
    pilot_chirps: Any,
    xp: Any,
) -> Any:
    """Construct all 90,624 first-layer activations."""
    full_pre, effective3 = basis_preactivations(
        first_weight,
        rotations[3],
        full_chirps_reordered,
        xp,
    )
    full_kerdock = signed_basis_rows(full_pre, xp)
    radius = mean_gaussian_radius()
    coordinate_rows = xp.stack(
        (radius * effective3, -radius * effective3),
        axis=1,
    ).reshape((-1, WIDTH))
    full_rows = xp.concatenate(
        (full_kerdock, coordinate_rows),
        axis=0,
    )

    pre0, _ = basis_preactivations(
        first_weight,
        rotations[0],
        pilot_chirps,
        xp,
    )
    pre1, _ = basis_preactivations(
        first_weight,
        rotations[1],
        pilot_chirps,
        xp,
    )
    pilot0 = signed_basis_rows(pre0, xp)
    pilot1 = signed_basis_rows(pre1, xp)
    activation = xp.maximum(
        xp.concatenate(
            (full_rows, pilot0, pilot1),
            axis=0,
        ),
        0.0,
    )
    if activation.shape != (TOTAL_ROWS, WIDTH):
        raise AssertionError(activation.shape)
    return activation


def reduce_multifidelity(activation: Any, xp: Any) -> Any:
    """Apply the frozen control-variate weights to final activations."""
    full = xp.mean(
        activation[:FULL_ROWS].astype(xp.float64),
        axis=0,
    )
    # The first 24 full-design bases are P3_S by construction.
    pilot3 = xp.mean(
        activation[:PILOT_ROWS].astype(xp.float64),
        axis=0,
    )
    pilot0 = xp.mean(
        activation[FULL_ROWS : FULL_ROWS + PILOT_ROWS].astype(
            xp.float64
        ),
        axis=0,
    )
    pilot1 = xp.mean(
        activation[FULL_ROWS + PILOT_ROWS :].astype(xp.float64),
        axis=0,
    )
    return full + (pilot0 + pilot1 - 2.0 * pilot3) / 16.0


def complete_prediction(
    weights: list[Any],
    rotations: dict[int, Any],
    full_chirps_reordered: Any,
    pilot_chirps: Any,
    recursion_depth: int,
    schedule: str,
    xp: Any,
) -> Any:
    activation = first_layer_multifidelity(
        weights[0],
        rotations,
        full_chirps_reordered,
        pilot_chirps,
        xp,
    )
    for weight in weights[1:]:
        if recursion_depth == 0:
            activation = activation @ weight
        else:
            activation = fast_matmul(
                activation,
                weight,
                recursion_depth,
                "winograd",
                schedule,
                xp,
            )
        activation = xp.maximum(activation, 0.0)
    final_mean = reduce_multifidelity(activation, xp)
    first_mean = (
        xp.sqrt(xp.sum(weights[0] * weights[0], axis=0))
        * INV_SQRT_2PI
    )
    rows = [xp.zeros(WIDTH) for _ in range(DEPTH)]
    rows[0] = first_mean
    rows[-1] = final_mean
    return xp.stack(rows, axis=0)


def numerical_drift(
    weights: np.ndarray,
    rotations: dict[int, np.ndarray],
    full_chirps_reordered: np.ndarray,
    pilot_chirps: np.ndarray,
    recursion_depth: int,
    schedule: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    baseline = first_layer_multifidelity(
        weights[0],
        rotations,
        full_chirps_reordered,
        pilot_chirps,
        np,
    )
    fast = baseline.copy()
    per_layer: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for layer, weight in enumerate(weights[1:], start=1):
        baseline_pre = baseline @ weight
        fast_pre = fast_matmul(
            fast,
            weight,
            recursion_depth,
            "winograd",
            schedule,
            np,
        )
        difference = (
            fast_pre.astype(np.float64)
            - baseline_pre.astype(np.float64)
        )
        mismatches = int(
            np.count_nonzero(
                (fast_pre > 0.0) != (baseline_pre > 0.0)
            )
        )
        per_layer.append(
            {
                "layer": layer,
                "preactivation_max_abs_difference": float(
                    np.max(np.abs(difference))
                ),
                "preactivation_rms_difference": float(
                    np.sqrt(np.mean(np.square(difference)))
                ),
                "gate_mismatches": mismatches,
                "gate_mismatch_fraction": float(
                    mismatches / difference.size
                ),
            }
        )
        baseline = np.maximum(baseline_pre, 0.0)
        fast = np.maximum(fast_pre, 0.0)
    elapsed = time.perf_counter() - started
    baseline_prediction = np.asarray(
        reduce_multifidelity(baseline, np)
    )
    fast_prediction = np.asarray(reduce_multifidelity(fast, np))
    difference = fast_prediction - baseline_prediction
    return baseline_prediction, fast_prediction, {
        "elapsed_s": elapsed,
        "final_max_abs_difference": float(
            np.max(np.abs(difference))
        ),
        "final_rms_difference": float(
            np.sqrt(np.mean(np.square(difference)))
        ),
        "total_gate_mismatches": int(
            sum(int(row["gate_mismatches"]) for row in per_layer)
        ),
        "maximum_layer_gate_mismatch_fraction": float(
            max(
                float(row["gate_mismatch_fraction"])
                for row in per_layer
            )
        ),
        "per_layer": per_layer,
    }


def instrumented_prediction(
    weights_np: np.ndarray,
    rotations_np: dict[int, np.ndarray],
    full_chirps_reordered_np: np.ndarray,
    pilot_chirps_np: np.ndarray,
    recursion_depth: int,
    schedule: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    with flopscope.BudgetContext(
        flop_budget=BUDGET,
        quiet=True,
    ) as context:
        weights = [
            fnp.asarray(weight).astype(fnp.float32)
            for weight in weights_np
        ]
        rotations = {
            seed: fnp.asarray(rotation)
            for seed, rotation in rotations_np.items()
        }
        full_chirps_reordered = fnp.asarray(
            full_chirps_reordered_np
        )
        pilot_chirps = fnp.asarray(pilot_chirps_np)
        prediction = complete_prediction(
            weights,
            rotations,
            full_chirps_reordered,
            pilot_chirps,
            recursion_depth,
            schedule,
            fnp,
        )
    return np.asarray(prediction), context.summary_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--recursion-depth", type=int, default=4)
    parser.add_argument(
        "--schedule",
        choices=(
            "packed",
            "depth_first",
            "hybrid_p1",
            "hybrid_p2",
            "hybrid_p3",
        ),
        default="hybrid_p2",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not 0 <= args.index < 50:
        raise ValueError("audit is restricted to selection IDs 0--49")

    name, weights, targets = _load_rows(
        args.data,
        [args.index],
    )[0]
    asset = np.load(args.asset)
    chirps = np.asarray(asset["chirps"], dtype=np.float32)
    remaining = [
        index for index in range(128) if index not in PILOT_BASES
    ]
    order = np.asarray((*PILOT_BASES, *remaining), dtype=np.int64)
    full_chirps_reordered = chirps[order]
    pilot_chirps = chirps[np.asarray(PILOT_BASES)]
    rotations = {
        seed: random_rotation(WIDTH, seed)
        for seed in (0, 1, 3)
    }

    baseline, fast, drift = numerical_drift(
        weights,
        rotations,
        full_chirps_reordered,
        pilot_chirps,
        args.recursion_depth,
        args.schedule,
    )
    instrumented, summary = instrumented_prediction(
        weights,
        rotations,
        full_chirps_reordered,
        pilot_chirps,
        args.recursion_depth,
        args.schedule,
    )
    target = targets[-1]
    baseline_mse = float(np.mean(np.square(baseline - target)))
    fast_mse = float(np.mean(np.square(fast - target)))
    instrumented_mse = float(
        np.mean(np.square(instrumented[-1] - target))
    )
    effective_compute = float(
        int(summary["flops_used"])
        + LAMBDA_FLOPS_PER_SECOND
        * float(summary["residual_wall_time_s"])
    )
    multiplier = max(0.1, effective_compute / BUDGET)
    payload = {
        "protocol": {
            "selection_index": args.index,
            "name": name,
            "holdout_loaded": False,
            "rows": TOTAL_ROWS,
            "full_rotation_seed": 3,
            "pilot_rotation_seeds": [0, 1],
            "pilot_bases": list(PILOT_BASES),
            "formula": "F3 + (P0_S + P1_S - 2*P3_S)/16",
            "recursion_depth": args.recursion_depth,
            "variant": "winograd",
            "schedule": args.schedule,
            "flopscope_version": getattr(
                flopscope,
                "__version__",
                "unknown",
            ),
        },
        "numerical": {
            "baseline_raw_final_mse": baseline_mse,
            "fast_raw_final_mse": fast_mse,
            "raw_mse_delta": fast_mse - baseline_mse,
            "drift": drift,
        },
        "authoritative_profile": {
            "tracked_flops": int(summary["flops_used"]),
            "wall_time_s": float(summary["wall_time_s"]),
            "backend_time_s": float(
                summary["flopscope_backend_time_s"]
            ),
            "overhead_time_s": float(
                summary["flopscope_overhead_time_s"]
            ),
            "residual_wall_time_s": float(
                summary["residual_wall_time_s"]
            ),
            "effective_compute": effective_compute,
            "score_multiplier": multiplier,
            "combined_budget_exhausted": bool(
                effective_compute > BUDGET
            ),
            "instrumented_raw_final_mse": instrumented_mse,
            "instrumented_adjusted_score": (
                instrumented_mse * multiplier
            ),
            "instrumented_vs_numpy_max_abs_difference": float(
                np.max(np.abs(instrumented[-1] - fast))
            ),
            "operation_breakdown": summary["operations"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        {
            "baseline_mse": baseline_mse,
            "fast_mse": fast_mse,
            "drift_max": drift["final_max_abs_difference"],
            "tracked_flops": summary["flops_used"],
            "residual_s": summary["residual_wall_time_s"],
            "effective_compute": effective_compute,
            "multiplier": multiplier,
            "adjusted": instrumented_mse * multiplier,
            "wrote": str(args.out),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
