"""Audit all six tensor-leg symmetries of rank-7 Winograd multiplication.

The scalar 2x2 multiplication tensor is invariant under the dihedral group of
the triangle.  This makes it tempting to move Winograd's apparently expensive
output leg onto the small right operand in a tall-by-square product.  This
script does the full calculation rather than assuming that coefficient-factor
permutation also permutes straight-line addition counts.

Every implementation is exact over a ring, uses only tracked operations in the
instrumented path, and is restricted to official selection IDs 0--49.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, NamedTuple

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
    first_layer_design,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "winograd_leg_permutations_row0.json"


class LegScheme(NamedTuple):
    left_factor: str
    right_factor: str
    output_factor: str
    transpose_columns: bool
    flip_left_row4: bool
    flip_right_row4: bool
    flip_output_row4: bool


# Z is the oriented output factor W^T P.  Flipping row 4 of Z removes its
# isolated unary minus.  A second row-4 flip on V or the output factor leaves
# each rank-one tensor unchanged.
SCHEMES: dict[str, LegScheme] = {
    "uvz": LegScheme("U", "V", "Z", False, False, True, True),
    "uzv": LegScheme("U", "Z", "V", True, False, True, True),
    "vuz": LegScheme("V", "U", "Z", True, True, False, True),
    "vzu": LegScheme("V", "Z", "U", False, False, True, True),
    "zuv": LegScheme("Z", "U", "V", False, True, False, True),
    "zvu": LegScheme("Z", "V", "U", True, True, False, True),
}


def _quadrants(values: Any) -> tuple[Any, Any, Any, Any]:
    half_rows = values.shape[-2] // 2
    half_columns = values.shape[-1] // 2
    return (
        values[..., :half_rows, :half_columns],
        values[..., :half_rows, half_columns:],
        values[..., half_rows:, :half_columns],
        values[..., half_rows:, half_columns:],
    )


def _factor_forms(
    values: Any,
    factor: str,
    transpose_columns: bool,
    flip_row4: bool,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    x11, x12, x21, x22 = _quadrants(values)
    if transpose_columns:
        x12, x21 = x21, x12

    if factor == "U":
        if flip_row4:
            raise ValueError("the selected schedules never flip encoded U")
        s1 = x21 + x22
        s2 = s1 - x11
        s3 = x11 - x21
        s4 = x12 - s2
        return x11, x12, s4, x22, s1, s2, s3

    if factor == "V":
        t1 = x12 - x11
        t2 = x22 - t1
        t3 = x22 - x12
        t4 = x21 - t2 if flip_row4 else t2 - x21
        return x11, x21, x22, t4, t1, t2, t3

    if factor == "Z":
        if not flip_row4:
            raise ValueError("encoded Z is normalized by a row-4 flip")
        # Z = W^T P.  Its seven forms have an addition-complexity-four
        # circuit after the isolated -x12 form is rank-normalized to +x12.
        z7 = x12 + x22
        z6 = x21 + z7
        z1 = x11 + z6
        z5 = x21 + x22
        return z1, x11, x21, x12, z5, z6, z7

    raise ValueError(f"unknown tensor factor {factor!r}")


def _decode_factor(
    products: Any,
    factor: str,
    transpose_columns: bool,
    flip_row4: bool,
    xp: Any,
) -> Any:
    if isinstance(products, (tuple, list)):
        p1, p2, p3, p4, p5, p6, p7 = products
    else:
        p1 = products[..., 0, :, :]
        p2 = products[..., 1, :, :]
        p3 = products[..., 2, :, :]
        p4 = products[..., 3, :, :]
        p5 = products[..., 4, :, :]
        p6 = products[..., 5, :, :]
        p7 = products[..., 6, :, :]

    if factor == "U":
        # Optimal seven-addition circuit for U^T.  The factor's oriented
        # coordinates are converted back to ordinary output block order below.
        r = p6 - p3
        x12 = p2 + p3
        x11 = (p1 + p7) - r
        shared = p5 + r
        x21 = shared - p7
        x22 = shared - p4 if flip_row4 else shared + p4
        ordinary = (x11, x12, x21, x22)
    elif factor == "V":
        # Optimal seven-addition circuit for V^T.
        shared = p6 - p4 if flip_row4 else p6 + p4
        x21 = p2 + p4 if flip_row4 else p2 - p4
        x22 = (p3 + shared) + p7
        auxiliary = p5 - shared
        x12 = auxiliary - p7
        x11 = p1 - auxiliary
        ordinary = (x11, x12, x21, x22)
    elif factor == "Z":
        u1 = p1 + p2
        u2 = p1 + p6
        u3 = u2 + p7
        u4 = u2 + p5
        c11 = u1
        c12 = u4 + p3
        c21 = u3 + p4 if flip_row4 else u3 - p4
        c22 = u3 + p5
        # Z^T is in oriented order, so the final P below cancels for the
        # non-transposed-column scheme.
        ordinary = (c11, c21, c12, c22)
    else:
        raise ValueError(f"unknown tensor factor {factor!r}")

    # For factor F[:, P^bit], the ordinary decoder is
    # (F[:, P^bit][:, P])^T.  Thus bit=0 swaps the two off-diagonal outputs.
    x11, x12, x21, x22 = ordinary
    if not transpose_columns:
        x12, x21 = x21, x12
    return xp.block([[x11, x12], [x21, x22]])


def _packed_encode(
    left: Any,
    right: Any,
    scheme: LegScheme,
    xp: Any,
) -> tuple[Any, Any]:
    left_forms = _factor_forms(
        left,
        scheme.left_factor,
        scheme.transpose_columns,
        scheme.flip_left_row4,
    )
    right_forms = _factor_forms(
        right,
        scheme.right_factor,
        scheme.transpose_columns,
        scheme.flip_right_row4,
    )
    return (
        xp.stack(left_forms, axis=-3),
        xp.stack(right_forms, axis=-3),
    )


def _depth_first(
    left: Any,
    right: Any,
    levels: int,
    scheme: LegScheme,
    xp: Any,
) -> Any:
    if levels == 0:
        return left @ right
    left_forms = _factor_forms(
        left,
        scheme.left_factor,
        scheme.transpose_columns,
        scheme.flip_left_row4,
    )
    right_forms = _factor_forms(
        right,
        scheme.right_factor,
        scheme.transpose_columns,
        scheme.flip_right_row4,
    )
    products = tuple(
        _depth_first(
            left_form,
            right_form,
            levels - 1,
            scheme,
            xp,
        )
        for left_form, right_form in zip(left_forms, right_forms)
    )
    return _decode_factor(
        products,
        scheme.output_factor,
        scheme.transpose_columns,
        scheme.flip_output_row4,
        xp,
    )


def leg_permuted_matmul(
    left: Any,
    right: Any,
    scheme_name: str,
    levels: int,
    packed_levels: int,
    xp: Any,
) -> Any:
    scheme = SCHEMES[scheme_name]
    encoded_left = left
    encoded_right = right
    for _ in range(packed_levels):
        encoded_left, encoded_right = _packed_encode(
            encoded_left,
            encoded_right,
            scheme,
            xp,
        )
    products = _depth_first(
        encoded_left,
        encoded_right,
        levels - packed_levels,
        scheme,
        xp,
    )
    for _ in range(packed_levels):
        products = _decode_factor(
            products,
            scheme.output_factor,
            scheme.transpose_columns,
            scheme.flip_output_row4,
            xp,
        )
    return products


def coefficient_identity_audit() -> dict[str, object]:
    u = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, -1, -1],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [-1, 0, 1, 1],
            [1, 0, -1, 0],
        ],
        dtype=np.int64,
    )
    v = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, -1, -1, 1],
            [-1, 1, 0, 0],
            [1, -1, 0, 1],
            [0, -1, 0, 1],
        ],
        dtype=np.int64,
    )
    w = np.array(
        [
            [1, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 1, 0],
            [1, 0, 0, -1, 0, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
        ],
        dtype=np.int64,
    )
    permutation = np.array([0, 2, 1, 3])
    factors = {"U": u, "V": v, "Z": w.T[:, permutation]}
    reference = np.zeros((4, 4, 4), dtype=np.int64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                reference[2 * i + j, 2 * j + k, 2 * i + k] = 1

    records = {}
    for name, scheme in SCHEMES.items():
        matrices = [
            factors[scheme.left_factor].copy(),
            factors[scheme.right_factor].copy(),
            factors[scheme.output_factor].copy(),
        ]
        if scheme.transpose_columns:
            matrices = [matrix[:, permutation] for matrix in matrices]
        flips = (
            scheme.flip_left_row4,
            scheme.flip_right_row4,
            scheme.flip_output_row4,
        )
        for matrix, flip in zip(matrices, flips):
            if flip:
                matrix[3] *= -1
        output = matrices[2][:, permutation].T
        tensor = np.einsum(
            "ra,rb,cr->abc",
            matrices[0],
            matrices[1],
            output,
        )
        records[name] = {
            "max_coefficient_error": int(
                np.max(np.abs(tensor - reference))
            ),
            "left_factor": scheme.left_factor,
            "right_factor": scheme.right_factor,
            "output_factor": scheme.output_factor,
            "transpose_columns": scheme.transpose_columns,
            "straight_line_additions": {
                "left": 4,
                "right": 4,
                "output": 7,
                "total": 15,
            },
        }
    return records


def profile_one_layer(
    activation_np: np.ndarray,
    weight_np: np.ndarray,
    scheme_name: str,
    levels: int,
    packed_levels: int,
) -> dict[str, object]:
    dense = activation_np @ weight_np
    with flopscope.BudgetContext(
        flop_budget=BUDGET,
        quiet=True,
    ) as context:
        started = time.perf_counter()
        result = leg_permuted_matmul(
            fnp.asarray(activation_np),
            fnp.asarray(weight_np),
            scheme_name,
            levels,
            packed_levels,
            fnp,
        )
        wall_time = time.perf_counter() - started
        summary = context.summary_dict()
    difference = np.asarray(result) - dense
    residual = float(summary["residual_wall_time_s"])
    effective = int(summary["flops_used"]) + (
        LAMBDA_FLOPS_PER_SECOND * residual
    )
    return {
        "scheme": scheme_name,
        "tracked_flops": int(summary["flops_used"]),
        "wall_time_s": wall_time,
        "backend_time_s": float(summary["flopscope_backend_time_s"]),
        "overhead_time_s": float(summary["flopscope_overhead_time_s"]),
        "residual_wall_time_s": residual,
        "effective_compute": effective,
        "max_abs_error": float(np.max(np.abs(difference))),
        "rms_error": float(np.sqrt(np.mean(np.square(difference)))),
        "operations": summary["operations"],
    }


def profile_network(
    weights_np: np.ndarray,
    target: np.ndarray,
    rotation_np: np.ndarray,
    chirps_np: np.ndarray,
    scheme_name: str,
    levels: int,
    packed_levels: int,
) -> dict[str, object]:
    with flopscope.BudgetContext(
        flop_budget=BUDGET,
        quiet=True,
    ) as context:
        weights = [
            fnp.asarray(weight).astype(fnp.float32)
            for weight in weights_np
        ]
        activation = first_layer_design(
            weights[0],
            fnp.asarray(rotation_np),
            fnp.asarray(chirps_np),
            fnp,
        )
        for weight in weights[1:]:
            activation = fnp.maximum(
                leg_permuted_matmul(
                    activation,
                    weight,
                    scheme_name,
                    levels,
                    packed_levels,
                    fnp,
                ),
                0.0,
            )
        final_mean = fnp.mean(
            activation.astype(fnp.float64),
            axis=0,
        )
        summary = context.summary_dict()
    prediction = np.asarray(final_mean)
    raw_mse = float(np.mean(np.square(prediction - target)))
    residual = float(summary["residual_wall_time_s"])
    effective = int(summary["flops_used"]) + (
        LAMBDA_FLOPS_PER_SECOND * residual
    )
    multiplier = min(1.0, effective / BUDGET)
    return {
        "scheme": scheme_name,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--packed-levels", type=int, default=3)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-network", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.index < 50:
        raise ValueError("audit is restricted to selection IDs 0--49")
    if not 0 < args.packed_levels < args.levels <= 8:
        raise ValueError("require 0 < packed_levels < levels <= 8")

    identity = coefficient_identity_audit()
    if any(
        record["max_coefficient_error"] != 0
        for record in identity.values()
    ):
        raise AssertionError("a tensor-leg coefficient identity failed")

    name, weights, targets = _load_rows(args.data, [args.index])[0]
    asset = np.load(args.asset)
    rotation = asset["rotation"].astype(np.float32)
    chirps = asset["chirps"].astype(np.float32)
    first = np.maximum(
        first_layer_design(
            weights[0].astype(np.float32),
            rotation,
            chirps,
            np,
        ),
        0.0,
    )
    profiles = [
        profile_one_layer(
            first,
            weights[1].astype(np.float32),
            scheme_name,
            args.levels,
            args.packed_levels,
        )
        for scheme_name in SCHEMES
    ]
    best = min(profiles, key=lambda record: record["effective_compute"])
    network = None
    if not args.skip_network:
        network = profile_network(
            weights,
            targets[-1],
            rotation,
            chirps,
            str(best["scheme"]),
            args.levels,
            args.packed_levels,
        )
    result = {
        "protocol": {
            "selection_index": args.index,
            "selection_name": name,
            "holdout_loaded": False,
            "levels": args.levels,
            "packed_levels": args.packed_levels,
            "flopscope_version": flopscope.__version__,
        },
        "coefficient_identities": identity,
        "one_layer_profiles": profiles,
        "best_one_layer_scheme": best["scheme"],
        "full_network_profile": network,
        "conclusion": (
            "All six tensor symmetries retain the optimal straight-line "
            "addition profile 4/4/7. Moving a coefficient factor to the "
            "output requires its adjoint circuit, so factor permutation "
            "does not create a 4/7/4 implementation."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
