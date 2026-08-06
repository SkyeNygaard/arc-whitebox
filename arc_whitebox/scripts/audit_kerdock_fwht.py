"""Independent algebraic, numerical, and Flopscope audit of the Kerdock design.

The companion research script spot-checks two mutually-unbiased bases and a
few fourth moments.  This audit checks all 8,128 nontrivial pairs of the 128
Walsh-chirp bases using exact integer Walsh spectra, verifies that the proposed
GF(128) arithmetic is a field, proves the spherical-5 moment condition through
the exact fourth frame potential, and compares a batched FWHT first layer with
the materialized dense construction.

Optionally, ``--profile-flopscope`` executes one complete structured prediction
inside the installed Flopscope 0.9.1 BudgetContext and saves the authoritative
operation ledger.  Challenge holdout rows are deliberately rejected.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import flopscope
import flopscope.numpy as fnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_kerdock_design import (  # noqa: E402
    FIELD_SIZE,
    N_BASES,
    N_POINTS,
    WIDTH,
    gf128_mul,
    gf128_pow,
    kerdock_chirp,
    make_kerdock_design,
    random_rotation,
)
from eval_sampling_official import DEFAULT_DATA, _load_rows  # noqa: E402
from eval_spherical_stein_cv import sphere_radius_mean  # noqa: E402


DEFAULT_OUT = ROOT / "results" / "kerdock_fwht_audit.json"


def integer_fwht(vector: np.ndarray) -> np.ndarray:
    """Unnormalized Walsh-Hadamard transform, exactly in int32."""
    transformed = np.asarray(vector, dtype=np.int32).copy()
    width = len(transformed)
    half = 1
    while half < width:
        groups = transformed.reshape(-1, 2 * half)
        left = groups[:, :half].copy()
        right = groups[:, half:].copy()
        groups[:, :half] = left + right
        groups[:, half:] = left - right
        half *= 2
    return transformed


def numpy_fwht_batch(values: np.ndarray) -> np.ndarray:
    """Apply the unnormalized FWHT on axis 1 of (batch, width, outputs)."""
    transformed = np.asarray(values).copy()
    batch, width, outputs = transformed.shape
    half = 1
    while half < width:
        groups = transformed.reshape(
            batch,
            width // (2 * half),
            2 * half,
            outputs,
        )
        left = groups[:, :, :half, :].copy()
        right = groups[:, :, half:, :].copy()
        groups[:, :, :half, :] = left + right
        groups[:, :, half:, :] = left - right
        half *= 2
    return transformed


def flopscope_fwht_batch(values: fnp.ndarray) -> fnp.ndarray:
    """Out-of-place, fully tracked FWHT on (batch, width, outputs)."""
    transformed = values
    batch, width, outputs = transformed.shape
    half = 1
    while half < width:
        groups = transformed.reshape(
            batch,
            width // (2 * half),
            2 * half,
            outputs,
        )
        left = groups[:, :, :half, :]
        right = groups[:, :, half:, :]
        transformed = fnp.concatenate(
            (left + right, left - right),
            axis=2,
        ).reshape(batch, width, outputs)
        half *= 2
    return transformed


def audit_field_and_mubs(chirps: np.ndarray) -> dict[str, object]:
    # x^7+x+1 yields a field iff every nonzero residue has a multiplicative
    # inverse.  Check both the exponent formula and permutation property.
    inverse_failures = []
    permutation_failures = []
    expected_nonzero = list(range(1, FIELD_SIZE))
    for value in expected_nonzero:
        inverse = gf128_pow(value, FIELD_SIZE - 2)
        if gf128_mul(value, inverse) != 1:
            inverse_failures.append(value)
        products = sorted(
            gf128_mul(value, other)
            for other in expected_nonzero
        )
        if products != expected_nonzero:
            permutation_failures.append(value)

    unique_chirps = len(np.unique(chirps, axis=0))
    bad_pairs = []
    spectrum_abs_values = set()
    for left in range(FIELD_SIZE):
        for right in range(left + 1, FIELD_SIZE):
            quotient = chirps[left].astype(np.int32) * chirps[
                right
            ].astype(np.int32)
            spectrum = integer_fwht(quotient)
            spectrum_abs_values.update(
                int(value) for value in np.unique(np.abs(spectrum))
            )
            if not np.all(np.abs(spectrum) == math.isqrt(WIDTH)):
                bad_pairs.append((left, right))
                if len(bad_pairs) >= 16:
                    break
        if len(bad_pairs) >= 16:
            break

    # Exact fourth-frame-potential certificate.  For M real MUBs:
    # within-basis ordered contribution is M*d; cross-basis contribution is
    # M*(M-1).  Equality with the real projective Welch bound is equivalent
    # to a projective 2-design.  Adding antipodes supplies all odd moments.
    mub_vectors = N_BASES * WIDTH
    exact_frame_potential = N_BASES * WIDTH + N_BASES * (N_BASES - 1)
    spherical_target_numerator = (
        3 * mub_vectors * mub_vectors
    )
    spherical_target_denominator = WIDTH * (WIDTH + 2)
    target_frame_potential = (
        spherical_target_numerator / spherical_target_denominator
    )

    return {
        "field_inverse_failures": inverse_failures,
        "field_permutation_failures": permutation_failures,
        "unique_chirps": unique_chirps,
        "chirp_count": len(chirps),
        "checked_nontrivial_basis_pairs": (
            FIELD_SIZE * (FIELD_SIZE - 1) // 2
        ),
        "bad_mub_pairs": bad_pairs,
        "pair_product_walsh_spectrum_abs_values": sorted(
            spectrum_abs_values
        ),
        "expected_walsh_spectrum_abs": math.isqrt(WIDTH),
        "fourth_frame_potential_exact": exact_frame_potential,
        "fourth_frame_potential_sphere_target": target_frame_potential,
        "fourth_frame_potential_error": (
            exact_frame_potential - target_frame_potential
        ),
        "antipodal_points": 2 * mub_vectors,
        "expected_antipodal_points": N_POINTS,
        "spherical_5_design_certificate": (
            not inverse_failures
            and not permutation_failures
            and unique_chirps == FIELD_SIZE
            and not bad_pairs
            and exact_frame_potential == target_frame_potential
            and 2 * mub_vectors == N_POINTS
        ),
    }


def structured_first_numpy(
    weight: np.ndarray,
    rotation: np.ndarray,
    chirps: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    rotated_weight = rotation @ weight
    scaled_weight = rotated_weight * np.float32(radius / math.sqrt(WIDTH))
    chirped = chirps[:, :, None] * scaled_weight[None, :, :]
    transformed = numpy_fwht_batch(chirped)
    paired = np.stack((transformed, -transformed), axis=1).reshape(
        2 * FIELD_SIZE * WIDTH,
        WIDTH,
    )
    axes = np.stack(
        (
            np.float32(radius) * rotated_weight,
            -np.float32(radius) * rotated_weight,
        ),
        axis=0,
    ).reshape(2 * WIDTH, WIDTH)
    return np.concatenate((paired, axes), axis=0), rotated_weight


def structured_prediction_numpy(
    weights: np.ndarray,
    rotation: np.ndarray,
    chirps: np.ndarray,
    radius: float,
) -> np.ndarray:
    preactivation, _ = structured_first_numpy(
        weights[0],
        rotation,
        chirps,
        radius,
    )
    activation = np.maximum(preactivation, 0.0)
    for weight in weights[1:]:
        activation = np.maximum(activation @ weight, 0.0)
    return activation.mean(axis=0, dtype=np.float64)


def numerical_equivalence(
    weights: np.ndarray,
    target: np.ndarray,
    rotation: np.ndarray,
    chirps: np.ndarray,
    radius: float,
) -> dict[str, float]:
    points = make_kerdock_design(radius)
    start = time.perf_counter()
    structured_pre, rotated_weight = structured_first_numpy(
        weights[0],
        rotation,
        chirps,
        radius,
    )
    structured_seconds = time.perf_counter() - start

    # The folded comparison isolates the FWHT construction.  The original
    # research association also quantifies float32 reassociation drift.
    dense_folded = points @ rotated_weight
    dense_original = (points @ rotation) @ weights[0]
    folded_difference = structured_pre - dense_folded
    original_difference = structured_pre - dense_original

    structured_prediction = structured_prediction_numpy(
        weights,
        rotation,
        chirps,
        radius,
    )
    dense_activation = points @ rotation
    for weight in weights:
        dense_activation = np.maximum(
            dense_activation @ weight,
            0.0,
        )
    dense_prediction = dense_activation.mean(axis=0, dtype=np.float64)
    prediction_difference = structured_prediction - dense_prediction
    return {
        "structured_first_seconds": structured_seconds,
        "folded_first_max_abs_error": float(
            np.max(np.abs(folded_difference))
        ),
        "folded_first_rms_error": float(
            np.sqrt(np.mean(np.square(folded_difference)))
        ),
        "original_association_first_max_abs_error": float(
            np.max(np.abs(original_difference))
        ),
        "original_association_first_rms_error": float(
            np.sqrt(np.mean(np.square(original_difference)))
        ),
        "final_prediction_max_abs_difference": float(
            np.max(np.abs(prediction_difference))
        ),
        "final_prediction_rms_difference": float(
            np.sqrt(np.mean(np.square(prediction_difference)))
        ),
        "structured_final_mse": float(
            np.mean(np.square(structured_prediction - target))
        ),
        "dense_original_final_mse": float(
            np.mean(np.square(dense_prediction - target))
        ),
    }


def structured_prediction_flopscope(
    weights: list[fnp.ndarray],
    rotation: fnp.ndarray,
    chirps: fnp.ndarray,
    radius: float,
) -> fnp.ndarray:
    rotated_weight = rotation @ weights[0]
    scaled_weight = rotated_weight * (
        radius / math.sqrt(WIDTH)
    )
    chirped = chirps[:, :, None] * scaled_weight[None, :, :]
    transformed = flopscope_fwht_batch(chirped)
    paired = fnp.stack(
        (transformed, -transformed),
        axis=1,
    ).reshape(2 * FIELD_SIZE * WIDTH, WIDTH)
    axes = fnp.stack(
        (
            radius * rotated_weight,
            -(radius * rotated_weight),
        ),
        axis=0,
    ).reshape(2 * WIDTH, WIDTH)
    activation = fnp.maximum(
        fnp.concatenate((paired, axes), axis=0),
        0.0,
    )
    for weight in weights[1:]:
        activation = fnp.maximum(activation @ weight, 0.0)
    return fnp.sum(
        activation.astype(fnp.float64),
        axis=0,
    ) / float(N_POINTS)


def flopscope_profile(
    weights_np: np.ndarray,
    rotation_np: np.ndarray,
    chirps_np: np.ndarray,
    radius: float,
) -> tuple[dict[str, object], np.ndarray]:
    # Assets and MLP arrays already exist before predict() in the grader.
    weights = [
        fnp.asarray(weight, dtype=fnp.float32)
        for weight in weights_np
    ]
    rotation = fnp.asarray(rotation_np, dtype=fnp.float32)
    chirps = fnp.asarray(chirps_np, dtype=fnp.float32)
    with flopscope.BudgetContext(
        flop_budget=272_000_000_000,
    ) as context:
        prediction = structured_prediction_flopscope(
            weights,
            rotation,
            chirps,
            radius,
        )
        summary = context.summary_dict()
    return summary, np.asarray(prediction)


def theoretical_flopscope_cost() -> dict[str, int]:
    # Installed 0.9.1 charges matmul m*k*(2*n-1), and two FLOPs per
    # pointwise/reduction scalar operation.
    remaining_dense_matmuls = (
        31 * N_POINTS * WIDTH * (2 * WIDTH - 1)
    )
    all_relu = 32 * 2 * N_POINTS * WIDTH
    rotate_first_weight = WIDTH * WIDTH * (2 * WIDTH - 1)
    scaled_weight = 2 * WIDTH * WIDTH
    chirp_signs = 2 * FIELD_SIZE * WIDTH * WIDTH
    fwht = 2 * FIELD_SIZE * WIDTH * WIDTH * 8
    negate_hadamard_blocks = 2 * FIELD_SIZE * WIDTH * WIDTH
    coordinate_scale_and_negation = 4 * WIDTH * WIDTH
    final_sum = 2 * (N_POINTS * WIDTH - WIDTH)
    final_divide = 2 * WIDTH
    total = sum(
        (
            remaining_dense_matmuls,
            all_relu,
            rotate_first_weight,
            scaled_weight,
            chirp_signs,
            fwht,
            negate_hadamard_blocks,
            coordinate_scale_and_negation,
            final_sum,
            final_divide,
        )
    )
    return {
        "remaining_31_dense_matmuls": remaining_dense_matmuls,
        "all_32_relus": all_relu,
        "rotate_first_weight": rotate_first_weight,
        "scale_rotated_weight": scaled_weight,
        "chirp_sign_multiply": chirp_signs,
        "eight_fwht_stages": fwht,
        "negate_hadamard_blocks": negate_hadamard_blocks,
        "coordinate_scale_and_negation": coordinate_scale_and_negation,
        "final_sum": final_sum,
        "final_divide": final_divide,
        "predicted_total": total,
        "predicted_budget_margin": 272_000_000_000 - total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--rotation-seed", type=int, default=3)
    parser.add_argument("--profile-flopscope", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not 0 <= args.index < 50:
        raise ValueError("audit is restricted to selection IDs 0--49")

    chirps = np.stack(
        [kerdock_chirp(index) for index in range(FIELD_SIZE)]
    ).astype(np.float32)
    algebra = audit_field_and_mubs(chirps)
    print({"algebra": algebra}, flush=True)
    if not algebra["spherical_5_design_certificate"]:
        raise AssertionError("full Kerdock/MUB audit failed")

    _, weights, targets = _load_rows(
        args.data,
        [args.index],
    )[0]
    rotation = random_rotation(WIDTH, args.rotation_seed)
    radius = sphere_radius_mean(WIDTH)
    numerical = numerical_equivalence(
        weights,
        targets[-1],
        rotation,
        chirps,
        radius,
    )
    print({"numerical": numerical}, flush=True)

    profile = None
    profile_prediction_difference = None
    if args.profile_flopscope:
        profile, profile_prediction = flopscope_profile(
            weights,
            rotation,
            chirps,
            radius,
        )
        numpy_prediction = structured_prediction_numpy(
            weights,
            rotation,
            chirps,
            radius,
        )
        profile_prediction_difference = {
            "max_abs": float(
                np.max(np.abs(profile_prediction - numpy_prediction))
            ),
            "rms": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            profile_prediction - numpy_prediction
                        )
                    )
                )
            ),
        }
        print(
            {
                "flopscope_flops_used": profile["flops_used"],
                "flopscope_margin": profile["flops_remaining"],
                "profile_prediction_difference": (
                    profile_prediction_difference
                ),
            },
            flush=True,
        )

    result = {
        "protocol": {
            "selection_id": args.index,
            "rotation_seed": args.rotation_seed,
            "holdout_loaded": False,
            "flopscope_version": getattr(
                flopscope,
                "__version__",
                "unknown",
            ),
        },
        "algebra": algebra,
        "theoretical_flopscope_cost": theoretical_flopscope_cost(),
        "numerical_equivalence": numerical,
        "flopscope_profile": profile,
        "profile_prediction_difference": profile_prediction_difference,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print({"out": str(args.out)}, flush=True)


if __name__ == "__main__":
    main()
