"""Evaluate a maximal-real-MUB spherical design on the ARC MLP integral.

In dimension ``d = 256 = 2^8``, the binary Kerdock code is a union of 128
cosets of the first-order Reed-Muller code.  After mapping bits to signs, each
coset supplies an orthonormal Hadamard basis of R^256, and distinct bases are
mutually unbiased.  Adding the coordinate basis gives the maximal 129 real
mutually unbiased bases.  Taking both signs of every basis vector produces
66,048 equal-weight sphere points and a spherical 5-design.

This is unusually well matched to the challenge:

* antipodes integrate all odd angular components exactly;
* the maximal MUB union also integrates degrees 2 and 4 exactly;
* each non-coordinate basis is a signed Walsh-Hadamard basis, so a submission
  can evaluate the first MLP layer with fast Walsh-Hadamard transforms rather
  than a dense 66,048-by-256 matrix multiplication.

This research harness materializes the points and uses ordinary dense forwards
for simplicity.  It validates the construction algebraically, then compares it
with official targets.  The structured first-layer implementation belongs in a
submission only if the quadrature error clears the experimental gate.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from eval_sampling_official import DEFAULT_DATA, _load_rows
from eval_spherical_stein_cv import sphere_radius_mean


WIDTH = 256
FIELD_SIZE = 128
N_BASES = 129
N_POINTS = 2 * WIDTH * N_BASES


def gf128_mul(a: int, b: int) -> int:
    """Multiply in GF(2^7) modulo x^7 + x + 1."""
    result = 0
    left = a
    right = b
    while right:
        if right & 1:
            result ^= left
        right >>= 1
        carry = left & 0x40
        left = (left << 1) & 0x7F
        if carry:
            left ^= 0x03
    return result


def gf128_square(a: int) -> int:
    return gf128_mul(a, a)


def gf128_pow(a: int, power: int) -> int:
    result = 1
    base = a
    exponent = power
    while exponent:
        if exponent & 1:
            result = gf128_mul(result, base)
        base = gf128_square(base)
        exponent >>= 1
    return result


def gf128_trace(a: int) -> int:
    """Absolute trace GF(2^7) -> GF(2), returned as the bit 0 or 1."""
    total = a
    term = a
    for _ in range(1, 7):
        term = gf128_square(term)
        total ^= term
    if total not in (0, 1):
        raise AssertionError(f"bad GF(128) trace value {total}")
    return total


def parity_table() -> np.ndarray:
    return np.asarray([int(i).bit_count() & 1 for i in range(WIDTH)], dtype=np.uint8)


def kerdock_chirp(u: int) -> np.ndarray:
    """The quadratic representative f_u on GF(2^7) x GF(2).

    Carlet's field construction for even n=8 writes m=n-1=7 and t=3:

        f(x, x_n) = Tr(sum_{j=1}^3 x^(2^j+1)) + x_n Tr(x),
        f_u(x, x_n) = f(u x, x_n).
    """
    bits = np.empty(WIDTH, dtype=np.uint8)
    for coordinate in range(WIDTH):
        x = coordinate & 0x7F
        x_n = coordinate >> 7
        ux = gf128_mul(u, x)
        polynomial = (
            gf128_pow(ux, 3)
            ^ gf128_pow(ux, 5)
            ^ gf128_pow(ux, 9)
        )
        bits[coordinate] = gf128_trace(polynomial) ^ (
            x_n & gf128_trace(ux)
        )
    return (1 - 2 * bits.astype(np.int16)).astype(np.float32)


def walsh_hadamard() -> np.ndarray:
    indices = np.arange(WIDTH, dtype=np.uint16)
    parity = parity_table()
    bits = parity[np.bitwise_and(indices[:, None], indices[None, :])]
    return (1 - 2 * bits.astype(np.int16)).astype(np.float32)


def make_kerdock_design(radius: float | None = None) -> np.ndarray:
    if radius is None:
        radius = sphere_radius_mean(WIDTH)
    hadamard = walsh_hadamard()
    scale = np.float32(radius / math.sqrt(WIDTH))
    blocks = []
    for u in range(FIELD_SIZE):
        basis = scale * (hadamard * kerdock_chirp(u)[None, :])
        blocks.extend((basis, -basis))
    axes = np.float32(radius) * np.eye(WIDTH, dtype=np.float32)
    blocks.extend((axes, -axes))
    design = np.concatenate(blocks, axis=0)
    if design.shape != (N_POINTS, WIDTH):
        raise AssertionError(design.shape)
    return design


def validate_design(points: np.ndarray) -> dict[str, float]:
    radius = float(np.linalg.norm(points[0].astype(np.float64)))
    unit = points.astype(np.float64) / radius
    covariance = unit.T @ unit / len(unit)
    expected_covariance = np.eye(WIDTH) / WIDTH

    # The full fourth-order tensor is too large.  Its permutation classes plus
    # random mixed monomials are sufficient to catch construction mistakes.
    diag4 = np.mean(np.power(unit[:, 0], 4))
    pair22 = np.mean(np.square(unit[:, 0]) * np.square(unit[:, 1]))
    target_diag4 = 3.0 / (WIDTH * (WIDTH + 2))
    target_pair22 = 1.0 / (WIDTH * (WIDTH + 2))
    rng = np.random.default_rng(2026)
    mixed4_max = 0.0
    for _ in range(32):
        indices = rng.choice(WIDTH, size=4, replace=False)
        moment = float(np.mean(np.prod(unit[:, indices], axis=1)))
        mixed4_max = max(mixed4_max, abs(moment))

    # Spot-check mutual unbiasedness before antipodal duplication.
    h = walsh_hadamard() / math.sqrt(WIDTH)
    basis_1 = h * kerdock_chirp(1)[None, :]
    basis_2 = h * kerdock_chirp(2)[None, :]
    cross = basis_1 @ basis_2.T
    target_cross_abs = 1.0 / math.sqrt(WIDTH)
    return {
        "points": float(len(points)),
        "max_norm_error": float(
            np.max(np.abs(np.linalg.norm(points, axis=1) - radius))
        ),
        "max_abs_mean": float(np.max(np.abs(unit.mean(axis=0)))),
        "max_covariance_error": float(
            np.max(np.abs(covariance - expected_covariance))
        ),
        "diag4": float(diag4),
        "diag4_target": target_diag4,
        "diag4_error": float(abs(diag4 - target_diag4)),
        "pair22": float(pair22),
        "pair22_target": target_pair22,
        "pair22_error": float(abs(pair22 - target_pair22)),
        "max_random_distinct_fourth_moment": mixed4_max,
        "max_mub_abs_inner_product_error": float(
            np.max(np.abs(np.abs(cross) - target_cross_abs))
        ),
    }


def random_rotation(width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((width, width))
    q, r = np.linalg.qr(gaussian)
    q *= np.where(np.diag(r) < 0.0, -1.0, 1.0)[None, :]
    return q.astype(np.float32)


def forward_final(
    weights: np.ndarray,
    points: np.ndarray,
    chunk: int,
    rotation: np.ndarray | None,
) -> tuple[np.ndarray, float]:
    total = np.zeros(weights.shape[-1], dtype=np.float64)
    start = time.perf_counter()
    for offset in range(0, len(points), chunk):
        activation = points[offset : offset + chunk]
        if rotation is not None:
            activation = activation @ rotation
        for weight in weights:
            activation = np.maximum(activation @ weight, 0.0)
        total += activation.sum(axis=0, dtype=np.float64)
    return total / len(points), time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(10)))
    parser.add_argument(
        "--rotation-seeds",
        type=int,
        nargs="+",
        default=[-1],
        help="-1 means the canonical unrotated design.",
    )
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    points = make_kerdock_design()
    validation = validate_design(points)
    print({"validation": validation}, flush=True)
    if (
        validation["max_covariance_error"] > 1e-10
        or validation["diag4_error"] > 1e-10
        or validation["pair22_error"] > 1e-10
        or validation["max_random_distinct_fourth_moment"] > 1e-10
        or validation["max_mub_abs_inner_product_error"] > 1e-6
    ):
        raise AssertionError("Kerdock design failed its exact-moment checks")

    rows = _load_rows(args.data, args.indices)
    rotations = {
        seed: None if seed < 0 else random_rotation(WIDTH, seed)
        for seed in args.rotation_seeds
    }
    records = []
    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        for seed, rotation in rotations.items():
            prediction, seconds = forward_final(
                weights, points, args.chunk, rotation
            )
            record = {
                "index": index,
                "name": name,
                "rotation_seed": seed,
                "points": len(points),
                "seconds": seconds,
                "final_mse": float(
                    np.mean(np.square(prediction - targets[-1]))
                ),
                "mean_prediction": float(np.mean(prediction)),
                "mean_target": float(np.mean(targets[-1])),
            }
            records.append(record)
            print(record, flush=True)

    summaries = []
    for seed in args.rotation_seeds:
        chosen = [record for record in records if record["rotation_seed"] == seed]
        summaries.append(
            {
                "rotation_seed": seed,
                "networks": len(chosen),
                "mean_final_mse": float(
                    np.mean([record["final_mse"] for record in chosen])
                ),
                "median_final_mse": float(
                    np.median([record["final_mse"] for record in chosen])
                ),
                "mean_seconds": float(
                    np.mean([record["seconds"] for record in chosen])
                ),
            }
        )
    result = {
        "construction": {
            "name": "binary Kerdock maximal real MUB spherical 5-design",
            "dimension": WIDTH,
            "bases": N_BASES,
            "points": N_POINTS,
            "field_polynomial": "x^7 + x + 1",
        },
        "validation": validation,
        "structured_flop_model": {
            "dense_layers_after_first": (
                N_POINTS * 31 * 2 * WIDTH * WIDTH
            ),
            "first_layer_fwht_additions": (
                2 * FIELD_SIZE * WIDTH * WIDTH * 8
            ),
            "coordinate_basis_first_layer_dense": (
                2 * WIDTH * 2 * WIDTH * WIDTH
            ),
            "note": (
                "Conservative sketch; ReLU/stacking and rotation costs omitted."
            ),
        },
        "summaries": summaries,
        "records": records,
    }
    print({"summaries": summaries}, flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
