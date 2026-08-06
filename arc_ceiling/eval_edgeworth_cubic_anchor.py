"""Oracle-pre Edgeworth anchor test for the late rank-4 cubic control.

The pointwise control from ``eval_crossfit_cumulant_control.py`` is

    g_k(x) = (u_k^T h(x)^2) (v_k^T h(x)) / r^2 - a_k,

where ``h`` is the layer-29 post-ReLU activation and

    a_k = u_k^T E[h^2 h^T] v_k / (width + 1).

This script tests whether the anchor can be recovered from the *oracle*
pre-ReLU bivariate moments through fourth order.  The Gaussian term is
analytic.  Edgeworth corrections are evaluated as derivatives of that
analytic Gaussian moment with respect to its two standardized means:

    F
    + sum K3[p,q] D[p,q] F / (p! q!)
    + sum K4[p,q] D[p,q] F / (p! q!)
    + 1/2 sum K3[p,q] K3[r,s] D[p+r,q+s] F
          / (p! q! r! s!).

The final term is required: kappa_3 squared is the same Edgeworth order as
kappa_4.  Derivatives use a centered high-order finite-difference stencil.
This is an oracle-state feasibility test, not a deployable estimator.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from whest.gaussmath import bvn_cdf  # noqa: E402
from eval_crossfit_cumulant_control import (  # noqa: E402
    crossfit_grid,
    empirical_c21_state,
    forward_layer_and_final,
    pointwise_features,
)
from eval_exact_anchor_residual import FULL_DATA  # noqa: E402
from eval_kerdock_design import (  # noqa: E402
    WIDTH,
    make_kerdock_design,
    random_rotation,
)
from eval_oracle_cumulant_bridge import moment_path  # noqa: E402
from eval_sampling_official import _load_rows  # noqa: E402
from exact_moments import sphere_radius_mean  # noqa: E402


INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def phi(value: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(value)) * INV_SQRT_2PI


def standardized_gaussian_m21(
    tx: np.ndarray,
    ty: np.ndarray,
    rho: np.ndarray,
    *,
    nodes: int,
) -> np.ndarray:
    """E[(tx+U)_+^2 (ty+V)_+] for standard correlated ``(U,V)``."""
    rho = np.clip(np.asarray(rho, dtype=np.float64), -1 + 1e-10, 1 - 1e-10)
    threshold_x = -np.asarray(tx, dtype=np.float64)
    threshold_y = -np.asarray(ty, dtype=np.float64)
    root = np.sqrt(np.maximum(1.0 - np.square(rho), 1e-20))

    probability = bvn_cdf(
        -threshold_x,
        -threshold_y,
        rho,
        n_nodes=nodes,
    )
    boundary_x = phi(threshold_x) * ndtr(
        (rho * threshold_x - threshold_y) / root
    )
    boundary_y = phi(threshold_y) * ndtr(
        (rho * threshold_y - threshold_x) / root
    )
    joint_boundary = np.exp(
        -0.5
        * (
            np.square(threshold_x)
            - 2.0 * rho * threshold_x * threshold_y
            + np.square(threshold_y)
        )
        / np.square(root)
    ) / (2.0 * math.pi * root)

    i10 = boundary_x + rho * boundary_y
    i01 = rho * boundary_x + boundary_y
    i11 = (
        rho
        * (
            probability
            + threshold_x * boundary_x
            + threshold_y * boundary_y
        )
        + np.square(root) * joint_boundary
    )
    i20 = (
        probability
        + threshold_x * boundary_x
        + np.square(rho) * threshold_y * boundary_y
        + rho * np.square(root) * joint_boundary
    )
    i21 = (
        rho * (np.square(threshold_x) + 2.0) * boundary_x
        + (
            1.0
            + np.square(rho) * (np.square(threshold_y) + 1.0)
        )
        * boundary_y
        + np.square(root)
        * (threshold_x + rho * threshold_y)
        * joint_boundary
    )
    return (
        np.square(tx) * ty * probability
        + np.square(tx) * i01
        + 2.0 * tx * ty * i10
        + 2.0 * tx * i11
        + ty * i20
        + i21
    )


def standardized_gaussian_relu_cube(t: np.ndarray) -> np.ndarray:
    """E[(t+Z)_+^3], Z standard normal."""
    return (
        (np.power(t, 3) + 3.0 * t) * ndtr(t)
        + (np.square(t) + 2.0) * phi(t)
    )


def derivative_weights(
    maximum_order: int,
    half_width: int,
    step: float,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    offsets = step * np.arange(-half_width, half_width + 1, dtype=np.float64)
    vandermonde = np.vstack(
        [np.power(offsets, degree) for degree in range(len(offsets))]
    )
    weights = {}
    for order in range(maximum_order + 1):
        target = np.zeros(len(offsets), dtype=np.float64)
        target[order] = math.factorial(order)
        weights[order] = np.linalg.solve(vandermonde, target)
    return offsets, weights


def raw_pair_moments(
    data: np.lib.npyio.NpzFile,
    layer: int,
) -> dict[tuple[int, int], np.ndarray]:
    mean = np.asarray(data["pre_mean"][layer], dtype=np.float64)
    second = np.asarray(data["pre_m2"][layer], dtype=np.float64)
    third = np.asarray(data["pre_m3"][layer], dtype=np.float64)
    fourth = np.asarray(data["pre_m4"][layer], dtype=np.float64)
    m11 = np.asarray(data["pre_M11"][layer], dtype=np.float64)
    m21 = np.asarray(data["pre_M21"][layer], dtype=np.float64)
    m31 = np.asarray(data["pre_M31"][layer], dtype=np.float64)
    m22 = np.asarray(data["pre_M22"][layer], dtype=np.float64)
    shape = (len(mean), len(mean))

    def rows(value: np.ndarray) -> np.ndarray:
        return np.broadcast_to(value[:, None], shape)

    def columns(value: np.ndarray) -> np.ndarray:
        return np.broadcast_to(value[None, :], shape)

    return {
        (0, 0): np.ones(shape, dtype=np.float64),
        (1, 0): rows(mean),
        (0, 1): columns(mean),
        (2, 0): rows(second),
        (0, 2): columns(second),
        (1, 1): m11,
        (3, 0): rows(third),
        (0, 3): columns(third),
        (2, 1): m21,
        (1, 2): m21.T,
        (4, 0): rows(fourth),
        (0, 4): columns(fourth),
        (3, 1): m31,
        (1, 3): m31.T,
        (2, 2): m22,
    }


def central_pair_moment(
    raw: dict[tuple[int, int], np.ndarray],
    mean_x: np.ndarray,
    mean_y: np.ndarray,
    power_x: int,
    power_y: int,
) -> np.ndarray:
    result = np.zeros_like(mean_x)
    for index_x in range(power_x + 1):
        for index_y in range(power_y + 1):
            result += (
                math.comb(power_x, index_x)
                * math.comb(power_y, index_y)
                * np.power(-mean_x, power_x - index_x)
                * np.power(-mean_y, power_y - index_y)
                * raw[index_x, index_y]
            )
    return result


def standardized_cumulants(
    data: np.lib.npyio.NpzFile,
    layer: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[tuple[int, int], np.ndarray],
    dict[tuple[int, int], np.ndarray],
]:
    raw = raw_pair_moments(data, layer)
    mean_x = raw[1, 0]
    mean_y = raw[0, 1]
    covariance_x = central_pair_moment(raw, mean_x, mean_y, 2, 0)
    covariance_y = central_pair_moment(raw, mean_x, mean_y, 0, 2)
    covariance_xy = central_pair_moment(raw, mean_x, mean_y, 1, 1)
    sd_x = np.sqrt(np.maximum(covariance_x, 1e-20))
    sd_y = np.sqrt(np.maximum(covariance_y, 1e-20))
    rho = np.clip(covariance_xy / (sd_x * sd_y), -1.0, 1.0)

    third = {}
    for power_x in range(4):
        power_y = 3 - power_x
        value = central_pair_moment(
            raw,
            mean_x,
            mean_y,
            power_x,
            power_y,
        )
        third[power_x, power_y] = value / (
            np.power(sd_x, power_x) * np.power(sd_y, power_y)
        )

    fourth = {}
    for power_x in range(5):
        power_y = 4 - power_x
        value = central_pair_moment(
            raw,
            mean_x,
            mean_y,
            power_x,
            power_y,
        )
        if (power_x, power_y) == (4, 0):
            value -= 3.0 * np.square(covariance_x)
        elif (power_x, power_y) == (3, 1):
            value -= 3.0 * covariance_x * covariance_xy
        elif (power_x, power_y) == (2, 2):
            value -= (
                covariance_x * covariance_y
                + 2.0 * np.square(covariance_xy)
            )
        elif (power_x, power_y) == (1, 3):
            value -= 3.0 * covariance_y * covariance_xy
        elif (power_x, power_y) == (0, 4):
            value -= 3.0 * np.square(covariance_y)
        fourth[power_x, power_y] = value / (
            np.power(sd_x, power_x) * np.power(sd_y, power_y)
        )

    mean = np.asarray(data["pre_mean"][layer], dtype=np.float64)
    marginal_sd = np.sqrt(
        np.maximum(
            np.asarray(data["pre_m2"][layer], dtype=np.float64)
            - np.square(mean),
            1e-20,
        )
    )
    return mean, marginal_sd, rho, third, fourth


def edgeworth_m21_matrices(
    data: np.lib.npyio.NpzFile,
    layer: int,
    *,
    step: float,
    half_width: int,
    chunk: int,
    nodes: int,
) -> dict[str, np.ndarray]:
    mean, marginal_sd, rho, third, fourth = standardized_cumulants(data, layer)
    width = len(mean)
    row, column = np.where(~np.eye(width, dtype=bool))
    tx = mean[row] / marginal_sd[row]
    ty = mean[column] / marginal_sd[column]
    pair_rho = rho[row, column]
    scale = np.square(marginal_sd[row]) * marginal_sd[column]
    offsets, weights = derivative_weights(6, half_width, step)
    center = half_width

    result_flat = {
        "gaussian": np.empty(len(row), dtype=np.float64),
        "third": np.empty(len(row), dtype=np.float64),
        "third_fourth": np.empty(len(row), dtype=np.float64),
        "full": np.empty(len(row), dtype=np.float64),
    }
    factorial = [math.factorial(order) for order in range(7)]

    for start in range(0, len(row), chunk):
        stop = min(start + chunk, len(row))
        selection = slice(start, stop)
        grid = standardized_gaussian_m21(
            tx[selection, None, None] + offsets[None, :, None],
            ty[selection, None, None] + offsets[None, None, :],
            pair_rho[selection, None, None],
            nodes=nodes,
        )
        derivatives = {}
        for order_x in range(7):
            for order_y in range(7 - order_x):
                if order_x + order_y not in (0, 3, 4, 6):
                    continue
                derivatives[order_x, order_y] = np.einsum(
                    "a,b,cab->c",
                    weights[order_x],
                    weights[order_y],
                    grid,
                    optimize=True,
                )

        correction_three = np.zeros(stop - start, dtype=np.float64)
        for order_x in range(4):
            order_y = 3 - order_x
            correction_three += (
                third[order_x, order_y][row[selection], column[selection]]
                * derivatives[order_x, order_y]
                / (factorial[order_x] * factorial[order_y])
            )

        correction_four = np.zeros(stop - start, dtype=np.float64)
        for order_x in range(5):
            order_y = 4 - order_x
            correction_four += (
                fourth[order_x, order_y][row[selection], column[selection]]
                * derivatives[order_x, order_y]
                / (factorial[order_x] * factorial[order_y])
            )

        correction_three_squared = np.zeros(stop - start, dtype=np.float64)
        for first_x in range(4):
            first_y = 3 - first_x
            first = third[first_x, first_y][row[selection], column[selection]]
            for second_x in range(4):
                second_y = 3 - second_x
                second = third[second_x, second_y][
                    row[selection],
                    column[selection],
                ]
                correction_three_squared += (
                    0.5
                    * first
                    * second
                    * derivatives[first_x + second_x, first_y + second_y]
                    / (
                        factorial[first_x]
                        * factorial[first_y]
                        * factorial[second_x]
                        * factorial[second_y]
                    )
                )

        base = grid[:, center, center]
        result_flat["gaussian"][selection] = scale[selection] * base
        result_flat["third"][selection] = scale[selection] * (
            base + correction_three
        )
        result_flat["third_fourth"][selection] = scale[selection] * (
            base + correction_three + correction_four
        )
        result_flat["full"][selection] = scale[selection] * (
            base
            + correction_three
            + correction_four
            + correction_three_squared
        )

    # The diagonal is the univariate ReLU cube.  Treating rho=1 through the
    # bivariate formula is both slower and numerically singular.
    diagonal_tx = mean / marginal_sd
    diagonal_grid = standardized_gaussian_relu_cube(
        diagonal_tx[:, None] + offsets[None, :]
    )
    diagonal_derivatives = {
        order: diagonal_grid @ weights[order]
        for order in (0, 3, 4, 6)
    }
    diagonal_third = np.diag(third[3, 0])
    diagonal_fourth = np.diag(fourth[4, 0])
    diagonal_values = {
        "gaussian": diagonal_derivatives[0],
        "third": (
            diagonal_derivatives[0]
            + diagonal_third * diagonal_derivatives[3] / 6.0
        ),
        "third_fourth": (
            diagonal_derivatives[0]
            + diagonal_third * diagonal_derivatives[3] / 6.0
            + diagonal_fourth * diagonal_derivatives[4] / 24.0
        ),
        "full": (
            diagonal_derivatives[0]
            + diagonal_third * diagonal_derivatives[3] / 6.0
            + diagonal_fourth * diagonal_derivatives[4] / 24.0
            + np.square(diagonal_third) * diagonal_derivatives[6] / 72.0
        ),
    }

    matrices = {}
    for label, flat in result_flat.items():
        matrix = np.empty((width, width), dtype=np.float64)
        matrix[row, column] = flat
        np.fill_diagonal(
            matrix,
            np.power(marginal_sd, 3) * diagonal_values[label],
        )
        matrices[label] = matrix
    return matrices


def contraction(
    left: np.ndarray,
    moment: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    return np.einsum("ik,ij,jk->k", left, moment, right) / (WIDTH + 1)


def sample_pre_moment_data(
    captured_pre: np.ndarray,
    radius: float,
) -> dict[str, np.ndarray]:
    """Estimate Gaussian pre moments from the same fixed-radius Kerdock cloud.

    The network is positively homogeneous, so total-degree ``k`` angular
    moments are converted to Gaussian moments by ``E[R^k] / radius^k``.
    """
    h = np.asarray(captured_pre, dtype=np.float32)
    count = len(h)
    squared = np.square(h)
    cubed = squared * h
    radial_two = WIDTH / np.square(radius)
    radial_three = (WIDTH + 1.0) / np.square(radius)
    radial_four = WIDTH * (WIDTH + 2.0) / np.power(radius, 4)

    mean = np.mean(h, axis=0, dtype=np.float64)
    m11 = (h.T @ h).astype(np.float64) * (radial_two / count)
    m21 = (squared.T @ h).astype(np.float64) * (radial_three / count)
    m22 = (squared.T @ squared).astype(np.float64) * (
        radial_four / count
    )
    m31 = (cubed.T @ h).astype(np.float64) * (radial_four / count)
    return {
        "pre_mean": mean[None, :],
        "pre_m2": np.diag(m11)[None, :],
        "pre_m3": np.diag(m21)[None, :],
        "pre_m4": np.diag(m22)[None, :],
        "pre_M11": m11[None, :, :],
        "pre_M21": m21[None, :, :],
        "pre_M22": m22[None, :, :],
        "pre_M31": m31[None, :, :],
    }


def moment_data_from_state(
    mean: np.ndarray,
    covariance: np.ndarray,
    c21: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct raw moments through order four with zero fourth cumulant."""
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    c21 = np.asarray(c21, dtype=np.float64)
    shape = covariance.shape
    mean_x = np.broadcast_to(mean[:, None], shape)
    mean_y = np.broadcast_to(mean[None, :], shape)
    variance = np.diag(covariance)
    variance_x = np.broadcast_to(variance[:, None], shape)
    variance_y = np.broadcast_to(variance[None, :], shape)
    marginal_c3 = np.diag(c21)

    central = {
        (0, 0): np.ones(shape, dtype=np.float64),
        (1, 0): np.zeros(shape, dtype=np.float64),
        (0, 1): np.zeros(shape, dtype=np.float64),
        (2, 0): variance_x,
        (0, 2): variance_y,
        (1, 1): covariance,
        (3, 0): np.broadcast_to(marginal_c3[:, None], shape),
        (0, 3): np.broadcast_to(marginal_c3[None, :], shape),
        (2, 1): c21,
        (1, 2): c21.T,
        # Kappa_4 = 0, so the fourth central moments are Wick pairings.
        (4, 0): 3.0 * np.square(variance_x),
        (0, 4): 3.0 * np.square(variance_y),
        (3, 1): 3.0 * variance_x * covariance,
        (1, 3): 3.0 * variance_y * covariance,
        (2, 2): (
            variance_x * variance_y + 2.0 * np.square(covariance)
        ),
    }

    def raw(power_x: int, power_y: int) -> np.ndarray:
        result = np.zeros(shape, dtype=np.float64)
        for index_x in range(power_x + 1):
            for index_y in range(power_y + 1):
                result += (
                    math.comb(power_x, index_x)
                    * math.comb(power_y, index_y)
                    * np.power(mean_x, power_x - index_x)
                    * np.power(mean_y, power_y - index_y)
                    * central[index_x, index_y]
                )
        return result

    m11 = raw(1, 1)
    m21 = raw(2, 1)
    m22 = raw(2, 2)
    m31 = raw(3, 1)
    return {
        "pre_mean": mean[None, :],
        "pre_m2": np.diag(raw(2, 0))[None, :],
        "pre_m3": np.diag(raw(3, 0))[None, :],
        "pre_m4": np.diag(raw(4, 0))[None, :],
        "pre_M11": m11[None, :, :],
        "pre_M21": m21[None, :, :],
        "pre_M22": m22[None, :, :],
        "pre_M31": m31[None, :, :],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=list(range(160, 168)),
    )
    parser.add_argument("--layer", type=int, default=29)
    parser.add_argument("--rotation-seed", type=int, default=3)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--step", type=float, default=0.2)
    parser.add_argument("--half-width", type=int, default=5)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--nodes", type=int, default=12)
    parser.add_argument(
        "--factorized-dir",
        type=Path,
        default=HERE / "results" / "factorized_k3_layer29",
    )
    parser.add_argument(
        "--factorized-scales",
        type=float,
        nargs="+",
        default=[1.0],
    )
    parser.add_argument(
        "--prediction-shrinks",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Shrink the correction from factorized-anchor predictions back "
            "toward the baseline. This keeps the exact same zero-mean "
            "control but scales its fitted coefficient."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "results" / "edgeworth_cubic_anchor_holdout8.json",
    )
    args = parser.parse_args()

    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    radius = sphere_radius_mean(WIDTH)
    rows = _load_rows(FULL_DATA, args.indices)
    records = []

    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        started = time.perf_counter()
        with np.load(moment_path(index)) as moment_data:
            true_raw_m21 = np.asarray(
                moment_data["M21"][args.layer],
                dtype=np.float64,
            )
            oracle_pre_mean = np.asarray(
                moment_data["pre_mean"][args.layer],
                dtype=np.float64,
            )
            oracle_pre_second = np.asarray(
                moment_data["pre_M11"][args.layer],
                dtype=np.float64,
            )
            oracle_pre_covariance = (
                oracle_pre_second
                - np.outer(oracle_pre_mean, oracle_pre_mean)
            )
            approximations = edgeworth_m21_matrices(
                moment_data,
                args.layer,
                step=args.step,
                half_width=args.half_width,
                chunk=args.chunk,
                nodes=args.nodes,
            )

        captured_pre, captured, final = forward_layer_and_final(
            weights,
            points,
            rotation,
            args.layer,
        )
        left, right, sample_raw_m21 = empirical_c21_state(
            captured,
            args.rank,
        )
        true_anchor = contraction(left, true_raw_m21, right)
        sample_anchor = (
            np.einsum("ik,ij,jk->k", left, sample_raw_m21, right)
            / np.square(radius)
        )
        anchors = {
            "oracle": true_anchor,
            "sample": sample_anchor,
            **{
                label: contraction(left, moment, right)
                for label, moment in approximations.items()
            },
        }
        sample_pre_data = sample_pre_moment_data(captured_pre, radius)
        sample_pre_approximations = edgeworth_m21_matrices(
            sample_pre_data,
            0,
            step=args.step,
            half_width=args.half_width,
            chunk=args.chunk,
            nodes=args.nodes,
        )
        anchors.update(
            {
                f"sample_pre_{label}": contraction(left, moment, right)
                for label, moment in sample_pre_approximations.items()
            }
        )
        factorized_path = args.factorized_dir / f"mlp_{index:05d}.npz"
        if factorized_path.exists():
            with np.load(factorized_path) as factorized:
                factorized_pre_mean = np.asarray(
                    factorized["pre_mean"],
                    dtype=np.float64,
                )
                factorized_pre_covariance = np.asarray(
                    factorized["pre_covariance"],
                    dtype=np.float64,
                )
                factorized_pre_c21 = np.asarray(
                    factorized["pre_c21"],
                    dtype=np.float64,
                )
            factorized_states = {
                "factorized_pre": (
                    factorized_pre_mean,
                    factorized_pre_covariance,
                ),
                "oracle_mean_cov_factorized_c21": (
                    oracle_pre_mean,
                    oracle_pre_covariance,
                ),
                "sample_mean_cov_factorized_c21": (
                    np.asarray(sample_pre_data["pre_mean"][0], dtype=np.float64),
                    (
                        np.asarray(
                            sample_pre_data["pre_M11"][0],
                            dtype=np.float64,
                        )
                        - np.outer(
                            sample_pre_data["pre_mean"][0],
                            sample_pre_data["pre_mean"][0],
                        )
                    ),
                ),
            }
            for state_label, (state_mean, state_covariance) in (
                factorized_states.items()
            ):
                state_data = moment_data_from_state(
                    state_mean,
                    state_covariance,
                    factorized_pre_c21,
                )
                state_approximation = edgeworth_m21_matrices(
                    state_data,
                    0,
                    step=args.step,
                    half_width=args.half_width,
                    chunk=args.chunk,
                    nodes=args.nodes,
                )
                gaussian_moment = state_approximation["gaussian"]
                third_delta = (
                    state_approximation["third"] - gaussian_moment
                )
                for scale in args.factorized_scales:
                    scaled_moment = gaussian_moment + scale * third_delta
                    anchors[
                        f"{state_label}_third_scale{scale:g}"
                    ] = contraction(left, scaled_moment, right)
                if state_label == "sample_mean_cov_factorized_c21":
                    transported_delta = contraction(
                        left,
                        (
                            state_approximation["third"]
                            - sample_pre_approximations["third"]
                        ),
                        right,
                    )
                    for scale in args.factorized_scales:
                        anchors[
                            "sample_anchor_plus_factorized_c21_delta"
                            f"_scale{scale:g}"
                        ] = sample_anchor + scale * transported_delta
                        anchors[
                            "sample_pre_third_plus_factorized_c21_delta"
                            f"_scale{scale:g}"
                        ] = (
                            anchors["sample_pre_third"]
                            + scale * transported_delta
                        )

        baseline_prediction = final.mean(axis=0, dtype=np.float64)
        baseline_mse = float(
            np.mean(np.square(baseline_prediction - targets[-1]))
        )
        method_mses = {}
        method_predictions = {}
        for label, anchor in anchors.items():
            features = pointwise_features(
                captured,
                left,
                right,
                anchor,
                radius,
            )
            predictions, _ = crossfit_grid(
                features,
                final,
                args.folds,
                [args.ridge],
            )
            prediction = predictions[args.ridge]
            method_predictions[label] = prediction
            method_mses[label] = float(
                np.mean(np.square(prediction - targets[-1]))
            )
        shrink_sources = {}
        for label, prediction in tuple(method_predictions.items()):
            if "factorized_c21" not in label:
                continue
            for shrink in args.prediction_shrinks:
                shrink_label = f"{label}_prediction_shrink{shrink:g}"
                shrunk_prediction = (
                    baseline_prediction
                    + shrink * (prediction - baseline_prediction)
                )
                method_mses[shrink_label] = float(
                    np.mean(np.square(shrunk_prediction - targets[-1]))
                )
                shrink_sources[shrink_label] = (label, shrink)

        same_cloud_error = max(
            float(np.linalg.norm(sample_anchor - true_anchor)),
            1e-30,
        )
        anchor_diagnostics = {
            label: {
                "absolute_error_norm": float(np.linalg.norm(anchor - true_anchor)),
                "relative_to_same_cloud": float(
                    np.linalg.norm(anchor - true_anchor) / same_cloud_error
                ),
                "error": (anchor - true_anchor).tolist(),
            }
            for label, anchor in anchors.items()
            if label != "oracle"
        }
        for label, (source, shrink) in shrink_sources.items():
            source_diagnostic = anchor_diagnostics[source]
            anchor_diagnostics[label] = {
                **source_diagnostic,
                "prediction_shrink": shrink,
                "source_anchor": source,
            }
        record = {
            "index": index,
            "name": name,
            "baseline_mse": baseline_mse,
            "method_mses": method_mses,
            "anchor_diagnostics": anchor_diagnostics,
            "seconds": time.perf_counter() - started,
        }
        records.append(record)
        best = min(method_mses, key=method_mses.get)
        print(
            f"[{index}] base={baseline_mse:.4e} best={best} "
            f"{method_mses[best] / baseline_mse:.4f}x "
            f"anchor(full/sample)="
            f"{anchor_diagnostics['full']['relative_to_same_cloud']:.3f} "
            f"({record['seconds']:.1f}s)",
            flush=True,
        )

    baseline = np.asarray([record["baseline_mse"] for record in records])
    labels = list(records[0]["method_mses"])
    summary = {}
    for label in labels:
        mse = np.asarray(
            [record["method_mses"][label] for record in records],
            dtype=np.float64,
        )
        relative_anchor = np.asarray(
            [
                record["anchor_diagnostics"][label][
                    "relative_to_same_cloud"
                ]
                for record in records
            ],
            dtype=np.float64,
        ) if label != "oracle" else np.zeros(len(records))
        summary[label] = {
            "mse_ratio": float(np.mean(mse) / np.mean(baseline)),
            "wins": int(np.sum(mse < baseline)),
            "worst": float(np.max(mse / baseline)),
            "mean_anchor_error_relative_to_same_cloud": float(
                np.mean(relative_anchor)
            ),
        }

    output = {
        "protocol": {
            "indices": args.indices,
            "layer": args.layer,
            "rotation_seed": args.rotation_seed,
            "rank": args.rank,
            "folds": args.folds,
            "ridge": args.ridge,
            "step": args.step,
            "half_width": args.half_width,
            "chunk": args.chunk,
            "nodes": args.nodes,
            "oracle_pre_moments": True,
        },
        "summary": summary,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
