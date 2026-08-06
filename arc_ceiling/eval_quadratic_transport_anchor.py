"""Positive Hermite-transport resummation for the late cubic anchor.

The ordinary third-order Edgeworth approximation applies the signed-density
operator

    1 + K3 : D^3 / 3!

to a Gaussian ReLU moment.  Here we instead push a Gaussian through the
quadratic Hermite map

    Z_a = G_a + K3[a,b,c] (G_b G_c - delta_bc) / 6.

Its third cumulant is K3 to first order, but evaluating the nonlinear ReLU
observable after the map automatically resums every power of K3 and always
corresponds to a positive distribution.  A second variant renormalizes the
known O(K3^2) covariance inflation before applying the target covariance.

This is an anchor-isolation experiment.  It compares oracle and factorized
pre-ReLU K3 while holding the pre mean/covariance oracle.  It intentionally
does not duplicate the sample-mean/covariance factorized-K3 scale sweep.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from eval_crossfit_cumulant_control import (  # noqa: E402
    crossfit_grid,
    empirical_c21_state,
    forward_layer_and_final,
    pointwise_features,
)
from eval_edgeworth_cubic_anchor import (  # noqa: E402
    central_pair_moment,
    contraction,
    edgeworth_m21_matrices,
    moment_data_from_state,
    raw_pair_moments,
    sample_pre_moment_data,
)
from eval_exact_anchor_residual import (  # noqa: E402
    FULL_DATA,
    ROWS_PER_BASIS,
)
from eval_kerdock_design import (  # noqa: E402
    N_BASES,
    WIDTH,
    make_kerdock_design,
    random_rotation,
)
from eval_oracle_cumulant_bridge import moment_path  # noqa: E402
from eval_sampling_official import _load_rows  # noqa: E402
from exact_moments import sphere_radius_mean  # noqa: E402


def pre_state(
    data: np.lib.npyio.NpzFile,
    layer: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pre-ReLU mean, covariance and connected K3[i,i,j]."""
    raw = raw_pair_moments(data, layer)
    mean = np.asarray(data["pre_mean"][layer], dtype=np.float64)
    mean_x = raw[1, 0]
    mean_y = raw[0, 1]
    covariance = central_pair_moment(raw, mean_x, mean_y, 1, 1)
    c21 = central_pair_moment(raw, mean_x, mean_y, 2, 1)
    return mean, 0.5 * (covariance + covariance.T), c21


def _quadratic_transport_pairs(
    mean: np.ndarray,
    covariance: np.ndarray,
    c21: np.ndarray,
    *,
    nodes: int,
    chunk: int,
    normalize_covariance: bool,
    whitening_rho_limit: float = 1.0 - 1e-7,
) -> np.ndarray:
    """Approximate E[X_+^2 Y_+] for every ordered coordinate pair."""
    width = len(mean)
    row, column = np.where(~np.eye(width, dtype=bool))
    variance = np.maximum(np.diag(covariance), 1e-20)
    sd = np.sqrt(variance)
    sx = sd[row]
    sy = sd[column]
    rho = np.clip(
        covariance[row, column] / (sx * sy),
        -1.0 + 1e-7,
        1.0 - 1e-7,
    )
    root = np.sqrt(np.maximum(1.0 - np.square(rho), 1e-14))
    whitening_rho = np.clip(
        rho,
        -whitening_rho_limit,
        whitening_rho_limit,
    )
    whitening_root = np.sqrt(
        np.maximum(1.0 - np.square(whitening_rho), 1e-14)
    )

    # X = mean + L Z, with lower-triangular pairwise square root L.
    pair_count = len(row)
    linear = np.zeros((pair_count, 2, 2), dtype=np.float64)
    linear[:, 0, 0] = sx
    linear[:, 1, 0] = sy * rho
    linear[:, 1, 1] = sy * root
    inverse = np.zeros_like(linear)
    inverse[:, 0, 0] = 1.0 / sx
    inverse[:, 1, 0] = -whitening_rho / (sx * whitening_root)
    inverse[:, 1, 1] = 1.0 / (sy * whitening_root)

    raw_k3 = np.empty((pair_count, 2, 2, 2), dtype=np.float64)
    kxxx = np.diag(c21)[row]
    kxxy = c21[row, column]
    kxyy = c21[column, row]
    kyyy = np.diag(c21)[column]
    raw_k3[:, 0, 0, 0] = kxxx
    raw_k3[:, 0, 0, 1] = kxxy
    raw_k3[:, 0, 1, 0] = kxxy
    raw_k3[:, 1, 0, 0] = kxxy
    raw_k3[:, 0, 1, 1] = kxyy
    raw_k3[:, 1, 0, 1] = kxyy
    raw_k3[:, 1, 1, 0] = kxyy
    raw_k3[:, 1, 1, 1] = kyyy
    white_k3 = np.einsum(
        "nai,nbj,nck,nijk->nabc",
        inverse,
        inverse,
        inverse,
        raw_k3,
        optimize=True,
    )

    abscissa, one_weights = hermgauss(nodes)
    standard_nodes = math.sqrt(2.0) * abscissa
    one_weights = one_weights / math.sqrt(math.pi)
    gx, gy = np.meshgrid(standard_nodes, standard_nodes, indexing="ij")
    wx, wy = np.meshgrid(one_weights, one_weights, indexing="ij")
    gaussian = np.stack([gx.ravel(), gy.ravel()], axis=1)
    weights = (wx * wy).ravel()
    centered_quadratic = (
        np.einsum("ma,mb->mab", gaussian, gaussian)
        - np.eye(2, dtype=np.float64)[None, :, :]
    )

    flat = np.empty(pair_count, dtype=np.float64)
    pair_mean = np.stack([mean[row], mean[column]], axis=1)
    for start in range(0, pair_count, chunk):
        stop = min(start + chunk, pair_count)
        k3 = white_k3[start:stop]
        delta = np.einsum(
            "nabc,mbc->nma",
            k3,
            centered_quadratic,
            optimize=True,
        ) / 6.0
        transported = gaussian[None, :, :] + delta

        if normalize_covariance:
            # Cov(delta)_ij = sum_ab K3[i,a,b] K3[j,a,b] / 18.
            inflation = (
                np.einsum("niab,njab->nij", k3, k3, optimize=True) / 18.0
            )
            s00 = np.maximum(1.0 + inflation[:, 0, 0], 1e-12)
            s11 = np.maximum(1.0 + inflation[:, 1, 1], 1e-12)
            s10 = inflation[:, 1, 0]
            chol00 = np.sqrt(s00)
            chol10 = s10 / chol00
            chol11 = np.sqrt(np.maximum(s11 - np.square(chol10), 1e-12))
            first = transported[:, :, 0] / chol00[:, None]
            second = (
                transported[:, :, 1] - chol10[:, None] * first
            ) / chol11[:, None]
            transported = np.stack([first, second], axis=2)

        values = (
            pair_mean[start:stop, None, :]
            + np.einsum(
                "nij,nmj->nmi",
                linear[start:stop],
                transported,
                optimize=True,
            )
        )
        relu = np.maximum(values, 0.0)
        flat[start:stop] = np.sum(
            np.square(relu[:, :, 0]) * relu[:, :, 1] * weights[None, :],
            axis=1,
        )

    # Univariate diagonal under the corresponding one-dimensional map.
    gamma = np.diag(c21) / np.power(sd, 3)
    q = (
        standard_nodes[None, :]
        + gamma[:, None]
        * (np.square(standard_nodes)[None, :] - 1.0)
        / 6.0
    )
    if normalize_covariance:
        q /= np.sqrt(1.0 + np.square(gamma) / 18.0)[:, None]
    diagonal = np.sum(
        np.power(
            np.maximum(mean[:, None] + sd[:, None] * q, 0.0),
            3,
        )
        * one_weights[None, :],
        axis=1,
    )

    result = np.empty((width, width), dtype=np.float64)
    result[row, column] = flat
    np.fill_diagonal(result, diagonal)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", default=[160, 161])
    parser.add_argument("--layer", type=int, default=29)
    parser.add_argument("--rotation-seed", type=int, default=3)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument(
        "--blend-alphas",
        type=float,
        nargs="+",
        default=[1.0],
        help=(
            "Fixed shrinkage applied to each control correction relative to "
            "the baseline estimator."
        ),
    )
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument(
        "--factorized-rho-limits",
        type=float,
        nargs="+",
        default=[0.99, 0.95, 0.9, 0.8],
    )
    parser.add_argument(
        "--factorized-scales",
        type=float,
        nargs="+",
        default=[1.0],
    )
    parser.add_argument(
        "--factorized-dir",
        type=Path,
        default=HERE / "results" / "factorized_k3_layer29",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "results" / "quadratic_transport_anchor_smoke.json",
    )
    args = parser.parse_args()

    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    radius = sphere_radius_mean(WIDTH)
    rows = _load_rows(FULL_DATA, args.indices)
    records = []

    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        started = time.perf_counter()
        with np.load(moment_path(index)) as data:
            true_raw_m21 = np.asarray(data["M21"][args.layer], dtype=np.float64)
            mean, covariance, oracle_c21 = pre_state(data, args.layer)
        factorized_path = args.factorized_dir / f"mlp_{index:05d}.npz"
        with np.load(factorized_path) as state:
            factorized_c21 = np.asarray(state["pre_c21"], dtype=np.float64)

        captured_pre, captured, final = forward_layer_and_final(
            weights,
            points,
            rotation,
            args.layer,
        )
        left, right, sample_raw_m21 = empirical_c21_state(captured, args.rank)
        true_anchor = contraction(left, true_raw_m21, right)
        sample_anchor = (
            np.einsum("ik,ij,jk->k", left, sample_raw_m21, right)
            / np.square(radius)
        )
        anchors = {"oracle": true_anchor, "sample": sample_anchor}
        for normalized in (False, True):
            label = f"oracle_k3_{'covnorm' if normalized else 'raw'}"
            moment = _quadratic_transport_pairs(
                mean,
                covariance,
                oracle_c21,
                nodes=args.nodes,
                chunk=args.chunk,
                normalize_covariance=normalized,
            )
            anchors[label] = contraction(left, moment, right)
        for scale in args.factorized_scales:
            for rho_limit in args.factorized_rho_limits:
                for normalized in (False, True):
                    label = (
                        f"factorized_k3_scale{scale:g}_rho{rho_limit:g}_"
                        f"{'covnorm' if normalized else 'raw'}"
                    )
                    moment = _quadratic_transport_pairs(
                        mean,
                        covariance,
                        scale * factorized_c21,
                        nodes=args.nodes,
                        chunk=args.chunk,
                        normalize_covariance=normalized,
                        whitening_rho_limit=rho_limit,
                    )
                    anchors[label] = contraction(left, moment, right)

        # A target-free scalar calibration of the factorized K3.  Unlike a
        # fixed scale sweep, this uses the aggregate 256x256 preactivation K3
        # observable from the Kerdock cloud.  The four eventual control
        # directions account for a negligible fraction of those entries, so
        # this tests whether global self-calibration can repair rollout
        # amplitude without learning from target errors.
        sample_data = sample_pre_moment_data(captured_pre, radius)
        sample_mean, sample_covariance, sample_c21 = pre_state(sample_data, 0)
        calibration_scales = {
            "ls": float(
                np.sum(factorized_c21 * sample_c21)
                / max(np.sum(np.square(factorized_c21)), 1e-30)
            ),
            "norm": float(
                np.linalg.norm(sample_c21)
                / max(np.linalg.norm(factorized_c21), 1e-30)
            ),
        }
        for scale_label, scale in calibration_scales.items():
            adaptive_data = moment_data_from_state(
                sample_mean,
                sample_covariance,
                scale * factorized_c21,
            )
            adaptive_moment = edgeworth_m21_matrices(
                adaptive_data,
                0,
                step=0.2,
                half_width=5,
                chunk=args.chunk,
                nodes=12,
            )["third"]
            anchors[f"adaptive_{scale_label}_edgeworth"] = contraction(
                left,
                adaptive_moment,
                right,
            )

        # Fully deployable measure of how strongly an analytic anchor moves
        # away from the same-cloud estimate, in Kerdock block standard-error
        # units.  This supports a single predeclared upper-threshold safety
        # gate without consulting the unknown target expectation.
        uncentered_features = (
            (np.square(captured) @ left) * (captured @ right)
            / np.square(radius)
        )
        block_means = uncentered_features.reshape(
            N_BASES,
            ROWS_PER_BASIS,
            args.rank,
        ).mean(axis=1)
        block_standard_error = np.std(
            block_means,
            axis=0,
            ddof=1,
        ) / np.sqrt(N_BASES)
        safe_standard_error = np.maximum(block_standard_error, 1e-30)
        deployable_anchor_shift_z = {
            label: float(
                np.sqrt(
                    np.mean(
                        np.square(
                            (anchor - sample_anchor)
                            / safe_standard_error
                        )
                    )
                )
            )
            for label, anchor in anchors.items()
        }

        baseline_prediction = final.mean(axis=0, dtype=np.float64)
        baseline_mse = float(
            np.mean(np.square(baseline_prediction - targets[-1]))
        )
        method_mses = {}
        method_anchor_labels = {}
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
            control_prediction = predictions[args.ridge]
            for alpha in args.blend_alphas:
                blended_prediction = baseline_prediction + alpha * (
                    control_prediction - baseline_prediction
                )
                blended_label = (
                    label
                    if alpha == 1.0
                    else f"{label}_blend{alpha:g}"
                )
                method_mses[blended_label] = float(
                    np.mean(
                        np.square(blended_prediction - targets[-1])
                    )
                )
                method_anchor_labels[blended_label] = label

        same_cloud_error = max(
            float(np.linalg.norm(sample_anchor - true_anchor)),
            1e-30,
        )
        anchor_diagnostics = {
            label: {
                "relative_to_same_cloud": float(
                    np.linalg.norm(anchor - true_anchor) / same_cloud_error
                ),
                "error": (anchor - true_anchor).tolist(),
            }
            for label, anchor in anchors.items()
            if label != "oracle"
        }
        record = {
            "index": index,
            "name": name,
            "baseline_mse": baseline_mse,
            "method_mses": method_mses,
            "method_anchor_labels": method_anchor_labels,
            "anchor_diagnostics": anchor_diagnostics,
            "deployable_anchor_shift_z": deployable_anchor_shift_z,
            "calibration_scales": calibration_scales,
            "seconds": time.perf_counter() - started,
        }
        records.append(record)
        print(
            f"[{index}] "
            + " ".join(
                f"{label}={mse / baseline_mse:.3f}x"
                for label, mse in method_mses.items()
            )
            + f" ({record['seconds']:.1f}s)",
            flush=True,
        )

    baseline = np.asarray([record["baseline_mse"] for record in records])
    labels = list(records[0]["method_mses"])
    summary = {}
    for label in labels:
        mse = np.asarray([r["method_mses"][label] for r in records])
        anchor_label = records[0]["method_anchor_labels"][label]
        summary[label] = {
            "mse_ratio": float(np.mean(mse) / np.mean(baseline)),
            "wins": int(np.sum(mse < baseline)),
            "worst": float(np.max(mse / baseline)),
            "mean_anchor_error_relative_to_same_cloud": (
                0.0
                if anchor_label == "oracle"
                else float(
                    np.mean(
                        [
                            r["anchor_diagnostics"][anchor_label][
                                "relative_to_same_cloud"
                            ]
                            for r in records
                        ]
                    )
                )
            ),
        }
    output = {
        "protocol": vars(args) | {
            "out": str(args.out),
            "factorized_dir": str(args.factorized_dir),
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
