"""Radially homogenized connected-cubic control with a c21-only anchor.

For a fixed-radius point ``x`` with ``rho = E[||X||]`` and a chosen centering
vector ``m``, define

    S_ij(x)
      = h_i^2 h_j / rho^2
        - d (2 m_i h_i h_j + m_j h_i^2) / (rho^2 (d + 1))
        + 2 m_i^2 h_j / (d + 1).

Positive homogeneity and chi radial moments give the exact spherical mean

    E_sphere[S_ij]
      = (M21_ij - 2 m_i M11_ij - m_j M2_i + 2 m_i^2 mu_j)
        / (d + 1).

When ``m=mu`` this is exactly connected ``c21_ij / (d+1)``.  The experiment
tests whether replacing that anchor with transported factorized c21 avoids
the severe raw-moment cancellation seen by the previous cubic control.
Pointwise coefficients are fitted target-free with held-out Kerdock bases.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from eval_crossfit_cumulant_control import (  # noqa: E402
    crossfit_grid,
    forward_layer_and_final,
)
from eval_exact_anchor_residual import FULL_DATA  # noqa: E402
from eval_kerdock_design import WIDTH, make_kerdock_design, random_rotation  # noqa: E402
from eval_oracle_cumulant_bridge import (  # noqa: E402
    connected_m21,
    moment_path,
    truncated_svd,
)
from eval_sampling_official import _load_rows  # noqa: E402
from exact_moments import sphere_radius_mean  # noqa: E402


def contracted_pointwise(
    activation: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Return the rankwise contraction of ``S_ij(x)``."""
    h = np.asarray(activation, dtype=np.float64)
    m = np.asarray(center, dtype=np.float64)
    h_right = h @ right
    squared_left = np.square(h) @ left
    raw_cubic = squared_left * h_right / np.square(radius)
    centered_bilinear = (
        2.0 * ((h * m[None, :]) @ left) * h_right
        + squared_left * (m @ right)[None, :]
    )
    linear_correction = (
        2.0
        * (np.square(m) @ left)[None, :]
        * h_right
        / (WIDTH + 1.0)
    )
    return (
        raw_cubic
        - WIDTH
        * centered_bilinear
        / (np.square(radius) * (WIDTH + 1.0))
        + linear_correction
    )


def exact_anchor_matrix(
    true_mean: np.ndarray,
    true_second: np.ndarray,
    true_raw_m21: np.ndarray,
    true_marginal_second: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    """Exact spherical anchor for arbitrary fixed centering ``center``."""
    mu = np.asarray(true_mean, dtype=np.float64)
    m = np.asarray(center, dtype=np.float64)
    return (
        np.asarray(true_raw_m21, dtype=np.float64)
        - 2.0 * m[:, None] * np.asarray(true_second, dtype=np.float64)
        - np.asarray(true_marginal_second, dtype=np.float64)[:, None]
        * m[None, :]
        + 2.0 * np.square(m)[:, None] * mu[None, :]
    ) / (WIDTH + 1.0)


def contract(
    left: np.ndarray,
    matrix: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    return np.einsum("ik,ij,jk->k", left, matrix, right)


def sample_direction_families(
    activation: np.ndarray,
    rank: int,
    radius: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return legacy and Gaussian-radially-corrected sample c21 SVDs."""
    h = np.asarray(activation, dtype=np.float64)
    count = len(h)
    mean = np.mean(h, axis=0)
    fixed_second = (h.T @ h) / count
    fixed_m21 = (np.square(h).T @ h) / count
    legacy_c21 = connected_m21(
        mean,
        fixed_second,
        fixed_m21,
        np.diag(fixed_second),
    )

    gaussian_second = WIDTH * fixed_second / np.square(radius)
    gaussian_m21 = (WIDTH + 1.0) * fixed_m21 / np.square(radius)
    corrected_c21 = connected_m21(
        mean,
        gaussian_second,
        gaussian_m21,
        np.diag(gaussian_second),
    )
    return {
        "legacy_dirs": truncated_svd(legacy_c21, rank),
        "radial_corrected_dirs": truncated_svd(corrected_c21, rank),
    }


def summarize(records: list[dict]) -> dict:
    baseline = np.asarray([record["baseline_mse"] for record in records])
    labels = list(records[0]["method_mses"])
    result = {}
    for label in labels:
        values = np.asarray(
            [record["method_mses"][label] for record in records]
        )
        result[label] = {
            "ratio": float(np.mean(values) / np.mean(baseline)),
            "wins": int(np.sum(values < baseline)),
            "worst": float(np.max(values / baseline)),
        }
    return result


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
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.6],
    )
    parser.add_argument(
        "--factorized-dir",
        type=Path,
        default=HERE / "results" / "factorized_k3_layer29",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "results" / "connected_cubic_selection8.json",
    )
    args = parser.parse_args()

    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    radius = sphere_radius_mean(WIDTH)
    rows = _load_rows(FULL_DATA, args.indices)
    records = []

    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        started = time.perf_counter()
        with np.load(moment_path(index)) as moments:
            true_mean = np.asarray(
                moments["mean"][args.layer],
                dtype=np.float64,
            )
            true_second = np.asarray(
                moments["M11"][args.layer],
                dtype=np.float64,
            )
            true_raw_m21 = np.asarray(
                moments["M21"][args.layer],
                dtype=np.float64,
            )
            true_marginal_second = np.asarray(
                moments["m2"][args.layer],
                dtype=np.float64,
            )
            true_c21 = connected_m21(
                true_mean,
                true_second,
                true_raw_m21,
                true_marginal_second,
            )

        _, captured, final = forward_layer_and_final(
            weights,
            points,
            rotation,
            args.layer,
        )
        sample_mean = np.mean(captured, axis=0, dtype=np.float64)

        factorized_path = args.factorized_dir / f"mlp_{index:05d}.npz"
        if not factorized_path.exists():
            raise FileNotFoundError(factorized_path)
        with np.load(factorized_path) as factorized:
            factorized_mean = np.asarray(
                factorized["mean"],
                dtype=np.float64,
            )
            factorized_c21 = np.asarray(
                factorized["c21"],
                dtype=np.float64,
            )

        centers = {
            "oracle_mean": true_mean,
            "sample_mean": sample_mean,
            "factorized_mean": factorized_mean,
        }
        direction_families = sample_direction_families(
            captured,
            args.rank,
            radius,
        )
        factorized_offdiag = factorized_c21.copy()
        np.fill_diagonal(factorized_offdiag, 0.0)
        direction_families.update(
            {
                "factorized_dirs": truncated_svd(
                    factorized_c21,
                    args.rank,
                ),
                "factorized_offdiag_dirs": truncated_svd(
                    factorized_offdiag,
                    args.rank,
                ),
            }
        )

        configurations = {}
        direction_anchor_diagnostics = {}
        for direction_label, (left, right) in direction_families.items():
            pointwise = {
                label: contracted_pointwise(
                    captured,
                    left,
                    right,
                    center,
                    radius,
                )
                for label, center in centers.items()
            }
            exact_anchors = {
                label: contract(
                    left,
                    exact_anchor_matrix(
                        true_mean,
                        true_second,
                        true_raw_m21,
                        true_marginal_second,
                        center,
                    ),
                    right,
                )
                for label, center in centers.items()
            }
            oracle_connected_anchor = (
                contract(left, true_c21, right) / (WIDTH + 1.0)
            )
            if not np.allclose(
                exact_anchors["oracle_mean"],
                oracle_connected_anchor,
                rtol=2e-7,
                atol=2e-10,
            ):
                raise AssertionError(
                    (
                        exact_anchors["oracle_mean"],
                        oracle_connected_anchor,
                    )
                )
            factorized_anchor_unscaled = (
                contract(left, factorized_c21, right) / (WIDTH + 1.0)
            )
            direction_anchor_diagnostics[direction_label] = {
                "factorized_c21_relative_error": float(
                    np.linalg.norm(
                        factorized_anchor_unscaled - oracle_connected_anchor
                    )
                    / max(np.linalg.norm(oracle_connected_anchor), 1e-30)
                ),
            }
            for center_label, values in pointwise.items():
                configurations[
                    f"{direction_label}_{center_label}_oracle_exact_anchor"
                ] = values - exact_anchors[center_label]
                for scale in args.scales:
                    configurations[
                        f"{direction_label}_{center_label}"
                        f"_factorized_c21_scale{scale:g}"
                    ] = values - scale * factorized_anchor_unscaled

        baseline_prediction = final.mean(axis=0, dtype=np.float64)
        baseline_mse = float(
            np.mean(np.square(baseline_prediction - targets[-1]))
        )
        method_mses = {}
        diagnostics = {}
        for label, features in configurations.items():
            predictions, fit = crossfit_grid(
                features,
                final,
                args.folds,
                [args.ridge],
            )
            method_mses[label] = float(
                np.mean(np.square(predictions[args.ridge] - targets[-1]))
            )
            diagnostics[label] = {
                **fit,
                "feature_mean_norm": float(
                    np.linalg.norm(np.mean(features, axis=0))
                ),
            }

        anchor_diagnostics = {
            "directions": direction_anchor_diagnostics,
            "sample_mean_relative_error": float(
                np.linalg.norm(sample_mean - true_mean)
                / max(np.linalg.norm(true_mean), 1e-30)
            ),
            "factorized_mean_relative_error": float(
                np.linalg.norm(factorized_mean - true_mean)
                / max(np.linalg.norm(true_mean), 1e-30)
            ),
        }
        record = {
            "index": index,
            "name": name,
            "baseline_mse": baseline_mse,
            "method_mses": method_mses,
            "diagnostics": diagnostics,
            "anchor_diagnostics": anchor_diagnostics,
            "seconds": time.perf_counter() - started,
        }
        records.append(record)
        best = min(method_mses, key=method_mses.get)
        print(
            f"[{index}] base={baseline_mse:.4e} best={best} "
            f"{method_mses[best] / baseline_mse:.4f}x "
            f"({record['seconds']:.1f}s)",
            flush=True,
        )

    summary = summarize(records)
    output = {
        "protocol": {
            "indices": args.indices,
            "layer": args.layer,
            "rotation_seed": args.rotation_seed,
            "rank": args.rank,
            "folds": args.folds,
            "ridge": args.ridge,
            "scales": args.scales,
            "target_leakage": False,
            "anchor_identity_validated": True,
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
