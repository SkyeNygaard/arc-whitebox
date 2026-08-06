"""Raw cubic control whose anchor correction uses connected K3 only.

The ordinary raw cubic feature

    r_k(x) = (u_k^T h(x)^2) (v_k^T h(x)) / E[R]^2

has a same-cloud anchor ``Q_K r_k`` that makes its control correction exactly
zero.  Reconstructing its analytic anchor from a K3 rollout normally also
requires an extraordinarily accurate target mean and covariance.

This experiment avoids that requirement.  Keep every marginal contribution
at its same-cloud value and replace only the connected-C21 contribution:

    a_hybrid
      = Q_K r
        + alpha (a_C21,predicted - a_C21,sample).

The pointwise regression still sees the numerically stable raw cubic feature,
while the analytic model supplies only the connected third-cumulant scalar it
can actually estimate.  No predicted target mean or covariance enters the
deployable variants.
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
    empirical_c21_state,
    forward_layer_and_final,
    pointwise_features,
)
from eval_exact_anchor_residual import FULL_DATA  # noqa: E402
from eval_kerdock_design import WIDTH, make_kerdock_design, random_rotation  # noqa: E402
from eval_oracle_cumulant_bridge import connected_m21, moment_path  # noqa: E402
from eval_sampling_official import _load_rows  # noqa: E402
from exact_moments import sphere_radius_mean  # noqa: E402


def radial_sample_c21(
    activation: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Convert fixed-radius sample moments to Gaussian connected C21."""
    h = np.asarray(activation, dtype=np.float64)
    mean = np.mean(h, axis=0)
    fixed_second = (h.T @ h) / len(h)
    fixed_raw_m21 = (np.square(h).T @ h) / len(h)
    gaussian_second = WIDTH * fixed_second / np.square(radius)
    gaussian_raw_m21 = (
        (WIDTH + 1.0) * fixed_raw_m21 / np.square(radius)
    )
    return connected_m21(
        mean,
        gaussian_second,
        gaussian_raw_m21,
        np.diag(gaussian_second),
    )


def contraction(
    left: np.ndarray,
    matrix: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    return (
        np.einsum("ik,ij,jk->k", left, matrix, right)
        / (WIDTH + 1.0)
    )


def summarize(records: list[dict]) -> dict:
    baseline = np.asarray([record["baseline_mse"] for record in records])
    labels = list(records[0]["method_mses"])
    result = {}
    for label in labels:
        values = np.asarray([r["method_mses"][label] for r in records])
        result[label] = {
            "mse_ratio": float(np.mean(values) / np.mean(baseline)),
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
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    )
    parser.add_argument(
        "--factorized-dir",
        type=Path,
        default=HERE / "results" / "factorized_k3_layer29_diagfix",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "results" / "c21_delta_hybrid_selection8.json",
    )
    args = parser.parse_args()

    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    radius = sphere_radius_mean(WIDTH)
    rows = _load_rows(FULL_DATA, args.indices)
    records = []

    for index, (name, weights, targets) in zip(
        args.indices,
        rows,
        strict=True,
    ):
        started = time.perf_counter()
        _, activation, final = forward_layer_and_final(
            weights,
            points,
            rotation,
            args.layer,
        )
        left, right, sample_raw_m21 = empirical_c21_state(
            activation,
            args.rank,
        )
        sample_raw_anchor = (
            np.einsum("ik,ij,jk->k", left, sample_raw_m21, right)
            / np.square(radius)
        )
        sample_c21 = radial_sample_c21(activation, radius)
        sample_c21_anchor = contraction(left, sample_c21, right)

        with np.load(moment_path(index)) as oracle:
            oracle_mean = np.asarray(
                oracle["mean"][args.layer],
                dtype=np.float64,
            )
            oracle_second = np.asarray(
                oracle["M11"][args.layer],
                dtype=np.float64,
            )
            oracle_raw_m21 = np.asarray(
                oracle["M21"][args.layer],
                dtype=np.float64,
            )
            oracle_c21 = connected_m21(
                oracle_mean,
                oracle_second,
                oracle_raw_m21,
                np.asarray(oracle["m2"][args.layer], dtype=np.float64),
            )
        oracle_c21_anchor = contraction(left, oracle_c21, right)

        factorized_path = args.factorized_dir / f"mlp_{index:05d}.npz"
        with np.load(factorized_path) as factorized:
            factorized_c21 = np.asarray(
                factorized["c21"],
                dtype=np.float64,
            )
        factorized_c21_anchor = contraction(
            left,
            factorized_c21,
            right,
        )

        anchors = {"same_cloud": sample_raw_anchor}
        for scale in args.scales:
            anchors[f"oracle_c21_delta_scale{scale:g}"] = (
                sample_raw_anchor
                + scale * (oracle_c21_anchor - sample_c21_anchor)
            )
            anchors[f"factorized_c21_delta_scale{scale:g}"] = (
                sample_raw_anchor
                + scale
                * (factorized_c21_anchor - sample_c21_anchor)
            )

        baseline_prediction = np.mean(final, axis=0, dtype=np.float64)
        baseline_mse = float(
            np.mean(np.square(baseline_prediction - targets[-1]))
        )
        method_mses = {}
        for label, anchor in anchors.items():
            features = pointwise_features(
                activation,
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
            method_mses[label] = float(
                np.mean(
                    np.square(predictions[args.ridge] - targets[-1])
                )
            )
        record = {
            "index": index,
            "name": name,
            "baseline_mse": baseline_mse,
            "method_mses": method_mses,
            "anchor_diagnostics": {
                "sample_c21": sample_c21_anchor.tolist(),
                "oracle_c21": oracle_c21_anchor.tolist(),
                "factorized_c21": factorized_c21_anchor.tolist(),
                "factorized_delta_relative_error": float(
                    np.linalg.norm(
                        (factorized_c21_anchor - sample_c21_anchor)
                        - (oracle_c21_anchor - sample_c21_anchor)
                    )
                    / max(
                        np.linalg.norm(
                            oracle_c21_anchor - sample_c21_anchor
                        ),
                        1e-30,
                    )
                ),
            },
            "seconds": time.perf_counter() - started,
        }
        records.append(record)
        best = min(method_mses, key=method_mses.get)
        print(
            f"[{index}] best={best}:"
            f"{method_mses[best] / baseline_mse:.3f}x "
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
            "factorized_dir": str(args.factorized_dir),
            "deployable_variants_use_target_k2": False,
            "oracle_used_for_diagnostics_and_ceiling_only": True,
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
