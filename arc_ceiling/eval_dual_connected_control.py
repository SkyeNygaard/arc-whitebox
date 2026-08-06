"""Test scalar adjoint-K3 anchors inside the connected cubic control.

This is deliberately stricter than comparing transported cumulants directly.
The Kerdock activation cloud supplies both the centering vector and the two
sample-SVD probe directions.  Transport is used only to predict the two scalar
connected-K3 contractions needed as anchors:

    q_k = sum_ij left[i,k] right[j,k] K3[i,i,j].

No deployable configuration reconstructs M21 or consumes the target
distribution's mean/covariance.  Oracle moments are loaded only for two
diagnostic ceilings:

* exact arbitrary-center anchor for the sample mean used pointwise;
* connected-K3 anchor, which is exact only if the center were the true mean.

All control coefficients are fitted target-free by held-Kerdock-basis
cross-fitting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mlp_kprop.kprop_harmonic import (
    SIMPLE,
    coerce_input,
    linear_kprop,
    nonlin_kprop,
)
from mlp_kprop.wick import relu_wick_coef

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))
# The specialized cumulant-propagation environment intentionally contains
# only the vendor research package.  Reuse the main ARC environment's
# Python-3.12 wheels for dataset IO (notably pyarrow) without shadowing the
# vendor environment's torch/mlp_kprop dependencies.
for site_packages in sorted((ROOT / ".venv" / "lib").glob("python*/site-packages")):
    sys.path.append(str(site_packages))

from eval_connected_cubic_control import (  # noqa: E402
    contract,
    contracted_pointwise,
    exact_anchor_matrix,
    sample_direction_families,
)
from eval_crossfit_cumulant_control import (  # noqa: E402
    crossfit_grid,
    forward_layer_and_final,
)
from eval_dual_contracted_k3 import (  # noqa: E402
    dual_contract,
    dual_contract_lowrank_probe,
    lower_only_source,
    transition,
)
from eval_exact_anchor_residual import FULL_DATA  # noqa: E402
from eval_kerdock_design import WIDTH, make_kerdock_design, random_rotation  # noqa: E402
from eval_oracle_cumulant_bridge import connected_m21, moment_path  # noqa: E402
from eval_sampling_official import _load_rows  # noqa: E402
from exact_moments import sphere_radius_mean  # noqa: E402
from predict_factorized_k3_anchor import load_weights, tensor  # noqa: E402


def cheap_dual_state(
    weights: torch.Tensor,
    layer: int,
    dtype: torch.dtype,
) -> tuple[
    list[torch.Tensor],
    list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    """Roll mean/covariance and retain only local lower-state K3 sources."""
    state = coerce_input(
        {
            1: torch.zeros(WIDTH, dtype=dtype),
            2: torch.eye(WIDTH, dtype=dtype),
        },
        k_max=2,
        kind=SIMPLE,
    )
    transitions = []
    sources = []
    with torch.no_grad():
        for current_layer in range(layer + 1):
            pre = linear_kprop(
                state,
                weights[current_layer],
                k_max=2,
            )
            sources.append(
                lower_only_source(
                    tensor(pre[1]),
                    tensor(pre[2]),
                )
            )
            transitions.append(transition(pre, weights[current_layer]))
            state = nonlin_kprop(
                pre,
                nonlin_wick_coef=relu_wick_coef,
                k_max=2,
                kind=SIMPLE,
                use_pK=True,
                factor=False,
            )
    return transitions, sources


def anchor_metric(
    predicted: np.ndarray,
    exact: np.ndarray,
    sample_quadrature_anchor: np.ndarray,
) -> dict[str, float]:
    """Measure anchor error both absolutely and on the relevant Q-E scale."""
    predicted = np.asarray(predicted, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    sample_quadrature_anchor = np.asarray(
        sample_quadrature_anchor,
        dtype=np.float64,
    )
    error = predicted - exact
    discrepancy = sample_quadrature_anchor - exact
    return {
        "relative_to_exact_anchor": float(
            np.linalg.norm(error) / max(np.linalg.norm(exact), 1e-30)
        ),
        "relative_to_quadrature_discrepancy": float(
            np.linalg.norm(error) / max(np.linalg.norm(discrepancy), 1e-30)
        ),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_absolute_error": float(np.mean(np.abs(error))),
        "cosine_with_exact": float(
            np.sum(predicted * exact)
            / max(
                np.linalg.norm(predicted) * np.linalg.norm(exact),
                1e-30,
            )
        ),
    }


def paired_summary(
    records: list[dict],
    labels: list[str],
) -> dict[str, dict[str, float | int | list[float]]]:
    baseline = np.asarray(
        [record["baseline_mse"] for record in records],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20260729)
    boot = rng.integers(0, len(records), size=(20_000, len(records)))
    result = {}
    for label in labels:
        values = np.asarray(
            [record["method_mses"][label] for record in records],
            dtype=np.float64,
        )
        boot_ratio = (
            np.mean(values[boot], axis=1)
            / np.mean(baseline[boot], axis=1)
        )
        result[label] = {
            "baseline_mean_mse": float(np.mean(baseline)),
            "method_mean_mse": float(np.mean(values)),
            "raw_mse_ratio": float(np.mean(values) / np.mean(baseline)),
            "ci95": [
                float(value)
                for value in np.percentile(boot_ratio, [2.5, 97.5])
            ],
            "wins": int(np.sum(values < baseline)),
            "worst_per_network_ratio": float(np.max(values / baseline)),
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
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=(
            ROOT
            / "submissions"
            / "whest_bounded_ml"
            / "data"
            / "official_weights_fresh"
        ),
    )
    parser.add_argument("--layer", type=int, default=29)
    parser.add_argument("--rotation-seed", type=int, default=3)
    parser.add_argument("--direction-rank", type=int, default=2)
    parser.add_argument("--probe-rank", type=int, default=32)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[1.7, 1.8, 1.9, 1.95, 2.0, 2.1, 2.2],
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            HERE
            / "results"
            / "dual_connected_control_rank2_selection8.json"
        ),
    )
    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    radius = sphere_radius_mean(WIDTH)
    rows = _load_rows(FULL_DATA, args.indices)
    records = []

    for index, (name, numpy_weights, targets) in zip(
        args.indices,
        rows,
        strict=True,
    ):
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
            numpy_weights,
            points,
            rotation,
            args.layer,
        )
        sample_mean = np.mean(captured, axis=0, dtype=np.float64)
        left, right = sample_direction_families(
            captured,
            args.direction_rank,
            radius,
        )["radial_corrected_dirs"]
        values = contracted_pointwise(
            captured,
            left,
            right,
            sample_mean,
            radius,
        )
        sample_quadrature_anchor = np.mean(
            values,
            axis=0,
            dtype=np.float64,
        )
        exact_arbitrary_anchor = contract(
            left,
            exact_anchor_matrix(
                true_mean,
                true_second,
                true_raw_m21,
                true_marginal_second,
                sample_mean,
            ),
            right,
        )
        oracle_connected_anchor = (
            contract(left, true_c21, right) / (WIDTH + 1.0)
        )

        torch_weights = load_weights(
            args.weights_dir / f"mlp_{index:05d}.npy",
            dtype,
        )
        transitions, sources = cheap_dual_state(
            torch_weights,
            args.layer,
            dtype,
        )
        cheap_q = dual_contract(
            transitions,
            sources,
            left,
            right,
        )
        probe_q = dual_contract_lowrank_probe(
            transitions,
            sources,
            left,
            right,
            args.probe_rank,
        )
        cheap_anchor_unscaled = cheap_q / (WIDTH + 1.0)
        probe_anchor_unscaled = probe_q / (WIDTH + 1.0)

        anchors = {
            "oracle_arbitrary_center": exact_arbitrary_anchor,
            "oracle_connected_approx": oracle_connected_anchor,
        }
        for scale in args.scales:
            anchors[f"cheap_dual_scale{scale:g}"] = (
                scale * cheap_anchor_unscaled
            )
            anchors[
                f"cheap_dual_probe{args.probe_rank}_scale{scale:g}"
            ] = scale * probe_anchor_unscaled

        configurations = {
            label: values - anchor
            for label, anchor in anchors.items()
        }
        baseline_prediction = np.mean(final, axis=0, dtype=np.float64)
        target = targets[-1]
        baseline_mse = float(
            np.mean(np.square(baseline_prediction - target))
        )
        method_mses = {}
        fit_diagnostics = {}
        for label, features in configurations.items():
            predictions, fit = crossfit_grid(
                features,
                final,
                args.folds,
                [args.ridge],
            )
            method_mses[label] = float(
                np.mean(
                    np.square(predictions[args.ridge] - target)
                )
            )
            fit_diagnostics[label] = {
                **fit,
                "feature_mean_norm": float(
                    np.linalg.norm(np.mean(features, axis=0))
                ),
            }

        anchor_diagnostics = {
            label: {
                **anchor_metric(
                    anchor,
                    exact_arbitrary_anchor,
                    sample_quadrature_anchor,
                ),
                "anchor": anchor.tolist(),
            }
            for label, anchor in anchors.items()
        }
        anchor_diagnostics["sample_quadrature_anchor"] = {
            **anchor_metric(
                sample_quadrature_anchor,
                exact_arbitrary_anchor,
                sample_quadrature_anchor,
            ),
            "anchor": sample_quadrature_anchor.tolist(),
        }
        record = {
            "index": index,
            "name": name,
            "baseline_mse": baseline_mse,
            "method_mses": method_mses,
            "fit_diagnostics": fit_diagnostics,
            "anchor_diagnostics": anchor_diagnostics,
            "exact_arbitrary_anchor": exact_arbitrary_anchor.tolist(),
            "oracle_connected_anchor": oracle_connected_anchor.tolist(),
            "sample_quadrature_anchor": sample_quadrature_anchor.tolist(),
            "cheap_q": cheap_q.tolist(),
            f"cheap_probe{args.probe_rank}_q": probe_q.tolist(),
            "sample_mean_relative_error": float(
                np.linalg.norm(sample_mean - true_mean)
                / max(np.linalg.norm(true_mean), 1e-30)
            ),
            "seconds": time.perf_counter() - started,
        }
        records.append(record)
        best = min(method_mses, key=method_mses.get)
        print(
            f"[{index}] base={baseline_mse:.4e} "
            f"oracle-exact="
            f"{method_mses['oracle_arbitrary_center'] / baseline_mse:.4f}x "
            f"oracle-connected="
            f"{method_mses['oracle_connected_approx'] / baseline_mse:.4f}x "
            f"best={best}:{method_mses[best] / baseline_mse:.4f}x "
            f"({record['seconds']:.1f}s)",
            flush=True,
        )

    labels = list(records[0]["method_mses"])
    summary = paired_summary(records, labels)
    pooled_exact = np.asarray(
        [record["exact_arbitrary_anchor"] for record in records],
        dtype=np.float64,
    )
    pooled_sample = np.asarray(
        [record["sample_quadrature_anchor"] for record in records],
        dtype=np.float64,
    )
    pooled_anchor_metrics = {}
    for label in labels:
        predicted = np.asarray(
            [
                record["anchor_diagnostics"][label]["anchor"]
                for record in records
            ],
            dtype=np.float64,
        )
        pooled_anchor_metrics[label] = anchor_metric(
            predicted,
            pooled_exact,
            pooled_sample,
        )
    pooled_anchor_metrics["sample_quadrature_anchor"] = anchor_metric(
        pooled_sample,
        pooled_exact,
        pooled_sample,
    )

    output = {
        "protocol": {
            "indices": args.indices,
            "weights_dir": str(args.weights_dir),
            "layer": args.layer,
            "rotation_seed": args.rotation_seed,
            "directions": "sample radial-corrected connected-c21 SVD",
            "direction_rank": args.direction_rank,
            "center": "same Kerdock activation sample mean",
            "folds": args.folds,
            "ridge": args.ridge,
            "scales": args.scales,
            "probe_rank": args.probe_rank,
            "dtype": args.dtype,
            "target_leakage_in_coefficients": False,
            "deployable_anchor_requires_target_mean_covariance": False,
            "deployable_anchor_reconstructs_raw_m21": False,
            "scope": "selection IDs only; no new holdout",
        },
        "summary": summary,
        "pooled_anchor_metrics": pooled_anchor_metrics,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(json.dumps(pooled_anchor_metrics, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
