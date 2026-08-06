"""Earlier-layer connected cubic using factor-state, not sample, directions.

The main connected-cubic harness chooses directions from the Kerdock cloud.
That is ideal for the oracle ceiling but can expose directions where the
factorized C21 error is unusually large.  This audit instead takes the rank-r
SVD of the factorized C21 itself.  It tests whether the state is accurate in
its own dominant subspace, while retaining the radially homogenized pointwise
identity from ``eval_connected_cubic_control.py``.
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

from eval_connected_cubic_control import (  # noqa: E402
    contract,
    contracted_pointwise,
    exact_anchor_matrix,
)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(160, 168)))
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--rotation-seed", type=int, default=3)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--scales", type=float, nargs="+", required=True)
    parser.add_argument(
        "--correction-multipliers",
        type=float,
        nargs="+",
        default=[1.0],
        help="Shrink each completed control prediction toward the full baseline.",
    )
    parser.add_argument("--factorized-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    radius = sphere_radius_mean(WIDTH)
    rows = _load_rows(FULL_DATA, args.indices)
    records = []

    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        started = time.perf_counter()
        with np.load(moment_path(index)) as moments:
            true_mean = np.asarray(moments["mean"][args.layer], dtype=np.float64)
            true_second = np.asarray(moments["M11"][args.layer], dtype=np.float64)
            true_raw_m21 = np.asarray(moments["M21"][args.layer], dtype=np.float64)
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
        with np.load(
            args.factorized_dir / f"mlp_{index:05d}.npz"
        ) as factorized:
            factorized_mean = np.asarray(
                factorized["mean"],
                dtype=np.float64,
            )
            factorized_c21 = np.asarray(
                factorized["c21"],
                dtype=np.float64,
            )

        _, captured, final = forward_layer_and_final(
            weights,
            points,
            rotation,
            args.layer,
        )
        baseline = np.mean(final, axis=0, dtype=np.float64)
        baseline_mse = float(np.mean(np.square(baseline - targets[-1])))
        method_mses = {}
        anchor_diagnostics = {}

        direction_families = {
            "factorized_dirs": truncated_svd(factorized_c21, args.rank),
            "oracle_dirs": truncated_svd(true_c21, args.rank),
        }
        for direction_label, (left, right) in direction_families.items():
            values = contracted_pointwise(
                captured,
                left,
                right,
                factorized_mean,
                radius,
            )
            true_anchor = contract(
                left,
                exact_anchor_matrix(
                    true_mean,
                    true_second,
                    true_raw_m21,
                    true_marginal_second,
                    factorized_mean,
                ),
                right,
            )
            predicted_anchor = (
                contract(left, factorized_c21, right) / (WIDTH + 1.0)
            )
            optimal_scale = float(
                np.sum(predicted_anchor * true_anchor)
                / max(np.sum(np.square(predicted_anchor)), 1e-30)
            )
            anchor_diagnostics[direction_label] = {
                "oracle_optimal_scale": optimal_scale,
                "unscaled_relative_error": float(
                    np.linalg.norm(predicted_anchor - true_anchor)
                    / max(np.linalg.norm(true_anchor), 1e-30)
                ),
                "optimal_scaled_relative_error": float(
                    np.linalg.norm(
                        optimal_scale * predicted_anchor - true_anchor
                    )
                    / max(np.linalg.norm(true_anchor), 1e-30)
                ),
            }
            configurations = {
                "oracle_exact": values - true_anchor,
                # Forbidden per-network oracle calibration. This isolates
                # whether scalar amplitude is the only remaining state error.
                "oracle_optimal_scale": (
                    values - optimal_scale * predicted_anchor
                ),
            }
            for scale in args.scales:
                configurations[f"factorized_scale{scale:g}"] = (
                    values - scale * predicted_anchor
                )
            for anchor_label, features in configurations.items():
                predictions, _ = crossfit_grid(
                    features,
                    final,
                    args.folds,
                    [args.ridge],
                )
                control_prediction = predictions[args.ridge]
                for multiplier in args.correction_multipliers:
                    shrunk_prediction = (
                        baseline
                        + multiplier * (control_prediction - baseline)
                    )
                    method_mses[
                        f"{direction_label}_{anchor_label}"
                        f"_correction{multiplier:g}"
                    ] = float(
                        np.mean(
                            np.square(shrunk_prediction - targets[-1])
                        )
                    )

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
            f"{method_mses[best] / baseline_mse:.4f}x",
            flush=True,
        )

    baseline = np.asarray([record["baseline_mse"] for record in records])
    labels = list(records[0]["method_mses"])
    summary = {}
    for label in labels:
        values = np.asarray(
            [record["method_mses"][label] for record in records]
        )
        summary[label] = {
            "ratio": float(np.mean(values) / np.mean(baseline)),
            "wins": int(np.sum(values < baseline)),
            "worst": float(np.max(values / baseline)),
        }
    output = {
        "protocol": {
            **vars(args),
            "factorized_dir": str(args.factorized_dir),
            "out": str(args.out),
            "target_leakage": False,
            "oracle_anchor_is_ceiling_only": True,
        },
        "summary": summary,
        "records": records,
    }
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
