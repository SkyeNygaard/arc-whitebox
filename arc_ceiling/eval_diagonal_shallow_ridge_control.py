"""Output-aligned shallow-ridge controls with exact anchors.

Let ``B`` be the empirical expected-gate input-to-output linearization.  For
each final output coordinate ``j`` define

    s_j(x) = ReLU(x @ B[:, j]),
    E[s_j(X)] = ||B[:, j]|| / sqrt(2 pi),  X ~ N(0, I).

Positive homogeneity makes the same anchor exact on the fixed-radius sphere
used by the Kerdock estimator.  Each output is fitted independently: output
``j`` sees only its own shallow ridge and/or degree-6/8/10/12 zonal harmonics
along ``B[:,j]``.  Whole Kerdock bases are held out, and no exact expectation
target participates in coefficient fitting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from eval_exact_anchor_residual import FULL_DATA, ROWS_PER_BASIS  # noqa: E402
from eval_exact_ridge_harmonic_control import (  # noqa: E402
    expected_gate_maps,
    forward_with_gates,
    normalized_ridge_harmonic,
)
from eval_kerdock_design import (  # noqa: E402
    N_BASES,
    WIDTH,
    make_kerdock_design,
    random_rotation,
)
from eval_sampling_official import _load_rows  # noqa: E402
from exact_moments import sphere_radius_mean  # noqa: E402


def diagonal_crossfit(
    features: np.ndarray,
    outputs: np.ndarray,
    *,
    folds: int,
    ridges: list[float],
) -> tuple[dict[float, np.ndarray], dict[str, float]]:
    """Cross-fit independent small regressions for every output coordinate.

    ``features`` has shape ``(points, outputs, components)``.  No coefficient
    couples distinct output coordinates.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(outputs, dtype=np.float64)
    if x.ndim != 3 or x.shape[:2] != y.shape:
        raise ValueError((x.shape, y.shape))
    if len(x) != N_BASES * ROWS_PER_BASIS:
        raise ValueError((len(x), N_BASES, ROWS_PER_BASIS))

    scale = np.sqrt(np.mean(np.square(x), axis=0))
    safe_scale = np.maximum(scale, 1e-14)
    x = x / safe_scale[None, :, :]
    gram_total = np.einsum("nop,noq->opq", x, x, optimize=True)
    cross_total = np.einsum("nop,no->op", x, y, optimize=True)

    block_ids = np.repeat(np.arange(N_BASES), ROWS_PER_BASIS)
    fold_ids = block_ids % folds
    estimates = {ridge: [] for ridge in ridges}
    sizes = []
    identity = np.eye(x.shape[2], dtype=np.float64)[None, :, :]
    max_condition = {ridge: 0.0 for ridge in ridges}
    for fold in range(folds):
        test = fold_ids == fold
        x_test = x[test]
        y_test = y[test]
        gram_train = (
            gram_total
            - np.einsum("nop,noq->opq", x_test, x_test, optimize=True)
        )
        cross_train = (
            cross_total
            - np.einsum("nop,no->op", x_test, y_test, optimize=True)
        )
        n_train = len(x) - int(np.sum(test))
        sizes.append(int(np.sum(test)))
        mean_x = np.mean(x_test, axis=0)
        mean_y = np.mean(y_test, axis=0)
        for ridge in ridges:
            system = gram_train + ridge * n_train * identity
            coefficient = np.linalg.solve(
                system,
                cross_train[..., None],
            )[..., 0]
            estimates[ridge].append(
                mean_y - np.einsum("op,op->o", mean_x, coefficient)
            )
            max_condition[ridge] = max(
                max_condition[ridge],
                float(np.max(np.linalg.cond(system))),
            )

    predictions = {
        ridge: np.average(values, axis=0, weights=sizes)
        for ridge, values in estimates.items()
    }
    diagnostics = {
        "components": int(x.shape[2]),
        "feature_rms_min": float(np.min(scale)),
        "feature_rms_max": float(np.max(scale)),
        **{
            f"condition_ridge_{ridge:g}": condition
            for ridge, condition in max_condition.items()
        },
    }
    return predictions, diagnostics


def cost_estimate(
    *,
    components: int,
    degrees: list[int],
    folds: int,
) -> dict[str, int]:
    points = N_BASES * ROWS_PER_BASIS
    outputs = WIDTH
    gate_tail_map = 2 * 31 * WIDTH**3
    shallow_projection = 2 * points * WIDTH**2
    gegenbauer = (
        0
        if components == 1
        else 6 * points * outputs * sum(degrees)
    )
    regression_products = (
        4 * points * outputs * components**2
        + 4 * points * outputs * components
    )
    regression_solves = int(
        folds
        * outputs
        * (
            (2.0 / 3.0) * components**3
            + 2.0 * components**2
        )
    )
    total = (
        gate_tail_map
        + shallow_projection
        + gegenbauer
        + regression_products
        + regression_solves
    )
    return {
        "components": components,
        "gate_tail_map": int(gate_tail_map),
        "shallow_projection": int(shallow_projection),
        "gegenbauer_evaluation": int(gegenbauer),
        "diagonal_crossfit_products": int(regression_products),
        "diagonal_crossfit_solves": int(regression_solves),
        "total_extra": int(total),
    }


def summarize(records: list[dict]) -> dict:
    baseline = np.asarray([record["baseline_mse"] for record in records])
    labels = list(records[0]["method_mses"])
    result = {}
    for label in labels:
        mse = np.asarray([record["method_mses"][label] for record in records])
        result[label] = {
            "mse_ratio": float(np.mean(mse) / np.mean(baseline)),
            "wins": int(np.sum(mse < baseline)),
            "worst": float(np.max(mse / baseline)),
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
    parser.add_argument("--rotation-seed", type=int, default=3)
    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=[6, 8, 10, 12],
    )
    parser.add_argument(
        "--ridges",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 1.0, 3.0],
    )
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            HERE
            / "results"
            / "diagonal_shallow_ridge_selection8.json"
        ),
    )
    args = parser.parse_args()
    if args.degrees != [6, 8, 10, 12]:
        raise ValueError("The bounded protocol fixes degrees 6,8,10,12")
    if args.ridges != [0.1, 0.3, 1.0, 3.0]:
        raise ValueError("The bounded protocol fixes ridges 0.1,0.3,1,3")

    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    radius = sphere_radius_mean(WIDTH)
    rows = _load_rows(FULL_DATA, args.indices)
    records = []

    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        started = time.perf_counter()
        _, final, gates = forward_with_gates(weights, points, rotation)
        end_to_end, _ = expected_gate_maps(weights, rotation, gates)
        shallow_pre = (
            np.asarray(points, dtype=np.float64) @ end_to_end
        )
        shallow_anchor = (
            np.linalg.norm(end_to_end, axis=0)
            / math.sqrt(2.0 * math.pi)
        )
        shallow = (
            np.maximum(shallow_pre, 0.0)
            - shallow_anchor[None, :]
        )

        harmonics = np.stack(
            [
                normalized_ridge_harmonic(
                    points,
                    end_to_end,
                    radius,
                    degree,
                )
                for degree in args.degrees
            ],
            axis=2,
        )
        configurations = {
            "shallow": shallow[:, :, None],
            "harmonics": harmonics,
            "shallow_plus_harmonics": np.concatenate(
                [shallow[:, :, None], harmonics],
                axis=2,
            ),
        }

        baseline_prediction = np.mean(final, axis=0, dtype=np.float64)
        baseline_mse = float(
            np.mean(np.square(baseline_prediction - targets[-1]))
        )
        method_mses = {}
        diagnostics = {}
        for label, features in configurations.items():
            predictions, fit = diagonal_crossfit(
                features,
                final,
                folds=args.folds,
                ridges=args.ridges,
            )
            for ridge in args.ridges:
                method_label = f"{label}_ridge{ridge:g}"
                method_mses[method_label] = float(
                    np.mean(
                        np.square(predictions[ridge] - targets[-1])
                    )
                )
                diagnostics[method_label] = fit

        record = {
            "index": index,
            "name": name,
            "baseline_mse": baseline_mse,
            "method_mses": method_mses,
            "diagnostics": diagnostics,
            "shallow_anchor_norm": float(
                np.linalg.norm(shallow_anchor)
            ),
            "shallow_sample_mean_norm": float(
                np.linalg.norm(np.mean(shallow, axis=0))
            ),
            "harmonic_sample_mean_norms": [
                float(np.linalg.norm(np.mean(harmonics[:, :, degree], axis=0)))
                for degree in range(len(args.degrees))
            ],
            "seconds": time.perf_counter() - started,
        }
        records.append(record)
        best = min(method_mses, key=method_mses.get)
        print(
            f"[{index}] base={baseline_mse:.4e} "
            f"best={best}:{method_mses[best] / baseline_mse:.4f}x "
            f"({record['seconds']:.1f}s)",
            flush=True,
        )

    summary = summarize(records)
    cost = {
        "shallow": cost_estimate(
            components=1,
            degrees=args.degrees,
            folds=args.folds,
        ),
        "harmonics": cost_estimate(
            components=4,
            degrees=args.degrees,
            folds=args.folds,
        ),
        "shallow_plus_harmonics": cost_estimate(
            components=5,
            degrees=args.degrees,
            folds=args.folds,
        ),
    }
    output = {
        "protocol": {
            "indices": args.indices,
            "rotation_seed": args.rotation_seed,
            "degrees": args.degrees,
            "ridges": args.ridges,
            "folds": args.folds,
            "coefficient_structure": "independent_per_output",
            "shallow_anchor": "exact_norm_over_sqrt_2pi",
            "harmonic_anchor": "exact_zero_spherical_mean",
            "uses_final_targets_for_construction": False,
        },
        "cost_estimate": cost,
        "summary": summary,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(json.dumps({"cost_estimate": cost}, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
