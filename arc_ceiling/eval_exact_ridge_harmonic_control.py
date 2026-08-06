"""Exact degree-6+ ridge-harmonic controls for the Kerdock estimator.

The maximal real-MUB rule is a spherical 5-design.  Its first non-zero
quadrature-error subspace is therefore harmonic degree 6.  For every fixed
unit vector ``a``,

    g_{a,l}(x) = sqrt(dim H_l) C_l^lambda(a'x/r) / C_l^lambda(1)

has *exactly zero* spherical expectation for ``l > 0``.  Unlike transported
moment anchors, this expectation has no closure or sample-centering error.

This experiment chooses a small, network-adapted ridge dictionary and fits
the final-output coefficients using held-Kerdock-basis cross-fitting:

* first-layer directions, ordered by expected-gate downstream influence;
* leading input singular directions of the expected-gate end-to-end map.

The control is deliberately evaluated only on the already-designated
selection networks unless explicit indices are supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import eval_gegenbauer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from eval_crossfit_cumulant_control import crossfit_grid  # noqa: E402
from eval_exact_anchor_residual import FULL_DATA  # noqa: E402
from eval_kerdock_design import (  # noqa: E402
    WIDTH,
    make_kerdock_design,
    random_rotation,
)
from eval_sampling_official import _load_rows  # noqa: E402
from exact_moments import sphere_radius_mean  # noqa: E402


def forward_with_gates(
    weights: np.ndarray,
    points: np.ndarray,
    rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Return first preactivations, final outputs, and empirical gate rates."""
    pre = points @ (rotation @ weights[0].astype(np.float32))
    first_pre = pre
    gates = [np.mean(pre > 0.0, axis=0, dtype=np.float64)]
    activation = np.maximum(pre, 0.0)
    for weight in weights[1:]:
        pre = activation @ weight
        gates.append(np.mean(pre > 0.0, axis=0, dtype=np.float64))
        activation = np.maximum(pre, 0.0)
    return first_pre, activation, gates


def expected_gate_maps(
    weights: np.ndarray,
    rotation: np.ndarray,
    gates: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the input-to-output map and post-layer-0 tail map."""
    tail = np.eye(WIDTH, dtype=np.float64)
    for layer in range(1, len(weights)):
        tail = tail @ np.asarray(weights[layer], dtype=np.float64)
        tail *= gates[layer][None, :]

    end_to_end = (
        np.asarray(rotation, dtype=np.float64)
        @ np.asarray(weights[0], dtype=np.float64)
    )
    end_to_end *= gates[0][None, :]
    end_to_end = end_to_end @ tail
    return end_to_end, tail


def normalized_ridge_harmonic(
    points: np.ndarray,
    directions: np.ndarray,
    radius: float,
    degree: int,
) -> np.ndarray:
    """Evaluate unit-L2 zonal spherical harmonics along direction columns."""
    norms = np.linalg.norm(directions, axis=0)
    keep = norms > 1e-14
    if not np.all(keep):
        directions = directions[:, keep]
        norms = norms[keep]
    unit = directions / norms[None, :]
    cosine = (
        np.asarray(points, dtype=np.float64) @ unit
    ) / float(radius)
    lam = (WIDTH - 2.0) / 2.0
    at_one = float(eval_gegenbauer(degree, lam, 1.0))
    dimension = (
        math.comb(WIDTH + degree - 1, degree)
        - math.comb(WIDTH + degree - 3, degree - 2)
    )
    return (
        math.sqrt(float(dimension))
        * eval_gegenbauer(degree, lam, cosine)
        / at_one
    )


def centered_angular_relu_power(
    points: np.ndarray,
    directions: np.ndarray,
    radius: float,
    power: int,
) -> np.ndarray:
    """A degree-one homogeneous angular ridge with exact Gaussian mean.

    On the design sphere this is

        rho * max(a' x / rho, 0)^power.

    Its homogeneous extension away from the sphere remains degree one, so
    the Gaussian expectation is ``E[chi_d]`` times a one-dimensional sphere
    moment.  Odd powers retain an infinite even-harmonic tail; even powers
    at most four are already integrated exactly by Kerdock.
    """
    norms = np.linalg.norm(directions, axis=0)
    unit = directions / np.maximum(norms, 1e-30)[None, :]
    cosine = (
        np.asarray(points, dtype=np.float64) @ unit
    ) / float(radius)
    values = float(radius) * np.maximum(cosine, 0.0) ** power
    positive_sphere_moment = (
        0.5
        * math.gamma(WIDTH / 2.0)
        * math.gamma((power + 1.0) / 2.0)
        / (
            math.sqrt(math.pi)
            * math.gamma((WIDTH + power) / 2.0)
        )
    )
    anchor = float(radius) * positive_sphere_moment
    return values - anchor


def paired_summary(records: list[dict]) -> dict:
    baseline = np.asarray(
        [record["baseline_mse"] for record in records],
        dtype=np.float64,
    )
    labels = list(records[0]["method_mses"])
    result = {}
    for label in labels:
        values = np.asarray(
            [record["method_mses"][label] for record in records],
            dtype=np.float64,
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
    parser.add_argument("--rotation-seed", type=int, default=3)
    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=[6, 8, 10, 12],
    )
    parser.add_argument(
        "--direction-counts",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
    )
    parser.add_argument(
        "--ridges",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0],
    )
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument(
        "--only-full-harmonic",
        action="store_true",
        help=(
            "Evaluate only the complete supplied degree list.  This is the "
            "frozen holdout mode and suppresses prefix/angular alternatives."
        ),
    )
    parser.add_argument(
        "--families",
        choices=["first", "linearized", "both"],
        nargs="+",
        default=["first", "linearized"],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            HERE / "results" / "exact_ridge_harmonic_selection8.json"
        ),
    )
    args = parser.parse_args()

    if any(degree <= 5 or degree % 2 for degree in args.degrees):
        raise ValueError("degrees must be even and greater than five")
    max_count = max(args.direction_counts)
    if max_count > WIDTH:
        raise ValueError(max_count)

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
        _, final, gates = forward_with_gates(weights, points, rotation)
        end_to_end, tail = expected_gate_maps(weights, rotation, gates)

        first_directions = (
            np.asarray(rotation, dtype=np.float64)
            @ np.asarray(weights[0], dtype=np.float64)
        )
        first_influence = np.linalg.norm(tail, axis=1)
        first_order = np.argsort(first_influence)[::-1]
        first_directions = first_directions[:, first_order[:max_count]]

        left_singular, singular_values, _ = np.linalg.svd(
            end_to_end,
            full_matrices=False,
        )
        linearized_directions = left_singular[:, :max_count]
        direction_sets = {
            "first": first_directions,
            "linearized": linearized_directions,
        }

        harmonic_cache: dict[tuple[str, int], np.ndarray] = {}
        for family in ("first", "linearized"):
            for degree in args.degrees:
                harmonic_cache[(family, degree)] = normalized_ridge_harmonic(
                    points,
                    direction_sets[family],
                    radius,
                    degree,
                )
            harmonic_cache[(family, -1)] = centered_angular_relu_power(
                points,
                direction_sets[family],
                radius,
                1,
            )
            for power in (3, 5):
                harmonic_cache[(family, -power)] = (
                    centered_angular_relu_power(
                        points,
                        direction_sets[family],
                        radius,
                        power,
                    )
                )

        configurations: dict[str, np.ndarray] = {}
        for family in args.families:
            component_families = (
                ("first", "linearized")
                if family == "both"
                else (family,)
            )
            for count in args.direction_counts:
                degree_counts = (
                    [len(args.degrees)]
                    if args.only_full_harmonic
                    else range(1, len(args.degrees) + 1)
                )
                for degree_count in degree_counts:
                    degrees = args.degrees[:degree_count]
                    features = np.concatenate(
                        [
                            harmonic_cache[(one_family, degree)][:, :count]
                            for one_family in component_families
                            for degree in degrees
                        ],
                        axis=1,
                    )
                    degree_label = (
                        f"degree{degrees[0]}"
                        if len(degrees) == 1
                        else "degrees" + "_".join(map(str, degrees))
                    )
                    configurations[
                        f"{family}_{degree_label}_q{count}"
                    ] = features
                if not args.only_full_harmonic:
                    configurations[f"{family}_angular_relu_p1_q{count}"] = (
                        np.concatenate(
                            [
                                harmonic_cache[(one_family, -1)][:, :count]
                                for one_family in component_families
                            ],
                            axis=1,
                        )
                    )
                    configurations[
                        f"{family}_angular_relu_p1_3_5_q{count}"
                    ] = np.concatenate(
                        [
                            harmonic_cache[(one_family, -power)][:, :count]
                            for one_family in component_families
                            for power in (1, 3, 5)
                        ],
                        axis=1,
                    )

        baseline_prediction = np.mean(final, axis=0, dtype=np.float64)
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
                args.ridges,
            )
            for ridge in args.ridges:
                method_label = f"{label}_ridge{ridge:g}"
                method_mses[method_label] = float(
                    np.mean(
                        np.square(predictions[ridge] - targets[-1])
                    )
                )
                diagnostics[method_label] = {
                    **fit,
                    "full_feature_mean_norm": float(
                        np.linalg.norm(np.mean(features, axis=0))
                    ),
                }

        record = {
            "index": index,
            "name": name,
            "baseline_mse": baseline_mse,
            "method_mses": method_mses,
            "diagnostics": diagnostics,
            "linearized_singular_values": singular_values[:16].tolist(),
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

    summary = paired_summary(records)
    output = {
        "protocol": {
            "indices": args.indices,
            "rotation_seed": args.rotation_seed,
            "degrees": args.degrees,
            "direction_counts": args.direction_counts,
            "ridges": args.ridges,
            "folds": args.folds,
            "families": args.families,
            "only_full_harmonic": args.only_full_harmonic,
            "anchor": "exact zero spherical mean",
            "target_leakage": False,
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
