"""Oracle test of the missing cumulant-state -> pointwise-control bridge.

The earlier first-layer controls answered the right statistical question but
did not transfer across held-out Kerdock rotations.  This experiment gives the
transported-cumulant proposal its strongest cheap test: use *oracle* deep-layer
moments to construct a pointwise control and fit its output coefficients with
oracle target errors on training rotations.

For a positively homogeneous network activation h_l and X ~ N(0, I_d), define

    phi_{u,v}(x)
      = (u^T h_l(x)^2) (v^T h_l(x)) / ||x||^2
        - u^T E[h_l(X)^2 h_l(X)^T] v / (d + 1).

The subtraction is an exact Gaussian anchor.  Writing X = R theta, the first
term is homogeneous of degree one and

    E[R^3] = (d + 1) E[R]

for a chi_d radius.  Hence E[phi] = 0.  The pointwise nonlinearity still
contains the network's ReLU kinks, so the degree-five spherical-design
exactness does not make its Kerdock error vanish.

Directions (u, v) are selected from SVDs of the oracle connected M21 slice and
its normalized symmetric/antisymmetric x1/x1a states.  A valid deployed
control would need to predict rotation-independent output coefficients from
weights/state.  Here those coefficients are fit using exact target errors on
training rotations and frozen on held-out rotations.  Failure at this oracle
ceiling falsifies the bridge before an expensive recursive state model is
built.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "arc_whitebox"
MOMENT_ROOT = (
    ROOT / "submissions" / "whest_bounded_ml" / "data"
)
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from eval_exact_anchor_residual import FULL_DATA  # noqa: E402
from eval_kerdock_design import WIDTH, make_kerdock_design, random_rotation  # noqa: E402
from eval_oracle_pointwise_bridge import oracle_fit  # noqa: E402
from eval_sampling_official import _load_rows  # noqa: E402
from exact_moments import sphere_radius_mean  # noqa: E402


@dataclass(frozen=True)
class DirectionFamily:
    name: str
    left: np.ndarray
    right: np.ndarray
    anchor: np.ndarray


def moment_path(index: int) -> Path:
    for directory in ("higher_fresh", "higher"):
        candidate = MOMENT_ROOT / directory / f"mlp_{index:05d}.npz"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no higher-moment file for official index {index}")


def connected_m21(
    mean: np.ndarray,
    second: np.ndarray,
    raw_m21: np.ndarray,
    marginal_second: np.ndarray,
) -> np.ndarray:
    """K[i,j] = E[(H_i-mu_i)^2 (H_j-mu_j)]."""
    return (
        raw_m21
        - 2.0 * mean[:, None] * second
        - marginal_second[:, None] * mean[None, :]
        + 2.0 * np.square(mean[:, None]) * mean[None, :]
    )


def truncated_svd(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    left, _, right_t = np.linalg.svd(matrix, full_matrices=False)
    return left[:, :rank], right_t.T[:, :rank]


def direction_families(
    moments: np.lib.npyio.NpzFile,
    layer: int,
    rank: int,
) -> dict[str, DirectionFamily]:
    mean = np.asarray(moments["mean"][layer], dtype=np.float64)
    second = np.asarray(moments["M11"][layer], dtype=np.float64)
    raw_m21 = np.asarray(moments["M21"][layer], dtype=np.float64)
    marginal_second = np.asarray(moments["m2"][layer], dtype=np.float64)
    cumulant = connected_m21(mean, second, raw_m21, marginal_second)

    covariance = second - np.outer(mean, mean)
    sigma = np.sqrt(np.maximum(np.diag(covariance), 1e-20))
    denominator = np.maximum(
        sigma[:, None] ** 3 + sigma[None, :] ** 3,
        1e-20,
    )
    x1 = (cumulant + cumulant.T) / denominator
    x1a = (cumulant - cumulant.T) / denominator

    matrices = {
        "c21": cumulant,
        "x1": x1,
        "x1a": x1a,
    }
    result = {}
    for name, matrix in matrices.items():
        left, right = truncated_svd(matrix, rank)
        # The directions come from a connected/normalized state, but the exact
        # anchor contracts the corresponding *raw* third moment.
        anchor = np.einsum("ik,ij,jk->k", left, raw_m21, right) / (WIDTH + 1)
        result[name] = DirectionFamily(name, left, right, anchor)
    return result


def forward_with_features(
    weights: np.ndarray,
    points: np.ndarray,
    rotation: np.ndarray,
    layer_families: dict[int, dict[str, DirectionFamily]],
    ranks: list[int],
    radius: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    activation = points @ (rotation @ weights[0].astype(np.float32))
    activation = np.maximum(activation, 0.0)
    feature_means: dict[str, np.ndarray] = {}

    for layer, weight in enumerate(weights):
        if layer > 0:
            activation = np.maximum(activation @ weight, 0.0)
        if layer not in layer_families:
            continue

        h = activation.astype(np.float64, copy=False)
        h_squared = np.square(h)
        family_means = {}
        for family_name, family in layer_families[layer].items():
            left_projection = h_squared @ family.left
            right_projection = h @ family.right
            quadrature = (
                np.mean(left_projection * right_projection, axis=0) / radius**2
                - family.anchor
            )
            family_means[family_name] = quadrature
            for rank in ranks:
                feature_means[
                    f"layer{layer:02d}_{family_name}_rank{rank}"
                ] = quadrature[:rank]

        for rank in ranks:
            feature_means[f"layer{layer:02d}_x1+x1a_rank{rank}"] = np.concatenate(
                (family_means["x1"][:rank], family_means["x1a"][:rank])
            )

    return activation, feature_means


def summaries(
    records: list[dict],
    labels: list[str],
) -> dict[str, dict[str, float | int | list[float]]]:
    baseline = np.asarray([record["baseline_mse"] for record in records])
    rng = np.random.default_rng(20260729)
    boot = rng.integers(0, len(records), size=(20000, len(records)))
    result = {}
    for label in labels:
        values = np.asarray([record["method_mses"][label] for record in records])
        ratios = values[boot].mean(axis=1) / baseline[boot].mean(axis=1)
        result[label] = {
            "ratio": float(values.mean() / baseline.mean()),
            "ci95": [float(x) for x in np.percentile(ratios, [2.5, 97.5])],
            "wins": int(np.sum(values < baseline)),
            "worst": float(np.max(values / baseline)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(100, 108)))
    parser.add_argument("--layers", type=int, nargs="+", default=[7, 15, 23, 29])
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--train-rotations", type=int, nargs="+", default=list(range(12)))
    parser.add_argument(
        "--test-rotations",
        type=int,
        nargs="+",
        default=list(range(12, 24)),
    )
    parser.add_argument("--ridges", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "results" / "oracle_cumulant_bridge_full8.json",
    )
    args = parser.parse_args()
    if set(args.train_rotations) & set(args.test_rotations):
        raise ValueError("rotation split overlaps")
    if not args.ranks or max(args.ranks) > WIDTH:
        raise ValueError(args.ranks)

    points = make_kerdock_design()
    radius = sphere_radius_mean(WIDTH)
    all_rotation_seeds = args.train_rotations + args.test_rotations
    rotations = {
        seed: random_rotation(WIDTH, seed)
        for seed in all_rotation_seeds
    }
    rows = _load_rows(FULL_DATA, args.indices)
    records = []

    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        started = time.perf_counter()
        path = moment_path(index)
        with np.load(path) as moment_data:
            if int(moment_data["global_index"]) != index:
                raise ValueError((path, moment_data["global_index"], index))
            if not np.array_equal(targets, moment_data["official_alm"]):
                raise ValueError(f"official target mismatch for {index}")
            layer_families = {
                layer: direction_families(moment_data, layer, max(args.ranks))
                for layer in args.layers
            }

        errors = {}
        features_by_rotation = {}
        for seed, rotation in rotations.items():
            final, feature_means = forward_with_features(
                weights,
                points,
                rotation,
                layer_families,
                args.ranks,
                radius,
            )
            errors[seed] = final.mean(axis=0, dtype=np.float64) - targets[-1]
            features_by_rotation[seed] = feature_means

        train_errors = np.stack([errors[seed] for seed in args.train_rotations])
        test_errors = np.stack([errors[seed] for seed in args.test_rotations])
        baseline_mse = float(np.mean(np.square(test_errors)))
        feature_labels = list(features_by_rotation[all_rotation_seeds[0]])
        method_mses = {}
        diagnostics = {}

        # Test each family/layer separately.
        for feature_label in feature_labels:
            q_train = np.stack(
                [features_by_rotation[s][feature_label] for s in args.train_rotations]
            )
            q_test = np.stack(
                [features_by_rotation[s][feature_label] for s in args.test_rotations]
            )
            for ridge in args.ridges:
                prediction, fit_diagnostics = oracle_fit(
                    q_train,
                    train_errors,
                    q_test,
                    ridge,
                )
                label = f"{feature_label}:ridge={ridge:g}"
                method_mses[label] = float(
                    np.mean(np.square(test_errors - prediction))
                )
                diagnostics[label] = {
                    **fit_diagnostics,
                    "anchor_mean_norm": float(
                        np.linalg.norm(
                            np.mean(
                                np.stack(
                                    [features_by_rotation[s][feature_label]
                                     for s in all_rotation_seeds]
                                ),
                                axis=0,
                            )
                        )
                    ),
                    "feature_rms": float(np.sqrt(np.mean(np.square(q_train)))),
                }

        # A joint x1/x1a dictionary across all requested deep layers is the
        # closest oracle analogue of a transported-cumulant state model.
        for rank in args.ranks:
            joint_labels = [
                f"layer{layer:02d}_x1+x1a_rank{rank}"
                for layer in args.layers
            ]
            q_train = np.stack(
                [
                    np.concatenate(
                        [features_by_rotation[s][label] for label in joint_labels]
                    )
                    for s in args.train_rotations
                ]
            )
            q_test = np.stack(
                [
                    np.concatenate(
                        [features_by_rotation[s][label] for label in joint_labels]
                    )
                    for s in args.test_rotations
                ]
            )
            for ridge in args.ridges:
                prediction, fit_diagnostics = oracle_fit(
                    q_train,
                    train_errors,
                    q_test,
                    ridge,
                )
                label = f"all_layers_x1+x1a_rank{rank}:ridge={ridge:g}"
                method_mses[label] = float(
                    np.mean(np.square(test_errors - prediction))
                )
                diagnostics[label] = fit_diagnostics

        best_label = min(method_mses, key=method_mses.get)
        record = {
            "index": index,
            "name": name,
            "moment_path": str(path),
            "baseline_mse": baseline_mse,
            "method_mses": method_mses,
            "diagnostics": diagnostics,
            "seconds": time.perf_counter() - started,
        }
        records.append(record)
        print(
            f"[{index:>4}] {name[:20]:<20} base={baseline_mse:.4e} "
            f"best={best_label} "
            f"{method_mses[best_label] / baseline_mse:.4f}x "
            f"({record['seconds']:.1f}s)",
            flush=True,
        )

    labels = list(records[0]["method_mses"])
    summary = summaries(records, labels)
    print("\nHeld-rotation deep-cumulant oracle ceiling", flush=True)
    for label in sorted(summary, key=lambda key: summary[key]["ratio"])[:30]:
        item = summary[label]
        print(
            f"{label:<48} ratio={item['ratio']:.5f} "
            f"CI=[{item['ci95'][0]:.5f},{item['ci95'][1]:.5f}] "
            f"wins={item['wins']}/{len(records)} worst={item['worst']:.2f}x",
            flush=True,
        )

    output = {
        "protocol": {
            "indices": args.indices,
            "layers": args.layers,
            "ranks": args.ranks,
            "train_rotations": args.train_rotations,
            "test_rotations": args.test_rotations,
            "ridges": args.ridges,
            "anchor": (
                "(u^T h^2)(v^T h)/||x||^2 - "
                "u^T E[h^2 h^T]v/(d+1)"
            ),
            "oracle_warning": (
                "Both direction state and training coefficients use oracle "
                "higher-moment/target data. This is an expressivity ceiling, "
                "not a deployable estimator."
            ),
        },
        "summary": summary,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
