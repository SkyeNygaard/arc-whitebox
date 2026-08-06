"""Test whether disjoint Kerdock bases can supply a deployable cubic anchor.

For a rank-r late-layer cubic statistic

    z_k(x) = (u_k^T h(x)^2) (v_k^T h(x)) / radius^2,

the ideal control estimate is ``Q[f] - beta (Q[z] - E[z])``.  Estimating
``E[z]`` from the same full Kerdock cloud makes the correction identically
zero.  This harness tests the remaining same-cloud possibilities:

1. cross-basis: fit directions, anchor, and beta on the complement of a fold,
   then apply the correction only to the held-out fold;
2. delete-fold Edgeworth: estimate preactivation moments on the complement,
   map them through the cubic Edgeworth closure, and anchor the held-out fold;
3. jackknife Edgeworth: bias-correct the nonlinear full-cloud plug-in anchor;
4. asymmetric pilot/evaluation splits and zero-mean basis contrasts.

All construction and coefficient fitting is target-free.  Official targets
are used only to score the completed estimators.
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
    empirical_c21_state,
    forward_layer_and_final,
)
from eval_edgeworth_cubic_anchor import (  # noqa: E402
    contraction,
    edgeworth_m21_matrices,
    sample_pre_moment_data,
)
from eval_exact_anchor_residual import FULL_DATA, ROWS_PER_BASIS  # noqa: E402
from eval_kerdock_design import (  # noqa: E402
    N_BASES,
    WIDTH,
    make_kerdock_design,
    random_rotation,
)
from eval_sampling_official import _load_rows  # noqa: E402
from exact_moments import sphere_radius_mean  # noqa: E402


def basis_mask(bases: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(bases, dtype=bool), ROWS_PER_BASIS)


def raw_features(
    activation: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    radius: float,
) -> np.ndarray:
    h = np.asarray(activation, dtype=np.float64)
    return (
        (np.square(h) @ left) * (h @ right) / np.square(radius)
    )


def fit_beta(
    features: np.ndarray,
    outputs: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Ridge regression without a target-derived intercept."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(outputs, dtype=np.float64)
    scale = np.sqrt(np.mean(np.square(x), axis=0))
    keep = scale > 1e-14
    if not np.any(keep):
        return np.zeros((x.shape[1], y.shape[1])), {
            "kept": 0.0,
            "condition": 1.0,
        }
    normalized = x[:, keep] / scale[keep]
    system = (
        normalized.T @ normalized
        + ridge * len(normalized) * np.eye(np.sum(keep))
    )
    coefficient_normalized = np.linalg.solve(
        system,
        normalized.T @ y,
    )
    coefficient = np.zeros((x.shape[1], y.shape[1]), dtype=np.float64)
    coefficient[keep] = coefficient_normalized / scale[keep, None]
    return coefficient, {
        "kept": float(np.sum(keep)),
        "condition": float(np.linalg.cond(system)),
    }


def edgeworth_anchor(
    captured_pre: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    radius: float,
    *,
    label: str,
    step: float,
    half_width: int,
    chunk: int,
    nodes: int,
) -> np.ndarray:
    state = sample_pre_moment_data(captured_pre, radius)
    moments = edgeworth_m21_matrices(
        state,
        0,
        step=step,
        half_width=half_width,
        chunk=chunk,
        nodes=nodes,
    )
    return contraction(left, moments[label], right)


def cross_basis_prediction(
    captured_pre: np.ndarray,
    captured: np.ndarray,
    final: np.ndarray,
    fold_ids: np.ndarray,
    *,
    rank: int,
    radius: float,
    ridge: float,
    anchor_kind: str,
    edgeworth_label: str,
    step: float,
    half_width: int,
    chunk: int,
    nodes: int,
) -> tuple[np.ndarray, dict]:
    predictions = []
    held_sizes = []
    diagnostics = []
    for fold in range(int(np.max(fold_ids)) + 1):
        held_bases = fold_ids == fold
        held = basis_mask(held_bases)
        train = ~held
        left, right, _ = empirical_c21_state(captured[train], rank)
        z = raw_features(captured, left, right, radius)
        if anchor_kind == "raw":
            anchor = np.mean(z[train], axis=0)
        elif anchor_kind == "edgeworth":
            anchor = edgeworth_anchor(
                captured_pre[train],
                left,
                right,
                radius,
                label=edgeworth_label,
                step=step,
                half_width=half_width,
                chunk=chunk,
                nodes=nodes,
            )
        else:
            raise ValueError(anchor_kind)
        beta, fit_diagnostics = fit_beta(
            z[train] - anchor,
            final[train],
            ridge,
        )
        prediction = (
            np.mean(final[held], axis=0, dtype=np.float64)
            - (np.mean(z[held], axis=0) - anchor) @ beta
        )
        predictions.append(prediction)
        held_sizes.append(int(np.sum(held)))
        diagnostics.append(
            {
                **fit_diagnostics,
                "anchor_norm": float(np.linalg.norm(anchor)),
                "held_feature_gap_norm": float(
                    np.linalg.norm(np.mean(z[held], axis=0) - anchor)
                ),
            }
        )
    return np.average(predictions, axis=0, weights=held_sizes), {
        "folds": diagnostics,
    }


def global_anchor_prediction(
    captured: np.ndarray,
    final: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    anchor: np.ndarray,
    fold_ids: np.ndarray,
    *,
    radius: float,
    ridge: float,
) -> np.ndarray:
    """Cross-fit beta while holding a single supplied anchor fixed."""
    z = raw_features(captured, left, right, radius)
    predictions = []
    held_sizes = []
    for fold in range(int(np.max(fold_ids)) + 1):
        held = basis_mask(fold_ids == fold)
        train = ~held
        beta, _ = fit_beta(z[train] - anchor, final[train], ridge)
        predictions.append(
            np.mean(final[held], axis=0, dtype=np.float64)
            - (np.mean(z[held], axis=0) - anchor) @ beta
        )
        held_sizes.append(int(np.sum(held)))
    return np.average(predictions, axis=0, weights=held_sizes)


def paired_summary(records: list[dict], labels: list[str]) -> dict:
    baseline = np.asarray([record["baseline_mse"] for record in records])
    rng = np.random.default_rng(20260729)
    bootstrap = rng.integers(0, len(records), size=(20000, len(records)))
    result = {}
    for label in labels:
        values = np.asarray([record["method_mses"][label] for record in records])
        ratios = (
            np.mean(values[bootstrap], axis=1)
            / np.mean(baseline[bootstrap], axis=1)
        )
        result[label] = {
            "ratio": float(np.mean(values) / np.mean(baseline)),
            "ci95": [float(x) for x in np.percentile(ratios, [2.5, 97.5])],
            "wins": int(np.sum(values < baseline)),
            "worst": float(np.max(values / baseline)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(160, 168)))
    parser.add_argument("--layer", type=int, default=29)
    parser.add_argument("--rotation-seed", type=int, default=3)
    parser.add_argument("--partition-seed", type=int, default=20260729)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--folds", type=int, nargs="+", default=[2, 3, 6])
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument(
        "--edgeworth-label",
        choices=["gaussian", "third", "third_fourth", "full"],
        default="third",
    )
    parser.add_argument("--skip-edgeworth", action="store_true")
    parser.add_argument("--step", type=float, default=0.2)
    parser.add_argument("--half-width", type=int, default=5)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--nodes", type=int, default=12)
    parser.add_argument("--pilot-bases", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument(
        "--contrast-scales",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.5],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "results" / "basis_jackknife_anchor_selection8.json",
    )
    args = parser.parse_args()

    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    radius = sphere_radius_mean(WIDTH)
    rows = _load_rows(FULL_DATA, args.indices)
    records = []

    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        started = time.perf_counter()
        captured_pre, captured, final = forward_layer_and_final(
            weights,
            points,
            rotation,
            args.layer,
        )
        baseline_prediction = np.mean(final, axis=0, dtype=np.float64)
        baseline_mse = float(
            np.mean(np.square(baseline_prediction - targets[-1]))
        )
        method_predictions: dict[str, np.ndarray] = {}
        method_diagnostics: dict[str, dict] = {}
        permutation = np.random.default_rng(
            args.partition_seed
        ).permutation(N_BASES)

        for folds in args.folds:
            fold_ids = np.empty(N_BASES, dtype=np.int64)
            fold_ids[permutation] = np.arange(N_BASES) % folds
            prediction, diagnostics = cross_basis_prediction(
                captured_pre,
                captured,
                final,
                fold_ids,
                rank=args.rank,
                radius=radius,
                ridge=args.ridge,
                anchor_kind="raw",
                edgeworth_label=args.edgeworth_label,
                step=args.step,
                half_width=args.half_width,
                chunk=args.chunk,
                nodes=args.nodes,
            )
            method_predictions[f"cross_raw_folds{folds}"] = prediction
            method_diagnostics[f"cross_raw_folds{folds}"] = diagnostics

            if not args.skip_edgeworth:
                prediction, diagnostics = cross_basis_prediction(
                    captured_pre,
                    captured,
                    final,
                    fold_ids,
                    rank=args.rank,
                    radius=radius,
                    ridge=args.ridge,
                    anchor_kind="edgeworth",
                    edgeworth_label=args.edgeworth_label,
                    step=args.step,
                    half_width=args.half_width,
                    chunk=args.chunk,
                    nodes=args.nodes,
                )
                method_predictions[
                    f"cross_edgeworth_{args.edgeworth_label}_folds{folds}"
                ] = prediction
                method_diagnostics[
                    f"cross_edgeworth_{args.edgeworth_label}_folds{folds}"
                ] = diagnostics

        # Nonlinear delete-fold jackknife of the Edgeworth plug-in anchor.
        if not args.skip_edgeworth:
            left_full, right_full, _ = empirical_c21_state(captured, args.rank)
            full_edgeworth_anchor = edgeworth_anchor(
                captured_pre,
                left_full,
                right_full,
                radius,
                label=args.edgeworth_label,
                step=args.step,
                half_width=args.half_width,
                chunk=args.chunk,
                nodes=args.nodes,
            )
            for folds in args.folds:
                fold_ids = np.empty(N_BASES, dtype=np.int64)
                fold_ids[permutation] = np.arange(N_BASES) % folds
                delete_anchors = []
                for fold in range(folds):
                    train = ~basis_mask(fold_ids == fold)
                    delete_anchors.append(
                        edgeworth_anchor(
                            captured_pre[train],
                            left_full,
                            right_full,
                            radius,
                            label=args.edgeworth_label,
                            step=args.step,
                            half_width=args.half_width,
                            chunk=args.chunk,
                            nodes=args.nodes,
                        )
                    )
                jackknife_anchor = (
                    folds * full_edgeworth_anchor
                    - (folds - 1) * np.mean(delete_anchors, axis=0)
                )
                method_predictions[
                    f"jackknife_edgeworth_{args.edgeworth_label}_folds{folds}"
                ] = global_anchor_prediction(
                    captured,
                    final,
                    left_full,
                    right_full,
                    jackknife_anchor,
                    fold_ids,
                    radius=radius,
                    ridge=args.ridge,
                )
                method_diagnostics[
                    f"jackknife_edgeworth_{args.edgeworth_label}_folds{folds}"
                ] = {
                    "full_anchor_norm": float(
                        np.linalg.norm(full_edgeworth_anchor)
                    ),
                    "jackknife_delta_norm": float(
                        np.linalg.norm(jackknife_anchor - full_edgeworth_anchor)
                    ),
                }

        # A deliberately asymmetric estimator.  It preserves a first-order
        # correction but gives up all pilot-basis outputs in the direct form.
        # The contrast form retains the full baseline and uses a rotation-mean
        # zero pilot/evaluation moment difference.
        for pilot_bases in args.pilot_bases:
            pilot_basis_mask = np.zeros(N_BASES, dtype=bool)
            pilot_basis_mask[permutation[:pilot_bases]] = True
            pilot = basis_mask(pilot_basis_mask)
            evaluation = ~pilot
            left, right, _ = empirical_c21_state(captured[pilot], args.rank)
            z = raw_features(captured, left, right, radius)
            anchor = np.mean(z[pilot], axis=0)
            beta, diagnostics = fit_beta(
                z[pilot] - anchor,
                final[pilot],
                args.ridge,
            )
            gap = np.mean(z[evaluation], axis=0) - anchor
            method_predictions[f"asymmetric_raw_pilot{pilot_bases}"] = (
                np.mean(final[evaluation], axis=0, dtype=np.float64)
                - gap @ beta
            )
            for scale in args.contrast_scales:
                method_predictions[
                    f"full_plus_contrast_pilot{pilot_bases}_scale{scale:g}"
                ] = baseline_prediction - scale * (gap @ beta)
            method_diagnostics[f"asymmetric_raw_pilot{pilot_bases}"] = {
                **diagnostics,
                "gap_norm": float(np.linalg.norm(gap)),
            }

        method_mses = {
            label: float(np.mean(np.square(prediction - targets[-1])))
            for label, prediction in method_predictions.items()
        }
        record = {
            "index": index,
            "name": name,
            "baseline_mse": baseline_mse,
            "method_mses": method_mses,
            "diagnostics": method_diagnostics,
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

    labels = list(records[0]["method_mses"])
    summary = paired_summary(records, labels)
    for label in sorted(summary, key=lambda item: summary[item]["ratio"]):
        item = summary[label]
        print(
            f"{label:<48} ratio={item['ratio']:.5f} "
            f"CI=[{item['ci95'][0]:.5f},{item['ci95'][1]:.5f}] "
            f"wins={item['wins']}/{len(records)} worst={item['worst']:.2f}",
            flush=True,
        )

    output = {
        "protocol": {
            "indices": args.indices,
            "layer": args.layer,
            "rotation_seed": args.rotation_seed,
            "partition_seed": args.partition_seed,
            "rank": args.rank,
            "folds": args.folds,
            "ridge": args.ridge,
            "edgeworth_label": args.edgeworth_label,
            "skip_edgeworth": args.skip_edgeworth,
            "target_leakage": False,
        },
        "summary": summary,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
