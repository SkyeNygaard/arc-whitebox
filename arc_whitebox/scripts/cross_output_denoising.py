"""Cross-output denoising of the strict N=32,768 Sobol estimator.

For a fixed penultimate activation ``a`` and a fresh final column
``w ~ N(0, 2/n)``,

    E_w[ReLU(a @ w)] = ||a|| / sqrt(pi*n).

The final 256 columns therefore provide an observable ensemble control for
shared directional error.  The identity is only exact in expectation over
weights, however; the benchmark fixes a finite set of columns.  This script
tests forced and partially shrunk identity controls, weight-norm shrinkage, and
low-rank ridge smoothing across final weight projections.

All tunable coefficients and model-family selection use mini MLPs 0--49 only,
with five-fold whole-MLP cross-validation.  The frozen rule is evaluated on
mini MLPs 50--99.  The script deliberately stops there unless the held-out
gain reaches 10%.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import gammaln, ndtr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_sampling_official import _load_rows, precompute_design  # noqa: E402


DEFAULT_DATA = ROOT / "data" / "official_phase1_mini" / "data"
DEFAULT_BASELINE = ROOT / "results" / "sobol_vectors_n32768.json"
DEFAULT_OUT = ROOT / "results" / "cross_output_denoising.json"
SQRT_2PI = math.sqrt(2.0 * math.pi)


@dataclass
class OutputData:
    indices: np.ndarray
    names: list[str]
    target: np.ndarray
    direct: np.ndarray
    norm_identity: np.ndarray
    corrections: dict[str, np.ndarray]
    seconds: list[float]


@dataclass
class FrozenRule:
    name: str
    kind: str
    cv_mse: float
    correction: str | None = None
    alpha: float | None = None
    ridge: float | None = None
    coef: np.ndarray | None = None
    correction_order: list[str] | None = None


def radial_moment(width: int, order: int) -> float:
    return float(
        math.exp(
            0.5 * order * math.log(2.0)
            + gammaln(0.5 * (width + order))
            - gammaln(0.5 * width)
        )
    )


def projection_smooth(
    direct: np.ndarray,
    feature: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Within-MLP ridge smoother with an unpenalized mean."""
    y_mean = float(np.mean(direct))
    centered_y = direct - y_mean
    centered_x = feature - np.mean(feature, axis=0)
    scale = np.sqrt(np.maximum(np.mean(np.square(centered_x), axis=0), 1e-20))
    x = centered_x / scale
    gram = x.T @ x / len(x)
    rhs = x.T @ centered_y / len(x)
    coef = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), rhs)
    # Centering makes the smoother preserve the observed output ensemble mean.
    return y_mean + x @ coef


def one_pass(
    weights: np.ndarray,
    input_blocks: list[np.ndarray],
    projection_ridges: tuple[float, ...],
    projection_ranks: tuple[int, ...],
) -> tuple[np.ndarray, float, dict[str, np.ndarray], float]:
    """Collect all observables during one strict QMC network pass."""
    width = weights.shape[-1]
    samples = sum(len(block) for block in input_blocks)
    direct_sum = np.zeros(width, dtype=np.float64)
    penultimate_sum = np.zeros(width, dtype=np.float64)
    penultimate_second = np.zeros((width, width), dtype=np.float64)
    norm_sum = 0.0

    start = time.perf_counter()
    for x in input_blocks:
        activation = x
        for weight in weights[:-1]:
            activation = np.maximum(activation @ weight, 0.0)
        penultimate = activation
        final = np.maximum(penultimate @ weights[-1], 0.0)
        direct_sum += final.sum(axis=0, dtype=np.float64)
        penultimate64 = penultimate.astype(np.float64)
        penultimate_sum += penultimate64.sum(axis=0)
        penultimate_second += penultimate64.T @ penultimate64
        norm_sum += float(np.sum(np.linalg.norm(penultimate64, axis=1)))
    elapsed = time.perf_counter() - start

    direct = direct_sum / samples
    expected_radius = radial_moment(width, 1)
    radial_second_factor = radial_moment(width, 2) / expected_radius**2
    mean_a = penultimate_sum / samples
    raw_second_a = penultimate_second / samples * radial_second_factor
    covariance_a = raw_second_a - np.outer(mean_a, mean_a)
    covariance_a = 0.5 * (covariance_a + covariance_a.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_a)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    norm_identity = norm_sum / samples / math.sqrt(math.pi * width)
    final_weights = weights[-1].astype(np.float64)
    column_norm = np.linalg.norm(final_weights, axis=0)
    norm_weight_target = norm_identity * column_norm / np.mean(column_norm)

    preactivation_mean = mean_a @ final_weights
    preactivation_variance = np.maximum(
        np.einsum(
            "io,ij,jo->o",
            final_weights,
            covariance_a,
            final_weights,
            optimize=True,
        ),
        1e-20,
    )
    preactivation_sd = np.sqrt(preactivation_variance)
    t = preactivation_mean / preactivation_sd
    phi = np.exp(-0.5 * np.square(t)) / SQRT_2PI
    gaussian = preactivation_mean * ndtr(t) + preactivation_sd * phi

    observed_mean = float(np.mean(direct))
    corrections: dict[str, np.ndarray] = {
        "identity_additive": np.full(width, norm_identity - observed_mean),
        "identity_scale": direct * (norm_identity / max(observed_mean, 1e-20) - 1.0),
        "weight_norm": norm_weight_target - direct,
        "gaussian_projection": gaussian - direct,
    }

    for rank in projection_ranks:
        vectors = eigenvectors[:, :rank]
        values = np.sqrt(eigenvalues[:rank])
        projection = final_weights.T @ vectors
        # Signed projections locate a column in the dominant activation
        # subspace; squared, eigenvalue-scaled projections expose its variance.
        feature = np.column_stack(
            (
                gaussian,
                preactivation_mean,
                preactivation_sd,
                column_norm,
                projection,
                np.square(projection) * values[None, :],
            )
        )
        for ridge in projection_ridges:
            smoothed = projection_smooth(direct, feature, ridge)
            corrections[f"projection_rank{rank}_ridge{ridge:g}"] = smoothed - direct

    return direct, norm_identity, corrections, elapsed


def collect(
    data_dir: Path,
    indices: list[int],
    input_blocks: list[np.ndarray],
    projection_ridges: tuple[float, ...],
    projection_ranks: tuple[int, ...],
    progress_every: int,
) -> OutputData:
    rows = _load_rows(data_dir, indices)
    names: list[str] = []
    targets: list[np.ndarray] = []
    directs: list[np.ndarray] = []
    identities: list[float] = []
    corrections: dict[str, list[np.ndarray]] = {}
    seconds: list[float] = []
    for position, (index, (name, weights, all_targets)) in enumerate(
        zip(indices, rows, strict=True), start=1
    ):
        direct, identity, row_corrections, elapsed = one_pass(
            weights,
            input_blocks,
            projection_ridges,
            projection_ranks,
        )
        names.append(name)
        targets.append(all_targets[-1])
        directs.append(direct)
        identities.append(identity)
        seconds.append(elapsed)
        for key, value in row_corrections.items():
            corrections.setdefault(key, []).append(value)
        if position == 1 or position % progress_every == 0 or position == len(rows):
            mse = np.mean(np.square(direct - all_targets[-1]))
            print(
                f"[{position:3d}/{len(rows)}] id={index:3d} "
                f"direct={mse:.3e} seconds={elapsed:.3f}",
                flush=True,
            )
    return OutputData(
        indices=np.asarray(indices, dtype=np.int64),
        names=names,
        target=np.stack(targets),
        direct=np.stack(directs),
        norm_identity=np.asarray(identities),
        corrections={
            key: np.stack(value) for key, value in corrections.items()
        },
        seconds=seconds,
    )


def subset(data: OutputData, mask: np.ndarray) -> OutputData:
    return OutputData(
        indices=data.indices[mask],
        names=[name for name, keep in zip(data.names, mask, strict=True) if keep],
        target=data.target[mask],
        direct=data.direct[mask],
        norm_identity=data.norm_identity[mask],
        corrections={key: value[mask] for key, value in data.corrections.items()},
        seconds=[value for value, keep in zip(data.seconds, mask, strict=True) if keep],
    )


def fit_alpha(
    data: OutputData,
    correction: str,
    mask: np.ndarray | None = None,
) -> float:
    if mask is None:
        mask = np.ones(len(data.indices), dtype=bool)
    delta = data.corrections[correction][mask].reshape(-1)
    residual = (data.target[mask] - data.direct[mask]).reshape(-1)
    alpha = float(delta @ residual / max(float(delta @ delta), 1e-30))
    return float(np.clip(alpha, 0.0, 1.0))


def alpha_prediction(
    data: OutputData,
    correction: str,
    alpha: float,
) -> np.ndarray:
    return np.maximum(
        data.direct + alpha * data.corrections[correction],
        0.0,
    )


def solve_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    rms = np.sqrt(np.maximum(np.mean(np.square(x), axis=0), 1e-30))
    normalized = x / rms
    gram = normalized.T @ normalized / len(normalized)
    rhs = normalized.T @ y / len(normalized)
    regularized = gram + ridge * np.eye(gram.shape[0])
    if ridge == 0.0:
        coef_normalized = np.linalg.lstsq(regularized, rhs, rcond=1e-12)[0]
    else:
        coef_normalized = np.linalg.solve(regularized, rhs)
    return coef_normalized / rms


def ridge_design(
    data: OutputData,
    correction_order: list[str],
) -> np.ndarray:
    return np.stack(
        [data.corrections[name] for name in correction_order],
        axis=-1,
    )


def ridge_prediction(
    data: OutputData,
    correction_order: list[str],
    coef: np.ndarray,
) -> np.ndarray:
    correction = ridge_design(data, correction_order) @ coef
    return np.maximum(data.direct + correction, 0.0)


def cv_alpha(data: OutputData, correction: str) -> float:
    folds = data.indices % 5
    squared_error = 0.0
    count = 0
    for fold in range(5):
        alpha = fit_alpha(data, correction, folds != fold)
        hold = folds == fold
        prediction = alpha_prediction(subset(data, hold), correction, alpha)
        squared_error += float(np.sum(np.square(prediction - data.target[hold])))
        count += prediction.size
    return squared_error / count


def cv_ridge(
    data: OutputData,
    correction_order: list[str],
    ridges: list[float],
) -> tuple[float, float]:
    folds = data.indices % 5
    design = ridge_design(data, correction_order)
    residual = data.target - data.direct
    scores = []
    for ridge in ridges:
        squared_error = 0.0
        count = 0
        for fold in range(5):
            fit = folds != fold
            hold = folds == fold
            x = design[fit].reshape(-1, design.shape[-1])
            y = residual[fit].reshape(-1)
            coef = solve_ridge(x, y, ridge)
            prediction = ridge_prediction(
                subset(data, hold),
                correction_order,
                coef,
            )
            squared_error += float(
                np.sum(np.square(prediction - data.target[hold]))
            )
            count += prediction.size
        scores.append(squared_error / count)
    best = int(np.argmin(scores))
    return ridges[best], scores[best]


def fit_rules(data: OutputData) -> list[FrozenRule]:
    rules: list[FrozenRule] = []
    for correction in data.corrections:
        cv_mse = cv_alpha(data, correction)
        rules.append(
            FrozenRule(
                name=f"partial_{correction}",
                kind="alpha",
                correction=correction,
                alpha=fit_alpha(data, correction),
                cv_mse=cv_mse,
            )
        )

    correction_order = list(data.corrections)
    ridges = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    ridge, cv_mse = cv_ridge(data, correction_order, ridges)
    design = ridge_design(data, correction_order)
    coef = solve_ridge(
        design.reshape(-1, design.shape[-1]),
        (data.target - data.direct).reshape(-1),
        ridge,
    )
    rules.append(
        FrozenRule(
            name="combined_ridge",
            kind="ridge",
            ridge=ridge,
            coef=coef,
            correction_order=correction_order,
            cv_mse=cv_mse,
        )
    )
    return rules


def rule_prediction(data: OutputData, rule: FrozenRule) -> np.ndarray:
    if rule.kind == "direct":
        return data.direct
    if rule.kind == "alpha":
        assert rule.correction is not None and rule.alpha is not None
        return alpha_prediction(data, rule.correction, rule.alpha)
    assert rule.coef is not None and rule.correction_order is not None
    return ridge_prediction(data, rule.correction_order, rule.coef)


def metrics(
    prediction: np.ndarray,
    data: OutputData,
) -> dict[str, float | int]:
    per_mlp = np.mean(np.square(prediction - data.target), axis=1)
    direct_per_mlp = np.mean(np.square(data.direct - data.target), axis=1)
    return {
        "mse": float(np.mean(per_mlp)),
        "gain_over_direct": float(
            np.mean(direct_per_mlp) / np.mean(per_mlp)
        ),
        "median_mlp_mse": float(np.median(per_mlp)),
        "p90_mlp_mse": float(np.quantile(per_mlp, 0.9)),
        "max_mlp_mse": float(np.max(per_mlp)),
        "max_mlp_id": int(data.indices[int(np.argmax(per_mlp))]),
        "fraction_mlps_improved": float(np.mean(per_mlp < direct_per_mlp)),
        "median_per_mlp_gain": float(
            np.median(direct_per_mlp / np.maximum(per_mlp, 1e-30))
        ),
    }


def identity_diagnostics(data: OutputData) -> dict[str, object]:
    observed_mean = np.mean(data.direct, axis=1)
    target_mean = np.mean(data.target, axis=1)
    proposal = data.norm_identity - observed_mean
    needed = target_mean - observed_mean
    fixed_column_gap = target_mean - data.norm_identity
    if np.std(proposal) > 0.0 and np.std(needed) > 0.0:
        correlation = float(np.corrcoef(proposal, needed)[0, 1])
    else:
        correlation = 0.0
    return {
        "proposal_q_minus_direct_mean_quantiles": [
            float(x) for x in np.quantile(proposal, (0.0, 0.1, 0.5, 0.9, 1.0))
        ],
        "needed_target_minus_direct_mean_quantiles": [
            float(x) for x in np.quantile(needed, (0.0, 0.1, 0.5, 0.9, 1.0))
        ],
        "fixed_column_target_mean_minus_q_quantiles": [
            float(x)
            for x in np.quantile(
                fixed_column_gap,
                (0.0, 0.1, 0.5, 0.9, 1.0),
            )
        ],
        "proposal_needed_correlation_across_mlps": correlation,
        "proposal_rms": float(np.sqrt(np.mean(np.square(proposal)))),
        "needed_rms": float(np.sqrt(np.mean(np.square(needed)))),
        "fixed_column_gap_rms": float(
            np.sqrt(np.mean(np.square(fixed_column_gap)))
        ),
    }


def baseline_reproduction(
    data: OutputData,
    baseline_path: Path,
) -> dict[str, float]:
    with baseline_path.open() as handle:
        artifact = json.load(handle)
    by_id = {int(run["index"]): run for run in artifact["runs"]}
    saved = np.stack(
        [
            np.asarray(by_id[int(index)]["final_prediction"], dtype=np.float64)
            for index in data.indices
        ]
    )
    difference = data.direct - saved
    return {
        "max_abs_vector_difference": float(np.max(np.abs(difference))),
        "mse_between_vectors": float(np.mean(np.square(difference))),
    }


def forced_metrics(data: OutputData) -> dict[str, dict[str, float | int]]:
    return {
        correction: metrics(alpha_prediction(data, correction, 1.0), data)
        for correction in (
            "identity_additive",
            "identity_scale",
            "weight_norm",
        )
    }


def rule_dict(
    rule: FrozenRule,
    train: OutputData,
    test: OutputData,
) -> dict[str, object]:
    return {
        "kind": rule.kind,
        "correction": rule.correction,
        "alpha": rule.alpha,
        "ridge": rule.ridge,
        "coef": None if rule.coef is None else rule.coef.tolist(),
        "correction_order": rule.correction_order,
        "whole_mlp_cv_mse": rule.cv_mse,
        "train": metrics(rule_prediction(train, rule), train),
        "test": metrics(rule_prediction(test, rule), test),
    }


def outliers(
    data: OutputData,
    prediction: np.ndarray,
    count: int = 8,
) -> list[dict[str, float | int | str]]:
    direct_mse = np.mean(np.square(data.direct - data.target), axis=1)
    model_mse = np.mean(np.square(prediction - data.target), axis=1)
    order = np.argsort(direct_mse)[::-1][:count]
    return [
        {
            "id": int(data.indices[i]),
            "name": data.names[i],
            "direct_mse": float(direct_mse[i]),
            "model_mse": float(model_mse[i]),
            "gain": float(direct_mse[i] / max(model_mse[i], 1e-30)),
        }
        for i in order
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--samples", type=int, default=32768)
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    input_blocks = precompute_design(
        "sobol",
        256,
        args.samples,
        args.seed,
        antithetic=True,
        sphere=True,
        chunk=args.chunk,
    )
    data = collect(
        args.data,
        list(range(100)),
        input_blocks,
        projection_ridges=(0.01, 0.1, 1.0),
        projection_ranks=(8, 16),
        progress_every=args.progress_every,
    )
    reproduction = baseline_reproduction(data, args.baseline)
    if reproduction["max_abs_vector_difference"] > 1e-10:
        raise AssertionError(f"strict direct vectors did not reproduce: {reproduction}")

    train = subset(data, data.indices < 50)
    test = subset(data, data.indices >= 50)
    rules = fit_rules(train)
    direct_rule = FrozenRule(
        name="direct",
        kind="direct",
        cv_mse=float(np.mean(np.square(train.direct - train.target))),
    )
    # Direct participates in family selection, preventing a noisy training
    # improvement from forcing a harmful correction onto the held-out MLPs.
    selected = min([direct_rule, *rules], key=lambda rule: rule.cv_mse)
    selected_prediction = rule_prediction(test, selected)
    selected_metrics = metrics(selected_prediction, test)

    result: dict[str, object] = {
        "protocol": {
            "samples": args.samples,
            "design": "scrambled_sobol_antithetic_sphere",
            "seed": args.seed,
            "reuse_inputs": True,
            "train_ids": [0, 49],
            "test_ids": [50, 99],
            "selection": "five-fold whole-MLP CV within train IDs only",
            "stop_threshold_gain": 1.1,
        },
        "baseline_reproduction": reproduction,
        "direct": {
            "train": metrics(train.direct, train),
            "test": metrics(test.direct, test),
        },
        "identity_diagnostics": {
            "train": identity_diagnostics(train),
            "test": identity_diagnostics(test),
        },
        "forced_controls": {
            "train": forced_metrics(train),
            "test": forced_metrics(test),
        },
        "frozen_rules": {
            rule.name: rule_dict(rule, train, test) for rule in rules
        },
        "selected_model": selected.name,
        "selected_cv_mse": selected.cv_mse,
        "selected_test": selected_metrics,
        "heldout_gain_reaches_10_percent": bool(
            selected_metrics["gain_over_direct"] >= 1.1
        ),
        "selected_test_outliers_by_direct_mse": outliers(
            test,
            selected_prediction,
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    print("\nRESULT_SUMMARY")
    print(
        json.dumps(
            {
                "identity_diagnostics": result["identity_diagnostics"],
                "forced_test": result["forced_controls"]["test"],
                "selected_model": selected.name,
                "selected_test": selected_metrics,
                "reaches_10_percent": result["heldout_gain_reaches_10_percent"],
                "out": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
