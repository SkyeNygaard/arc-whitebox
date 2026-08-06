"""Held-out terminal-marginal smoothing on the strict Sobol sphere design.

For every MLP, a single forward pass over the same 32,768 input points returns
the direct final ReLU mean and the first four raw moments of the final
pre-activation.  The latter produce Gaussian, third-order Edgeworth, and
fourth-order Edgeworth marginal estimates.

The sphere construction integrates the radial coordinate analytically for a
positively homogeneous ReLU MLP.  Direct means use E[R], while raw k-th moments
must use E[R**k]; applying the first-moment scale to all four moments would
silently bias the variance, skewness, and kurtosis.

Mini MLPs 0--49 are the only fitting set.  Ridge strengths and the gated
marginal are chosen by five-fold whole-MLP cross-validation within those 50
MLPs.  Mini MLPs 50--99 are evaluated exactly once after freezing the rule.
If the selected rule improves held-out MSE by more than 15%, the same frozen
rule is also evaluated on a disjoint slice of the official full split.
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


SQRT_2PI = math.sqrt(2.0 * math.pi)
DEFAULT_MINI = ROOT / "data" / "official_phase1_mini" / "data"
DEFAULT_FULL = ROOT / "data" / "official_phase1_full" / "data"
DEFAULT_BASELINE = ROOT / "results" / "sobol_vectors_n32768.json"
DEFAULT_OUT = ROOT / "results" / "terminal_marginal_smoothing.json"


@dataclass
class TerminalData:
    indices: np.ndarray
    names: list[str]
    target: np.ndarray
    direct: np.ndarray
    gaussian: np.ndarray
    edgeworth3: np.ndarray
    edgeworth4: np.ndarray
    t: np.ndarray
    skew: np.ndarray
    kurtosis: np.ndarray
    seconds: list[float]


@dataclass
class FrozenModel:
    name: str
    kind: str
    ridge: float
    coef: np.ndarray
    marginal: str | None
    cv_mse: float


def radial_moment(width: int, order: int) -> float:
    """E[R**order] for R distributed as chi(width)."""
    return float(
        math.exp(
            0.5 * order * math.log(2.0)
            + gammaln(0.5 * (width + order))
            - gammaln(0.5 * width)
        )
    )


def terminal_pass(
    weights: np.ndarray,
    input_blocks: list[np.ndarray],
) -> tuple[dict[str, np.ndarray], float]:
    """Compute all terminal estimates in one network sampling pass."""
    width = weights.shape[-1]
    samples = sum(len(block) for block in input_blocks)
    direct_sum = np.zeros(width, dtype=np.float64)
    raw_sum = np.zeros((4, width), dtype=np.float64)

    start = time.perf_counter()
    for x in input_blocks:
        activation = x
        for layer, weight in enumerate(weights):
            preactivation = activation @ weight
            activation = np.maximum(preactivation, 0.0)
            if layer + 1 == len(weights):
                h = preactivation.astype(np.float64)
                h2 = np.square(h)
                direct_sum += activation.sum(axis=0, dtype=np.float64)
                raw_sum[0] += h.sum(axis=0)
                raw_sum[1] += h2.sum(axis=0)
                raw_sum[2] += (h2 * h).sum(axis=0)
                raw_sum[3] += np.square(h2).sum(axis=0)
    elapsed = time.perf_counter() - start

    direct = direct_sum / samples
    raw = raw_sum / samples

    # Design.next() puts every direction at E[R].  Positive homogeneity makes
    # the first moment exact, but raw higher moments need their own radial
    # factors to recover moments under the original Gaussian input.
    expected_radius = radial_moment(width, 1)
    for order in range(1, 5):
        raw[order - 1] *= radial_moment(width, order) / expected_radius**order

    mu = raw[0]
    variance = np.maximum(raw[1] - np.square(mu), 1e-24)
    sigma = np.sqrt(variance)
    centered3 = raw[2] - 3.0 * mu * raw[1] + 2.0 * np.power(mu, 3)
    centered4 = (
        raw[3]
        - 4.0 * mu * raw[2]
        + 6.0 * np.square(mu) * raw[1]
        - 3.0 * np.power(mu, 4)
    )
    cumulant4 = centered4 - 3.0 * np.square(variance)
    t = mu / sigma
    skew = centered3 / np.power(sigma, 3)
    kurtosis = cumulant4 / np.square(variance)

    phi = np.exp(-0.5 * np.square(t)) / SQRT_2PI
    gaussian = mu * ndtr(t) + sigma * phi
    edgeworth3 = gaussian - sigma * t * phi * skew / 6.0
    edgeworth4 = (
        edgeworth3
        + sigma * (np.square(t) - 1.0) * phi * kurtosis / 24.0
    )

    return {
        "direct": np.maximum(direct, 0.0),
        "gaussian": np.maximum(gaussian, 0.0),
        "edgeworth3": np.maximum(edgeworth3, 0.0),
        "edgeworth4": np.maximum(edgeworth4, 0.0),
        "t": np.nan_to_num(t, nan=0.0, posinf=20.0, neginf=-20.0),
        "skew": np.nan_to_num(skew, nan=0.0, posinf=20.0, neginf=-20.0),
        "kurtosis": np.nan_to_num(
            kurtosis, nan=0.0, posinf=100.0, neginf=-100.0
        ),
    }, elapsed


def collect(
    data_dir: Path,
    indices: list[int],
    input_blocks: list[np.ndarray],
    progress_every: int,
) -> TerminalData:
    rows = _load_rows(data_dir, indices)
    names: list[str] = []
    target: list[np.ndarray] = []
    fields: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "direct",
            "gaussian",
            "edgeworth3",
            "edgeworth4",
            "t",
            "skew",
            "kurtosis",
        )
    }
    seconds: list[float] = []
    for position, (index, (name, weights, all_targets)) in enumerate(
        zip(indices, rows, strict=True), start=1
    ):
        estimates, elapsed = terminal_pass(weights, input_blocks)
        names.append(name)
        target.append(all_targets[-1])
        for key in fields:
            fields[key].append(estimates[key])
        seconds.append(elapsed)
        if position == 1 or position % progress_every == 0 or position == len(rows):
            mse = np.mean(np.square(estimates["direct"] - all_targets[-1]))
            print(
                f"[{position:3d}/{len(rows)}] id={index:4d} "
                f"direct={mse:.3e} seconds={elapsed:.3f}",
                flush=True,
            )
    return TerminalData(
        indices=np.asarray(indices, dtype=np.int64),
        names=names,
        target=np.stack(target),
        seconds=seconds,
        **{key: np.stack(value) for key, value in fields.items()},
    )


def correction_components(data: TerminalData) -> np.ndarray:
    """Orthogonalized direct-to-marginal corrections."""
    return np.stack(
        (
            data.gaussian - data.direct,
            data.edgeworth3 - data.gaussian,
            data.edgeworth4 - data.edgeworth3,
        ),
        axis=-1,
    )


def gate_basis(data: TerminalData) -> np.ndarray:
    """Bounded observable features used to vary marginal shrinkage."""
    t = np.clip(data.t, -8.0, 8.0)
    skew = np.clip(data.skew, -6.0, 6.0)
    kurtosis = np.clip(data.kurtosis, -20.0, 30.0)
    return np.stack(
        (
            np.ones_like(t),
            np.tanh(t / 2.0),
            np.tanh(np.abs(t) / 2.0),
            np.tanh(skew),
            np.tanh(np.abs(skew)),
            np.tanh(kurtosis / 3.0),
            np.tanh(np.abs(kurtosis) / 3.0),
            np.tanh(t * skew / 2.0),
            np.tanh((np.square(t) - 1.0) * kurtosis / 6.0),
        ),
        axis=-1,
    )


def solve_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Scale-normalized ridge without a free additive intercept."""
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


def flattened(data: TerminalData, mask: np.ndarray, value: np.ndarray) -> np.ndarray:
    return value[mask].reshape(-1, *value.shape[2:])


def global_design(data: TerminalData) -> np.ndarray:
    return correction_components(data)


def gated_design(data: TerminalData, marginal: str) -> np.ndarray:
    delta = getattr(data, marginal) - data.direct
    return delta[..., None] * gate_basis(data)


def predict_global(data: TerminalData, coef: np.ndarray) -> np.ndarray:
    correction = global_design(data) @ coef
    return np.maximum(data.direct + correction, 0.0)


def predict_gated(
    data: TerminalData,
    marginal: str,
    coef: np.ndarray,
) -> np.ndarray:
    # A gate is intentionally bounded: the frozen rule may choose direct or
    # the smooth marginal, but cannot extrapolate beyond both on a new MLP.
    gate = np.clip(gate_basis(data) @ coef, 0.0, 1.0)
    smooth = getattr(data, marginal)
    return data.direct + gate * (smooth - data.direct)


def cv_ridge(
    data: TerminalData,
    kind: str,
    marginal: str | None,
    ridges: list[float],
) -> tuple[float, float]:
    """Choose ridge by five-fold, whole-MLP cross-validation."""
    folds = data.indices % 5
    residual = data.target - data.direct
    design = global_design(data) if kind == "global" else gated_design(data, marginal)
    scores = []
    for ridge in ridges:
        squared_error = 0.0
        count = 0
        for fold in range(5):
            fit_mask = folds != fold
            hold_mask = folds == fold
            x_fit = flattened(data, fit_mask, design)
            y_fit = residual[fit_mask].reshape(-1)
            coef = solve_ridge(x_fit, y_fit, ridge)
            if kind == "global":
                prediction = predict_global(subset(data, hold_mask), coef)
            else:
                prediction = predict_gated(
                    subset(data, hold_mask), marginal, coef
                )
            squared_error += float(
                np.sum(np.square(prediction - data.target[hold_mask]))
            )
            count += prediction.size
        scores.append(squared_error / count)
    best = int(np.argmin(scores))
    return ridges[best], scores[best]


def subset(data: TerminalData, mask: np.ndarray) -> TerminalData:
    return TerminalData(
        indices=data.indices[mask],
        names=[name for name, keep in zip(data.names, mask, strict=True) if keep],
        target=data.target[mask],
        direct=data.direct[mask],
        gaussian=data.gaussian[mask],
        edgeworth3=data.edgeworth3[mask],
        edgeworth4=data.edgeworth4[mask],
        t=data.t[mask],
        skew=data.skew[mask],
        kurtosis=data.kurtosis[mask],
        seconds=[value for value, keep in zip(data.seconds, mask, strict=True) if keep],
    )


def fit_models(data: TerminalData, ridges: list[float]) -> list[FrozenModel]:
    residual = (data.target - data.direct).reshape(-1)
    models = []

    ridge, cv_mse = cv_ridge(data, "global", None, ridges)
    x = global_design(data).reshape(-1, 3)
    models.append(
        FrozenModel(
            name="global_ridge",
            kind="global",
            ridge=ridge,
            coef=solve_ridge(x, residual, ridge),
            marginal=None,
            cv_mse=cv_mse,
        )
    )

    for marginal in ("gaussian", "edgeworth3", "edgeworth4"):
        ridge, cv_mse = cv_ridge(data, "gated", marginal, ridges)
        x = gated_design(data, marginal).reshape(-1, gate_basis(data).shape[-1])
        models.append(
            FrozenModel(
                name=f"gated_{marginal}",
                kind="gated",
                ridge=ridge,
                coef=solve_ridge(x, residual, ridge),
                marginal=marginal,
                cv_mse=cv_mse,
            )
        )
    return models


def model_prediction(data: TerminalData, model: FrozenModel) -> np.ndarray:
    if model.kind == "global":
        return predict_global(data, model.coef)
    assert model.marginal is not None
    return predict_gated(data, model.marginal, model.coef)


def metrics(
    prediction: np.ndarray,
    data: TerminalData,
    baseline: np.ndarray | None = None,
) -> dict[str, float | int]:
    per_mlp = np.mean(np.square(prediction - data.target), axis=1)
    result: dict[str, float | int] = {
        "mse": float(np.mean(per_mlp)),
        "median_mlp_mse": float(np.median(per_mlp)),
        "p90_mlp_mse": float(np.quantile(per_mlp, 0.9)),
        "max_mlp_mse": float(np.max(per_mlp)),
        "max_mlp_id": int(data.indices[int(np.argmax(per_mlp))]),
    }
    if baseline is not None:
        baseline_per_mlp = np.mean(np.square(baseline - data.target), axis=1)
        result.update(
            {
                "gain_over_direct": float(
                    np.mean(baseline_per_mlp) / np.mean(per_mlp)
                ),
                "fraction_mlps_improved": float(np.mean(per_mlp < baseline_per_mlp)),
                "median_per_mlp_gain": float(
                    np.median(
                        baseline_per_mlp / np.maximum(per_mlp, 1e-30)
                    )
                ),
            }
        )
    return result


def summarize_estimators(data: TerminalData) -> dict[str, dict[str, float | int]]:
    return {
        name: metrics(getattr(data, name), data, data.direct if name != "direct" else None)
        for name in ("direct", "gaussian", "edgeworth3", "edgeworth4")
    }


def feature_summary(data: TerminalData) -> dict[str, list[float]]:
    return {
        name: [
            float(value)
            for value in np.quantile(
                getattr(data, name), (0.0, 0.01, 0.5, 0.99, 1.0)
            )
        ]
        for name in ("t", "skew", "kurtosis")
    }


def outliers(
    data: TerminalData,
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


def baseline_reproduction(
    data: TerminalData,
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


def model_dict(
    model: FrozenModel,
    train: TerminalData,
    test: TerminalData,
) -> dict[str, object]:
    return {
        "kind": model.kind,
        "marginal": model.marginal,
        "ridge": model.ridge,
        "whole_mlp_cv_mse": model.cv_mse,
        "coef": model.coef.tolist(),
        "train": metrics(model_prediction(train, model), train, train.direct),
        "test": metrics(model_prediction(test, model), test, test.direct),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-data", type=Path, default=DEFAULT_MINI)
    parser.add_argument("--full-data", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--samples", type=int, default=32768)
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full-count", type=int, default=50)
    parser.add_argument("--validation-threshold", type=float, default=0.15)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    indices = list(range(100))
    width = 256
    input_blocks = precompute_design(
        "sobol",
        width,
        args.samples,
        args.seed,
        antithetic=True,
        sphere=True,
        chunk=args.chunk,
    )
    mini = collect(args.mini_data, indices, input_blocks, args.progress_every)
    reproduction = baseline_reproduction(mini, args.baseline)
    if reproduction["max_abs_vector_difference"] > 1e-10:
        raise AssertionError(
            f"strict direct baseline did not reproduce: {reproduction}"
        )

    train = subset(mini, mini.indices < 50)
    test = subset(mini, mini.indices >= 50)
    ridges = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    models = fit_models(train, ridges)
    # Model-family selection is frozen using only whole-MLP CV on IDs 0--49.
    selected = min(models, key=lambda model: model.cv_mse)
    selected_test_prediction = model_prediction(test, selected)
    selected_test_metrics = metrics(
        selected_test_prediction, test, test.direct
    )
    test_gain = float(selected_test_metrics["gain_over_direct"])

    result: dict[str, object] = {
        "protocol": {
            "samples": args.samples,
            "design": "scrambled_sobol_antithetic_sphere",
            "seed": args.seed,
            "reuse_inputs": True,
            "exact_radial_moments": True,
            "train_ids": [0, 49],
            "test_ids": [50, 99],
            "selection": "five-fold whole-MLP CV within train IDs only",
        },
        "baseline_reproduction": reproduction,
        "feature_quantiles_min_p01_median_p99_max": {
            "train": feature_summary(train),
            "test": feature_summary(test),
        },
        "base_estimators": {
            "train": summarize_estimators(train),
            "test": summarize_estimators(test),
        },
        "learned_models": {
            model.name: model_dict(model, train, test) for model in models
        },
        "selected_model": selected.name,
        "selected_test_gain": test_gain,
        "selected_test_outliers_by_direct_mse": outliers(
            test, selected_test_prediction
        ),
        "full_validation": None,
    }

    if test_gain > 1.0 + args.validation_threshold:
        full_indices = list(range(args.full_count))
        full = collect(
            args.full_data,
            full_indices,
            input_blocks,
            args.progress_every,
        )
        full_prediction = model_prediction(full, selected)
        result["full_validation"] = {
            "ids": [0, args.full_count - 1],
            "disjoint_from_mini": True,
            "base_estimators": summarize_estimators(full),
            "selected_model": selected.name,
            "selected_metrics": metrics(
                full_prediction, full, full.direct
            ),
            "outliers_by_direct_mse": outliers(full, full_prediction),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print("\nRESULT_SUMMARY")
    print(
        json.dumps(
            {
                "base_test": result["base_estimators"]["test"],
                "selected_model": selected.name,
                "selected_cv_mse": selected.cv_mse,
                "selected_test": selected_test_metrics,
                "full_validation": result["full_validation"],
                "out": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
