"""Learn a transferable layerwise closure from the official public trajectories.

The key reparameterisation is a one-dimensional "implied Gaussian sigma".
For a pre-activation with known mean ``mu`` and true post-ReLU mean ``y``,
Jensen's inequality gives ``y >= relu(mu)``.  Therefore there is a unique
``sigma_eff >= 0`` such that

    y = mu * Phi(mu / sigma_eff) + sigma_eff * phi(mu / sigma_eff).

This effective sigma is only a representation of the first moment: it absorbs
all covariance and higher-cumulant errors into one positive scalar.  It is not
claimed to equal the true standard deviation.

This script is deliberately a small, dependency-light prototype:

* whole MLPs, never neurons, define the train/validation/test split;
* exact Gaussian propagation supplies the baseline covariance trajectory;
* a shared random-feature ridge model predicts log(sigma_eff / sigma_GP);
* the learned closure is evaluated by a free rollout, not teacher forcing;
* an optional tiny Monte-Carlo pilot supplies noisy per-neuron shape features.

The public labels are used only during offline fitting.  Every model input used
at evaluation time is computed from the weights, the Gaussian baseline, or the
explicitly costed pilot.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from whest import gaussmath as gm  # noqa: E402
from whest.estimators import gauss_prop  # noqa: E402
from whest.nets import MLP  # noqa: E402


EPS = 1e-12


@dataclass
class Record:
    mlp_id: int
    seed: int
    split: str
    truth: np.ndarray
    gp_y: np.ndarray
    gp_mu: np.ndarray
    gp_sigma: np.ndarray
    teacher_mu: np.ndarray
    target_log_ratio: np.ndarray
    target_sigma: np.ndarray
    structural: np.ndarray
    pilot: np.ndarray | None


@dataclass
class RFModel:
    mean: np.ndarray
    scale: np.ndarray
    projection: np.ndarray
    bias: np.ndarray
    coef: np.ndarray
    include_pilot: bool
    calibration: np.ndarray | None = None


def _split(mlp_id: int) -> str:
    """A deterministic 60/20/20 split by complete MLP."""
    bucket = mlp_id % 5
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "validation"
    return "train"


def _train_validation_split(mlp_id: int) -> str:
    """An 80/20 split used when an independent external test set exists."""
    return "validation" if mlp_id % 5 == 1 else "train"


def implied_sigma(mu: np.ndarray, y: np.ndarray, initial: np.ndarray) -> np.ndarray:
    """Invert the Gaussian ReLU mean in sigma by vectorised bisection."""
    mu = np.asarray(mu, dtype=np.float64)
    # The float32 baked means can violate Jensen at the last few ulps.
    y = np.maximum(np.asarray(y, dtype=np.float64), np.maximum(mu, 0.0))
    initial = np.maximum(np.asarray(initial, dtype=np.float64), 1e-10)

    lo = np.maximum(initial * 1e-7, 1e-12)
    hi = np.maximum(initial * 4.0, np.abs(mu) + gm.SQRT2PI * y + 1e-6)
    for _ in range(8):
        need = gm.relu_mean(mu, hi) < y
        hi = np.where(need, hi * 2.0, hi)

    for _ in range(44):
        mid = 0.5 * (lo + hi)
        low_price = gm.relu_mean(mu, mid) < y
        lo = np.where(low_price, mid, lo)
        hi = np.where(low_price, hi, mid)
    return 0.5 * (lo + hi)


def pilot_moments(
    weights: np.ndarray,
    n_samples: int,
    seed: int,
) -> dict[str, np.ndarray] | None:
    """Return full-pilot and split-half pre/post-activation statistics."""
    if n_samples <= 0:
        return None
    if n_samples % 2:
        raise ValueError("pilot sample count must be even")

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, weights.shape[-1])).astype(np.float32)
    halves = (slice(0, n_samples // 2), slice(n_samples // 2, n_samples))
    depth, width = weights.shape[:2]

    names = ("mh", "sd", "k3", "k4", "p", "ma")
    out = {name: np.zeros((3, depth, width), dtype=np.float64) for name in names}
    a = x
    for li, weight in enumerate(weights):
        h = a @ weight.T
        a = np.maximum(h, 0.0)
        for gi, sl in enumerate((slice(None),) + halves):
            hs = h[sl].astype(np.float64)
            aa = a[sl].astype(np.float64)
            mean = hs.mean(0)
            centered = hs - mean
            var = np.mean(centered * centered, axis=0)
            sd = np.sqrt(np.maximum(var, 1e-30))
            out["mh"][gi, li] = mean
            out["sd"][gi, li] = sd
            out["k3"][gi, li] = np.mean(centered**3, axis=0) / sd**3
            out["k4"][gi, li] = np.mean(centered**4, axis=0) / sd**4 - 3.0
            out["p"][gi, li] = np.mean(hs > 0.0, axis=0)
            out["ma"][gi, li] = aa.mean(0)
    return out


def structural_features(
    weights: np.ndarray,
    stats: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Cheap permutation-equivariant row and covariance summaries."""
    depth, width = weights.shape[:2]
    out = np.zeros((depth, width, 13), dtype=np.float64)
    for li, (weight, (_, covariance)) in enumerate(zip(weights, stats, strict=True)):
        row2 = np.sum(weight.astype(np.float64) ** 2, axis=1)
        row_norm = np.sqrt(np.maximum(row2, EPS))
        sd = np.sqrt(np.maximum(np.diag(covariance), EPS))
        correlation = covariance / np.outer(sd, sd)
        correlation = np.clip(correlation, -1.0, 1.0)
        trace = float(np.trace(covariance))
        eff_rank = trace * trace / max(float(np.sum(covariance * covariance)), EPS)

        layer = li / max(depth - 1, 1)
        out[li, :, 0] = layer
        out[li, :, 1] = layer * layer
        out[li, :, 2] = row_norm / np.sqrt(2.0)
        out[li, :, 3] = np.sum(weight, axis=1) / np.sqrt(2.0)
        out[li, :, 4] = (
            np.sum(weight.astype(np.float64) ** 3, axis=1)
            / np.maximum(row_norm**3, EPS)
            * np.sqrt(width)
        )
        out[li, :, 5] = (
            np.sum(weight.astype(np.float64) ** 4, axis=1)
            / np.maximum(row2**2, EPS)
            * width
            / 3.0
        )
        out[li, :, 6] = np.mean(weight > 0.0, axis=1) - 0.5
        out[li, :, 7] = (np.sum(correlation, axis=1) - 1.0) / np.sqrt(width)
        out[li, :, 8] = (np.sum(np.abs(correlation), axis=1) - 1.0) / np.sqrt(width)
        out[li, :, 9] = (np.sum(correlation * correlation, axis=1) - 1.0) / width
        out[li, :, 10] = np.log(max(eff_rank, 1e-6) / width)
        out[li, :, 11] = np.mean(np.abs(correlation))
        out[li, :, 12] = np.std(sd) / max(float(np.mean(sd)), EPS)
    return out


def pack_pilot(
    raw: dict[str, np.ndarray] | None,
    gp_sigma: np.ndarray,
) -> np.ndarray | None:
    """Pack raw pilot quantities that do not depend on the rollout mean."""
    if raw is None:
        return None
    depth, width = gp_sigma.shape
    packed = np.zeros((depth, width, 12), dtype=np.float64)
    packed[:, :, 0] = raw["mh"][0]
    packed[:, :, 1] = np.log(np.maximum(raw["sd"][0], 1e-10))
    packed[:, :, 2] = np.clip(raw["k3"][0], -4.0, 4.0)
    packed[:, :, 3] = np.clip(raw["k4"][0], -6.0, 12.0)
    packed[:, :, 4] = raw["p"][0]
    packed[:, :, 5] = raw["ma"][0]
    packed[:, :, 6] = raw["mh"][1] - raw["mh"][2]
    packed[:, :, 7] = np.log(np.maximum(raw["sd"][1], 1e-10)) - np.log(
        np.maximum(raw["sd"][2], 1e-10)
    )
    packed[:, :, 8] = raw["ma"][1] - raw["ma"][2]

    sigma_full = implied_sigma(raw["mh"][0], raw["ma"][0], gp_sigma)
    sigma_a = implied_sigma(raw["mh"][1], raw["ma"][1], gp_sigma)
    sigma_b = implied_sigma(raw["mh"][2], raw["ma"][2], gp_sigma)
    log_full = np.log(
        np.maximum(sigma_full, 1e-12) / np.maximum(gp_sigma, 1e-12)
    )
    log_a = np.log(np.maximum(sigma_a, 1e-12) / np.maximum(gp_sigma, 1e-12))
    log_b = np.log(np.maximum(sigma_b, 1e-12) / np.maximum(gp_sigma, 1e-12))
    packed[:, :, 9] = np.clip(log_full, -4.0, 4.0)
    packed[:, :, 10] = np.clip(0.5 * (log_a + log_b), -4.0, 4.0)
    packed[:, :, 11] = np.clip(log_a - log_b, -6.0, 6.0)
    return packed


def dynamic_features(
    record: Record,
    layer: int,
    mu: np.ndarray,
    previous_mean: np.ndarray,
    include_pilot: bool,
) -> np.ndarray:
    """Features for one layer at either a teacher-forced or rolled-out state."""
    sigma = record.gp_sigma[layer]
    t = np.clip(mu / sigma, -7.0, 7.0)
    gp_t = np.clip(record.gp_mu[layer] / sigma, -7.0, 7.0)
    base = gm.relu_mean(mu, sigma)

    prev_avg = float(np.mean(previous_mean)) if previous_mean.size else 0.0
    prev_sd = float(np.std(previous_mean)) if previous_mean.size else 0.0
    mu_avg = float(np.mean(mu))
    mu_sd = float(np.std(mu))
    dynamic = np.column_stack(
        [
            t,
            np.abs(t),
            np.minimum(t * t, 25.0),
            gm.phi(t),
            gm.Phi(t),
            t - gp_t,
            np.log(np.maximum(sigma, 1e-12)),
            np.full_like(t, prev_avg),
            np.full_like(t, prev_sd),
            np.full_like(t, mu_avg),
            np.full_like(t, mu_sd),
            base / sigma,
        ]
    )
    pieces = [dynamic, record.structural[layer]]

    if include_pilot:
        if record.pilot is None:
            raise ValueError("pilot features requested but record has no pilot")
        pilot = record.pilot[layer]
        pilot_dynamic = np.column_stack(
            [
                (pilot[:, 0] - mu) / sigma,
                pilot[:, 1] - np.log(np.maximum(sigma, 1e-12)),
                pilot[:, 2],
                pilot[:, 3],
                pilot[:, 4] - gm.Phi(t),
                (pilot[:, 5] - base) / sigma,
                pilot[:, 6] / sigma,
                pilot[:, 7],
                pilot[:, 8] / sigma,
                pilot[:, 9],
                pilot[:, 10],
                np.abs(pilot[:, 11]),
            ]
        )
        pieces.append(pilot_dynamic)
    return np.concatenate(pieces, axis=1)


def teacher_matrix(record: Record, include_pilot: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten layers 2..L into model inputs, targets, and price-space weights."""
    xs, ys, weights = [], [], []
    depth = record.truth.shape[0]
    for li in range(1, depth):
        x = dynamic_features(record, li, record.teacher_mu[li], record.truth[li - 1], include_pilot)
        t = record.teacher_mu[li] / record.gp_sigma[li]
        # d price / d log(sigma), evaluated at the GP scale.
        vega_log = record.gp_sigma[li] * gm.phi(t)
        # Later local defects matter more to the final layer.  This smooth
        # approximation avoids hard-coding one seed's exact Jacobian profile.
        sensitivity = 0.05 + 0.95 * (li / (depth - 1)) ** 1.35
        w = sensitivity * sensitivity * (vega_log * vega_log + 1e-8)
        xs.append(x)
        ys.append(np.clip(record.target_log_ratio[li], -3.0, 3.0))
        weights.append(w)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)


def hidden_features(z: np.ndarray, projection: np.ndarray, bias: np.ndarray) -> np.ndarray:
    nonlinear = np.tanh(z @ projection + bias)
    return np.concatenate([np.ones((len(z), 1)), z, nonlinear], axis=1)


def feature_normalisation(records: list[Record], include_pilot: bool) -> tuple[np.ndarray, np.ndarray]:
    total = None
    total2 = None
    count = 0
    for record in records:
        x, _, _ = teacher_matrix(record, include_pilot)
        if total is None:
            total = np.zeros(x.shape[1])
            total2 = np.zeros(x.shape[1])
        total += x.sum(0)
        total2 += np.sum(x * x, axis=0)
        count += len(x)
    assert total is not None and total2 is not None
    mean = total / count
    var = np.maximum(total2 / count - mean * mean, 1e-8)
    return mean, np.sqrt(var)


def fit_models(
    records: list[Record],
    include_pilot: bool,
    random_features: int,
    lambdas: list[float],
    seed: int,
) -> list[RFModel]:
    mean, scale = feature_normalisation(records, include_pilot)
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((len(mean), random_features)) / np.sqrt(len(mean))
    bias = rng.uniform(-1.5, 1.5, size=random_features)
    dim = 1 + len(mean) + random_features
    gram = np.zeros((dim, dim), dtype=np.float64)
    rhs = np.zeros(dim, dtype=np.float64)
    weight_sum = 0.0

    for record in records:
        x, target, weight = teacher_matrix(record, include_pilot)
        z = np.clip((x - mean) / scale, -8.0, 8.0)
        h = hidden_features(z, projection, bias)
        gram += (h.T * weight) @ h
        rhs += h.T @ (weight * target)
        weight_sum += float(np.sum(weight))

    gram /= weight_sum
    rhs /= weight_sum
    penalty = np.eye(dim)
    penalty[0, 0] = 0.0
    models = []
    for ridge in lambdas:
        coef = np.linalg.solve(gram + ridge * penalty, rhs)
        models.append(RFModel(mean, scale, projection, bias, coef, include_pilot))
    return models


def predict_log_ratio(model: RFModel, x: np.ndarray) -> np.ndarray:
    z = np.clip((x - model.mean) / model.scale, -8.0, 8.0)
    h = hidden_features(z, model.projection, model.bias)
    return np.clip(h @ model.coef, -1.5, 1.5)


def rollout(record: Record, weights: np.ndarray, model: RFModel) -> np.ndarray:
    depth, width = record.truth.shape
    out = np.zeros((depth, width), dtype=np.float64)
    # Layer 1 is analytically exact.
    out[0] = np.linalg.norm(weights[0].astype(np.float64), axis=1) / gm.SQRT2PI
    for li in range(1, depth):
        mu = weights[li].astype(np.float64) @ out[li - 1]
        x = dynamic_features(record, li, mu, out[li - 1], model.include_pilot)
        log_ratio = predict_log_ratio(model, x)
        sigma_eff = record.gp_sigma[li] * np.exp(log_ratio)
        out[li] = gm.relu_mean(mu, sigma_eff)
    return out


def defect_rollout(
    record: Record,
    weights: np.ndarray,
    model: RFModel,
    linearised: bool = False,
) -> np.ndarray:
    """Propagate signed, baseline-evaluated local closure defects.

    A volatility-only closure cannot lower ``ReLU(mu)`` when an upstream mean
    error makes ``mu`` too positive.  Defect correction avoids that exposure
    failure: infer each non-Gaussian local defect on the fixed GP trajectory,
    then propagate the signed innovations through either the nonlinear map or
    its Jacobian.  This is the standard deferred-correction form used for
    learned numerical closures.
    """
    depth, width = record.truth.shape
    local_defect = np.zeros((depth, width), dtype=np.float64)
    for li in range(1, depth):
        mu0 = record.gp_mu[li]
        x = dynamic_features(record, li, mu0, record.gp_y[li - 1], model.include_pilot)
        log_ratio = predict_log_ratio(model, x)
        corrected_local = gm.relu_mean(mu0, record.gp_sigma[li] * np.exp(log_ratio))
        local_defect[li] = corrected_local - record.gp_y[li]

    out = np.zeros((depth, width), dtype=np.float64)
    out[0] = np.linalg.norm(weights[0].astype(np.float64), axis=1) / gm.SQRT2PI
    if linearised:
        error = out[0] - record.gp_y[0]
        for li in range(1, depth):
            beta = gm.Phi(record.gp_mu[li] / record.gp_sigma[li])
            scale = 1.0 if model.calibration is None else model.calibration[li]
            error = beta * (weights[li].astype(np.float64) @ error) + scale * local_defect[li]
            out[li] = record.gp_y[li] + error
    else:
        for li in range(1, depth):
            mu = weights[li].astype(np.float64) @ out[li - 1]
            scale = 1.0 if model.calibration is None else model.calibration[li]
            out[li] = gm.relu_mean(mu, record.gp_sigma[li]) + scale * local_defect[li]
    return out


def teacher_predictions(record: Record, model: RFModel) -> np.ndarray:
    depth, width = record.truth.shape
    out = np.zeros((depth, width), dtype=np.float64)
    out[0] = record.gp_y[0]
    for li in range(1, depth):
        mu = record.teacher_mu[li]
        x = dynamic_features(record, li, mu, record.truth[li - 1], model.include_pilot)
        log_ratio = predict_log_ratio(model, x)
        out[li] = gm.relu_mean(mu, record.gp_sigma[li] * np.exp(log_ratio))
    return out


def local_defects(record: Record, model: RFModel) -> np.ndarray:
    """Predicted local innovations evaluated on the fixed GP trajectory."""
    depth, width = record.truth.shape
    defect = np.zeros((depth, width), dtype=np.float64)
    for li in range(1, depth):
        mu0 = record.gp_mu[li]
        x = dynamic_features(record, li, mu0, record.gp_y[li - 1], model.include_pilot)
        log_ratio = predict_log_ratio(model, x)
        defect[li] = (
            gm.relu_mean(mu0, record.gp_sigma[li] * np.exp(log_ratio)) - record.gp_y[li]
        )
    return defect


def defect_response_design(
    record: Record,
    weights: np.ndarray,
    model: RFModel,
) -> np.ndarray:
    """Final-layer response to independently scaling each local defect."""
    depth, width = record.truth.shape
    defects = local_defects(record, model)
    response = np.zeros((width, depth - 1), dtype=np.float64)
    for li in range(1, depth):
        beta = gm.Phi(record.gp_mu[li] / record.gp_sigma[li])
        response = beta[:, None] * (weights[li].astype(np.float64) @ response)
        response[:, li - 1] += defects[li]
    return response


def fit_defect_calibration(
    records: list[Record],
    dataset,
    index_by_id: dict[int, int],
    model: RFModel,
    ridge: float = 1e-6,
) -> None:
    """Fit 31 transferable layer scales against final error on training MLPs."""
    depth = records[0].truth.shape[0]
    gram = np.zeros((depth - 1, depth - 1), dtype=np.float64)
    rhs = np.zeros(depth - 1, dtype=np.float64)
    count = 0
    for record in records:
        weights = dataset[index_by_id[record.mlp_id]]["weights"].astype(np.float32).swapaxes(-1, -2)
        design = defect_response_design(record, weights, model)
        target = record.truth[-1] - record.gp_y[-1]
        gram += design.T @ design
        rhs += design.T @ target
        count += len(target)
    gram /= count
    rhs /= count
    # The implied-sigma model already says the natural scale is one; shrink
    # toward that physical prior rather than toward deleting the correction.
    alpha = np.linalg.solve(gram + ridge * np.eye(depth - 1), rhs + ridge)
    calibration = np.ones(depth, dtype=np.float64)
    calibration[1:] = np.clip(alpha, -5.0, 5.0)
    model.calibration = calibration


def evaluate(
    records: list[Record],
    dataset,
    index_by_id: dict[int, int],
    model: RFModel,
) -> dict[str, float | list[float]]:
    gp_mses = []
    teacher_mses = []
    rollout_mses = []
    defect_mses = []
    linear_defect_mses = []
    oracle_mses = []
    for record in records:
        # The HuggingFace array is serialized in the benchmark's sample-major
        # convention (input, output); our local kernels use (output, input).
        weights = dataset[index_by_id[record.mlp_id]]["weights"].astype(np.float32).swapaxes(-1, -2)
        teacher = teacher_predictions(record, model)
        rolled = rollout(record, weights, model)
        defect = defect_rollout(record, weights, model, linearised=False)
        linear_defect = defect_rollout(record, weights, model, linearised=True)
        oracle = gm.relu_mean(record.teacher_mu, record.target_sigma)
        gp_mses.append(float(np.mean((record.gp_y[-1] - record.truth[-1]) ** 2)))
        teacher_mses.append(float(np.mean((teacher[-1] - record.truth[-1]) ** 2)))
        rollout_mses.append(float(np.mean((rolled[-1] - record.truth[-1]) ** 2)))
        defect_mses.append(float(np.mean((defect[-1] - record.truth[-1]) ** 2)))
        linear_defect_mses.append(float(np.mean((linear_defect[-1] - record.truth[-1]) ** 2)))
        oracle_mses.append(float(np.mean((oracle[-1] - record.truth[-1]) ** 2)))
    return {
        "gp_mse": float(np.mean(gp_mses)),
        "teacher_forced_mse": float(np.mean(teacher_mses)),
        "rollout_mse": float(np.mean(rollout_mses)),
        "defect_rollout_mse": float(np.mean(defect_mses)),
        "linear_defect_rollout_mse": float(np.mean(linear_defect_mses)),
        "oracle_implied_sigma_mse": float(np.mean(oracle_mses)),
        "rollout_per_mlp": rollout_mses,
        "defect_per_mlp": defect_mses,
    }


def make_record(row: dict, pilot_samples: int, nodes: int) -> Record:
    # Public parquet stores each linear map as (input, output), whereas the
    # local whest helpers use h @ W.T with W=(output, input).
    weights = row["weights"].astype(np.float32).swapaxes(-1, -2)
    truth = row["all_layer_means"].astype(np.float64)
    mlp = MLP(tuple(weights), weights.shape[1], weights.shape[0], int(row["mlp_seed"]))
    gp_y, _, stats = gauss_prop(mlp, mode="exact", nodes=nodes, return_stats=True)
    gp_mu = np.stack([mean for mean, _ in stats])
    gp_sigma = np.stack(
        [np.sqrt(np.maximum(np.diag(covariance), 1e-30)) for _, covariance in stats]
    )

    teacher_mu = np.zeros_like(truth)
    for li in range(1, len(weights)):
        teacher_mu[li] = weights[li].astype(np.float64) @ truth[li - 1]
    target_sigma = implied_sigma(teacher_mu, truth, gp_sigma)
    target_log_ratio = np.log(np.maximum(target_sigma, 1e-12) / gp_sigma)

    raw_pilot = pilot_moments(
        weights,
        pilot_samples,
        seed=(int(row["mlp_id"]) * 1_000_003 + 91_337) % (2**32),
    )
    return Record(
        mlp_id=int(row["mlp_id"]),
        seed=int(row["mlp_seed"]),
        split=_split(int(row["mlp_id"])),
        truth=truth,
        gp_y=gp_y,
        gp_mu=gp_mu,
        gp_sigma=gp_sigma,
        teacher_mu=teacher_mu,
        target_log_ratio=target_log_ratio,
        target_sigma=target_sigma,
        structural=structural_features(weights, stats),
        pilot=pack_pilot(raw_pilot, gp_sigma),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(
            os.path.dirname(__file__), "..", "data", "official_phase1_mini", "data", "*.parquet"
        ),
    )
    parser.add_argument(
        "--external-test-data",
        default="",
        help=(
            "Optional independent parquet glob. When supplied, --data is split "
            "80/20 into train/validation and every external row is held out for test."
        ),
    )
    parser.add_argument("--pilot-samples", type=int, default=512)
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--random-features", type=int, default=128)
    parser.add_argument("--cache-dir", default="/tmp/arc-whest-hf-cache")
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("deterministic", "pilot"),
        default=["deterministic", "pilot"],
    )
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    files = sorted(glob.glob(args.data))
    if not files:
        raise FileNotFoundError(args.data)
    os.environ.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    dataset = load_dataset("parquet", data_files=files, split="train").with_format("numpy")
    ids = np.asarray(dataset["mlp_id"], dtype=np.int64)
    order = np.argsort(ids)
    # Keep dataset indexing explicit: parquet shard order is not guaranteed to be id order.
    index_by_id = {int(ids[i]): int(i) for i in range(len(ids))}

    records: list[Record] = []
    for i, raw_index in enumerate(order):
        row = dataset[int(raw_index)]
        record = make_record(row, args.pilot_samples, args.nodes)
        if args.external_test_data:
            record.split = _train_validation_split(record.mlp_id)
        records.append(record)
        if (i + 1) % args.progress_every == 0 or i == 0 or i + 1 == len(dataset):
            print(
                f"[fit {i + 1:4d}/{len(dataset)}] id={record.mlp_id:4d} "
                f"split={record.split:10s} "
                f"gp={np.mean((record.gp_y[-1] - record.truth[-1])**2):.3e}",
                flush=True,
            )

    by_split = {
        name: [record for record in records if record.split == name]
        for name in ("train", "validation")
    }
    test_dataset = dataset
    test_index_by_id = index_by_id
    if args.external_test_data:
        test_files = sorted(glob.glob(args.external_test_data))
        if not test_files:
            raise FileNotFoundError(args.external_test_data)
        test_dataset = load_dataset(
            "parquet", data_files=test_files, split="train"
        ).with_format("numpy")
        test_ids = np.asarray(test_dataset["mlp_id"], dtype=np.int64)
        test_order = np.argsort(test_ids)
        test_index_by_id = {
            int(test_ids[i]): int(i) for i in range(len(test_ids))
        }
        test_records = []
        for i, raw_index in enumerate(test_order):
            row = test_dataset[int(raw_index)]
            record = make_record(row, args.pilot_samples, args.nodes)
            record.split = "external_test"
            test_records.append(record)
            if (i + 1) % args.progress_every == 0 or i == 0 or i + 1 == len(test_dataset):
                print(
                    f"[test {i + 1:4d}/{len(test_dataset)}] id={record.mlp_id:4d} "
                    f"gp={np.mean((record.gp_y[-1] - record.truth[-1])**2):.3e}",
                    flush=True,
                )
    else:
        test_records = [record for record in records if record.split == "test"]

    lambdas = args.lambdas
    result: dict[str, object] = {
        "split": {
            **{name: [r.mlp_id for r in part] for name, part in by_split.items()},
            "test": [r.mlp_id for r in test_records],
        },
        "fit_data": args.data,
        "external_test_data": args.external_test_data or None,
        "pilot_samples": args.pilot_samples,
        "random_features": args.random_features,
        "variants": {},
    }

    for variant, include_pilot in (("deterministic", False), ("pilot", True)):
        if variant not in args.variants:
            continue
        if include_pilot and args.pilot_samples <= 0:
            continue
        print(f"\nFitting {variant} closure ...", flush=True)
        models = fit_models(
            by_split["train"],
            include_pilot=include_pilot,
            random_features=args.random_features,
            lambdas=lambdas,
            seed=20260727 + int(include_pilot),
        )
        # This second-stage fit is also train-only.  It calibrates how much of
        # each layer's predicted innovation should be exposed to the remainder
        # of the network, using complete training MLP trajectories.
        for model in models:
            fit_defect_calibration(
                by_split["train"],
                dataset,
                index_by_id,
                model,
                ridge=1e-6,
            )
        validation = [
            evaluate(by_split["validation"], dataset, index_by_id, model) for model in models
        ]
        # Select on the best deployable closure form, fixed before seeing test.
        best_index = int(np.argmin([metrics["defect_rollout_mse"] for metrics in validation]))
        best = models[best_index]
        test = evaluate(test_records, test_dataset, test_index_by_id, best)
        result["variants"][variant] = {
            "ridge": lambdas[best_index],
            "trajectory_calibration_ridge": 1e-6,
            "trajectory_calibration": best.calibration.tolist(),
            "validation": validation[best_index],
            "test": test,
            "validation_ladder": [
                {
                    "ridge": ridge,
                    "rollout_mse": metrics["rollout_mse"],
                    "defect_rollout_mse": metrics["defect_rollout_mse"],
                }
                for ridge, metrics in zip(lambdas, validation, strict=True)
            ],
        }
        print(
            f"{variant}: ridge={lambdas[best_index]:.1e} "
            f"val rollout={validation[best_index]['rollout_mse']:.4e} "
            f"test rollout={test['rollout_mse']:.4e} "
            f"test defect={test['defect_rollout_mse']:.4e} "
            f"test linear-defect={test['linear_defect_rollout_mse']:.4e} "
            f"test teacher={test['teacher_forced_mse']:.4e} "
            f"test GP={test['gp_mse']:.4e}",
            flush=True,
        )

    print("\nRESULT_JSON")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
