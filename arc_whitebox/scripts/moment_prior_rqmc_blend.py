"""Use deterministic moment chains as low-dimensional RQMC priors.

The strict RQMC prediction is the saved N=32,768 Sobol-antithetic-sphere
vector.  Candidate deterministic priors are:

* exact K2 / Gaussian moment propagation;
* exact factored K3-simple from ARC's reference implementation;
* the frozen rank-64, 256-column sketched K3-simple approximation.

Only mini IDs 0--49 fit coefficients or select a model.  Five-fold
whole-MLP CV compares a global scalar shrink, a two-coefficient decomposition
into per-MLP mean and output-centered corrections, and small combined ridges.
The frozen rules are then evaluated on IDs 50--99.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RQMC = ROOT / "results" / "sobol_vectors_n32768.json"
DEFAULT_K2 = ROOT / "results" / "k2seq_mini100.npz"
DEFAULT_K3 = ROOT / "results" / "kprop_priors_mini100.npz"
DEFAULT_OUT = ROOT / "results" / "moment_prior_rqmc_blend.json"

WIDTH = 256
DEPTH = 32
SAMPLES = 32768
CHALLENGE_BUDGET = 272_000_000_000
SIMPLE_SUM_CARRIER_COLUMNS = 405_504
FINAL_CARRIER_COLUMNS = 24_576
SKETCH256_ADDED_FLOPS = 8_906_604_544


@dataclass
class Data:
    indices: np.ndarray
    target: np.ndarray
    rqmc: np.ndarray
    priors: dict[str, np.ndarray]


@dataclass
class Rule:
    name: str
    priors: list[str]
    mode: str
    coef: np.ndarray
    cv_mse: float
    ridge: float


def load_data(rqmc_path: Path, k2_path: Path, k3_path: Path) -> tuple[Data, dict]:
    with rqmc_path.open() as handle:
        runs = json.load(handle)["runs"]
    indices = np.asarray([run["index"] for run in runs], dtype=np.int64)
    rqmc = np.stack(
        [np.asarray(run["final_prediction"], dtype=np.float64) for run in runs]
    )
    target = np.stack(
        [np.asarray(run["final_target"], dtype=np.float64) for run in runs]
    )

    k2 = np.load(k2_path)
    k3 = np.load(k3_path)
    if not np.array_equal(indices, k2["mlp_id"]):
        raise AssertionError("K2 IDs do not align with RQMC")
    if not np.array_equal(indices, k3["mlp_id"]):
        raise AssertionError("K3 IDs do not align with RQMC")
    if np.max(np.abs(target - k2["targets"][:, -1])) != 0.0:
        raise AssertionError("K2 targets do not match RQMC targets")
    if np.max(np.abs(target - k3["target"])) != 0.0:
        raise AssertionError("K3 targets do not match RQMC targets")

    data = Data(
        indices=indices,
        target=target,
        rqmc=rqmc,
        priors={
            "k2": k2["base"][:, -1].astype(np.float64),
            "k3_simple": k3["k3_simple"].astype(np.float64),
            "k3_sketch256": k3["k3_sketch256"].astype(np.float64),
        },
    )
    timing = {
        "k2_isolated_id36_seconds": 0.10821870784275234,
        "k3_simple_parallel_export_mean_seconds": float(
            np.mean(k3["k3_simple_seconds"])
        ),
        "k3_sketch256_parallel_export_mean_seconds": float(
            np.mean(k3["k3_sketch256_seconds"])
        ),
        "k3_simple_isolated_id36_seconds": 1.8700502079445869,
        "k3_sketch256_isolated_id36_seconds": 1.6016167500056326,
        "k3_augment_isolated_id36_seconds": 18.322075208183378,
    }
    return data, timing


def subset(data: Data, mask: np.ndarray) -> Data:
    return Data(
        indices=data.indices[mask],
        target=data.target[mask],
        rqmc=data.rqmc[mask],
        priors={name: value[mask] for name, value in data.priors.items()},
    )


def prior_components(data: Data, prior_names: list[str], mode: str) -> np.ndarray:
    components = []
    for name in prior_names:
        delta = data.priors[name] - data.rqmc
        if mode == "scalar":
            components.append(delta)
        elif mode == "mean_center":
            mean = np.broadcast_to(np.mean(delta, axis=1, keepdims=True), delta.shape)
            components.extend((mean, delta - mean))
        else:
            raise ValueError(mode)
    return np.stack(components, axis=-1)


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


def prediction(data: Data, rule: Rule) -> np.ndarray:
    design = prior_components(data, rule.priors, rule.mode)
    return np.maximum(data.rqmc + design @ rule.coef, 0.0)


def cv_score(
    data: Data,
    prior_names: list[str],
    mode: str,
    ridge: float,
) -> float:
    folds = data.indices % 5
    design = prior_components(data, prior_names, mode)
    residual = data.target - data.rqmc
    squared_error = 0.0
    count = 0
    for fold in range(5):
        fit = folds != fold
        hold = folds == fold
        x = design[fit].reshape(-1, design.shape[-1])
        y = residual[fit].reshape(-1)
        coef = solve_ridge(x, y, ridge)
        held_prediction = np.maximum(
            data.rqmc[hold] + design[hold] @ coef,
            0.0,
        )
        squared_error += float(
            np.sum(np.square(held_prediction - data.target[hold]))
        )
        count += held_prediction.size
    return squared_error / count


def fit_rule(
    data: Data,
    name: str,
    prior_names: list[str],
    mode: str,
    ridges: list[float],
) -> Rule:
    scores = [
        cv_score(data, prior_names, mode, ridge)
        for ridge in ridges
    ]
    best = int(np.argmin(scores))
    ridge = ridges[best]
    design = prior_components(data, prior_names, mode)
    coef = solve_ridge(
        design.reshape(-1, design.shape[-1]),
        (data.target - data.rqmc).reshape(-1),
        ridge,
    )
    return Rule(
        name=name,
        priors=prior_names,
        mode=mode,
        coef=coef,
        cv_mse=scores[best],
        ridge=ridge,
    )


def fit_rules(train: Data) -> list[Rule]:
    rules = []
    for prior in train.priors:
        rules.append(
            fit_rule(
                train,
                f"{prior}_scalar",
                [prior],
                "scalar",
                [0.0],
            )
        )
        rules.append(
            fit_rule(
                train,
                f"{prior}_mean_center",
                [prior],
                "mean_center",
                [0.0],
            )
        )

    ridge_grid = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    rules.append(
        fit_rule(
            train,
            "combined_k3_mean_center",
            ["k3_simple", "k3_sketch256"],
            "mean_center",
            ridge_grid,
        )
    )
    rules.append(
        fit_rule(
            train,
            "combined_all_mean_center",
            ["k2", "k3_simple", "k3_sketch256"],
            "mean_center",
            ridge_grid,
        )
    )
    return rules


def metrics(estimate: np.ndarray, data: Data) -> dict[str, float | int]:
    per_mlp = np.mean(np.square(estimate - data.target), axis=1)
    rqmc_per_mlp = np.mean(np.square(data.rqmc - data.target), axis=1)
    return {
        "mse": float(np.mean(per_mlp)),
        "gain_over_rqmc": float(np.mean(rqmc_per_mlp) / np.mean(per_mlp)),
        "median_mlp_mse": float(np.median(per_mlp)),
        "p90_mlp_mse": float(np.quantile(per_mlp, 0.9)),
        "max_mlp_mse": float(np.max(per_mlp)),
        "max_mlp_id": int(data.indices[int(np.argmax(per_mlp))]),
        "fraction_mlps_improved": float(np.mean(per_mlp < rqmc_per_mlp)),
        "median_per_mlp_gain": float(
            np.median(rqmc_per_mlp / np.maximum(per_mlp, 1e-30))
        ),
    }


def prior_metrics(data: Data, name: str) -> dict[str, float | int]:
    result = metrics(data.priors[name], data)
    rqmc_error = (data.rqmc - data.target).reshape(-1)
    prior_error = (data.priors[name] - data.target).reshape(-1)
    result["error_correlation_with_rqmc"] = float(
        np.corrcoef(rqmc_error, prior_error)[0, 1]
    )
    return result


def rule_dict(rule: Rule, train: Data, test: Data) -> dict[str, object]:
    return {
        "priors": rule.priors,
        "mode": rule.mode,
        "ridge": rule.ridge,
        "coef": rule.coef.tolist(),
        "whole_mlp_cv_mse": rule.cv_mse,
        "train": metrics(prediction(train, rule), train),
        "test": metrics(prediction(test, rule), test),
    }


def rule_cost(
    rule: Rule,
    costs: dict[str, object],
    test_metrics: dict[str, float | int],
    rqmc_test_mse: float,
) -> dict[str, float | int]:
    prior_flops = sum(
        int(costs[name]["prior_flops"])
        for name in set(rule.priors)
    )
    rqmc_flops = int(costs["rqmc"]["flops"])
    multiplier = (rqmc_flops + prior_flops) / rqmc_flops
    score_ratio = (
        float(test_metrics["mse"]) / rqmc_test_mse * multiplier
    )
    return {
        "combined_flops": rqmc_flops + prior_flops,
        "cost_multiplier_vs_rqmc": multiplier,
        "cost_adjusted_score_ratio_vs_rqmc": score_ratio,
        "cost_adjusted_gain_vs_rqmc": 1.0 / score_ratio,
    }


def compute_costs() -> dict[str, object]:
    rqmc_flops = 2 * SAMPLES * DEPTH * WIDTH**2
    k2_flops = 4 * DEPTH * WIDTH**3
    # Each CP carrier column contracts three factors through a dense W:
    # 3 * (2 n^2) FLOPs. Exclude the final carrier, which is not propagated.
    k3_factor_matmul_lower_bound = (
        6
        * WIDTH**2
        * (SIMPLE_SUM_CARRIER_COLUMNS - FINAL_CARRIER_COLUMNS)
    )
    k3_simple_lower_bound = k2_flops + k3_factor_matmul_lower_bound
    k3_sketch_estimate = k3_simple_lower_bound + SKETCH256_ADDED_FLOPS

    def record(prior_flops: int) -> dict[str, float | int]:
        combined = rqmc_flops + prior_flops
        return {
            "prior_flops": prior_flops,
            "prior_budget_fraction": prior_flops / CHALLENGE_BUDGET,
            "rqmc_plus_prior_flops": combined,
            "rqmc_plus_prior_budget_fraction": combined / CHALLENGE_BUDGET,
            "cost_multiplier_vs_rqmc": combined / rqmc_flops,
        }

    return {
        "convention": "multiply and add count as two FLOPs",
        "rqmc": {
            "flops": rqmc_flops,
            "budget_fraction": rqmc_flops / CHALLENGE_BUDGET,
        },
        "k2": {
            **record(k2_flops),
            "note": "dense covariance propagation estimate",
        },
        "k3_simple": {
            **record(k3_simple_lower_bound),
            "note": (
                "lower bound: K2 plus dense W contractions of all propagated "
                "CP factors; excludes nonlinear factor assembly"
            ),
        },
        "k3_sketch256": {
            **record(k3_sketch_estimate),
            "sketch_added_flops": SKETCH256_ADDED_FLOPS,
            "note": (
                "K3-simple lower bound plus the published sketch/eigh model; "
                "still excludes nonlinear factor assembly"
            ),
        },
        "challenge_budget": CHALLENGE_BUDGET,
    }


def outliers(
    data: Data,
    estimate: np.ndarray,
    count: int = 8,
) -> list[dict[str, float | int]]:
    rqmc_mse = np.mean(np.square(data.rqmc - data.target), axis=1)
    estimate_mse = np.mean(np.square(estimate - data.target), axis=1)
    order = np.argsort(rqmc_mse)[::-1][:count]
    return [
        {
            "id": int(data.indices[i]),
            "rqmc_mse": float(rqmc_mse[i]),
            "adjusted_mse": float(estimate_mse[i]),
            "gain": float(rqmc_mse[i] / max(estimate_mse[i], 1e-30)),
        }
        for i in order
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rqmc", type=Path, default=DEFAULT_RQMC)
    parser.add_argument("--k2", type=Path, default=DEFAULT_K2)
    parser.add_argument("--k3", type=Path, default=DEFAULT_K3)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    data, timing = load_data(args.rqmc, args.k2, args.k3)
    train = subset(data, data.indices < 50)
    test = subset(data, data.indices >= 50)
    rules = fit_rules(train)
    direct_cv_mse = float(np.mean(np.square(train.rqmc - train.target)))
    selected = min(rules, key=lambda rule: rule.cv_mse)
    if direct_cv_mse <= selected.cv_mse:
        selected_name = "rqmc"
        selected_test = test.rqmc
        selected_cv = direct_cv_mse
    else:
        selected_name = selected.name
        selected_test = prediction(test, selected)
        selected_cv = selected.cv_mse

    costs = compute_costs()
    selected_metrics = metrics(selected_test, test)
    rqmc_test_mse = float(np.mean(np.square(test.rqmc - test.target)))
    rules_payload = {}
    for rule in rules:
        payload = rule_dict(rule, train, test)
        payload["compute"] = rule_cost(
            rule,
            costs,
            payload["test"],
            rqmc_test_mse,
        )
        rules_payload[rule.name] = payload
    if selected_name == "rqmc":
        selected_compute = {
            "combined_flops": int(costs["rqmc"]["flops"]),
            "cost_multiplier_vs_rqmc": 1.0,
            "cost_adjusted_score_ratio_vs_rqmc": 1.0,
            "cost_adjusted_gain_vs_rqmc": 1.0,
        }
    else:
        selected_compute = rules_payload[selected_name]["compute"]
    result = {
        "protocol": {
            "train_ids": [0, 49],
            "test_ids": [50, 99],
            "selection": "five-fold whole-MLP CV within train IDs only",
            "rqmc": "N=32768 reused scrambled Sobol antithetic sphere, seed 0",
        },
        "raw": {
            "train": {
                "rqmc": metrics(train.rqmc, train),
                **{
                    name: prior_metrics(train, name)
                    for name in train.priors
                },
            },
            "test": {
                "rqmc": metrics(test.rqmc, test),
                **{
                    name: prior_metrics(test, name)
                    for name in test.priors
                },
            },
        },
        "rules": rules_payload,
        "selected_model": selected_name,
        "selected_cv_mse": selected_cv,
        "selected_test": selected_metrics,
        "selected_compute": selected_compute,
        "selected_test_outliers_by_rqmc_mse": outliers(
            test,
            selected_test,
        ),
        "timing": timing,
        "compute_cost": costs,
        "cost_conclusion": (
            "K2 is cheap enough to combine with strict RQMC but does not "
            "transfer. K3 priors provide only a few-percent raw MSE gain while "
            "roughly doubling the RQMC compute, so their cost-adjusted score is "
            "strictly worse."
        ),
        "augment_excluded": {
            "reason": "not affordable for strict 100-MLP coverage",
            "isolated_id36_seconds": timing["k3_augment_isolated_id36_seconds"],
            "projected_100_mlp_seconds": (
                100 * timing["k3_augment_isolated_id36_seconds"]
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    print(
        json.dumps(
            {
                "raw_test": result["raw"]["test"],
                "selected_model": selected_name,
                "selected_test": selected_metrics,
                "cost": costs,
                "out": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
