#!/usr/bin/env python3
"""Exact-Kerdock network-specific coreset diagnostic.

Reproduces the 20-network experiment in
NETWORK_SPECIFIC_KERNEL_CORESET_EXACT_KERDOCK_REPORT.md.

This is an offline research script, not a challenge-runtime estimator. It uses
NumPy and the existing kerdock_mub5_seed3.npz asset.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

WIDTH = 256
DEPTH = 32
KERDOCK_BASES = 128
ALL_BASES = 129
PAIRS_PER_BASIS = 256
TOTAL_PAIRS = ALL_BASES * PAIRS_PER_BASIS
TOTAL_ROWS = 2 * TOTAL_PAIRS
RADIUS = math.sqrt(2.0) * math.exp(
    math.lgamma((WIDTH + 1.0) / 2.0) - math.lgamma(WIDTH / 2.0)
)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def gen_weights(seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    scale = math.sqrt(2.0 / WIDTH)
    return [
        (rng.normal(size=(WIDTH, WIDTH)) * scale).astype(np.float32)
        for _ in range(DEPTH)
    ]


def fwht_axis_one(values: np.ndarray) -> np.ndarray:
    values = values.copy()
    span = 1
    while span < WIDTH:
        grouped = values.reshape(
            (KERDOCK_BASES, WIDTH // (2 * span), 2, span, values.shape[-1])
        )
        left = grouped[:, :, 0, :, :]
        right = grouped[:, :, 1, :, :]
        values = np.stack((left + right, left - right), axis=2).reshape(
            (KERDOCK_BASES, WIDTH, values.shape[-1])
        )
        span *= 2
    return values


def first_activation(
    first_weight: np.ndarray,
    chirps: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    effective_weight = rotation @ first_weight.astype(np.float32)
    weighted = chirps[:, :, None] * effective_weight[None, :, :]
    preactivation = fwht_axis_one(weighted) * (RADIUS / math.sqrt(WIDTH))
    kerdock_rows = np.stack((preactivation, -preactivation), axis=2).reshape(
        (-1, WIDTH)
    )
    coordinate_rows = np.stack(
        (RADIUS * effective_weight, -RADIUS * effective_weight), axis=1
    ).reshape((-1, WIDTH))
    return relu(np.concatenate((kerdock_rows, coordinate_rows), axis=0)).astype(
        np.float32
    )


def propagate_to_anchor(
    weights: list[np.ndarray],
    chirps: np.ndarray,
    rotation: np.ndarray,
    anchor_depth: int = 28,
) -> np.ndarray:
    h = first_activation(weights[0], chirps, rotation)
    for layer in range(1, anchor_depth):
        h = relu(h @ weights[layer])
    return h


def pair_average(rows: np.ndarray) -> np.ndarray:
    return rows.reshape((TOTAL_PAIRS, 2, rows.shape[1])).mean(
        axis=1, dtype=np.float64
    )


def pilot_rows(pairs_per_basis: int = 8) -> np.ndarray:
    rows: list[int] = []
    for basis in range(ALL_BASES):
        base = basis * 2 * WIDTH
        for pair in range(pairs_per_basis):
            rows.extend((base + 2 * pair, base + 2 * pair + 1))
    return np.asarray(rows, dtype=np.int32)


def pilot_affine_params(
    pilot_anchor: np.ndarray,
    tail_weights: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    probabilities: list[np.ndarray] = []
    intercepts: list[np.ndarray] = []
    h = pilot_anchor.astype(np.float64)
    for weight in tail_weights:
        z = h @ weight.astype(np.float64)
        a = relu(z)
        p = (z > 0).mean(axis=0).astype(np.float64)
        c = a.mean(axis=0) - p * z.mean(axis=0)
        probabilities.append(p)
        intercepts.append(c)
        h = a
    return probabilities, intercepts


def build_suffix_features(
    anchor: np.ndarray,
    tail_weights: list[np.ndarray],
    pilot_indices: np.ndarray,
    q_each: int,
) -> np.ndarray:
    weights = [weight.astype(np.float64) for weight in tail_weights]
    h64 = anchor.astype(np.float64)
    probabilities, intercepts = pilot_affine_params(
        h64[pilot_indices], tail_weights
    )
    tail_length = len(weights)

    importances: list[np.ndarray] = []
    pilot_h = h64[pilot_indices]
    pilot_activations: list[np.ndarray] = []
    for weight in weights:
        pilot_h = relu(pilot_h @ weight)
        pilot_activations.append(pilot_h)

    for layer in range(tail_length):
        if layer == tail_length - 1:
            importance = pilot_activations[-1].var(axis=0) + 1e-8
        else:
            transport = weights[layer + 1].copy()
            transport *= probabilities[layer + 1][None, :]
            for later in range(layer + 2, tail_length):
                transport = transport @ weights[later]
                transport *= probabilities[later][None, :]
            importance = np.linalg.norm(transport, axis=1) + 1e-12
        importances.append(importance)

    selected = [
        np.argsort(importances[layer])[-q_each:]
        for layer in range(tail_length)
    ]

    affine_matrices: list[np.ndarray] = []
    affine_offsets: list[np.ndarray] = []
    matrix = weights[0].copy()
    offset = np.zeros(WIDTH, dtype=np.float64)
    for layer in range(tail_length):
        affine_matrices.append(matrix.copy())
        affine_offsets.append(offset.copy())
        if layer < tail_length - 1:
            matrix = (matrix * probabilities[layer][None, :]) @ weights[layer + 1]
            offset = (
                offset * probabilities[layer] + intercepts[layer]
            ) @ weights[layer + 1]

    residuals: list[np.ndarray] = []
    approximated_activations: list[np.ndarray] = []
    for layer in range(tail_length):
        index = selected[layer]
        zhat = h64 @ affine_matrices[layer][:, index] + affine_offsets[layer][index]
        for earlier in range(layer):
            transport = weights[earlier + 1][selected[earlier], :].copy()
            for middle in range(earlier + 1, layer):
                transport = (
                    transport * probabilities[middle][None, :]
                ) @ weights[middle + 1]
            zhat += residuals[earlier] @ transport[:, index]
        ahat = relu(zhat)
        residual = ahat - (
            zhat * probabilities[layer][index] + intercepts[layer][index]
        )
        approximated_activations.append(ahat)
        residuals.append(residual)

    return np.concatenate(
        [h64, *residuals[:-1], approximated_activations[-1]], axis=1
    )


def standardize(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = features.mean(axis=0)
    sd = features.std(axis=0)
    keep = sd > eps
    return (features[:, keep] - mean[keep]) / sd[keep]


def quotas(total_selected_pairs: int) -> np.ndarray:
    base = total_selected_pairs // ALL_BASES
    remainder = total_selected_pairs - base * ALL_BASES
    result = np.full(ALL_BASES, base, dtype=np.int32)
    result[:remainder] += 1
    return result


def random_balanced_selection(
    quota: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    selected: list[int] = []
    for basis, count in enumerate(quota):
        candidates = np.arange(
            basis * PAIRS_PER_BASIS,
            (basis + 1) * PAIRS_PER_BASIS,
        )
        selected.extend(rng.choice(candidates, size=int(count), replace=False))
    return np.asarray(sorted(selected), dtype=np.int32)


def base_weights(selection: np.ndarray, quota: np.ndarray) -> np.ndarray:
    basis = selection // PAIRS_PER_BASIS
    return np.asarray(
        [1.0 / (ALL_BASES * quota[b]) for b in basis], dtype=np.float64
    )


def objective(features: np.ndarray, selection: np.ndarray, quota: np.ndarray) -> float:
    vector = features[selection].T @ base_weights(selection, quota)
    return float(vector @ vector)


def best_multistart(
    features: np.ndarray,
    quota: np.ndarray,
    nstarts: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best: np.ndarray | None = None
    best_objective = math.inf
    for _ in range(nstarts):
        selection = random_balanced_selection(quota, rng)
        value = objective(features, selection, quota)
        if value < best_objective:
            best = selection
            best_objective = value
    if best is None:
        raise RuntimeError("No multistart candidate generated")
    return best


def exchange_selection(
    features: np.ndarray,
    selection: np.ndarray,
    quota: np.ndarray,
    max_iterations: int,
    shortlist: int = 4,
) -> tuple[np.ndarray, int]:
    mask = np.zeros(len(features), dtype=bool)
    mask[selection] = True
    weights = base_weights(selection, quota)
    mean_vector = features[selection].T @ weights

    completed = 0
    for iteration in range(max_iterations):
        best_delta = 0.0
        best_pair: tuple[int, int] | None = None
        best_scale = 0.0
        projection = features @ mean_vector

        for basis, count in enumerate(quota):
            candidates = np.arange(
                basis * PAIRS_PER_BASIS,
                (basis + 1) * PAIRS_PER_BASIS,
            )
            inside = candidates[mask[candidates]]
            outside = candidates[~mask[candidates]]
            scale = 1.0 / (ALL_BASES * count)
            remove = inside[
                np.argsort(projection[inside])[-min(shortlist, len(inside)) :]
            ]
            add = outside[
                np.argsort(projection[outside])[: min(shortlist, len(outside))]
            ]
            differences = (
                features[add][:, None, :] - features[remove][None, :, :]
            ) * scale
            deltas = 2.0 * np.tensordot(
                differences, mean_vector, axes=([2], [0])
            ) + np.sum(differences * differences, axis=2)
            i, j = np.unravel_index(np.argmin(deltas), deltas.shape)
            value = float(deltas[i, j])
            if value < best_delta:
                best_delta = value
                best_pair = int(remove[j]), int(add[i])
                best_scale = scale

        if best_pair is None or best_delta > -1e-14:
            break
        removed, added = best_pair
        mask[removed] = False
        mask[added] = True
        mean_vector += (features[added] - features[removed]) * best_scale
        completed = iteration + 1

    return np.flatnonzero(mask).astype(np.int32), completed


def calibrated_weights(
    selected_features: np.ndarray,
    selection: np.ndarray,
    quota: np.ndarray,
    min_relative: float = 0.05,
    max_relative: float = 4.0,
    min_ess_fraction: float = 0.8,
) -> tuple[np.ndarray, dict[str, float | None]]:
    count, feature_count = selected_features.shape
    basis = selection // PAIRS_PER_BASIS
    uniform = base_weights(selection, quota)
    target_error = selected_features.T @ uniform

    within_basis = selected_features.copy()
    for basis_id in range(ALL_BASES):
        indices = np.flatnonzero(basis == basis_id)
        within_basis[indices] -= selected_features[indices].mean(
            axis=0, keepdims=True
        )

    gram = within_basis.T @ within_basis
    best_weights = uniform
    best_objective = float(target_error @ target_error)
    best_info: dict[str, float | None] = {
        "lambda": None,
        "min_relative": 1.0,
        "max_relative": 1.0,
        "ess_fraction": 1.0 / (np.sum(uniform * uniform) * count),
    }

    for scale in np.logspace(-8, 4, 40):
        regularization = scale * count
        try:
            coefficient = np.linalg.solve(
                gram + regularization * np.eye(feature_count), target_error
            )
        except np.linalg.LinAlgError:
            continue
        weights = uniform - within_basis @ coefficient
        relative = weights / uniform
        if relative.min() < min_relative or relative.max() > max_relative:
            continue
        ess = 1.0 / np.sum(weights * weights)
        if ess < min_ess_fraction * count:
            continue
        error = selected_features.T @ weights
        value = float(error @ error)
        if value < best_objective:
            best_weights = weights
            best_objective = value
            best_info = {
                "lambda": float(regularization),
                "min_relative": float(relative.min()),
                "max_relative": float(relative.max()),
                "ess_fraction": float(ess / count),
            }

    return best_weights, best_info


def added_mse(
    outputs: np.ndarray,
    selection: np.ndarray,
    weights: np.ndarray,
    full_mean: np.ndarray,
) -> float:
    estimate = weights @ outputs[selection]
    return float(np.mean((estimate - full_mean) ** 2))


def run_network(
    seed: int,
    chirps: np.ndarray,
    rotation: np.ndarray,
    q_each: int,
    selected_pairs: int,
    nstarts: int,
    max_swaps: int,
) -> dict[str, Any]:
    started = time.time()
    network_weights = gen_weights(seed)
    anchor = propagate_to_anchor(network_weights, chirps, rotation, 28)
    prefix_time = time.time() - started

    final = anchor
    for weight in network_weights[-4:]:
        final = relu(final @ weight)
    output_pairs = pair_average(final)
    anchor_pairs = pair_average(anchor)
    full_mean = output_pairs.mean(axis=0)

    quota = quotas(selected_pairs)
    random_selection = random_balanced_selection(
        quota, np.random.default_rng(seed + 777)
    )
    random_weights = base_weights(random_selection, quota)

    anchor_features = standardize(anchor_pairs)
    random_anchor_weights, _ = calibrated_weights(
        anchor_features[random_selection], random_selection, quota
    )

    anchor_selection = best_multistart(
        anchor_features, quota, nstarts, seed + 100
    )
    anchor_selection, anchor_iterations = exchange_selection(
        anchor_features, anchor_selection, quota, max_swaps
    )
    anchor_weights, _ = calibrated_weights(
        anchor_features[anchor_selection], anchor_selection, quota
    )

    feature_started = time.time()
    row_features = build_suffix_features(
        anchor, network_weights[-4:], pilot_rows(8), q_each
    )
    suffix_features = standardize(pair_average(row_features))
    suffix_selection = best_multistart(
        suffix_features, quota, nstarts, seed + 200
    )
    suffix_selection, suffix_iterations = exchange_selection(
        suffix_features, suffix_selection, quota, max_swaps
    )
    suffix_weights, suffix_info = calibrated_weights(
        suffix_features[suffix_selection], suffix_selection, quota
    )
    feature_time = time.time() - feature_started

    oracle_features = standardize(output_pairs)
    same_support_oracle_weights, same_support_info = calibrated_weights(
        oracle_features[suffix_selection], suffix_selection, quota
    )
    oracle_selection = best_multistart(
        oracle_features, quota, nstarts, seed + 300
    )
    oracle_selection, oracle_iterations = exchange_selection(
        oracle_features, oracle_selection, quota, max_swaps
    )
    oracle_weights, oracle_info = calibrated_weights(
        oracle_features[oracle_selection], oracle_selection, quota
    )

    result: dict[str, Any] = {
        "seed": seed,
        "prefix_time": prefix_time,
        "feature_time": feature_time,
        "total_time": time.time() - started,
        "K1_iterations": anchor_iterations,
        "K2_iterations": suffix_iterations,
        "O2_iterations": oracle_iterations,
        "K2_ess": suffix_info["ess_fraction"],
        "O1_ess": same_support_info["ess_fraction"],
        "O2_ess": oracle_info["ess_fraction"],
        "R0_added": added_mse(
            output_pairs, random_selection, random_weights, full_mean
        ),
        "R1_added": added_mse(
            output_pairs, random_selection, random_anchor_weights, full_mean
        ),
        "K1_added": added_mse(
            output_pairs, anchor_selection, anchor_weights, full_mean
        ),
        "K2_added": added_mse(
            output_pairs, suffix_selection, suffix_weights, full_mean
        ),
        "O1_added": added_mse(
            output_pairs,
            suffix_selection,
            same_support_oracle_weights,
            full_mean,
        ),
        "O2_added": added_mse(
            output_pairs, oracle_selection, oracle_weights, full_mean
        ),
    }
    return result


def parse_seeds(value: str) -> list[int]:
    if ":" in value:
        start, stop = value.split(":", 1)
        return list(range(int(start), int(stop)))
    return [int(part) for part in value.split(",") if part.strip()]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method in ("R0", "R1", "K1", "K2", "O1", "O2"):
        values = np.asarray([row[f"{method}_added"] for row in records])
        summary[method] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "worst": float(values.max()),
            "below_1.1e-8": int(np.sum(values <= 1.1e-8)),
            "below_2.2e-8": int(np.sum(values <= 2.2e-8)),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--seeds", default="44000:44020")
    parser.add_argument("--output", type=Path, default=Path("exact_kerdock_results.json"))
    parser.add_argument("--q-each", type=int, default=8)
    parser.add_argument("--selected-pairs", type=int, default=4096)
    parser.add_argument("--nstarts", type=int, default=4)
    parser.add_argument("--max-swaps", type=int, default=32)
    args = parser.parse_args()

    asset = np.load(args.asset)
    chirps = asset["chirps"].astype(np.float32)
    rotation = asset["rotation"].astype(np.float32)
    seeds = parse_seeds(args.seeds)

    records: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, 1):
        result = run_network(
            seed,
            chirps,
            rotation,
            args.q_each,
            args.selected_pairs,
            args.nstarts,
            args.max_swaps,
        )
        records.append(result)
        print(
            f"[{index}/{len(seeds)}] seed={seed} "
            f"K2={result['K2_added']:.3e} "
            f"O1={result['O1_added']:.3e} "
            f"O2={result['O2_added']:.3e} "
            f"time={result['total_time']:.1f}s",
            flush=True,
        )
        args.output.write_text(
            json.dumps(
                {"records": records, "summary": summarize(records)}, indent=2
            )
        )

    print(json.dumps(summarize(records), indent=2))


if __name__ == "__main__":
    main()
