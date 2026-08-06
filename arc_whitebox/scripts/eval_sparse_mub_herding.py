"""Target-independent sparse MUB cubature by polynomial kernel herding.

Every antipodal orthonormal basis is an exact spherical 3-design.  The full
129-basis Kerdock rule is an exact 5-design, but its adjusted score pays for
all 66,048 rows.  This experiment searches a pool of 516 bases from frozen
rotations 0, 1, 3, and 5 for much smaller equal-weight unions.

Selection never sees a network target.  Fixed random spherical probes encode
the degree-4, degree-6, and optionally degree-8 moment residual of each basis.
Greedy kernel herding followed by one-for-one exchange minimizes the norm of
the average residual.  An independent probe set audits generalization of the
moment objective.

Network evaluation uses a precomputed selection-only cache of per-basis final
predictions.  This makes it possible to sweep many subset sizes without
rerunning the MLPs; targets are consulted only to report MSE after each sparse
rule has been frozen by geometry.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval_kerdock_design import (
    N_BASES,
    WIDTH,
    make_kerdock_design,
    random_rotation,
)
from eval_sampling_official import DEFAULT_DATA, _load_rows
from eval_spherical_stein_cv import sphere_radius_mean


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "kerdock_basis_predictions_selection.npz"
OUT = ROOT / "results" / "sparse_mub_herding_selection.json"
SUBSETS = (24, 32, 48, 64, 80, 96, 112, 129)
ROTATION_SEEDS = (0, 1, 3, 5)
PROBES_PER_DEGREE = 384
DEGREES = (4, 6, 8)


@dataclass(frozen=True)
class FeatureSet:
    features: np.ndarray
    block_scales: dict[int, float]


def unit_basis_pool() -> tuple[np.ndarray, np.ndarray]:
    points = make_kerdock_design()
    radius = sphere_radius_mean(WIDTH)
    canonical = np.empty((N_BASES, WIDTH, WIDTH), dtype=np.float32)
    for basis in range(N_BASES):
        start = basis * 2 * WIDTH
        canonical[basis] = points[start : start + WIDTH] / radius
    pool = []
    labels = []
    for seed in ROTATION_SEEDS:
        rotation = random_rotation(WIDTH, seed)
        pool.append(canonical @ rotation)
        labels.extend((seed, basis) for basis in range(N_BASES))
    return np.concatenate(pool, axis=0), np.asarray(labels, dtype=np.int16)


def sphere_moment(degree: int) -> float:
    if degree % 2:
        return 0.0
    numerator = 1
    for value in range(1, degree, 2):
        numerator *= value
    denominator = 1
    for offset in range(0, degree, 2):
        denominator *= WIDTH + offset
    return numerator / denominator


def polynomial_features(
    bases: np.ndarray,
    probe_seed: int,
    block_scales: dict[int, float] | None = None,
) -> FeatureSet:
    rng = np.random.default_rng(probe_seed)
    blocks = []
    scales = {} if block_scales is None else dict(block_scales)
    for degree in DEGREES:
        probes = rng.standard_normal((WIDTH, PROBES_PER_DEGREE))
        probes /= np.linalg.norm(probes, axis=0, keepdims=True)
        target = sphere_moment(degree)
        values = np.empty((len(bases), PROBES_PER_DEGREE), dtype=np.float64)
        for offset in range(0, len(bases), 16):
            projection = bases[offset : offset + 16].astype(np.float64) @ probes
            values[offset : offset + 16] = np.mean(
                np.power(projection, degree),
                axis=1,
            )
        relative = values / target - 1.0
        if block_scales is None:
            # Equalize typical candidate energy across polynomial degrees.
            scales[degree] = float(np.sqrt(np.mean(np.square(relative))))
        blocks.append(relative / scales[degree])
    return FeatureSet(
        features=np.concatenate(blocks, axis=1).astype(np.float32),
        block_scales=scales,
    )


def weighted_features(
    features: np.ndarray,
    degree_weights: tuple[float, float, float],
) -> np.ndarray:
    result = features.copy()
    for block, weight in enumerate(degree_weights):
        start = block * PROBES_PER_DEGREE
        stop = start + PROBES_PER_DEGREE
        result[:, start:stop] *= math.sqrt(weight)
    return result


def greedy_herding(features: np.ndarray, count: int) -> list[int]:
    norms = np.sum(np.square(features), axis=1, dtype=np.float64)
    selected: list[int] = []
    available = np.ones(len(features), dtype=bool)
    total = np.zeros(features.shape[1], dtype=np.float64)
    for _ in range(count):
        scores = norms + 2.0 * (features.astype(np.float64) @ total)
        scores[~available] = np.inf
        chosen = int(np.argmin(scores))
        selected.append(chosen)
        available[chosen] = False
        total += features[chosen]
    return selected


def exchange_refine(
    features: np.ndarray,
    selected: list[int],
    max_passes: int = 8,
) -> list[int]:
    chosen = np.asarray(selected, dtype=np.int32)
    mask = np.zeros(len(features), dtype=bool)
    mask[chosen] = True
    total = np.sum(features[chosen], axis=0, dtype=np.float64)
    feature64 = features.astype(np.float64)
    for _ in range(max_passes):
        current = float(total @ total)
        best = current
        best_remove = None
        best_add = None
        outside = np.flatnonzero(~mask)
        outside_features = feature64[outside]
        outside_norms = np.sum(
            np.square(outside_features),
            axis=1,
        )
        for position, remove in enumerate(chosen):
            residual = total - feature64[remove]
            scores = (
                outside_norms
                + 2.0 * (outside_features @ residual)
                + float(residual @ residual)
            )
            candidate_position = int(np.argmin(scores))
            score = float(scores[candidate_position])
            if score + 1e-12 < best:
                best = score
                best_remove = position
                best_add = int(outside[candidate_position])
        if best_remove is None or best_add is None:
            break
        old = int(chosen[best_remove])
        total += feature64[best_add] - feature64[old]
        mask[old] = False
        mask[best_add] = True
        chosen[best_remove] = best_add
    return chosen.tolist()


def moment_audit(
    selected: list[int],
    train: FeatureSet,
    audit: FeatureSet,
) -> dict[str, object]:
    result = {}
    for label, feature_set in (("train", train), ("audit", audit)):
        average = np.mean(feature_set.features[selected], axis=0)
        by_degree = {}
        for block, degree in enumerate(DEGREES):
            start = block * PROBES_PER_DEGREE
            stop = start + PROBES_PER_DEGREE
            normalized = average[start:stop]
            relative = normalized * feature_set.block_scales[degree]
            by_degree[str(degree)] = {
                "relative_rms": float(
                    np.sqrt(np.mean(np.square(relative)))
                ),
                "relative_max_abs": float(np.max(np.abs(relative))),
            }
        result[label] = by_degree
    return result


def estimated_flops(count: int, labels: np.ndarray) -> int:
    rows = count * 2 * WIDTH
    selected_seeds = len(set(int(seed) for seed in labels[:, 0]))
    dense = 31 * rows * WIDTH * (2 * WIDTH - 1)
    relu = 32 * 2 * rows * WIDTH
    rotations = selected_seeds * WIDTH * WIDTH * (2 * WIDTH - 1)
    # Per selected non-coordinate basis: chirp multiply, eight FWHT stages,
    # and antipodal negation.  Flopscope charges two FLOPs per scalar op.
    noncoordinate = int(np.count_nonzero(labels[:, 1] != 128))
    structured_first = noncoordinate * (2 + 16 + 2) * WIDTH * WIDTH
    coordinate = int(np.count_nonzero(labels[:, 1] == 128))
    coordinate_cost = coordinate * 4 * WIDTH * WIDTH
    final_sum = 2 * (rows * WIDTH - WIDTH)
    final_divide = 2 * WIDTH
    return int(
        dense
        + relu
        + rotations
        + structured_first
        + coordinate_cost
        + final_sum
        + final_divide
    )


def evaluate_rule(
    selected: list[int],
    labels: np.ndarray,
    cache: np.ndarray,
    targets: np.ndarray,
) -> dict[str, object]:
    predictions = cache.reshape(
        (cache.shape[0], len(ROTATION_SEEDS) * N_BASES, WIDTH)
    )[:, selected].mean(axis=1, dtype=np.float64)
    per_network = np.mean(np.square(predictions - targets), axis=1)
    chosen_labels = labels[selected]
    flops = estimated_flops(len(selected), chosen_labels)
    multiplier = flops / 272_000_000_000
    raw = float(np.mean(per_network))
    return {
        "basis_count": len(selected),
        "row_count": len(selected) * 2 * WIDTH,
        "estimated_tracked_flops": flops,
        "estimated_compute_multiplier": multiplier,
        "raw_mean_final_mse": raw,
        "estimated_adjusted_score": raw * multiplier,
        "median_final_mse": float(np.median(per_network)),
        "selected_pool_indices": selected,
        "selected_seed_basis": chosen_labels.tolist(),
        "rotation_counts": {
            str(seed): int(np.count_nonzero(chosen_labels[:, 0] == seed))
            for seed in ROTATION_SEEDS
        },
        "per_network_final_mse": per_network.tolist(),
    }


def baseline_subsets(labels: np.ndarray, count: int) -> dict[str, list[int]]:
    seed3_start = ROTATION_SEEDS.index(3) * N_BASES
    rng_seed3 = np.random.default_rng(3103)
    rng_pool = np.random.default_rng(4516)
    return {
        "seed3_prefix": list(range(seed3_start, seed3_start + count)),
        "seed3_random": (
            seed3_start + rng_seed3.permutation(N_BASES)[:count]
        ).tolist(),
        "pool_random": rng_pool.permutation(len(labels))[:count].tolist(),
    }


def main() -> None:
    if not CACHE.exists():
        raise FileNotFoundError(
            f"basis cache not found: {CACHE}; run "
            "eval_kerdock_basis_variance_selector.py first"
        )
    basis_pool, labels = unit_basis_pool()
    train = polynomial_features(basis_pool, probe_seed=66171)
    audit = polynomial_features(
        basis_pool,
        probe_seed=88193,
        block_scales=train.block_scales,
    )
    cache_payload = np.load(CACHE)
    cache = cache_payload["basis_predictions"]
    if tuple(cache_payload["rotation_seeds"].tolist()) != ROTATION_SEEDS:
        raise AssertionError("basis-cache rotation order mismatch")
    rows = _load_rows(DEFAULT_DATA, list(range(50)))
    targets = np.stack([target[-1] for _, _, target in rows])

    methods = {
        "herd_q4": (1.0, 0.0, 0.0),
        "herd_q4_q6_quarter": (1.0, 0.25, 0.0),
        "herd_q4_q6": (1.0, 1.0, 0.0),
        "herd_q4_q6_q8": (1.0, 0.5, 0.25),
    }
    records = []
    for count in SUBSETS:
        for method, subset in baseline_subsets(labels, count).items():
            evaluation = evaluate_rule(subset, labels, cache, targets)
            evaluation["method"] = method
            evaluation["moment_audit"] = moment_audit(
                subset,
                train,
                audit,
            )
            records.append(evaluation)
            print(
                {
                    "count": count,
                    "method": method,
                    "raw": evaluation["raw_mean_final_mse"],
                    "adjusted": evaluation["estimated_adjusted_score"],
                },
                flush=True,
            )
        for method, weights in methods.items():
            feature = weighted_features(train.features, weights)
            subset = greedy_herding(feature, count)
            subset = exchange_refine(feature, subset)
            evaluation = evaluate_rule(subset, labels, cache, targets)
            evaluation["method"] = method
            evaluation["degree_weights"] = list(weights)
            evaluation["moment_audit"] = moment_audit(
                subset,
                train,
                audit,
            )
            records.append(evaluation)
            print(
                {
                    "count": count,
                    "method": method,
                    "raw": evaluation["raw_mean_final_mse"],
                    "adjusted": evaluation["estimated_adjusted_score"],
                },
                flush=True,
            )
            seed3_start = ROTATION_SEEDS.index(3) * N_BASES
            seed3_pool = np.arange(
                seed3_start,
                seed3_start + N_BASES,
                dtype=np.int32,
            )
            local_feature = feature[seed3_pool]
            local_subset = greedy_herding(local_feature, count)
            local_subset = exchange_refine(local_feature, local_subset)
            local_subset = seed3_pool[local_subset].tolist()
            local_evaluation = evaluate_rule(
                local_subset,
                labels,
                cache,
                targets,
            )
            local_evaluation["method"] = f"seed3_{method}"
            local_evaluation["degree_weights"] = list(weights)
            local_evaluation["moment_audit"] = moment_audit(
                local_subset,
                train,
                audit,
            )
            records.append(local_evaluation)
            print(
                {
                    "count": count,
                    "method": local_evaluation["method"],
                    "raw": local_evaluation["raw_mean_final_mse"],
                    "adjusted": local_evaluation[
                        "estimated_adjusted_score"
                    ],
                },
                flush=True,
            )

    full_seed3 = list(
        range(
            ROTATION_SEEDS.index(3) * N_BASES,
            (ROTATION_SEEDS.index(3) + 1) * N_BASES,
        )
    )
    full = evaluate_rule(full_seed3, labels, cache, targets)
    best = min(records, key=lambda row: float(row["estimated_adjusted_score"]))
    payload = {
        "protocol": {
            "selection_indices": list(range(50)),
            "holdout_loaded": False,
            "target_independent_subset_construction": True,
            "rotation_seeds": list(ROTATION_SEEDS),
            "subset_sizes": list(SUBSETS),
            "train_probe_seed": 66171,
            "audit_probe_seed": 88193,
            "probes_per_degree": PROBES_PER_DEGREE,
            "degrees": list(DEGREES),
        },
        "feature_block_scales": {
            str(key): value for key, value in train.block_scales.items()
        },
        "full_seed3_reference": full,
        "best_estimated_adjusted": {
            key: best[key]
            for key in (
                "method",
                "basis_count",
                "row_count",
                "raw_mean_final_mse",
                "estimated_adjusted_score",
                "estimated_tracked_flops",
            )
        },
        "records": records,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        {
            "out": str(OUT),
            "full_seed3": {
                "raw": full["raw_mean_final_mse"],
                "adjusted": full["estimated_adjusted_score"],
            },
            "best": payload["best_estimated_adjusted"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
