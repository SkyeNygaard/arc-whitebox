"""Selection-only sparse Kerdock-basis adjusted-score frontier.

The cached basis predictions make it possible to explore equal-weight unions
of antipodal orthonormal bases without rerunning the networks.  Every such
union is an exact spherical 3-design.  We compare:

* bases chosen only from the frozen seed-3 rotation;
* unrestricted bases from rotations 0, 1, 3, and 5;
* matched basis-index groups averaged over all four rotations;
* balanced four-rotation unions with a fixed quota per rotation.

Target-aware subsets are always selected on four folds and evaluated on the
fifth.  Both a nested greedy path and greedy-plus-one-swap local search are
reported.  This script refuses to consume any index beyond official ID 49.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "results" / "kerdock_basis_selection_cache_0135.npz"
DEFAULT_OUT = ROOT / "results" / "sparse_kerdock_frontier_selection.json"
DEFAULT_SIZES = [24, 32, 48, 64, 80, 96, 112, 129]


@dataclass
class CandidateFamily:
    name: str
    predictions: np.ndarray
    row_cost_per_candidate: int
    groups: np.ndarray | None = None
    quota_scale: int = 1


def train_statistics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    candidates = predictions.transpose((1, 0, 2)).reshape(
        (predictions.shape[1], -1)
    )
    target = targets.reshape(-1)
    gram = candidates @ candidates.T
    correlation = candidates @ target
    target_norm = float(target @ target)
    return gram, correlation, target_norm


def objective_numerator(
    selected: list[int],
    gram: np.ndarray,
    correlation: np.ndarray,
    target_norm: float,
) -> float:
    size = len(selected)
    indices = np.asarray(selected)
    return float(
        np.sum(gram[np.ix_(indices, indices)])
        - 2.0 * size * np.sum(correlation[indices])
        + size * size * target_norm
    )


def greedy_path(
    maximum: int,
    gram: np.ndarray,
    correlation: np.ndarray,
    target_norm: float,
    groups: np.ndarray | None = None,
    quota: int | None = None,
) -> list[list[int]]:
    candidates = len(correlation)
    selected: list[int] = []
    selected_mask = np.zeros(candidates, dtype=bool)
    group_counts = (
        np.zeros(int(np.max(groups)) + 1, dtype=int)
        if groups is not None
        else None
    )
    path = []
    quadratic = 0.0
    linear = 0.0
    summed_gram = np.zeros(candidates)
    for step in range(maximum):
        new_size = step + 1
        new_quadratic = quadratic + 2.0 * summed_gram + np.diag(gram)
        new_linear = linear + correlation
        values = (
            new_quadratic
            - 2.0 * new_size * new_linear
            + new_size * new_size * target_norm
        )
        values[selected_mask] = np.inf
        if groups is not None and quota is not None and group_counts is not None:
            blocked_groups = np.flatnonzero(group_counts >= quota)
            if len(blocked_groups):
                values[np.isin(groups, blocked_groups)] = np.inf
        chosen = int(np.argmin(values))
        if not np.isfinite(values[chosen]):
            raise RuntimeError("greedy path exhausted its quota")
        selected.append(chosen)
        selected_mask[chosen] = True
        if groups is not None and group_counts is not None:
            group_counts[groups[chosen]] += 1
        quadratic = float(new_quadratic[chosen])
        linear = float(new_linear[chosen])
        summed_gram += gram[:, chosen]
        path.append(selected.copy())
    return path


def local_swaps(
    initial: list[int],
    gram: np.ndarray,
    correlation: np.ndarray,
    target_norm: float,
    groups: np.ndarray | None = None,
    max_iterations: int = 100,
) -> list[int]:
    selected = initial.copy()
    size = len(selected)
    candidates = len(correlation)
    selected_mask = np.zeros(candidates, dtype=bool)
    selected_mask[selected] = True
    current = objective_numerator(
        selected, gram, correlation, target_norm
    )
    for _ in range(max_iterations):
        selected_array = np.asarray(selected)
        summed_gram = np.sum(gram[:, selected_array], axis=1)
        quadratic = float(np.sum(gram[np.ix_(selected_array, selected_array)]))
        linear = float(np.sum(correlation[selected_array]))
        best_value = current
        best_swap: tuple[int, int] | None = None
        for position, old in enumerate(selected):
            allowed = ~selected_mask
            if groups is not None:
                allowed &= groups == groups[old]
            new_indices = np.flatnonzero(allowed)
            if not len(new_indices):
                continue
            new_quadratic = (
                quadratic
                - 2.0 * summed_gram[old]
                + gram[old, old]
                + 2.0 * (summed_gram[new_indices] - gram[new_indices, old])
                + gram[new_indices, new_indices]
            )
            new_linear = linear - correlation[old] + correlation[new_indices]
            values = (
                new_quadratic
                - 2.0 * size * new_linear
                + size * size * target_norm
            )
            index = int(np.argmin(values))
            if values[index] < best_value - 1e-12:
                best_value = float(values[index])
                best_swap = (position, int(new_indices[index]))
        if best_swap is None:
            break
        position, new = best_swap
        old = selected[position]
        selected_mask[old] = False
        selected_mask[new] = True
        selected[position] = new
        current = best_value
    return selected


def prediction_mse(
    predictions: np.ndarray,
    targets: np.ndarray,
    selected: list[int],
) -> float:
    estimate = predictions[:, selected].mean(axis=1, dtype=np.float64)
    return float(np.mean(np.square(estimate - targets)))


def mean_pairwise_jaccard(subsets: list[list[int]]) -> float:
    values = []
    for left in range(len(subsets)):
        a = set(subsets[left])
        for right in range(left):
            b = set(subsets[right])
            values.append(len(a & b) / len(a | b))
    return float(np.mean(values)) if values else 1.0


def build_families(cache: np.lib.npyio.NpzFile) -> list[CandidateFamily]:
    predictions = cache["predictions"].astype(np.float64)
    seeds = cache["seeds"].tolist()
    seed_index = {int(seed): index for index, seed in enumerate(seeds)}
    seed3 = predictions[:, seed_index[3]]
    unrestricted = predictions.reshape(
        (predictions.shape[0], -1, predictions.shape[-1])
    )
    matched4 = predictions.mean(axis=1)
    groups = np.repeat(np.arange(len(seeds)), predictions.shape[2])
    return [
        CandidateFamily("seed3_only", seed3, 512),
        CandidateFamily("unrestricted_four_rotations", unrestricted, 512),
        CandidateFamily("matched_four_rotation_groups", matched4, 4 * 512, quota_scale=4),
        CandidateFamily(
            "balanced_four_rotations",
            unrestricted,
            512,
            groups=groups,
            quota_scale=1,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    args = parser.parse_args()
    cache = np.load(args.cache)
    if not np.array_equal(cache["indices"], np.arange(50)):
        raise ValueError("cache must contain official selection IDs 0--49 only")
    targets = cache["targets"].astype(np.float64)
    folds = np.arange(50) % 5
    families = build_families(cache)
    records = []
    selections: dict[tuple[str, int, str], list[list[int]]] = {}

    for family in families:
        valid_sizes = [
            size
            for size in args.sizes
            if size % family.quota_scale == 0
            and size // family.quota_scale <= family.predictions.shape[1]
            and (
                family.name != "balanced_four_rotations"
                or size % 4 == 0
            )
        ]
        candidate_sizes = [size // family.quota_scale for size in valid_sizes]
        maximum = max(candidate_sizes)
        for fold in range(5):
            train = folds != fold
            test = ~train
            gram, correlation, target_norm = train_statistics(
                family.predictions[train],
                targets[train],
            )
            quota = (
                maximum // 4
                if family.name == "balanced_four_rotations"
                else None
            )
            # For balanced paths, each requested size is divisible by four and
            # the maximum quota yields a path with equal group counts only at
            # the requested checkpoints.  Recompute per size below to enforce
            # its exact smaller quota.
            shared_path = (
                None
                if family.name == "balanced_four_rotations"
                else greedy_path(
                    maximum,
                    gram,
                    correlation,
                    target_norm,
                )
            )
            for total_bases, candidate_count in zip(
                valid_sizes, candidate_sizes, strict=True
            ):
                if family.name == "balanced_four_rotations":
                    path = greedy_path(
                        candidate_count,
                        gram,
                        correlation,
                        target_norm,
                        groups=family.groups,
                        quota=candidate_count // 4,
                    )
                else:
                    if shared_path is None:
                        raise AssertionError
                    path = shared_path
                greedy = path[candidate_count - 1]
                swapped = local_swaps(
                    greedy,
                    gram,
                    correlation,
                    target_norm,
                    groups=family.groups
                    if family.name == "balanced_four_rotations"
                    else None,
                )
                for method, selected in [
                    ("greedy", greedy),
                    ("greedy_swaps", swapped),
                ]:
                    mse = prediction_mse(
                        family.predictions[test],
                        targets[test],
                        selected,
                    )
                    key = (family.name, total_bases, method)
                    selections.setdefault(key, []).append(selected)
                    record = {
                        "family": family.name,
                        "total_bases": total_bases,
                        "rows": total_bases * 512,
                        "fold": fold,
                        "method": method,
                        "test_mse": mse,
                    }
                    records.append(record)
                    print(record, flush=True)

    summaries = []
    for key, fold_subsets in selections.items():
        family_name, total_bases, method = key
        family = next(
            value for value in families if value.name == family_name
        )
        candidate_count = total_bases // family.quota_scale
        all_gram, all_correlation, all_target_norm = train_statistics(
            family.predictions,
            targets,
        )
        if family.name == "balanced_four_rotations":
            all_path = greedy_path(
                candidate_count,
                all_gram,
                all_correlation,
                all_target_norm,
                groups=family.groups,
                quota=candidate_count // 4,
            )
        else:
            all_path = greedy_path(
                candidate_count,
                all_gram,
                all_correlation,
                all_target_norm,
            )
        all_greedy = all_path[candidate_count - 1]
        all_selected = (
            all_greedy
            if method == "greedy"
            else local_swaps(
                all_greedy,
                all_gram,
                all_correlation,
                all_target_norm,
                groups=family.groups
                if family.name == "balanced_four_rotations"
                else None,
            )
        )
        chosen = [
            record
            for record in records
            if record["family"] == family_name
            and record["total_bases"] == total_bases
            and record["method"] == method
        ]
        summaries.append(
            {
                "family": family_name,
                "total_bases": total_bases,
                "rows": total_bases * 512,
                "method": method,
                "nested_fivefold_raw_mse": float(
                    np.mean([record["test_mse"] for record in chosen])
                ),
                "fold_raw_mses": [
                    record["test_mse"]
                    for record in sorted(chosen, key=lambda value: value["fold"])
                ],
                "mean_pairwise_selection_jaccard": mean_pairwise_jaccard(
                    fold_subsets
                ),
                "all_selection_raw_mse": prediction_mse(
                    family.predictions,
                    targets,
                    all_selected,
                ),
                "all_selection_subset": all_selected,
            }
        )
    summaries.sort(
        key=lambda value: (
            value["total_bases"],
            value["nested_fivefold_raw_mse"],
        )
    )
    result = {
        "protocol": {
            "split": "official IDs 0--49 only",
            "fold": "index modulo 5",
            "candidate_rotation_seeds": cache["seeds"].tolist(),
            "points_per_antipodal_basis": 512,
            "moment_exactness": "every rule is a spherical 3-design",
        },
        "summaries": summaries,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print({"summaries": summaries}, flush=True)


if __name__ == "__main__":
    main()
