"""Nested consensus selection for sparse Kerdock basis unions.

For each outer 10-network test fold, four greedy paths are fitted on distinct
30-network inner training subsets.  Candidate bases are ranked first by how
often they appear and then by mean greedy rank.  The consensus subset is
evaluated only on the outer test fold.

This is a stricter stability-oriented companion to
``eval_sparse_kerdock_frontier.py`` and is limited to IDs 0--49.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_sparse_kerdock_frontier import (
    DEFAULT_CACHE,
    DEFAULT_SIZES,
    ROOT,
    build_families,
    greedy_path,
    prediction_mse,
    train_statistics,
)


DEFAULT_OUT = ROOT / "results" / "sparse_kerdock_consensus_selection.json"


def consensus_subset(
    paths: list[list[int]],
    count: int,
    groups: np.ndarray | None,
) -> list[int]:
    candidates = max(max(path) for path in paths) + 1
    frequency = np.zeros(candidates, dtype=int)
    rank_sum = np.zeros(candidates, dtype=float)
    missing_rank = max(len(path) for path in paths) + 1
    for path in paths:
        ranks = np.full(candidates, missing_rank, dtype=float)
        for rank, candidate in enumerate(path):
            frequency[candidate] += 1
            ranks[candidate] = rank
        rank_sum += ranks
    order = sorted(
        range(candidates),
        key=lambda candidate: (
            -frequency[candidate],
            rank_sum[candidate],
            candidate,
        ),
    )
    if groups is None:
        return order[:count]
    quota = count // (int(np.max(groups)) + 1)
    selected = []
    group_counts = np.zeros(int(np.max(groups)) + 1, dtype=int)
    for candidate in order:
        group = groups[candidate]
        if group_counts[group] >= quota:
            continue
        selected.append(candidate)
        group_counts[group] += 1
        if len(selected) == count:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    args = parser.parse_args()
    cache = np.load(args.cache)
    if not np.array_equal(cache["indices"], np.arange(50)):
        raise ValueError("cache must contain IDs 0--49")
    targets = cache["targets"].astype(np.float64)
    families = build_families(cache)
    outer_folds = np.arange(50) % 5
    records = []
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
        maximum = max(size // family.quota_scale for size in valid_sizes)
        for outer in range(5):
            outer_train = outer_folds != outer
            outer_test = ~outer_train
            train_indices = np.flatnonzero(outer_train)
            inner_paths = []
            for inner in range(4):
                inner_train = outer_train.copy()
                inner_train[
                    train_indices[np.arange(len(train_indices)) % 4 == inner]
                ] = False
                gram, correlation, target_norm = train_statistics(
                    family.predictions[inner_train],
                    targets[inner_train],
                )
                path = greedy_path(
                    maximum,
                    gram,
                    correlation,
                    target_norm,
                    groups=family.groups
                    if family.name == "balanced_four_rotations"
                    else None,
                    quota=maximum // 4
                    if family.name == "balanced_four_rotations"
                    else None,
                )
                inner_paths.append(path[-1])
            for total_bases in valid_sizes:
                count = total_bases // family.quota_scale
                truncated_paths = [path[:count] for path in inner_paths]
                selected = consensus_subset(
                    truncated_paths,
                    count,
                    family.groups
                    if family.name == "balanced_four_rotations"
                    else None,
                )
                record = {
                    "family": family.name,
                    "total_bases": total_bases,
                    "rows": 512 * total_bases,
                    "outer_fold": outer,
                    "test_mse": prediction_mse(
                        family.predictions[outer_test],
                        targets[outer_test],
                        selected,
                    ),
                }
                records.append(record)
                print(record, flush=True)
    summaries = []
    keys = sorted(
        {
            (record["family"], record["total_bases"])
            for record in records
        },
        key=lambda value: (value[1], value[0]),
    )
    for family, total_bases in keys:
        chosen = [
            record
            for record in records
            if record["family"] == family
            and record["total_bases"] == total_bases
        ]
        summaries.append(
            {
                "family": family,
                "total_bases": total_bases,
                "rows": 512 * total_bases,
                "nested_consensus_raw_mse": float(
                    np.mean([record["test_mse"] for record in chosen])
                ),
                "fold_raw_mses": [
                    record["test_mse"]
                    for record in sorted(
                        chosen, key=lambda value: value["outer_fold"]
                    )
                ],
            }
        )
    result = {
        "protocol": {
            "split": "IDs 0--49 only",
            "outer_fold": "index modulo 5",
            "inner_models_per_outer_fold": 4,
            "selection": "frequency then mean greedy rank",
        },
        "summaries": summaries,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print({"summaries": summaries}, flush=True)


if __name__ == "__main__":
    main()
