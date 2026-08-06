"""Internal-replication diagnostics for choosing a Kerdock orientation.

Each full Kerdock/MUB rule is an average of 129 antipodal orthonormal bases.
The basis estimates act as internal replicates.  A small number of bases can
therefore pilot several candidate rotations, after which the orientation with
the smallest across-basis variance (or split discrepancy) can be evaluated
with the full rule.

This script computes basis-level final predictions for fixed rotations
0, 1, 3, and 5 on selection IDs only, then retrospectively evaluates
target-free pilot selectors.  The official target is used solely to score the
chosen orientation.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval_kerdock_design import (
    N_BASES,
    WIDTH,
    make_kerdock_design,
    random_rotation,
)
from eval_sampling_official import DEFAULT_DATA, _load_rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "kerdock_basis_variance_selector.json"
DEFAULT_CACHE_OUT = ROOT / "results" / "kerdock_basis_predictions_selection.npz"
ROWS_PER_BASIS = 2 * WIDTH
DEFAULT_SEEDS = (0, 1, 3, 5)
PILOT_SIZES = (4, 8, 16, 24, 32)


def pilot_order() -> np.ndarray:
    rng = np.random.default_rng(20260728)
    # Keep the coordinate basis out of very small pilots; the 128 Kerdock
    # bases are exchangeable under this diagnostic.
    order = rng.permutation(128)
    return np.concatenate((order, np.asarray([128])))


def forward_basis_means(
    weights: np.ndarray,
    points: np.ndarray,
    rotation: np.ndarray,
    chunk_bases: int = 4,
) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    chunks = []
    chunk_rows = chunk_bases * ROWS_PER_BASIS
    for offset in range(0, len(points), chunk_rows):
        activation = points[offset : offset + chunk_rows] @ rotation
        for weight in weights:
            activation = np.maximum(activation @ weight, 0.0)
        chunks.append(
            activation.reshape((-1, ROWS_PER_BASIS, WIDTH)).mean(
                axis=1,
                dtype=np.float64,
            )
        )
    basis_means = np.concatenate(chunks, axis=0)
    if basis_means.shape != (N_BASES, WIDTH):
        raise AssertionError(basis_means.shape)
    return basis_means, time.perf_counter() - start


def diagnostics(
    basis_means: np.ndarray,
    order: np.ndarray,
) -> dict[str, float]:
    result = {
        "full_basis_variance": float(
            np.mean(np.var(basis_means, axis=0, ddof=1))
        ),
        "full_odd_even_discrepancy": float(
            np.mean(
                np.square(
                    basis_means[::2].mean(axis=0)
                    - basis_means[1::2].mean(axis=0)
                )
            )
        ),
    }
    for size in PILOT_SIZES:
        selected = basis_means[order[:size]]
        first = selected[: size // 2].mean(axis=0)
        second = selected[size // 2 :].mean(axis=0)
        cumulative_half = selected[: size // 2].mean(axis=0)
        cumulative_full = selected.mean(axis=0)
        result[f"pilot_variance_{size}"] = float(
            np.mean(np.var(selected, axis=0, ddof=1))
        )
        result[f"pilot_split_discrepancy_{size}"] = float(
            np.mean(np.square(first - second))
        )
        result[f"pilot_stabilization_{size}"] = float(
            np.mean(np.square(cumulative_full - cumulative_half))
        )
    return result


def selector_summary(
    records: list[dict[str, object]],
) -> dict[str, object]:
    features = [
        "full_basis_variance",
        "full_odd_even_discrepancy",
    ]
    for size in PILOT_SIZES:
        features.extend(
            (
                f"pilot_variance_{size}",
                f"pilot_split_discrepancy_{size}",
                f"pilot_stabilization_{size}",
            )
        )
    by_index: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_index[int(record["index"])].append(record)
    summaries = {}
    for feature in features:
        errors = []
        seeds = []
        ranks = []
        for candidates in by_index.values():
            chosen = min(
                candidates,
                key=lambda row: float(row["diagnostics"][feature]),
            )
            errors.append(float(chosen["final_mse"]))
            seeds.append(int(chosen["rotation_seed"]))
            diagnostic_values = np.asarray(
                [
                    float(row["diagnostics"][feature])
                    for row in candidates
                ]
            )
            target_values = np.asarray(
                [float(row["final_mse"]) for row in candidates]
            )
            rank_x = np.argsort(np.argsort(diagnostic_values))
            rank_y = np.argsort(np.argsort(target_values))
            ranks.append(float(np.corrcoef(rank_x, rank_y)[0, 1]))
        summaries[feature] = {
            "mean_selected_final_mse": float(np.mean(errors)),
            "median_selected_final_mse": float(np.median(errors)),
            "mean_within_network_spearman": float(np.mean(ranks)),
            "selection_counts": {
                str(seed): seeds.count(seed) for seed in DEFAULT_SEEDS
            },
        }
    fixed3 = [
        float(record["final_mse"])
        for record in records
        if int(record["rotation_seed"]) == 3
    ]
    oracle = []
    for candidates in by_index.values():
        oracle.append(min(float(row["final_mse"]) for row in candidates))
    return {
        "fixed_seed3_mean_final_mse": float(np.mean(fixed3)),
        "oracle_mean_final_mse": float(np.mean(oracle)),
        "selectors": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=list(range(50)),
    )
    parser.add_argument(
        "--rotation-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-out", type=Path, default=DEFAULT_CACHE_OUT)
    args = parser.parse_args()
    if not args.indices or min(args.indices) < 0 or max(args.indices) >= 50:
        raise ValueError("basis-variance study is restricted to IDs 0--49")
    if tuple(args.rotation_seeds) != DEFAULT_SEEDS:
        raise ValueError(f"expected frozen candidates {DEFAULT_SEEDS}")

    points = make_kerdock_design()
    rotations = {
        seed: random_rotation(WIDTH, seed)
        for seed in args.rotation_seeds
    }
    order = pilot_order()
    rows = _load_rows(args.data, args.indices)
    records: list[dict[str, object]] = []
    cache = np.empty(
        (
            len(args.indices),
            len(args.rotation_seeds),
            N_BASES,
            WIDTH,
        ),
        dtype=np.float32,
    )
    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        row_position = args.indices.index(index)
        for seed_position, (seed, rotation) in enumerate(rotations.items()):
            basis_means, seconds = forward_basis_means(
                weights,
                points,
                rotation,
            )
            cache[row_position, seed_position] = basis_means.astype(np.float32)
            prediction = basis_means.mean(axis=0)
            record = {
                "index": index,
                "name": name,
                "rotation_seed": seed,
                "seconds": seconds,
                "final_mse": float(
                    np.mean(np.square(prediction - targets[-1]))
                ),
                "diagnostics": diagnostics(basis_means, order),
            }
            records.append(record)
        print({"index": index, "rotations": len(rotations)}, flush=True)
    summary = selector_summary(records)
    payload = {
        "protocol": {
            "selection_indices": args.indices,
            "holdout_loaded": False,
            "rotation_seeds": args.rotation_seeds,
            "pilot_basis_order": order.tolist(),
            "pilot_sizes": list(PILOT_SIZES),
            "diagnostics_use_targets": False,
        },
        "summary": summary,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    args.cache_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.cache_out,
        basis_predictions=cache,
        indices=np.asarray(args.indices, dtype=np.int16),
        rotation_seeds=np.asarray(args.rotation_seeds, dtype=np.int16),
        pilot_order=order.astype(np.int16),
    )
    print(
        {
            "out": str(args.out),
            "cache_out": str(args.cache_out),
            "summary": summary,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
