"""Quantify factorized post-ReLU state accuracy as a function of depth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from eval_oracle_cumulant_bridge import (  # noqa: E402
    connected_m21,
    moment_path,
    truncated_svd,
)


def relative_error(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.linalg.norm(predicted - target)
        / max(np.linalg.norm(target), 1e-30)
    )


def cosine(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.sum(predicted * target)
        / max(np.linalg.norm(predicted) * np.linalg.norm(target), 1e-30)
    )


def optimal_scale(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.sum(predicted * target)
        / max(np.sum(np.square(predicted)), 1e-30)
    )


def contracted(
    left: np.ndarray,
    matrix: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    return np.einsum("ik,ij,jk->k", left, matrix, right)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(160, 168)))
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[12, 16, 20, 24, 27, 29],
    )
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument(
        "--factorized-root",
        type=Path,
        default=HERE / "results" / "factorized_k3_depth_audit",
    )
    parser.add_argument(
        "--timings",
        type=Path,
        default=HERE / "results" / "factorized_k3_depth_audit_timings.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "results" / "factorized_k3_depth_accuracy.json",
    )
    args = parser.parse_args()

    by_layer: dict[int, list[dict]] = {layer: [] for layer in args.layers}
    pooled: dict[int, dict[str, list[np.ndarray]]] = {
        layer: {
            "true_c21": [],
            "factorized_c21": [],
            "true_oracle_contraction": [],
            "factorized_oracle_contraction": [],
            "true_factorized_contraction": [],
            "factorized_factorized_contraction": [],
            "true_factorcenter_contraction": [],
            "factorized_factorcenter_contraction": [],
        }
        for layer in args.layers
    }

    for index in args.indices:
        with np.load(moment_path(index)) as oracle:
            for layer in args.layers:
                true_mean = np.asarray(oracle["mean"][layer], dtype=np.float64)
                true_second = np.asarray(
                    oracle["M11"][layer],
                    dtype=np.float64,
                )
                true_covariance = (
                    true_second - np.outer(true_mean, true_mean)
                )
                true_c21 = connected_m21(
                    true_mean,
                    true_second,
                    np.asarray(oracle["M21"][layer], dtype=np.float64),
                    np.asarray(oracle["m2"][layer], dtype=np.float64),
                )
                factorized_path = (
                    args.factorized_root
                    / f"layer{layer}"
                    / f"mlp_{index:05d}.npz"
                )
                with np.load(factorized_path) as factorized:
                    predicted_mean = np.asarray(
                        factorized["mean"],
                        dtype=np.float64,
                    )
                    predicted_covariance = np.asarray(
                        factorized["covariance"],
                        dtype=np.float64,
                    )
                    predicted_c21 = np.asarray(
                        factorized["c21"],
                        dtype=np.float64,
                    )

                oracle_left, oracle_right = truncated_svd(
                    true_c21,
                    args.rank,
                )
                factorized_left, factorized_right = truncated_svd(
                    predicted_c21,
                    args.rank,
                )
                true_oracle_contraction = contracted(
                    oracle_left,
                    true_c21,
                    oracle_right,
                )
                predicted_oracle_contraction = contracted(
                    oracle_left,
                    predicted_c21,
                    oracle_right,
                )
                true_factorized_contraction = contracted(
                    factorized_left,
                    true_c21,
                    factorized_right,
                )
                predicted_factorized_contraction = contracted(
                    factorized_left,
                    predicted_c21,
                    factorized_right,
                )
                # The deployed connected pointwise feature is centered on the
                # factorized mean, not the oracle mean. Even a small mean error
                # can materially shift a third cumulant, so calibrate against
                # the exact anchor for that same fixed center.
                factor_center_true_matrix = (
                    np.asarray(oracle["M21"][layer], dtype=np.float64)
                    - 2.0
                    * predicted_mean[:, None]
                    * true_second
                    - np.asarray(
                        oracle["m2"][layer],
                        dtype=np.float64,
                    )[:, None]
                    * predicted_mean[None, :]
                    + 2.0
                    * np.square(predicted_mean)[:, None]
                    * true_mean[None, :]
                )
                true_factorcenter_contraction = contracted(
                    factorized_left,
                    factor_center_true_matrix,
                    factorized_right,
                )
                predicted_factorcenter_contraction = (
                    predicted_factorized_contraction
                )

                record = {
                    "index": index,
                    "mean_relative_error": relative_error(
                        predicted_mean,
                        true_mean,
                    ),
                    "covariance_relative_error": relative_error(
                        predicted_covariance,
                        true_covariance,
                    ),
                    "covariance_cosine": cosine(
                        predicted_covariance,
                        true_covariance,
                    ),
                    "c21_relative_error": relative_error(
                        predicted_c21,
                        true_c21,
                    ),
                    "c21_cosine": cosine(predicted_c21, true_c21),
                    "c21_norm_ratio": float(
                        np.linalg.norm(predicted_c21)
                        / max(np.linalg.norm(true_c21), 1e-30)
                    ),
                    "c21_optimal_scale": optimal_scale(
                        predicted_c21,
                        true_c21,
                    ),
                    "oracle_direction_relative_error": relative_error(
                        predicted_oracle_contraction,
                        true_oracle_contraction,
                    ),
                    "oracle_direction_optimal_scale": optimal_scale(
                        predicted_oracle_contraction,
                        true_oracle_contraction,
                    ),
                    "factorized_direction_relative_error": relative_error(
                        predicted_factorized_contraction,
                        true_factorized_contraction,
                    ),
                    "factorized_direction_optimal_scale": optimal_scale(
                        predicted_factorized_contraction,
                        true_factorized_contraction,
                    ),
                    "factorcenter_direction_optimal_scale": optimal_scale(
                        predicted_factorcenter_contraction,
                        true_factorcenter_contraction,
                    ),
                }
                by_layer[layer].append(record)
                pooled[layer]["true_c21"].append(true_c21)
                pooled[layer]["factorized_c21"].append(predicted_c21)
                pooled[layer]["true_oracle_contraction"].append(
                    true_oracle_contraction
                )
                pooled[layer]["factorized_oracle_contraction"].append(
                    predicted_oracle_contraction
                )
                pooled[layer]["true_factorized_contraction"].append(
                    true_factorized_contraction
                )
                pooled[layer]["factorized_factorized_contraction"].append(
                    predicted_factorized_contraction
                )
                pooled[layer]["true_factorcenter_contraction"].append(
                    true_factorcenter_contraction
                )
                pooled[layer]["factorized_factorcenter_contraction"].append(
                    predicted_factorcenter_contraction
                )

    timing_data = json.loads(args.timings.read_text())
    timing_by_layer = {
        layer: np.asarray(
            [
                record["layers"][str(layer)]["cumulative_seconds"]
                for record in timing_data["records"]
            ],
            dtype=np.float64,
        )
        for layer in args.layers
    }
    # These are generated by the vendor's cached exact FLOP polynomial for
    # SIMPLE/factored k=3 at width 256.
    known_flops = {
        12: 51_021_562_512,
        16: 87_369_318_076,
        20: 133_431_081_704,
        24: 189_206_853_396,
        27: 237_413_499_957,
        29: 272_586_891_851,
    }

    summary = {}
    for layer in args.layers:
        records = by_layer[layer]
        true_c21 = np.stack(pooled[layer]["true_c21"])
        predicted_c21 = np.stack(pooled[layer]["factorized_c21"])
        true_oracle = np.stack(
            pooled[layer]["true_oracle_contraction"]
        )
        predicted_oracle = np.stack(
            pooled[layer]["factorized_oracle_contraction"]
        )
        true_factorized = np.stack(
            pooled[layer]["true_factorized_contraction"]
        )
        predicted_factorized = np.stack(
            pooled[layer]["factorized_factorized_contraction"]
        )
        true_factorcenter = np.stack(
            pooled[layer]["true_factorcenter_contraction"]
        )
        predicted_factorcenter = np.stack(
            pooled[layer]["factorized_factorcenter_contraction"]
        )
        global_scale = optimal_scale(predicted_c21, true_c21)
        oracle_direction_scale = optimal_scale(
            predicted_oracle,
            true_oracle,
        )
        factorized_direction_scale = optimal_scale(
            predicted_factorized,
            true_factorized,
        )
        factorcenter_direction_scale = optimal_scale(
            predicted_factorcenter,
            true_factorcenter,
        )
        summary[str(layer)] = {
            "mean_relative_error_mean": float(
                np.mean([r["mean_relative_error"] for r in records])
            ),
            "covariance_relative_error_mean": float(
                np.mean([r["covariance_relative_error"] for r in records])
            ),
            "c21_relative_error_pooled": relative_error(
                predicted_c21,
                true_c21,
            ),
            "c21_cosine_pooled": cosine(predicted_c21, true_c21),
            "c21_global_optimal_scale": global_scale,
            "c21_global_scaled_relative_error": relative_error(
                global_scale * predicted_c21,
                true_c21,
            ),
            "c21_per_network_scale_mean": float(
                np.mean([r["c21_optimal_scale"] for r in records])
            ),
            "c21_per_network_scale_std": float(
                np.std([r["c21_optimal_scale"] for r in records], ddof=1)
            ),
            "oracle_direction_optimal_scale": oracle_direction_scale,
            "oracle_direction_scaled_relative_error": relative_error(
                oracle_direction_scale * predicted_oracle,
                true_oracle,
            ),
            "factorized_direction_optimal_scale": factorized_direction_scale,
            "factorized_direction_scaled_relative_error": relative_error(
                factorized_direction_scale * predicted_factorized,
                true_factorized,
            ),
            "factorcenter_direction_optimal_scale": (
                factorcenter_direction_scale
            ),
            "factorcenter_direction_scaled_relative_error": relative_error(
                factorcenter_direction_scale * predicted_factorcenter,
                true_factorcenter,
            ),
            "local_seconds_mean": float(np.mean(timing_by_layer[layer])),
            "local_seconds_std": float(
                np.std(timing_by_layer[layer], ddof=1)
            ),
            "analytical_flops": known_flops.get(layer),
        }

    output = {
        "protocol": {
            "indices": args.indices,
            "layers": args.layers,
            "rank": args.rank,
            "factorized_root": str(args.factorized_root),
            "oracle_samples": 100_000_000,
        },
        "summary": summary,
        "records": {
            str(layer): records
            for layer, records in by_layer.items()
        },
    }
    args.out.write_text(json.dumps(output, indent=2))
    for layer in args.layers:
        item = summary[str(layer)]
        print(
            f"L{layer:>2}: mean={item['mean_relative_error_mean']:.3f} "
            f"cov={item['covariance_relative_error_mean']:.3f} "
            f"c21={item['c21_relative_error_pooled']:.3f} "
            f"cos={item['c21_cosine_pooled']:.3f} "
            f"scale={item['c21_global_optimal_scale']:.3f} "
            f"scaled={item['c21_global_scaled_relative_error']:.3f} "
            f"dirscale={item['factorized_direction_optimal_scale']:.3f} "
            f"direrr={item['factorized_direction_scaled_relative_error']:.3f} "
            f"time={item['local_seconds_mean']:.2f}s "
            f"flops={item['analytical_flops'] / 1e9:.1f}B",
            flush=True,
        )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
