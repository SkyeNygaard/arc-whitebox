"""Compare dual contracted-K3 predictions with oracle connected C21."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def connected_c21(data: np.lib.npyio.NpzFile, layer: int) -> np.ndarray:
    mean = np.asarray(data["mean"][layer], dtype=np.float64)
    second = np.asarray(data["M11"][layer], dtype=np.float64)
    raw = np.asarray(data["M21"][layer], dtype=np.float64)
    marginal_second = np.asarray(data["m2"][layer], dtype=np.float64)
    return (
        raw
        - 2.0 * mean[:, None] * second
        - marginal_second[:, None] * mean[None, :]
        + 2.0 * np.square(mean)[:, None] * mean[None, :]
    )


def metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
    scale = float(
        np.sum(predicted * target)
        / max(np.sum(np.square(predicted)), 1e-30)
    )
    flat_predicted = predicted.ravel()
    flat_target = target.ravel()
    per_network_scales = np.sum(predicted * target, axis=1) / np.maximum(
        np.sum(np.square(predicted), axis=1),
        1e-30,
    )
    return {
        "relative_error": float(
            np.linalg.norm(predicted - target)
            / max(np.linalg.norm(target), 1e-30)
        ),
        "cosine": float(
            np.sum(predicted * target)
            / max(
                np.linalg.norm(predicted) * np.linalg.norm(target),
                1e-30,
            )
        ),
        "pearson": float(np.corrcoef(flat_predicted, flat_target)[0, 1]),
        "optimal_scale": scale,
        "scaled_relative_error": float(
            np.linalg.norm(scale * predicted - target)
            / max(np.linalg.norm(target), 1e-30)
        ),
        "per_network_scale_mean": float(np.mean(per_network_scales)),
        "per_network_scale_std": float(np.std(per_network_scales, ddof=1)),
        "per_network_scale_min": float(np.min(per_network_scales)),
        "per_network_scale_max": float(np.max(per_network_scales)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-results", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dual = json.loads(args.dual_results.read_text())
    layer = int(dual["protocol"]["layer"])
    rank = int(dual["protocol"]["rank"])
    oracle_targets = []
    factor_targets = []
    predictions = {
        label: []
        for label in (
            "direct_cp",
            "exact_dual",
            "frozen_lower_dual",
            "cheap_lower_dual",
        )
    }
    lowrank_keys = sorted(
        dual["records"][0].get("cheap_lower_dual_lowrank", {}).keys(),
        key=int,
    )
    for key in lowrank_keys:
        predictions[f"cheap_lower_dual_probe_rank{key}"] = []
    records = []
    for record in dual["records"]:
        index = int(record["index"])
        with np.load(
            args.artifact_dir / f"mlp_{index:05d}.npz"
        ) as artifact:
            factor_c21 = np.asarray(artifact["c21"], dtype=np.float64)
        with np.load(
            args.oracle_dir / f"mlp_{index:05d}.npz"
        ) as oracle:
            true_c21 = connected_c21(oracle, layer)
        u, _, vt = np.linalg.svd(factor_c21, full_matrices=False)
        left = u[:, :rank]
        right = vt[:rank].T
        true_target = np.einsum(
            "ik,ij,jk->k",
            left,
            true_c21,
            right,
        )
        factor_target = np.einsum(
            "ik,ij,jk->k",
            left,
            factor_c21,
            right,
        )
        oracle_targets.append(true_target)
        factor_targets.append(factor_target)
        for label in predictions:
            if label.startswith("cheap_lower_dual_probe_rank"):
                key = label.removeprefix("cheap_lower_dual_probe_rank")
                predictions[label].append(
                    record["cheap_lower_dual_lowrank"][key]
                )
            else:
                predictions[label].append(record[label])
        records.append(
            {
                "index": index,
                "oracle_target": true_target.tolist(),
                "factor_target": factor_target.tolist(),
            }
        )

    oracle_targets_array = np.asarray(oracle_targets)
    factor_targets_array = np.asarray(factor_targets)
    summary = {
        "factor_artifact_vs_oracle": metrics(
            factor_targets_array,
            oracle_targets_array,
        )
    }
    for label, values in predictions.items():
        array = np.asarray(values, dtype=np.float64)
        summary[f"{label}_vs_oracle"] = metrics(
            array,
            oracle_targets_array,
        )
        summary[f"{label}_vs_factor"] = metrics(
            array,
            factor_targets_array,
        )
    output = {
        "protocol": {
            "dual_results": str(args.dual_results),
            "artifact_dir": str(args.artifact_dir),
            "oracle_dir": str(args.oracle_dir),
            "layer": layer,
            "rank": rank,
        },
        "summary": summary,
        "records": records,
    }
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
