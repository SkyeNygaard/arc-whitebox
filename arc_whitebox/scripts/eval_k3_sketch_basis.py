"""Tune K3 covariance sketch size, then evaluate once on frozen holdout."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))

from factor_k3_fused_proxy_ablation import run_fused  # noqa: E402
from factor_k3_subspace_ablation import KINDS, load_official  # noqa: E402


RANK = 64
RESIDUAL_SCALE = 0.75
SKETCH_SEED = 2026
SKETCH_SIZES = [64, 128, 256]
SELECTION_IDS = list(range(10))
HOLDOUT_IDS = list(range(50, 60))
WIDTH = 256
DEPTH = 32
SIMPLE_SUM_CARRIER_COLUMNS = 405_504
FINAL_CARRIER_COLUMNS = 24_576


def target_mse(prediction: torch.Tensor, target: np.ndarray) -> float:
    return float(np.mean((prediction.cpu().numpy() - target) ** 2))


def run_source(mlp_id: int, source: str, sketch_columns: int) -> dict:
    weights, target = load_official(
        f"/tmp/phase1_mlp{mlp_id}.npz", torch.float32
    )
    prediction, seconds = run_fused(
        weights,
        rank=RANK,
        residual_scale=RESIDUAL_SCALE,
        basis_source=source,
        kind=KINDS["simple"],
        sketch_columns=sketch_columns,
        sketch_seed=SKETCH_SEED,
    )
    return {
        "mlp_id": mlp_id,
        "target_mse": target_mse(prediction, target),
        "seconds": seconds,
    }


def attach_ratios(records: list[dict], baseline_by_id: dict[int, dict]) -> None:
    for record in records:
        baseline = baseline_by_id[record["mlp_id"]]
        record["mse_ratio_vs_baseline"] = (
            record["target_mse"] / baseline["exact_target_mse"]
        )
        record["runtime_ratio_vs_baseline"] = (
            record["seconds"] / baseline["exact_seconds"]
        )


def summarize(records: list[dict], baseline_by_id: dict[int, dict]) -> dict:
    baseline_mean = statistics.fmean(
        baseline_by_id[r["mlp_id"]]["exact_target_mse"]
        for r in records
    )
    result_mean = statistics.fmean(r["target_mse"] for r in records)
    return {
        "networks": len(records),
        "improved": sum(r["mse_ratio_vs_baseline"] < 1 for r in records),
        "mean_target_mse": result_mean,
        "ratio_of_mean_mse_vs_baseline": result_mean / baseline_mean,
        "median_per_network_mse_ratio": statistics.median(
            r["mse_ratio_vs_baseline"] for r in records
        ),
        "mean_runtime_ratio_vs_baseline": statistics.fmean(
            r["runtime_ratio_vs_baseline"] for r in records
        ),
    }


def flop_model(sample_columns: int) -> dict:
    transitions = DEPTH - 1
    scanned_columns = SIMPLE_SUM_CARRIER_COLUMNS - FINAL_CARRIER_COLUMNS
    norm_scan = 6 * WIDTH * scanned_columns
    sampled_covariance = (
        6 * WIDTH * WIDTH * sample_columns * transitions
    )
    fused_wp = 4 * WIDTH * WIDTH * RANK * transitions
    estimated_eigh = 9 * WIDTH**3 * transitions
    total = norm_scan + sampled_covariance + fused_wp + estimated_eigh
    return {
        "sample_columns": sample_columns,
        "norm_scan_lower_bound": norm_scan,
        "sampled_covariance_matmuls": sampled_covariance,
        "fused_WP_matmuls": fused_wp,
        "estimated_full_eigh": estimated_eigh,
        "estimated_total_added": total,
        "challenge_budget_fraction": total / 272_000_000_000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "k3_sketch_basis.json",
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float32)

    k2_artifact = json.loads(
        (ROOT / "results" / "k3_fused_k2_proxy.json").read_text()
    )
    true_artifact = json.loads(
        (ROOT / "results" / "k3_soft_shrink_holdout.json").read_text()
    )
    selection_baseline = {
        r["mlp_id"]: r for r in k2_artifact["selection"]["records"]
    }
    holdout_baseline = {
        r["mlp_id"]: r for r in k2_artifact["holdout"]["records"]
    }

    # Selection phase. All sketch sizes and the exact K3 basis are compared
    # before any new holdout sketch run occurs.
    true_selection = [
        run_source(mlp_id, "fused_k3", sketch_columns=64)
        for mlp_id in SELECTION_IDS
    ]
    attach_ratios(true_selection, selection_baseline)
    sketch_selection: dict[int, dict] = {}
    for sample_columns in SKETCH_SIZES:
        records = [
            run_source(
                mlp_id,
                "fused_k3_sketch",
                sketch_columns=sample_columns,
            )
            for mlp_id in SELECTION_IDS
        ]
        attach_ratios(records, selection_baseline)
        sketch_selection[sample_columns] = {
            "summary": summarize(records, selection_baseline),
            "records": records,
        }
    chosen_size = min(
        SKETCH_SIZES,
        key=lambda size: sketch_selection[size]["summary"][
            "ratio_of_mean_mse_vs_baseline"
        ],
    )

    # Holdout is opened once with the selected sketch size.
    holdout_records = [
        run_source(
            mlp_id,
            "fused_k3_sketch",
            sketch_columns=chosen_size,
        )
        for mlp_id in HOLDOUT_IDS
    ]
    attach_ratios(holdout_records, holdout_baseline)

    artifact = {
        "frozen": {
            "rank": RANK,
            "residual_leg_scale": RESIDUAL_SCALE,
            "sketch_seed": SKETCH_SEED,
            "sampling": (
                "with replacement, p proportional to squared gauge-balanced "
                "leg norm, count/(m p) covariance correction"
            ),
        },
        "selection": {
            "ids": SELECTION_IDS,
            "sketches": {
                str(size): sketch_selection[size]
                for size in SKETCH_SIZES
            },
            "chosen_sample_columns": chosen_size,
            "true_k3_basis": {
                "summary": summarize(
                    true_selection, selection_baseline
                ),
                "records": true_selection,
            },
            "k2_proxy_summary": k2_artifact["selection"]["summary"],
        },
        "holdout": {
            "ids": HOLDOUT_IDS,
            "summary": summarize(holdout_records, holdout_baseline),
            "records": holdout_records,
            "baseline_summary": {
                "mean_target_mse": statistics.fmean(
                    r["exact_target_mse"]
                    for r in holdout_baseline.values()
                )
            },
            "true_k3_basis_summary": true_artifact["simple"]["summary"],
            "k2_proxy_summary": k2_artifact["holdout"]["summary"],
        },
        "id50_comparison": {
            "baseline": holdout_baseline[50]["exact_target_mse"],
            "k2_proxy": holdout_baseline[50]["proxy_target_mse"],
            "true_k3_basis": next(
                r["shrink_target_mse"]
                for r in true_artifact["simple"]["records"]
                if r["mlp_id"] == 50
            ),
            "sketch": next(
                r["target_mse"]
                for r in holdout_records
                if r["mlp_id"] == 50
            ),
        },
        "flops": {
            "convention": "multiply and add count as two FLOPs",
            "models": {
                str(size): flop_model(size) for size in SKETCH_SIZES
            },
            "note": (
                "Norm scan and sampled covariance exclude inexpensive log, "
                "gauge scaling, sampling, and linear-combination operations. "
                "The full-eigh constant is implementation-dependent."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                str(size): sketch_selection[size]["summary"]
                for size in SKETCH_SIZES
            }
        ),
        flush=True,
    )
    print(json.dumps({"chosen_sample_columns": chosen_size}), flush=True)
    print(json.dumps(artifact["holdout"]["summary"]), flush=True)
    print(json.dumps(artifact["id50_comparison"]), flush=True)
    print(json.dumps(artifact["flops"]["models"][str(chosen_size)]), flush=True)
    print(args.output)


if __name__ == "__main__":
    main()
