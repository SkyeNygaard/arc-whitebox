"""Staged selection and frozen holdout for depth-dependent augment K3 shrink."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))

from factor_k3_fused_proxy_ablation import (  # noqa: E402
    fused_weight,
    linear_kprop_split_k3,
    sketched_k3_basis,
)
from factor_k3_subspace_ablation import (  # noqa: E402
    KINDS,
    initial_tower,
    load_official,
)
from mlp_kprop.factor_k3 import FactoredTensor  # noqa: E402
from mlp_kprop.kprop_harmonic import nonlin_kprop  # noqa: E402
from mlp_kprop.wick import relu_wick_coef  # noqa: E402


RANK = 64
SKETCH_COLUMNS = 256
SKETCH_SEED = 2026


def run_schedule(weights, schedule):
    tower = initial_tower(256, weights[0].dtype, KINDS["augment"])
    pending_basis = None
    pending_scale = 1.0
    generator = torch.Generator().manual_seed(SKETCH_SEED)
    start_time = time.perf_counter()
    for layer, weight in enumerate(weights):
        k3_weight = fused_weight(weight, pending_basis, pending_scale)
        tower = linear_kprop_split_k3(tower, weight, k3_weight)
        tower = nonlin_kprop(
            tower,
            nonlin_wick_coef=relu_wick_coef,
            k_max=3,
            kind=KINDS["augment"],
            factor=True,
        )
        carrier = tower[3]
        assert isinstance(carrier, FactoredTensor)
        if layer >= schedule["start"] and layer < len(weights) - 1:
            pending_basis = sketched_k3_basis(
                carrier,
                rank=RANK,
                sample_columns=SKETCH_COLUMNS,
                generator=generator,
            )
            if schedule["kind"] == "constant":
                pending_scale = schedule["lambda"]
            else:
                # Linear from one at layer zero to 0.75 at the last layer.
                pending_scale = 1.0 - 0.25 * layer / (len(weights) - 1)
        else:
            pending_basis = None
            pending_scale = 1.0
    return tower[1].core.detach().clone(), time.perf_counter() - start_time


def evaluate(mlp_id, schedule):
    weights, target = load_official(
        f"/tmp/phase1_mlp{mlp_id}.npz", torch.float32
    )
    prediction, seconds = run_schedule(weights, schedule)
    return {
        "mlp_id": mlp_id,
        "target_mse": float(
            np.mean((prediction.cpu().numpy() - target) ** 2)
        ),
        "seconds": seconds,
    }


def baseline(mlp_id):
    schedule = {
        "name": "baseline",
        "kind": "constant",
        "start": 32,
        "lambda": 1.0,
    }
    return evaluate(mlp_id, schedule)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "augment_depth_shrink.json",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float32)

    schedules = [
        {
            "name": f"start{start}_lambda{value}",
            "kind": "constant",
            "start": start,
            "lambda": value,
        }
        for start in (8, 16, 24)
        for value in (0.85, 0.9, 0.95)
    ]
    schedules.append(
        {
            "name": "linear_ramp_1_to_075",
            "kind": "ramp",
            "start": 0,
            "lambda": None,
        }
    )
    baseline_by_id = {0: baseline(0)}
    screen = []
    for schedule in schedules:
        result = evaluate(0, schedule)
        screen.append(
            {
                "schedule": schedule,
                "result": result,
                "ratio": (
                    result["target_mse"]
                    / baseline_by_id[0]["target_mse"]
                ),
            }
        )
        print(
            json.dumps(
                {
                    "screen": schedule["name"],
                    "ratio": screen[-1]["ratio"],
                }
            ),
            flush=True,
        )
    if args.screen_only:
        artifact = {
            "configuration": {
                "rank": RANK,
                "sketch_columns": SKETCH_COLUMNS,
                "sketch_seed": SKETCH_SEED,
            },
            "screen_id0": screen,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2) + "\n")
        print(args.output)
        return
    top_two = sorted(screen, key=lambda row: row["ratio"])[:2]

    stage_two = []
    for candidate in top_two:
        rows = [candidate["result"]]
        for mlp_id in range(1, 5):
            baseline_by_id.setdefault(mlp_id, baseline(mlp_id))
            rows.append(evaluate(mlp_id, candidate["schedule"]))
        ratio = (
            statistics.fmean(row["target_mse"] for row in rows)
            / statistics.fmean(
                baseline_by_id[i]["target_mse"] for i in range(5)
            )
        )
        stage_two.append(
            {
                "schedule": candidate["schedule"],
                "rows": rows,
                "ratio": ratio,
            }
        )
    winner = min(stage_two, key=lambda row: row["ratio"])

    selection_rows = list(winner["rows"])
    for mlp_id in range(5, 10):
        baseline_by_id[mlp_id] = baseline(mlp_id)
        selection_rows.append(evaluate(mlp_id, winner["schedule"]))
    selection_ratio = (
        statistics.fmean(r["target_mse"] for r in selection_rows)
        / statistics.fmean(
            baseline_by_id[i]["target_mse"] for i in range(10)
        )
    )

    holdout = None
    if selection_ratio <= 0.9:
        holdout_baseline = {}
        holdout_rows = []
        for mlp_id in range(60, 70):
            holdout_baseline[mlp_id] = baseline(mlp_id)
            holdout_rows.append(evaluate(mlp_id, winner["schedule"]))
        holdout = {
            "ids": [60, 69],
            "baseline": list(holdout_baseline.values()),
            "rows": holdout_rows,
            "ratio": (
                statistics.fmean(r["target_mse"] for r in holdout_rows)
                / statistics.fmean(
                    r["target_mse"]
                    for r in holdout_baseline.values()
                )
            ),
        }
    artifact = {
        "configuration": {
            "rank": RANK,
            "sketch_columns": SKETCH_COLUMNS,
            "sketch_seed": SKETCH_SEED,
            "selection_ids": [0, 9],
            "holdout_gate": "selection ratio <= 0.9",
        },
        "screen_id0": screen,
        "stage_two_ids0_4": stage_two,
        "selection": {
            "winner": winner["schedule"],
            "baseline": list(baseline_by_id.values()),
            "rows": selection_rows,
            "ratio": selection_ratio,
        },
        "holdout": holdout,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "winner": winner["schedule"],
                "selection_ratio": selection_ratio,
                "holdout_opened": holdout is not None,
                "holdout_ratio": (
                    holdout["ratio"] if holdout is not None else None
                ),
            },
            indent=2,
        )
    )
    print(args.output)


if __name__ == "__main__":
    main()
