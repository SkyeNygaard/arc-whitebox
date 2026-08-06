"""Frozen holdout evaluation for the K3 shared-subspace shrinkage ablation.

Selection set: official mini IDs 0--9.
Frozen hyperparameters: shared rank 64, residual leg scale 0.75.
Holdout: official mini IDs 50--59.

The script expects small NPZ extracts at ``/tmp/phase1_mlp<ID>.npz`` and writes
one compact JSON artifact. It does not tune or branch on holdout outcomes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from factor_k3_subspace_ablation import KINDS, load_official, run  # noqa: E402


FROZEN_RANK = 64
FROZEN_RESIDUAL_SCALE = 0.75
WIDTH = 256
SIMPLE_IDS = list(range(50, 60))
AUGMENT_IDS = [36, 50, 51, 52]
NAMES = {
    36: "paul-beasley",
    50: "joshua-collins",
    51: "brittney-massey",
    52: "elaine-lopez",
    53: "alexandra-reid",
    54: "desiree-maynard",
    55: "tyler-garcia",
    56: "thomas-smith",
    57: "ashley-smith",
    58: "tommy-arias",
    59: "shane-long",
}


def evaluate_one(mlp_id: int, kind_name: str, dtype: torch.dtype) -> dict:
    path = f"/tmp/phase1_mlp{mlp_id}.npz"
    weights, target = load_official(path, dtype)
    kind = KINDS[kind_name]
    exact_mean, _, _, ranks, exact_seconds = run(
        weights,
        projection_rank=None,
        kind=kind,
    )
    shrink_mean, _, captures, shrink_ranks, shrink_seconds = run(
        weights,
        projection_rank=FROZEN_RANK,
        residual_scale=FROZEN_RESIDUAL_SCALE,
        kind=kind,
    )
    assert ranks == shrink_ranks
    exact_target_mse = float(
        np.mean((exact_mean.cpu().numpy() - target) ** 2)
    )
    shrink_target_mse = float(
        np.mean((shrink_mean.cpu().numpy() - target) ** 2)
    )

    # Matmul-only lower bound for the added work:
    #   3 balanced-factor covariance products: 6 n^2 R
    #   project 3 factors out and back:          12 n r R
    # This excludes norms, balancing, and eigendecompositions.
    sum_carrier_columns = sum(ranks)
    added_matmul_flops = (
        6 * WIDTH * WIDTH + 12 * WIDTH * FROZEN_RANK
    ) * sum_carrier_columns
    return {
        "mlp_id": mlp_id,
        "mlp_name": NAMES[mlp_id],
        "kind": kind_name,
        "exact_target_mse": exact_target_mse,
        "shrink_target_mse": shrink_target_mse,
        "mse_ratio": shrink_target_mse / exact_target_mse,
        "mse_vs_exact_mean": float(
            (shrink_mean - exact_mean).square().mean()
        ),
        "exact_seconds": exact_seconds,
        "shrink_seconds": shrink_seconds,
        "runtime_ratio": shrink_seconds / exact_seconds,
        "sum_carrier_columns": sum_carrier_columns,
        "added_matmul_flops_lower_bound": added_matmul_flops,
        "capture_min": min(captures),
        "capture_mean": statistics.fmean(captures),
        "capture_final": captures[-1],
    }


def summarize(records: list[dict]) -> dict:
    exact = [record["exact_target_mse"] for record in records]
    shrink = [record["shrink_target_mse"] for record in records]
    return {
        "networks": len(records),
        "improved": sum(
            record["shrink_target_mse"] < record["exact_target_mse"]
            for record in records
        ),
        "mean_exact_target_mse": statistics.fmean(exact),
        "mean_shrink_target_mse": statistics.fmean(shrink),
        "ratio_of_mean_mse": statistics.fmean(shrink)
        / statistics.fmean(exact),
        "median_per_network_mse_ratio": statistics.median(
            record["mse_ratio"] for record in records
        ),
        "median_runtime_ratio": statistics.median(
            record["runtime_ratio"] for record in records
        ),
        "mean_runtime_ratio": statistics.fmean(
            record["runtime_ratio"] for record in records
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "results"
            / "k3_soft_shrink_holdout.json"
        ),
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float32)

    simple_records = [
        evaluate_one(mlp_id, "simple", torch.float32)
        for mlp_id in SIMPLE_IDS
    ]
    augment_records = [
        evaluate_one(mlp_id, "augment", torch.float32)
        for mlp_id in AUGMENT_IDS
    ]
    artifact = {
        "selection": {
            "selection_ids": list(range(10)),
            "holdout_ids": SIMPLE_IDS,
            "frozen_shared_rank": FROZEN_RANK,
            "frozen_residual_leg_scale": FROZEN_RESIDUAL_SCALE,
            "note": "No hyperparameters were changed after opening IDs 50--59.",
        },
        "simple": {
            "summary": summarize(simple_records),
            "records": simple_records,
        },
        "augment": {
            "summary": summarize(augment_records),
            "records": augment_records,
        },
        "flop_accounting": {
            "convention": "multiply and add count as two FLOPs",
            "scope": (
                "Added balanced-covariance and out/back projection matmuls only; "
                "excludes norms, balancing, eigendecomposition, and unchanged K3."
            ),
            "challenge_budget": 272_000_000_000,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact["simple"]["summary"]), flush=True)
    print(json.dumps(artifact["augment"]["summary"]), flush=True)
    print(args.output)


if __name__ == "__main__":
    main()
