"""Frozen evaluation of the cheap fused K2-basis proxy for K3 shrinkage."""

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

from factor_k3_fused_proxy_ablation import run_fused  # noqa: E402
from factor_k3_subspace_ablation import KINDS, load_official  # noqa: E402


RANK = 64
RESIDUAL_SCALE = 0.75
WIDTH = 256
DEPTH = 32
SELECTION_IDS = list(range(10))
HOLDOUT_IDS = list(range(50, 60))
NAMES = {
    0: "daniel-harrison",
    1: "dustin-robinson",
    2: "cole-martin",
    3: "donna-clarke",
    4: "rebecca-walker",
    5: "monique-stevens",
    6: "keith-brock",
    7: "robert-huff",
    8: "kimberly-caldwell",
    9: "annette-hansen",
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


def evaluate(mlp_id: int) -> dict:
    weights, target = load_official(
        f"/tmp/phase1_mlp{mlp_id}.npz", torch.float32
    )
    exact, exact_seconds = run_fused(
        weights,
        rank=RANK,
        residual_scale=RESIDUAL_SCALE,
        basis_source="exact",
        kind=KINDS["simple"],
    )
    proxy, proxy_seconds = run_fused(
        weights,
        rank=RANK,
        residual_scale=RESIDUAL_SCALE,
        basis_source="fused_k2",
        kind=KINDS["simple"],
    )
    exact_mse = float(np.mean((exact.cpu().numpy() - target) ** 2))
    proxy_mse = float(np.mean((proxy.cpu().numpy() - target) ** 2))
    return {
        "mlp_id": mlp_id,
        "mlp_name": NAMES[mlp_id],
        "exact_target_mse": exact_mse,
        "proxy_target_mse": proxy_mse,
        "mse_ratio": proxy_mse / exact_mse,
        "mse_vs_exact_mean": float((proxy - exact).square().mean()),
        "exact_seconds": exact_seconds,
        "proxy_seconds": proxy_seconds,
        "runtime_ratio": proxy_seconds / exact_seconds,
    }


def summarize(records: list[dict]) -> dict:
    exact = statistics.fmean(r["exact_target_mse"] for r in records)
    proxy = statistics.fmean(r["proxy_target_mse"] for r in records)
    return {
        "networks": len(records),
        "improved": sum(r["mse_ratio"] < 1 for r in records),
        "mean_exact_target_mse": exact,
        "mean_proxy_target_mse": proxy,
        "ratio_of_mean_mse": proxy / exact,
        "median_per_network_mse_ratio": statistics.median(
            r["mse_ratio"] for r in records
        ),
        "mean_runtime_ratio": statistics.fmean(
            r["runtime_ratio"] for r in records
        ),
        "median_runtime_ratio": statistics.median(
            r["runtime_ratio"] for r in records
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            HERE.parents[0] / "results" / "k3_fused_k2_proxy.json"
        ),
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float32)

    selection = [evaluate(mlp_id) for mlp_id in SELECTION_IDS]
    holdout = [evaluate(mlp_id) for mlp_id in HOLDOUT_IDS]
    transitions = DEPTH - 1
    fused_projector_matmul_flops = (
        4 * WIDTH * WIDTH * RANK * transitions
    )
    # A conventional full symmetric eigendecomposition is about 9 n^3 FLOPs;
    # exact constants depend on the LAPACK algorithm. Report it separately.
    estimated_eigh_flops = 9 * WIDTH**3 * transitions
    artifact = {
        "frozen": {
            "rank": RANK,
            "residual_leg_scale": RESIDUAL_SCALE,
            "basis": "top eigenvectors of current preactivation K2 covariance",
        },
        "selection": {
            "ids": SELECTION_IDS,
            "summary": summarize(selection),
            "records": selection,
        },
        "holdout": {
            "ids": HOLDOUT_IDS,
            "summary": summarize(holdout),
            "records": holdout,
        },
        "flops": {
            "convention": "multiply and add count as two FLOPs",
            "fused_WP_matmuls": fused_projector_matmul_flops,
            "estimated_full_eigh": estimated_eigh_flops,
            "estimated_total_added": (
                fused_projector_matmul_flops + estimated_eigh_flops
            ),
            "challenge_budget": 272_000_000_000,
            "estimated_budget_fraction": (
                fused_projector_matmul_flops + estimated_eigh_flops
            )
            / 272_000_000_000,
            "note": (
                "No carrier-rank-dependent covariance or projection remains. "
                "The eigendecomposition constant is implementation-dependent."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact["selection"]["summary"]), flush=True)
    print(json.dumps(artifact["holdout"]["summary"]), flush=True)
    print(json.dumps(artifact["flops"]), flush=True)
    print(args.output)


if __name__ == "__main__":
    main()
