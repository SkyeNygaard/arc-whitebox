"""Export deterministic K3 priors for the RQMC shrinkage experiment.

This helper intentionally uses the vendored ARC cumulant-propagation runtime.
Input rows are small challenge-format NPZ files with ``weights`` and ``means``.
It exports exact factored K3-simple and the already-frozen 256-column sketched
K3-simple prediction.  K3-augment is optional because it is roughly ten times
slower at width 256.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from factor_k3_fused_proxy_ablation import run_fused  # noqa: E402
from factor_k3_subspace_ablation import KINDS, load_official  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", type=int, required=True)
    parser.add_argument("--input-pattern", default="/tmp/phase1_mlp{id}.npz")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--include-augment", action="store_true")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float32)

    predictions: dict[str, list[np.ndarray]] = {
        "k3_simple": [],
        "k3_sketch256": [],
    }
    if args.include_augment:
        predictions["k3_augment"] = []
    targets: list[np.ndarray] = []
    seconds: dict[str, list[float]] = {key: [] for key in predictions}

    for position, mlp_id in enumerate(args.ids, start=1):
        weights, target = load_official(
            args.input_pattern.format(id=mlp_id),
            torch.float32,
        )
        simple, elapsed = run_fused(
            weights,
            rank=64,
            residual_scale=0.75,
            basis_source="exact",
            kind=KINDS["simple"],
        )
        predictions["k3_simple"].append(simple.cpu().numpy())
        seconds["k3_simple"].append(elapsed)

        sketch, elapsed = run_fused(
            weights,
            rank=64,
            residual_scale=0.75,
            basis_source="fused_k3_sketch",
            kind=KINDS["simple"],
            sketch_columns=256,
            sketch_seed=2026,
        )
        predictions["k3_sketch256"].append(sketch.cpu().numpy())
        seconds["k3_sketch256"].append(elapsed)

        if args.include_augment:
            augment, elapsed = run_fused(
                weights,
                rank=64,
                residual_scale=0.75,
                basis_source="exact",
                kind=KINDS["augment"],
            )
            predictions["k3_augment"].append(augment.cpu().numpy())
            seconds["k3_augment"].append(elapsed)
        targets.append(target)
        print(
            f"[{position:3d}/{len(args.ids)}] id={mlp_id:3d} "
            f"simple={seconds['k3_simple'][-1]:.3f}s "
            f"sketch={seconds['k3_sketch256'][-1]:.3f}s",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "mlp_id": np.asarray(args.ids, dtype=np.int64),
        "target": np.stack(targets),
    }
    for key, value in predictions.items():
        payload[key] = np.stack(value)
        payload[f"{key}_seconds"] = np.asarray(seconds[key], dtype=np.float64)
    np.savez_compressed(args.out, **payload)
    print(args.out)


if __name__ == "__main__":
    main()
