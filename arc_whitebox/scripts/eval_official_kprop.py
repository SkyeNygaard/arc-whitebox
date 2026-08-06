"""Evaluate ARC's reference cumulant propagation on a challenge-format MLP.

The input is a small ``npz`` extracted from an official WhestBench parquet row
with keys ``weights`` (depth, width, width) and ``means`` (depth, width).
This script intentionally lives outside the submission implementation: it is a
research oracle for deciding which parts of kprop are worth porting to
``flopscope.numpy``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "mlp_cumulant_propagation" / "src"
sys.path.insert(0, str(VENDOR))

from mlp_kprop.kprop_harmonic import AUGMENT, BASE, SIMPLE, mlp_kprop  # noqa: E402
from mlp_kprop.mlp import MLP  # noqa: E402


KINDS = {"simple": SIMPLE, "augment": AUGMENT, "base": BASE}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz")
    parser.add_argument("--k-max", type=int, default=2)
    parser.add_argument("--kind", choices=KINDS, default="simple")
    parser.add_argument("--factor", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--layer-errors",
        action="store_true",
        help="Also report the activation-mean MSE after every challenge layer.",
    )
    parser.add_argument("--save", type=Path, help="Optional NPZ path for predictions.")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.set_default_dtype(dtype)

    data = np.load(args.npz)
    weights = np.asarray(data["weights"])
    targets = np.asarray(data["means"])
    depth, width, width2 = weights.shape
    assert width == width2
    assert targets.shape == (depth, width)

    # ARC's MLP object has no activation after its final linear layer. Add an
    # identity readout so that challenge layer depth-1 remains a ReLU layer.
    mlp = MLP(
        input_dim=width,
        hidden_dim=width,
        output_dim=width,
        num_layers=depth + 1,
        nonlin="relu",
        init_kind="manual",
        w_var=[2.0] * depth + [1.0],
        b_var=0.0,
        b_mean=0.0,
    ).to(dtype=dtype)
    with torch.no_grad():
        for layer, weight in enumerate(weights):
            # WhestBench uses row-vector activations: z = h @ W. PyTorch
            # Linear stores the transposed (out, in) matrix.
            mlp.Ws[layer].weight.copy_(torch.as_tensor(weight.T, dtype=dtype))
        mlp.Ws[-1].weight.copy_(torch.eye(width, dtype=dtype))

    k_in = {
        1: torch.zeros(width, dtype=dtype),
        2: torch.eye(width, dtype=dtype),
    }
    start = time.perf_counter()
    result = mlp_kprop(
        mlp,
        k_in,
        k_max=args.k_max,
        kind=KINDS[args.kind],
        factor=args.factor,
        use_avg_metric=True,
        output_all=args.layer_errors,
        output_d_max=1,
    )
    elapsed = time.perf_counter() - start
    if args.layer_errors:
        prediction = result[f"act{depth - 1}"][1].to_tensor().detach().cpu().numpy()
    else:
        prediction = result[1].to_tensor().detach().cpu().numpy()
    target = targets[-1]
    mse = float(np.mean((prediction - target) ** 2))
    print(
        {
            "k_max": args.k_max,
            "kind": args.kind,
            "factor": args.factor,
            "dtype": args.dtype,
            "threads": torch.get_num_threads(),
            "seconds": elapsed,
            "mse": mse,
            "prediction_norm": float(np.linalg.norm(prediction)),
            "target_norm": float(np.linalg.norm(target)),
            "max_abs_error": float(np.max(np.abs(prediction - target))),
        }
    )
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save,
            prediction=prediction,
            target=target,
            all_targets=targets,
            k_max=args.k_max,
            kind=args.kind,
            factor=args.factor,
        )
    if args.layer_errors:
        print(
            {
                "layer_mse": [
                    float(
                        np.mean(
                            (
                                result[f"act{layer}"][1]
                                .to_tensor()
                                .detach()
                                .cpu()
                                .numpy()
                                - targets[layer]
                            )
                            ** 2
                        )
                    )
                    for layer in range(depth)
                ]
            }
        )


if __name__ == "__main__":
    main()
