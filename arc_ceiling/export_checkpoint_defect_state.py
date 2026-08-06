"""Export compact factorized-K3 state for checkpoint defect assimilation.

Unlike ``predict_factorized_k3_depth_audit.py --save-k3-factors``, this does
not write the enormous CP factors.  It rolls once to the target, then stores:

* the corrected target post-activation c21 slice;
* for each checkpoint, the part of target c21 inherited from factor columns
  already present at that checkpoint;
* the factorized expected-gate tail map from checkpoint post-activation to
  target post-activation.

Those are sufficient to replace the inherited contribution by a cheap
Kerdock checkpoint estimate.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from mlp_kprop.kprop_harmonic import (
    SIMPLE,
    coerce_input,
    linear_kprop,
    nonlin_kprop,
)
from mlp_kprop.wick import relu_wick_coef

from predict_factorized_k3_anchor import (
    full_c21_slice,
    load_weights,
    tensor,
)


def factor_c21_torch(factors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    a, b, c = factors
    return (
        (a * b) @ c.T
        + (a * c) @ b.T
        + (b * c) @ a.T
    ) / 3.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[20])
    parser.add_argument("--target-layer", type=int, default=29)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = sorted(set(args.checkpoints))
    if (
        not checkpoints
        or checkpoints[0] < 0
        or checkpoints[-1] >= args.target_layer
    ):
        raise ValueError((checkpoints, args.target_layer))
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for index in args.indices:
        started = time.perf_counter()
        weights = load_weights(
            args.weights_dir / f"mlp_{index:05d}.npy",
            dtype,
        )
        state = coerce_input(
            {
                1: torch.zeros(256, dtype=dtype),
                2: torch.eye(256, dtype=dtype),
            },
            k_max=3,
            kind=SIMPLE,
        )
        checkpoint_ranks: dict[int, int] = {}
        tail_maps: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for layer in range(args.target_layer + 1):
                pre = linear_kprop(state, weights[layer], k_max=3)
                pre_mean = tensor(pre[1])
                pre_covariance = tensor(pre[2])
                pre_variance = torch.diag(pre_covariance)
                gate = relu_wick_coef(
                    mean=pre_mean,
                    var=pre_variance,
                    k=1,
                    p=1,
                )

                # Existing checkpoint factors first undergo this layer's
                # linear map, then one expected ReLU derivative per tensor leg.
                for checkpoint in list(tail_maps):
                    tail_maps[checkpoint] = (
                        gate[:, None]
                        * (weights[layer] @ tail_maps[checkpoint])
                    )

                post = nonlin_kprop(
                    pre,
                    nonlin_wick_coef=relu_wick_coef,
                    k_max=3,
                    kind=SIMPLE,
                    use_pK=True,
                    factor=True,
                )
                state = post
                if layer in checkpoints:
                    if 3 not in post or not hasattr(post[3], "factors"):
                        raise TypeError((layer, type(post.get(3))))
                    checkpoint_ranks[layer] = post[3].factors[0].shape[1]
                    tail_maps[layer] = torch.eye(256, dtype=dtype)

            if 3 not in post or not hasattr(post[3], "factors"):
                raise TypeError(type(post.get(3)))
            target_factors = post[3].factors
            arrays: dict[str, np.ndarray | int | float] = {
                "global_index": index,
                "target_layer": args.target_layer,
                "target_mean": (
                    tensor(post[1]).detach().cpu().numpy().astype(np.float64)
                ),
                "target_covariance": (
                    tensor(post[2]).detach().cpu().numpy().astype(np.float64)
                ),
                "target_c21": (
                    full_c21_slice(post[3])
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                ),
            }
            for checkpoint in checkpoints:
                rank = checkpoint_ranks[checkpoint]
                inherited = factor_c21_torch(
                    tuple(factor[:, :rank] for factor in target_factors)
                )
                arrays[f"checkpoint{checkpoint}_rank"] = rank
                arrays[f"checkpoint{checkpoint}_inherited_c21"] = (
                    inherited.detach().cpu().numpy().astype(np.float64)
                )
                arrays[f"checkpoint{checkpoint}_tail_map"] = (
                    tail_maps[checkpoint]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )

        output = args.out_dir / f"mlp_{index:05d}.npz"
        np.savez_compressed(output, **arrays)
        print(
            f"[{index:>4}] wrote {output} "
            f"({time.perf_counter() - started:.2f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
