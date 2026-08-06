"""Export ARC factorized-K3 post-ReLU state for a late-layer anchor test.

Run this with the `mlp_cumulant_propagation` environment.  The output contains
the predicted post-activation mean, covariance and connected M21 diagonal slice
at one layer.  A separate Kerdock experiment contracts that state against
sample-derived low-rank directions.
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


def tensor(value) -> torch.Tensor:
    if hasattr(value, "to_tensor"):
        value = value.to_tensor()
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value


def full_c21_slice(value) -> torch.Tensor:
    """Return ``K[i,i,j]`` including the all-equal ``K[i,i,i]`` diagonal.

    The harmonic/factored API stores diagonal slices by integer partition.
    ``get_dslice((2, 1))`` therefore contains only ``i != j`` entries and
    deliberately zeros its matrix diagonal; the missing all-equal entries
    live in ``get_dslice((3,))``.  Recombining both partitions reconstructs
    the ordinary M21 matrix slice.
    """
    off_diagonal = tensor(value.get_dslice((2, 1))).clone()
    marginal_third = tensor(value.get_dslice((3,)))
    if off_diagonal.ndim != 2 or off_diagonal.shape[0] != off_diagonal.shape[1]:
        raise ValueError(f"bad (2,1) slice shape {tuple(off_diagonal.shape)}")
    if marginal_third.shape != (off_diagonal.shape[0],):
        raise ValueError(f"bad (3,) slice shape {tuple(marginal_third.shape)}")
    off_diagonal.diagonal().copy_(marginal_third)
    return off_diagonal


def load_weights(path: Path, dtype: torch.dtype) -> torch.Tensor:
    weights = np.asarray(np.load(path), dtype=np.float64)
    if weights.shape != (32, 256, 256):
        raise ValueError((path, weights.shape))
    # mlp_kprop's linear convention is the transpose of the challenge's x @ W.
    return torch.as_tensor(
        weights.transpose(0, 2, 1).copy(),
        dtype=dtype,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=29)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
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
        with torch.no_grad():
            for layer in range(args.layer + 1):
                pre = linear_kprop(state, weights[layer], k_max=3)
                if layer == args.layer:
                    pre_mean = (
                        tensor(pre[1])
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    pre_covariance = (
                        tensor(pre[2])
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    if 3 in pre:
                        pre_c21 = (
                            full_c21_slice(pre[3])
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float64)
                        )
                    else:
                        pre_c21 = np.zeros((256, 256), dtype=np.float64)
                post = nonlin_kprop(
                    pre,
                    nonlin_wick_coef=relu_wick_coef,
                    k_max=3,
                    kind=SIMPLE,
                    use_pK=True,
                    factor=True,
                )
                state = post

            mean = tensor(post[1]).detach().cpu().numpy().astype(np.float64)
            covariance = tensor(post[2]).detach().cpu().numpy().astype(np.float64)
            if 3 in post:
                c21 = (
                    full_c21_slice(post[3])
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
            else:
                c21 = np.zeros((256, 256), dtype=np.float64)

        output = args.out_dir / f"mlp_{index:05d}.npz"
        np.savez_compressed(
            output,
            global_index=index,
            layer=args.layer,
            mean=mean,
            covariance=covariance,
            c21=c21,
            pre_mean=pre_mean,
            pre_covariance=pre_covariance,
            pre_c21=pre_c21,
        )
        print(
            f"[{index:>4}] wrote {output} "
            f"({time.perf_counter() - started:.2f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
