"""Export factorized-K3 states at several depths in one cumulative rollout.

This is a research-only companion to ``predict_factorized_k3_anchor.py``.
Running the single-layer exporter independently at six depths repeats almost
all propagation work.  Here each network is propagated once to the deepest
requested layer and snapshots are written in the same per-layer format.
"""

from __future__ import annotations

import argparse
import json
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

from predict_factorized_k3_anchor import full_c21_slice, load_weights, tensor


def state_arrays(state) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = tensor(state[1]).detach().cpu().numpy().astype(np.float64)
    covariance = tensor(state[2]).detach().cpu().numpy().astype(np.float64)
    if 3 in state:
        c21 = (
            full_c21_slice(state[3])
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    else:
        c21 = np.zeros((256, 256), dtype=np.float64)
    return mean, covariance, c21


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[12, 16, 20, 24, 27, 29],
    )
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--timing-out", type=Path)
    parser.add_argument(
        "--save-k3-factors",
        action="store_true",
        help="Save post-activation factor matrices for checkpoint defect transport.",
    )
    args = parser.parse_args()

    layers = sorted(set(args.layers))
    if not layers or layers[0] < 0 or layers[-1] >= 32:
        raise ValueError(layers)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    for layer in layers:
        (args.out_root / f"layer{layer}").mkdir(parents=True, exist_ok=True)

    timing_records = []
    for index in args.indices:
        network_started = time.perf_counter()
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
        layer_timings = {}
        with torch.no_grad():
            for layer in range(layers[-1] + 1):
                layer_started = time.perf_counter()
                pre = linear_kprop(state, weights[layer], k_max=3)
                post = nonlin_kprop(
                    pre,
                    nonlin_wick_coef=relu_wick_coef,
                    k_max=3,
                    kind=SIMPLE,
                    use_pK=True,
                    factor=True,
                )
                state = post
                layer_timings[layer] = {
                    "incremental_seconds": time.perf_counter() - layer_started,
                    "cumulative_seconds": time.perf_counter() - network_started,
                }
                if layer not in layers:
                    continue

                pre_mean, pre_covariance, pre_c21 = state_arrays(pre)
                mean, covariance, c21 = state_arrays(post)
                output = (
                    args.out_root
                    / f"layer{layer}"
                    / f"mlp_{index:05d}.npz"
                )
                arrays = dict(
                    global_index=index,
                    layer=layer,
                    mean=mean,
                    covariance=covariance,
                    c21=c21,
                    pre_mean=pre_mean,
                    pre_covariance=pre_covariance,
                    pre_c21=pre_c21,
                    cumulative_seconds=layer_timings[layer][
                        "cumulative_seconds"
                    ],
                )
                if args.save_k3_factors and 3 in post:
                    if not hasattr(post[3], "factors"):
                        raise TypeError(
                            f"layer {layer} K3 has no factor representation: "
                            f"{type(post[3])}"
                        )
                    for factor_index, factor in enumerate(post[3].factors):
                        arrays[f"k3_factor{factor_index}"] = (
                            factor.detach()
                            .cpu()
                            .numpy()
                            .astype(np.float64)
                        )
                np.savez_compressed(output, **arrays)
                print(
                    f"[{index:>4}] layer={layer:>2} "
                    f"t={layer_timings[layer]['cumulative_seconds']:.2f}s "
                    f"wrote {output}",
                    flush=True,
                )

        timing_records.append(
            {
                "index": index,
                "dtype": args.dtype,
                "layers": {
                    str(layer): layer_timings[layer]
                    for layer in range(layers[-1] + 1)
                },
                "total_seconds": time.perf_counter() - network_started,
            }
        )

    timing_out = args.timing_out or args.out_root / "timings.json"
    timing_out.write_text(
        json.dumps(
            {
                "indices": args.indices,
                "snapshot_layers": layers,
                "dtype": args.dtype,
                "records": timing_records,
            },
            indent=2,
        )
    )
    print(f"wrote {timing_out}", flush=True)


if __name__ == "__main__":
    main()
