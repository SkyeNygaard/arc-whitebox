"""Fixed-rank K3 closure that preserves every repeated slice exactly.

The cheap Gaussian/Born adjoint discards incoming K3 before each layer.  This
experiment keeps a compact forward carrier instead.  After every ReLU it:

1. optionally retains the heaviest CP columns (gauge-invariant product of leg
   norms), which carry an approximation to the all-distinct tensor;
2. adds the exact difference of the ``(3,)`` and ``(2,1)`` slices as a
   rank-width tensor supported only on repeated indices.

Thus the state has fixed rank but its marginal skew and C21 slice are exact at
the compression boundary.  The next dense linear map can turn the retained
all-distinct component into new repeated structure; ``repeated_only`` tests
the rank-width zero-all-distinct endpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from mlp_kprop.diagslice import DSTensor
from mlp_kprop.factor_k3 import FactoredTensor
from mlp_kprop.flop_utils import NamedFlopCounter
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


def carrier_factors(carrier: FactoredTensor) -> tuple[torch.Tensor, ...]:
    factors = getattr(carrier, "_factors", None)
    if factors is None:
        raise TypeError(type(carrier))
    return tuple(factors)


def repeated_matched_closure(
    carrier: FactoredTensor,
    cap: int,
    adjoints: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
    ] | None = None,
) -> FactoredTensor:
    """Keep ``cap-width`` heavy columns and repair repeated slices exactly."""
    width = carrier.n
    keep_count = cap - width
    if keep_count < 0:
        raise ValueError(f"cap {cap} is below width {width}")
    factors = carrier_factors(carrier)
    if keep_count == 0:
        return FactoredTensor.from_dstensor(carrier.get_repeated())
    if adjoints is None:
        scores = torch.ones(
            factors[0].shape[1],
            dtype=factors[0].dtype,
            device=factors[0].device,
        )
        for factor in factors:
            scores = scores * torch.linalg.vector_norm(factor, dim=0)
    else:
        a, c, d = factors
        control_contributions = []
        for matrix_or_basis, vector, eigenvalues in adjoints:
            if eigenvalues is None:
                ma = matrix_or_basis @ a
                mc = matrix_or_basis @ c
                md = matrix_or_basis @ d
                ac = torch.sum(a * mc, dim=0)
                ad = torch.sum(a * md, dim=0)
                cd = torch.sum(c * md, dim=0)
            else:
                pa = matrix_or_basis.T @ a
                pc = matrix_or_basis.T @ c
                pd = matrix_or_basis.T @ d
                ac = torch.sum(eigenvalues[:, None] * pa * pc, dim=0)
                ad = torch.sum(eigenvalues[:, None] * pa * pd, dim=0)
                cd = torch.sum(eigenvalues[:, None] * pc * pd, dim=0)
            contribution = (
                ac * (d.T @ vector)
                + ad * (c.T @ vector)
                + cd * (a.T @ vector)
            ) / 3.0
            control_contributions.append(contribution)
        # A dual-weighted residual criterion: retain the columns with the
        # largest predicted effect on either final scalar anchor.
        scores = torch.sum(
            torch.stack(control_contributions).square(),
            dim=0,
        )
    keep = torch.topk(
        scores,
        k=min(keep_count, scores.numel()),
        largest=True,
        sorted=False,
    ).indices
    retained = FactoredTensor(
        n=width,
        d=3,
        factors=tuple(factor[:, keep] for factor in factors),
        device=carrier.device,
        dtype=carrier.dtype,
    )
    exact_repeated = carrier.get_repeated()
    retained_repeated = retained.get_repeated()
    delta = DSTensor(
        {
            partition: (
                exact_repeated.slices[partition]
                - retained_repeated.slices[partition]
            )
            for partition in exact_repeated.slices
        },
        n=width,
        d=3,
        device=carrier.device,
        dtype=carrier.dtype,
    )
    return retained + FactoredTensor.from_dstensor(delta)


def rollout(
    weights: torch.Tensor,
    target_layer: int,
    cap: int,
    adjoints_by_layer: list[
        list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]
    ] | None = None,
) -> tuple[dict, list[int]]:
    width = weights.shape[-1]
    state = coerce_input(
        {
            1: torch.zeros(width, dtype=weights.dtype),
            2: torch.eye(width, dtype=weights.dtype),
        },
        k_max=3,
        kind=SIMPLE,
    )
    ranks = []
    for layer in range(target_layer + 1):
        pre = linear_kprop(state, weights[layer], k_max=3)
        post = nonlin_kprop(
            pre,
            nonlin_wick_coef=relu_wick_coef,
            k_max=3,
            kind=SIMPLE,
            use_pK=True,
            factor=True,
        )
        post[3] = repeated_matched_closure(
            post[3],
            cap,
            (
                adjoints_by_layer[layer]
                if adjoints_by_layer is not None
                else None
            ),
        )
        ranks.append(carrier_factors(post[3])[0].shape[1])
        state = post
    return state, ranks


def k2_response_maps(
    weights: torch.Tensor,
    target_layer: int,
) -> list[torch.Tensor]:
    """Cheap lower-state pass used only to choose dual-aware heavy columns."""
    width = weights.shape[-1]
    state = coerce_input(
        {
            1: torch.zeros(width, dtype=weights.dtype),
            2: torch.eye(width, dtype=weights.dtype),
        },
        k_max=2,
        kind=SIMPLE,
    )
    maps = []
    for layer in range(target_layer + 1):
        pre = linear_kprop(state, weights[layer], k_max=2)
        gate = relu_wick_coef(
            mean=tensor(pre[1]),
            var=torch.diag(tensor(pre[2])),
            k=1,
            p=1,
        )
        maps.append(gate[:, None] * weights[layer])
        state = nonlin_kprop(
            pre,
            nonlin_wick_coef=relu_wick_coef,
            k_max=2,
            kind=SIMPLE,
            use_pK=True,
            factor=False,
        )
    return maps


def backward_adjoint_states(
    maps: list[torch.Tensor],
    left: torch.Tensor,
    right: torch.Tensor,
    matrix_rank: int | None,
    handoff: int,
) -> list[
    list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]
]:
    """Return target-observable adjoints at every post-activation layer."""
    controls = []
    for control in range(left.shape[1]):
        controls.append(
            (
                torch.diag(left[:, control]),
                right[:, control].clone(),
                None,
            )
        )
    states = [None] * len(maps)
    for layer in range(len(maps) - 1, -1, -1):
        states[layer] = controls
        response = maps[layer]
        next_controls = []
        for matrix_or_basis, vector, eigenvalues in controls:
            if (
                eigenvalues is None
                and matrix_rank is not None
                and layer == handoff
            ):
                matrix = 0.5 * (matrix_or_basis + matrix_or_basis.T)
                values, vectors = torch.linalg.eigh(matrix)
                order = torch.argsort(
                    torch.abs(values),
                    descending=True,
                )[:matrix_rank]
                matrix_or_basis = vectors[:, order]
                eigenvalues = values[order]
            if eigenvalues is None:
                next_matrix = (
                    response.T @ matrix_or_basis @ response
                )
            else:
                next_matrix = response.T @ matrix_or_basis
            next_controls.append(
                (
                    next_matrix,
                    response.T @ vector,
                    eigenvalues,
                )
            )
        controls = next_controls
    return states


def contraction(
    left: np.ndarray,
    matrix: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    return np.einsum("ik,ij,jk->k", left, matrix, right)


def arbitrary_center_matrix(
    c21: np.ndarray,
    mean: np.ndarray,
    second: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    """Connected-cubic numerator for pointwise centering vector ``center``."""
    delta = np.asarray(mean) - np.asarray(center)
    return (
        np.asarray(c21)
        + 2.0 * delta[:, None] * np.asarray(second)
        + np.diag(second)[:, None] * delta[None, :]
        + 2.0
        * (np.square(center) - np.square(mean))[:, None]
        * mean[None, :]
    )


def relative_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.linalg.norm(estimate - truth)
        / max(np.linalg.norm(truth), 1e-30)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(160, 168)))
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--directions-dir", type=Path, required=True)
    parser.add_argument("--factorized-dir", type=Path, required=True)
    parser.add_argument("--direction-family", choices=["sample", "oracle"], default="sample")
    parser.add_argument("--target-layer", type=int, default=29)
    parser.add_argument("--caps", type=int, nargs="+", default=[256, 384, 512, 768])
    parser.add_argument(
        "--score-mode",
        choices=["norm", "dual", "dual_sample_gates"],
        default="norm",
        help="Rank retained columns by tensor norm or final-observable contribution.",
    )
    parser.add_argument("--dual-matrix-rank", type=int, default=32)
    parser.add_argument("--dual-handoff", type=int, default=24)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--profile-flops",
        action="store_true",
        help="Use the vendor exact named counter (slower, but counts closure too).",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    records = []
    for index in args.indices:
        weights = load_weights(
            args.weights_dir / f"mlp_{index:05d}.npy",
            dtype,
        )
        with np.load(args.directions_dir / f"mlp_{index:05d}.npz") as directions:
            left = np.asarray(
                directions[f"{args.direction_family}_left"],
                dtype=np.float64,
            )
            right = np.asarray(
                directions[f"{args.direction_family}_right"],
                dtype=np.float64,
            )
            oracle_c21 = np.asarray(directions["oracle_c21"], dtype=np.float64)
            sample_mean = np.asarray(directions["sample_mean"], dtype=np.float64)
            oracle_mean = np.asarray(directions["oracle_mean"], dtype=np.float64)
            oracle_second = np.asarray(
                directions["oracle_second"],
                dtype=np.float64,
            )
            sample_gates = np.asarray(
                directions["sample_gates"],
                dtype=np.float64,
            )
        with np.load(args.factorized_dir / f"mlp_{index:05d}.npz") as factorized:
            full_c21 = np.asarray(factorized["c21"], dtype=np.float64)
            factorized_mean = np.asarray(factorized["mean"], dtype=np.float64)
            factorized_covariance = np.asarray(
                factorized["covariance"],
                dtype=np.float64,
            )
        factorized_second = (
            factorized_covariance
            + np.outer(factorized_mean, factorized_mean)
        )
        if args.score_mode in ("dual", "dual_sample_gates"):
            if args.score_mode == "dual":
                maps = k2_response_maps(weights, args.target_layer)
            else:
                maps = [
                    torch.as_tensor(
                        sample_gates[layer],
                        dtype=dtype,
                    )[:, None]
                    * weights[layer]
                    for layer in range(args.target_layer + 1)
                ]
            adjoints_by_layer = backward_adjoint_states(
                maps,
                torch.as_tensor(left, dtype=dtype),
                torch.as_tensor(right, dtype=dtype),
                args.dual_matrix_rank,
                args.dual_handoff,
            )
        else:
            adjoints_by_layer = None
        full_truth = contraction(left, full_c21, right)
        oracle_truth = contraction(left, oracle_c21, right)
        variants = {}
        for cap in args.caps:
            started = time.perf_counter()
            if args.profile_flops:
                with NamedFlopCounter(strict=False) as counter:
                    state, ranks = rollout(
                        weights,
                        args.target_layer,
                        cap,
                        adjoints_by_layer,
                    )
                flops = counter.total()
                flop_breakdown = counter.flop_dict()
            else:
                state, ranks = rollout(
                    weights,
                    args.target_layer,
                    cap,
                    adjoints_by_layer,
                )
                flops = None
                flop_breakdown = None
            c21 = (
                full_c21_slice(state[3])
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            predicted_mean = (
                tensor(state[1])
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            predicted_covariance = (
                tensor(state[2])
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            predicted_second = (
                predicted_covariance
                + np.outer(predicted_mean, predicted_mean)
            )
            estimate = contraction(left, c21, right)
            variants[f"cap{cap}"] = {
                "estimate": estimate.tolist(),
                "predicted_mean": predicted_mean.tolist(),
                "sample_center_contraction": contraction(
                    left,
                    arbitrary_center_matrix(
                        c21,
                        predicted_mean,
                        predicted_second,
                        sample_mean,
                    ),
                    right,
                ).tolist(),
                "factorized_center_contraction": contraction(
                    left,
                    arbitrary_center_matrix(
                        c21,
                        predicted_mean,
                        predicted_second,
                        factorized_mean,
                    ),
                    right,
                ).tolist(),
                "relative_error_vs_full": relative_error(estimate, full_truth),
                "relative_error_vs_oracle": relative_error(estimate, oracle_truth),
                "ranks": ranks,
                "seconds": time.perf_counter() - started,
                "vendor_flops": flops,
                "vendor_flop_breakdown": flop_breakdown,
            }
            print(
                f"[{index:>4}] cap={cap:<4} "
                f"full={variants[f'cap{cap}']['relative_error_vs_full']:.4f} "
                f"oracle={variants[f'cap{cap}']['relative_error_vs_oracle']:.4f} "
                f"time={variants[f'cap{cap}']['seconds']:.2f}s",
                flush=True,
            )
        records.append(
            {
                "index": index,
                "full_truth": full_truth.tolist(),
                "oracle_truth": oracle_truth.tolist(),
                "sample_mean": sample_mean.tolist(),
                "factorized_mean": factorized_mean.tolist(),
                "full_sample_center_contraction": contraction(
                    left,
                    arbitrary_center_matrix(
                        full_c21,
                        factorized_mean,
                        factorized_second,
                        sample_mean,
                    ),
                    right,
                ).tolist(),
                "oracle_sample_center_contraction": contraction(
                    left,
                    arbitrary_center_matrix(
                        oracle_c21,
                        oracle_mean,
                        oracle_second,
                        sample_mean,
                    ),
                    right,
                ).tolist(),
                "oracle_factorized_center_contraction": contraction(
                    left,
                    arbitrary_center_matrix(
                        oracle_c21,
                        oracle_mean,
                        oracle_second,
                        factorized_mean,
                    ),
                    right,
                ).tolist(),
                "variants": variants,
            }
        )

    labels = list(records[0]["variants"])
    pooled_full = np.concatenate(
        [np.asarray(record["full_truth"]) for record in records]
    )
    pooled_oracle = np.concatenate(
        [np.asarray(record["oracle_truth"]) for record in records]
    )
    summary = {}
    for label in labels:
        estimate = np.concatenate(
            [
                np.asarray(record["variants"][label]["estimate"])
                for record in records
            ]
        )
        full_scale = float(np.dot(estimate, pooled_full) / np.dot(estimate, estimate))
        oracle_scale = float(
            np.dot(estimate, pooled_oracle) / np.dot(estimate, estimate)
        )
        summary[label] = {
            "relative_error_vs_full": relative_error(estimate, pooled_full),
            "optimal_scale_to_full": full_scale,
            "scaled_relative_error_vs_full": relative_error(
                full_scale * estimate,
                pooled_full,
            ),
            "relative_error_vs_oracle": relative_error(estimate, pooled_oracle),
            "optimal_scale_to_oracle": oracle_scale,
            "scaled_relative_error_vs_oracle": relative_error(
                oracle_scale * estimate,
                pooled_oracle,
            ),
            "cosine_vs_oracle": float(
                np.dot(estimate, pooled_oracle)
                / max(
                    np.linalg.norm(estimate) * np.linalg.norm(pooled_oracle),
                    1e-30,
                )
            ),
        }
    output = {
        "protocol": {
            "indices": args.indices,
            "target_layer": args.target_layer,
            "direction_family": args.direction_family,
            "caps": args.caps,
            "score_mode": args.score_mode,
            "dual_matrix_rank": args.dual_matrix_rank,
            "dual_handoff": args.dual_handoff,
            "closure": (
                "retain cap-width CP columns by product of leg norms; "
                "repair (3,) and (2,1) slices exactly"
            ),
            "dtype": args.dtype,
            "profile_flops": args.profile_flops,
        },
        "summary": summary,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
