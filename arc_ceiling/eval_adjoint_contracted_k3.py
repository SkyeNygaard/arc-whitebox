"""Contract factorized K3 with two observables without materializing C21.

For a symmetric third cumulant ``K`` define

    L[A, b](K) = K : Sym(A (x) b),       A = A.T.

The late-layer C21 observable ``u.T K[i,i,j] v`` is ``L[diag(u), v]``.
If one layer transports the inherited CP carrier by ``M = diag(E relu') W``,
the exact adjoint update is

    A <- M.T A M,       b <- M.T b.

For a CP source ``Sym(a_r (x) c_r (x) d_r)`` its scalar contribution is

    1/3 sum_r [
      (a_r.T A c_r)(d_r.T b) +
      (a_r.T A d_r)(c_r.T b) +
      (c_r.T A d_r)(a_r.T b)
    ].

This script has two source modes:

* ``full``: audit-only.  Run the ordinary full factorized-K3 rollout, split
  newly appended source columns from inherited columns, and verify that the
  adjoint/Duhamel contraction reproduces the direct full-factor anchor.
* ``born``: deployable approximation.  Generate each layer's local K3 source
  from the K1/K2/K4 state, discard K3 before the next layer, then contract the
  sources backward.  This removes the quadratic-in-depth CP transport cost.

The matrix A may remain dense or be truncated once at a chosen backward
handoff.  A rank-q representation ``P diag(lam) P.T`` stays rank q exactly
under the adjoint linear pullback, so no later n-by-n eigendecompositions are
needed.
"""

from __future__ import annotations

import argparse
import gc
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

from predict_factorized_k3_anchor import (
    full_c21_slice,
    load_weights,
    tensor,
)


def carrier_factors(carrier) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return internal factors without the public property's three full clones."""
    factors = getattr(carrier, "_factors", None)
    if factors is None or len(factors) != 3:
        raise TypeError(type(carrier))
    return tuple(factors)


def source_pair_dense(
    matrix: torch.Tensor,
    vector: torch.Tensor,
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    a, c, d = factors
    if a.shape[1] == 0:
        return torch.zeros((), dtype=matrix.dtype, device=matrix.device)
    ma = matrix @ a
    mc = matrix @ c
    md = matrix @ d
    ac = torch.sum(a * mc, dim=0)
    ad = torch.sum(a * md, dim=0)
    cd = torch.sum(c * md, dim=0)
    ab = a.T @ vector
    cb = c.T @ vector
    db = d.T @ vector
    return torch.sum(ac * db + ad * cb + cd * ab) / 3.0


def source_pair_lowrank(
    basis: torch.Tensor,
    eigenvalues: torch.Tensor,
    vector: torch.Tensor,
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    a, c, d = factors
    if a.shape[1] == 0:
        return torch.zeros((), dtype=basis.dtype, device=basis.device)
    pa = basis.T @ a
    pc = basis.T @ c
    pd = basis.T @ d
    ac = torch.sum(eigenvalues[:, None] * pa * pc, dim=0)
    ad = torch.sum(eigenvalues[:, None] * pa * pd, dim=0)
    cd = torch.sum(eigenvalues[:, None] * pc * pd, dim=0)
    ab = a.T @ vector
    cb = c.T @ vector
    db = d.T @ vector
    return torch.sum(ac * db + ad * cb + cd * ab) / 3.0


def truncate_symmetric(
    matrix: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    order = torch.argsort(torch.abs(eigenvalues), descending=True)[:rank]
    return eigenvectors[:, order], eigenvalues[order]


def direct_factor_contraction(
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Contract ``K[i,i,j]`` directly, exploiting the diagonal target A."""
    a, c, d = factors
    ac = torch.sum(left[:, None] * a * c, dim=0)
    ad = torch.sum(left[:, None] * a * d, dim=0)
    cd = torch.sum(left[:, None] * c * d, dim=0)
    ab = a.T @ right
    cb = c.T @ right
    db = d.T @ right
    return torch.sum(ac * db + ad * cb + cd * ab) / 3.0


def contracted_matrix(
    matrix: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    return np.einsum("ik,ij,jk->k", left, matrix, right)


def rollout_sources(
    weights: torch.Tensor,
    target_layer: int,
    source_mode: str,
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    list[torch.Tensor],
    object,
    list[int],
]:
    width = weights.shape[-1]
    state = coerce_input(
        {
            1: torch.zeros(width, dtype=weights.dtype),
            2: torch.eye(width, dtype=weights.dtype),
        },
        k_max=3,
        kind=SIMPLE,
    )
    sources = []
    maps = []
    ranks = []
    for layer in range(target_layer + 1):
        pre = linear_kprop(state, weights[layer], k_max=3)
        pre_mean = tensor(pre[1])
        pre_variance = torch.diag(tensor(pre[2]))
        gate = relu_wick_coef(
            mean=pre_mean,
            var=pre_variance,
            k=1,
            p=1,
        )
        maps.append(gate[:, None] * weights[layer])
        inherited_rank = (
            carrier_factors(pre[3])[0].shape[1]
            if 3 in pre
            else 0
        )
        post = nonlin_kprop(
            pre,
            nonlin_wick_coef=relu_wick_coef,
            k_max=3,
            kind=SIMPLE,
            use_pK=True,
            factor=True,
        )
        factors = carrier_factors(post[3])
        if source_mode == "full":
            source = tuple(
                factor[:, inherited_rank:].detach().clone()
                for factor in factors
            )
            state = post
        elif source_mode == "born":
            if inherited_rank != 0:
                raise AssertionError(inherited_rank)
            source = tuple(factor.detach().clone() for factor in factors)
            # Keep the ordinary moment/radial state, but deliberately prevent
            # K3 from feeding the next nonlinearity or being transported.
            state = {
                order: value
                for order, value in post.items()
                if order != 3
            }
        else:
            raise ValueError(source_mode)
        sources.append(source)
        ranks.append(source[0].shape[1])
    return sources, maps, post, ranks


def adjoint_contraction(
    sources: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    maps: list[torch.Tensor],
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    rank: int | None,
    handoff: int,
) -> float:
    """Backward Duhamel sum with a dense tail and optional rank-q prefix."""
    matrix = torch.diag(left)
    vector = right.clone()
    basis = None
    eigenvalues = None
    total = torch.zeros((), dtype=left.dtype, device=left.device)

    # handoff == len(sources) means truncate before the target-layer source.
    if rank is not None and handoff == len(sources):
        basis, eigenvalues = truncate_symmetric(matrix, rank)
        matrix = None

    for layer in range(len(sources) - 1, -1, -1):
        if basis is None:
            total = total + source_pair_dense(matrix, vector, sources[layer])
            if rank is not None and layer == handoff:
                basis, eigenvalues = truncate_symmetric(matrix, rank)
                matrix = None
        else:
            total = total + source_pair_lowrank(
                basis,
                eigenvalues,
                vector,
                sources[layer],
            )

        response = maps[layer]
        vector = response.T @ vector
        if basis is None:
            matrix = response.T @ matrix @ response
        else:
            # P diag(lam) P.T -> (M.T P) diag(lam) (M.T P).T.
            basis = response.T @ basis
    return float(total)


def matmul_flops(m: int, k: int, n: int) -> int:
    """The challenge/flopscope analytic matmul convention."""
    return m * n * (2 * k - 1)


def adjoint_flop_model(
    source_ranks: list[int],
    width: int,
    controls: int,
    matrix_rank: int | None,
    handoff: int,
) -> dict[str, int]:
    """Exact arithmetic model for the operations in ``adjoint_contraction``.

    Array construction and the one-off handoff eigendecomposition are reported
    separately because the latter's flopscope price depends on dtype/version.
    """
    categories = {
        "source_projection": 0,
        "source_vector_dots": 0,
        "source_elementwise_reductions": 0,
        "adjoint_matrix_pullback": 0,
        "adjoint_vector_pullback": 0,
    }
    for layer, source_rank in enumerate(source_ranks):
        lowrank = (
            matrix_rank is not None
            and (
                handoff == len(source_ranks)
                or layer < handoff
            )
        )
        if lowrank:
            q = int(matrix_rank)
            categories["source_projection"] += (
                3 * matmul_flops(q, width, source_rank)
            )
            # Three q-wise products, each two multiplies and one q reduction;
            # then multiply by the b-dot.  Finally add three length-R vectors
            # and reduce them to a scalar.
            categories["source_elementwise_reductions"] += (
                3 * (2 * q * source_rank + (q - 1) * source_rank + source_rank)
                + 2 * source_rank
                + max(source_rank - 1, 0)
                + 1
            )
            categories["adjoint_matrix_pullback"] += matmul_flops(
                width,
                width,
                q,
            )
        else:
            categories["source_projection"] += (
                3 * matmul_flops(width, width, source_rank)
            )
            categories["source_elementwise_reductions"] += (
                3 * width * source_rank
                + 3 * (width - 1) * source_rank
                + 3 * source_rank
                + 2 * source_rank
                + max(source_rank - 1, 0)
                + 1
            )
            categories["adjoint_matrix_pullback"] += (
                matmul_flops(width, width, width)
                + matmul_flops(width, width, width)
            )
        categories["source_vector_dots"] += (
            3 * matmul_flops(source_rank, width, 1)
        )
        categories["adjoint_vector_pullback"] += matmul_flops(
            width,
            width,
            1,
        )
    categories = {
        key: value * controls
        for key, value in categories.items()
    }
    categories["total_excluding_handoff_eigh"] = sum(categories.values())
    return categories


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
    parser.add_argument("--source-mode", choices=["full", "born"], default="full")
    parser.add_argument("--direction-family", choices=["sample", "oracle"], default="sample")
    parser.add_argument("--target-layer", type=int, default=29)
    parser.add_argument("--matrix-ranks", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument(
        "--handoffs",
        type=int,
        nargs="+",
        default=[30, 29, 27, 24, 20],
        help="Dense through this source layer, then truncate. target+1 truncates immediately.",
    )
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    records = []
    width = None
    controls = None
    for index in args.indices:
        started = time.perf_counter()
        weights = load_weights(
            args.weights_dir / f"mlp_{index:05d}.npy",
            dtype,
        )
        sources, maps, post, source_ranks = rollout_sources(
            weights,
            args.target_layer,
            args.source_mode,
        )
        with np.load(args.directions_dir / f"mlp_{index:05d}.npz") as directions:
            left_np = np.asarray(
                directions[f"{args.direction_family}_left"],
                dtype=np.float64,
            )
            right_np = np.asarray(
                directions[f"{args.direction_family}_right"],
                dtype=np.float64,
            )
            oracle_c21 = np.asarray(directions["oracle_c21"], dtype=np.float64)
        with np.load(args.factorized_dir / f"mlp_{index:05d}.npz") as factorized:
            factorized_c21 = np.asarray(factorized["c21"], dtype=np.float64)

        left = torch.as_tensor(left_np, dtype=dtype)
        right = torch.as_tensor(right_np, dtype=dtype)
        width = int(weights.shape[-1])
        controls = int(left.shape[1])
        factorized_truth = contracted_matrix(factorized_c21, left_np, right_np)
        oracle_truth = contracted_matrix(oracle_c21, left_np, right_np)

        # In full mode, independently validate the saved C21 truth against the
        # direct final CP factors.  This catches diagonal-slice mistakes.
        direct_factor = None
        if args.source_mode == "full":
            final_factors = carrier_factors(post[3])
            direct_factor = np.asarray(
                [
                    direct_factor_contraction(
                        final_factors,
                        left[:, control],
                        right[:, control],
                    ).item()
                    for control in range(left.shape[1])
                ],
                dtype=np.float64,
            )

        estimates = {}
        if args.dense:
            estimates["dense"] = [
                adjoint_contraction(
                    sources,
                    maps,
                    left[:, control],
                    right[:, control],
                    rank=None,
                    handoff=-1,
                )
                for control in range(left.shape[1])
            ]
        for rank in args.matrix_ranks:
            for handoff in args.handoffs:
                if not 0 <= handoff <= args.target_layer + 1:
                    raise ValueError(handoff)
                label = f"q{rank}_h{handoff}"
                estimates[label] = [
                    adjoint_contraction(
                        sources,
                        maps,
                        left[:, control],
                        right[:, control],
                        rank=rank,
                        handoff=handoff,
                    )
                    for control in range(left.shape[1])
                ]
        estimates = {
            key: np.asarray(value, dtype=np.float64)
            for key, value in estimates.items()
        }
        record = {
            "index": index,
            "source_ranks": source_ranks,
            "factorized_truth": factorized_truth.tolist(),
            "oracle_truth": oracle_truth.tolist(),
            "direct_factor": (
                direct_factor.tolist()
                if direct_factor is not None
                else None
            ),
            "direct_factor_relative_error": (
                relative_error(direct_factor, factorized_truth)
                if direct_factor is not None
                else None
            ),
            "estimates": {
                key: value.tolist()
                for key, value in estimates.items()
            },
            "relative_errors_vs_factorized": {
                key: relative_error(value, factorized_truth)
                for key, value in estimates.items()
            },
            "relative_errors_vs_oracle": {
                key: relative_error(value, oracle_truth)
                for key, value in estimates.items()
            },
            "seconds": time.perf_counter() - started,
        }
        records.append(record)
        best = min(
            record["relative_errors_vs_factorized"],
            key=record["relative_errors_vs_factorized"].get,
        )
        print(
            f"[{index:>4}] source={args.source_mode} "
            f"best={best} "
            f"factor_err={record['relative_errors_vs_factorized'][best]:.4f} "
            f"oracle_err={record['relative_errors_vs_oracle'][best]:.4f} "
            f"({record['seconds']:.1f}s)",
            flush=True,
        )
        # Full-source audits temporarily hold roughly the entire 37,888-column
        # CP carrier twice.  Release it between networks instead of relying on
        # a later cyclic-GC pass; otherwise an eight-network audit can exceed
        # the grader-like 64 GiB host limit despite modest live-set memory.
        del sources, maps, post, weights, left, right
        gc.collect()

    labels = list(records[0]["estimates"])
    pooled_factorized = np.concatenate(
        [np.asarray(record["factorized_truth"]) for record in records]
    )
    pooled_oracle = np.concatenate(
        [np.asarray(record["oracle_truth"]) for record in records]
    )
    summary = {}
    representative_ranks = records[0]["source_ranks"]
    for label in labels:
        pooled_estimate = np.concatenate(
            [np.asarray(record["estimates"][label]) for record in records]
        )
        if label == "dense":
            matrix_rank = None
            handoff = -1
        else:
            q_text, h_text = label.split("_")
            matrix_rank = int(q_text[1:])
            handoff = int(h_text[1:])
        summary[label] = {
            "relative_error_vs_factorized": relative_error(
                pooled_estimate,
                pooled_factorized,
            ),
            "relative_error_vs_oracle": relative_error(
                pooled_estimate,
                pooled_oracle,
            ),
            "cosine_vs_factorized": float(
                np.dot(pooled_estimate, pooled_factorized)
                / max(
                    np.linalg.norm(pooled_estimate)
                    * np.linalg.norm(pooled_factorized),
                    1e-30,
                )
            ),
            "adjoint_flops_representative": adjoint_flop_model(
                representative_ranks,
                width=width,
                controls=controls,
                matrix_rank=matrix_rank,
                handoff=handoff,
            ),
        }
    output = {
        "protocol": {
            "indices": args.indices,
            "source_mode": args.source_mode,
            "direction_family": args.direction_family,
            "target_layer": args.target_layer,
            "control_rank": controls,
            "matrix_ranks": args.matrix_ranks,
            "handoffs": args.handoffs,
            "dtype": args.dtype,
            "weights_dir": str(args.weights_dir),
            "directions_dir": str(args.directions_dir),
            "factorized_dir": str(args.factorized_dir),
            "full_mode_is_oracle_audit_only": args.source_mode == "full",
            "born_mode_discards_k3_feedback": args.source_mode == "born",
        },
        "summary": summary,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    for label in sorted(
        summary,
        key=lambda key: summary[key]["relative_error_vs_factorized"],
    ):
        item = summary[label]
        flops = item["adjoint_flops_representative"][
            "total_excluding_handoff_eigh"
        ]
        print(
            f"{label:<10} factor={item['relative_error_vs_factorized']:.5f} "
            f"oracle={item['relative_error_vs_oracle']:.5f} "
            f"cos={item['cosine_vs_factorized']:.5f} "
            f"adjoint={flops / 1e9:.3f}B",
            flush=True,
        )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
