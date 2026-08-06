"""Adjoint evaluation of selected contractions of a factorized K3 rollout.

For a symmetric third cumulant ``K`` and a probe

    q(M, v) = sum_abc M[a,b] v[c] K[a,b,c],

the affine/expected-gate transport ``K_out = T^⊗3 K_in + S`` has adjoint

    M_in = T.T @ M_out @ T
    v_in = T.T @ v_out
    q_out = q_in + <M_out ⊗ v_out, S>.

The connected-cubic anchor ``sum_ij u[i] v[j] K[i,i,j]`` corresponds to
``M=diag(u)``.  This script validates that recurrence exactly against the full
factorized rollout, then replaces each local source by the source generated
from mean/covariance only.  Two lower-state variants are evaluated:

* frozen: mean/covariance and gates from the full factorized rollout;
* cheap: a separate k=2 rollout, with no propagated K3 state.

The exact recurrence is a validation oracle because its residual sources still
depend on the incoming full K3/K4 state.
"""

from __future__ import annotations

import argparse
import json
import math
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

from predict_factorized_k3_anchor import load_weights, tensor


def factor_tuple(state) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if 3 not in state:
        empty = torch.empty(
            (256, 0),
            dtype=tensor(state[1]).dtype,
            device=tensor(state[1]).device,
        )
        return empty, empty.clone(), empty.clone()
    return tuple(state[3]._factors)  # noqa: SLF001 - research instrumentation


def source_after_old_rank(
    post,
    old_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(factor[:, old_rank:].clone() for factor in factor_tuple(post))


def active_cp_rank(
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> int:
    if factors[0].shape[1] == 0:
        return 0
    active = torch.ones(
        factors[0].shape[1],
        dtype=torch.bool,
        device=factors[0].device,
    )
    for factor in factors:
        active &= torch.linalg.vector_norm(factor, dim=0) > 1e-30
    return int(torch.sum(active))


def lower_only_source(
    mean: torch.Tensor,
    covariance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lower = coerce_input(
        {1: mean, 2: covariance},
        k_max=3,
        kind=SIMPLE,
    )
    post = nonlin_kprop(
        lower,
        nonlin_wick_coef=relu_wick_coef,
        k_max=3,
        kind=SIMPLE,
        use_pK=True,
        factor=True,
    )
    return tuple(factor.clone() for factor in factor_tuple(post))


def transition(
    pre,
    weight: torch.Tensor,
) -> torch.Tensor:
    mean = tensor(pre[1])
    covariance = tensor(pre[2])
    variance = torch.diag(covariance)
    gate = relu_wick_coef(
        mean=mean,
        var=variance,
        k=1,
        p=1,
    )
    return gate[:, None] * weight


def cp_probe_contraction(
    matrix: torch.Tensor,
    vector: torch.Tensor,
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Contract ``M⊗v`` against a symmetric CP third-order tensor."""
    a, b, c = factors
    if a.shape[1] == 0:
        return torch.zeros((), dtype=matrix.dtype, device=matrix.device)
    ma = matrix @ a
    mb = matrix @ b
    # M is symmetric. Sym(A⊗B⊗C) gives three distinct paired terms.
    ab = torch.sum(ma * b, dim=0) * (vector @ c)
    ac = torch.sum(ma * c, dim=0) * (vector @ b)
    bc = torch.sum(mb * c, dim=0) * (vector @ a)
    return torch.sum(ab + ac + bc) / 3.0


def dual_contract(
    transitions: list[torch.Tensor],
    sources: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    values = []
    for column in range(left.shape[1]):
        matrix = torch.diag(
            torch.as_tensor(
                left[:, column],
                dtype=transitions[0].dtype,
            )
        )
        vector = torch.as_tensor(
            right[:, column],
            dtype=transitions[0].dtype,
        )
        total = torch.zeros((), dtype=matrix.dtype)
        for layer in reversed(range(len(transitions))):
            total = total + cp_probe_contraction(
                matrix,
                vector,
                sources[layer],
            )
            transform = transitions[layer]
            matrix = transform.T @ matrix @ transform
            vector = transform.T @ vector
        values.append(float(total))
    return np.asarray(values, dtype=np.float64)


def lowrank_cp_probe_contraction(
    basis: torch.Tensor,
    eigenvalues: torch.Tensor,
    vector: torch.Tensor,
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Contract when ``M = basis @ diag(eigenvalues) @ basis.T``."""
    a, b, c = factors
    if a.shape[1] == 0:
        return torch.zeros((), dtype=basis.dtype, device=basis.device)
    qa = basis.T @ a
    qb = basis.T @ b
    qc = basis.T @ c
    weighted = eigenvalues[:, None]
    ab = torch.sum(weighted * qa * qb, dim=0) * (vector @ c)
    ac = torch.sum(weighted * qa * qc, dim=0) * (vector @ b)
    bc = torch.sum(weighted * qb * qc, dim=0) * (vector @ a)
    return torch.sum(ab + ac + bc) / 3.0


def dual_contract_lowrank_probe(
    transitions: list[torch.Tensor],
    sources: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    left: np.ndarray,
    right: np.ndarray,
    probe_rank: int,
) -> np.ndarray:
    """Best coordinate-rank truncation of the final diagonal probe."""
    values = []
    width = left.shape[0]
    for column in range(left.shape[1]):
        chosen = np.argsort(np.abs(left[:, column]))[-probe_rank:]
        basis = torch.zeros(
            (width, probe_rank),
            dtype=transitions[0].dtype,
        )
        basis[chosen, np.arange(probe_rank)] = 1.0
        eigenvalues = torch.as_tensor(
            left[chosen, column],
            dtype=transitions[0].dtype,
        )
        vector = torch.as_tensor(
            right[:, column],
            dtype=transitions[0].dtype,
        )
        total = torch.zeros((), dtype=basis.dtype)
        for layer in reversed(range(len(transitions))):
            total = total + lowrank_cp_probe_contraction(
                basis,
                eigenvalues,
                vector,
                sources[layer],
            )
            transform = transitions[layer]
            basis = transform.T @ basis
            vector = transform.T @ vector
        values.append(float(total))
    return np.asarray(values, dtype=np.float64)


def direct_factor_contraction(
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    identity = torch.eye(factors[0].shape[0], dtype=factors[0].dtype)
    values = []
    for column in range(left.shape[1]):
        matrix = torch.diag(
            torch.as_tensor(left[:, column], dtype=identity.dtype)
        )
        vector = torch.as_tensor(right[:, column], dtype=identity.dtype)
        values.append(
            float(cp_probe_contraction(matrix, vector, factors))
        )
    return np.asarray(values, dtype=np.float64)


def metric(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    scale = float(
        np.sum(predicted * target)
        / max(np.sum(np.square(predicted)), 1e-30)
    )
    return {
        "relative_error": float(
            np.linalg.norm(predicted - target)
            / max(np.linalg.norm(target), 1e-30)
        ),
        "cosine": float(
            np.sum(predicted * target)
            / max(
                np.linalg.norm(predicted) * np.linalg.norm(target),
                1e-30,
            )
        ),
        "optimal_scale": scale,
        "scaled_relative_error": float(
            np.linalg.norm(scale * predicted - target)
            / max(np.linalg.norm(target), 1e-30)
        ),
    }


def generic_dual_flops(
    width: int,
    layers: int,
    ranks: int,
    total_source_rank: int,
) -> int:
    """Dense generic implementation, counting multiply and add separately."""
    # Per probe/layer: T.T@M and result@T (4d^3), plus T.T@v (2d^2).
    adjoint = ranks * layers * (
        4 * width**3 + 2 * width**2
    )
    # Per source column/probe: M@A and M@B (4d^2), paired reductions
    # and vector dots (~9d). M@C is unnecessary in the implemented formula.
    source = ranks * total_source_rank * (
        4 * width**2 + 9 * width
    )
    return adjoint + source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(160, 168)))
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=29)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument(
        "--probe-ranks",
        type=int,
        nargs="*",
        default=[],
        help="Optional low-rank truncations of the final diagonal M probe.",
    )
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    records = []

    for index in args.indices:
        started = time.perf_counter()
        weights = load_weights(
            args.weights_dir / f"mlp_{index:05d}.npy",
            dtype,
        )
        full_state = coerce_input(
            {
                1: torch.zeros(256, dtype=dtype),
                2: torch.eye(256, dtype=dtype),
            },
            k_max=3,
            kind=SIMPLE,
        )
        cheap_state = coerce_input(
            {
                1: torch.zeros(256, dtype=dtype),
                2: torch.eye(256, dtype=dtype),
            },
            k_max=2,
            kind=SIMPLE,
        )
        exact_transitions = []
        cheap_transitions = []
        exact_sources = []
        frozen_sources = []
        cheap_sources = []
        source_ranks = {"exact": [], "frozen": [], "cheap": []}
        active_source_ranks = {"exact": [], "frozen": [], "cheap": []}

        with torch.no_grad():
            for layer in range(args.layer + 1):
                pre_full = linear_kprop(
                    full_state,
                    weights[layer],
                    k_max=3,
                )
                old_rank = factor_tuple(pre_full)[0].shape[1]
                post_full = nonlin_kprop(
                    pre_full,
                    nonlin_wick_coef=relu_wick_coef,
                    k_max=3,
                    kind=SIMPLE,
                    use_pK=True,
                    factor=True,
                )
                exact_source = source_after_old_rank(post_full, old_rank)
                pre_full_mean = tensor(pre_full[1])
                pre_full_covariance = tensor(pre_full[2])
                frozen_source = lower_only_source(
                    pre_full_mean,
                    pre_full_covariance,
                )
                exact_transitions.append(
                    transition(pre_full, weights[layer])
                )
                exact_sources.append(exact_source)
                frozen_sources.append(frozen_source)
                source_ranks["exact"].append(exact_source[0].shape[1])
                source_ranks["frozen"].append(frozen_source[0].shape[1])
                active_source_ranks["exact"].append(
                    active_cp_rank(exact_source)
                )
                active_source_ranks["frozen"].append(
                    active_cp_rank(frozen_source)
                )
                full_state = post_full

                pre_cheap = linear_kprop(
                    cheap_state,
                    weights[layer],
                    k_max=2,
                )
                pre_cheap_mean = tensor(pre_cheap[1])
                pre_cheap_covariance = tensor(pre_cheap[2])
                cheap_source = lower_only_source(
                    pre_cheap_mean,
                    pre_cheap_covariance,
                )
                cheap_transitions.append(
                    transition(pre_cheap, weights[layer])
                )
                cheap_sources.append(cheap_source)
                source_ranks["cheap"].append(cheap_source[0].shape[1])
                active_source_ranks["cheap"].append(
                    active_cp_rank(cheap_source)
                )
                cheap_state = nonlin_kprop(
                    pre_cheap,
                    nonlin_wick_coef=relu_wick_coef,
                    k_max=2,
                    kind=SIMPLE,
                    use_pK=True,
                    factor=False,
                )

        full_c21 = (
            full_state[3]
            .get_dslice((2, 1))
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        with np.load(
            args.artifact_dir / f"mlp_{index:05d}.npz"
        ) as artifact:
            artifact_c21 = np.asarray(artifact["c21"], dtype=np.float64)
        u, _, vt = np.linalg.svd(artifact_c21, full_matrices=False)
        left = u[:, : args.rank]
        right = vt[: args.rank].T
        target = np.einsum(
            "ik,ij,jk->k",
            left,
            artifact_c21,
            right,
        )
        direct_cp = direct_factor_contraction(
            factor_tuple(full_state),
            left,
            right,
        )
        exact_dual = dual_contract(
            exact_transitions,
            exact_sources,
            left,
            right,
        )
        frozen_dual = dual_contract(
            exact_transitions,
            frozen_sources,
            left,
            right,
        )
        cheap_dual = dual_contract(
            cheap_transitions,
            cheap_sources,
            left,
            right,
        )
        cheap_lowrank = {
            probe_rank: dual_contract_lowrank_probe(
                cheap_transitions,
                cheap_sources,
                left,
                right,
                probe_rank,
            )
            for probe_rank in args.probe_ranks
        }
        record = {
            "index": index,
            "target": target.tolist(),
            "direct_cp": direct_cp.tolist(),
            "exact_dual": exact_dual.tolist(),
            "frozen_lower_dual": frozen_dual.tolist(),
            "cheap_lower_dual": cheap_dual.tolist(),
            "cheap_lower_dual_lowrank": {
                str(probe_rank): value.tolist()
                for probe_rank, value in cheap_lowrank.items()
            },
            "artifact_full_c21_relative_error": float(
                np.linalg.norm(full_c21 - artifact_c21)
                / max(np.linalg.norm(artifact_c21), 1e-30)
            ),
            "exact_dual_vs_direct_cp_relative_error": float(
                np.linalg.norm(exact_dual - direct_cp)
                / max(np.linalg.norm(direct_cp), 1e-30)
            ),
            "source_ranks": source_ranks,
            "active_source_ranks": active_source_ranks,
            "seconds": time.perf_counter() - started,
        }
        records.append(record)
        print(
            f"[{index}] exact={metric(exact_dual, target)['relative_error']:.2e} "
            f"frozen={metric(frozen_dual, target)['relative_error']:.3f} "
            f"cheap={metric(cheap_dual, target)['relative_error']:.3f} "
            f"active-ranks(ex/fr/ch)="
            f"{sum(active_source_ranks['exact'])}/"
            f"{sum(active_source_ranks['frozen'])}/"
            f"{sum(active_source_ranks['cheap'])} raw="
            f"{sum(source_ranks['exact'])}/"
            f"{sum(source_ranks['frozen'])}/"
            f"{sum(source_ranks['cheap'])} "
            f"({record['seconds']:.1f}s)",
            flush=True,
        )

    pooled_target = np.asarray(
        [record["target"] for record in records],
        dtype=np.float64,
    )
    summary = {}
    for label in (
        "direct_cp",
        "exact_dual",
        "frozen_lower_dual",
        "cheap_lower_dual",
    ):
        predicted = np.asarray(
            [record[label] for record in records],
            dtype=np.float64,
        )
        summary[label] = metric(predicted, pooled_target)
    for probe_rank in args.probe_ranks:
        predicted = np.asarray(
            [
                record["cheap_lower_dual_lowrank"][str(probe_rank)]
                for record in records
            ],
            dtype=np.float64,
        )
        summary[f"cheap_lower_dual_probe_rank{probe_rank}"] = metric(
            predicted,
            pooled_target,
        )

    source_rank_summary = {}
    for kind in ("exact", "frozen", "cheap"):
        totals = np.asarray(
            [
                sum(record["source_ranks"][kind])
                for record in records
            ],
            dtype=np.float64,
        )
        source_rank_summary[kind] = {
            "mean_total_rank": float(np.mean(totals)),
            "min_total_rank": int(np.min(totals)),
            "max_total_rank": int(np.max(totals)),
            "generic_dual_flops_rank2": generic_dual_flops(
                256,
                args.layer + 1,
                args.rank,
                int(round(np.mean(totals))),
            ),
            "mean_active_total_rank": float(
                np.mean(
                    [
                        sum(record["active_source_ranks"][kind])
                        for record in records
                    ]
                )
            ),
            "active_rank_generic_dual_flops_rank2": generic_dual_flops(
                256,
                args.layer + 1,
                args.rank,
                int(
                    round(
                        np.mean(
                            [
                                sum(record["active_source_ranks"][kind])
                                for record in records
                            ]
                        )
                    )
                ),
            ),
        }
    output = {
        "protocol": {
            **vars(args),
            "weights_dir": str(args.weights_dir),
            "artifact_dir": str(args.artifact_dir),
            "out": str(args.out),
            "directions": "top SVD directions of full factorized C21",
        },
        "summary": summary,
        "source_rank_summary": source_rank_summary,
        "records": records,
    }
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(json.dumps(source_rank_summary, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
