"""Ablate unbiased rank capping for ARC's factorized third cumulant.

The reference K=3 propagator represents the symmetric third-cumulant tensor as

    Sym(sum_r A[:, r] x B[:, r] x C[:, r]).

Its rank grows by O(width) per layer.  This script caps that rank without
changing the vendor implementation.  Each selected column is importance
weighted, so the capped tensor is unbiased conditional on the current tower.

The ``*_matched`` schemes additionally add a rank-``width`` correction whose
tensor is supported only on repeated indices.  This makes the (3) and (2, 1)
diagonal slices exactly equal to the uncapped carrier while leaving the
all-distinct estimate unbiased.  These slices are heavily used by the next
nonlinear propagation step.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


VENDOR = (
    Path(__file__).resolve().parents[1]
    / "vendor"
    / "mlp_cumulant_propagation"
    / "src"
)
sys.path.insert(0, str(VENDOR))

from mlp_kprop.diagslice import DSTensor  # noqa: E402
from mlp_kprop.factor_k3 import FactoredTensor  # noqa: E402
from mlp_kprop.kprop_harmonic import (  # noqa: E402
    SIMPLE,
    coerce_input,
    linear_kprop,
    nonlin_kprop,
)
from mlp_kprop.wick import relu_wick_coef  # noqa: E402


@dataclass
class RunResult:
    prediction: torch.Tensor
    carrier: FactoredTensor
    ranks: list[int]
    seconds: float


def _column_scores(ft: FactoredTensor) -> torch.Tensor:
    """Product of leg norms, the norm of each unsymmetrized rank-one term."""
    factors = ft.factors
    scores = torch.ones(
        factors[0].shape[1], dtype=factors[0].dtype, device=factors[0].device
    )
    for factor in factors:
        scores *= torch.linalg.vector_norm(factor, dim=0)
    return scores


def _balanced_shared_basis(
    ft: FactoredTensor, rank: int
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Gauge-balance the CP carrier and return its shared top eigenspace."""
    factors = ft.factors
    norms = torch.stack(
        [torch.linalg.vector_norm(factor, dim=0) for factor in factors]
    )
    tiny = torch.finfo(norms.dtype).tiny
    geometric = torch.exp(torch.log(norms.clamp_min(tiny)).mean(dim=0))
    balanced = tuple(
        factor
        * torch.where(
            norms[leg] > 0,
            geometric / norms[leg].clamp_min(tiny),
            torch.zeros_like(geometric),
        )[None, :]
        for leg, factor in enumerate(factors)
    )
    covariance = sum(factor @ factor.T for factor in balanced)
    _, eigenvectors = torch.linalg.eigh(covariance)
    basis = eigenvectors[:, -rank:]
    return balanced, basis


def _shared_subspace_scores(ft: FactoredTensor, rank: int) -> torch.Tensor:
    """Gauge-invariant contribution scores inside a shared neuron subspace."""
    balanced, basis = _balanced_shared_basis(ft, rank)
    projected_norms = [
        torch.linalg.vector_norm(basis.T @ factor, dim=0)
        for factor in balanced
    ]
    return projected_norms[0] * projected_norms[1] * projected_norms[2]


def _sample_tail(
    factors: tuple[torch.Tensor, ...],
    indices: torch.Tensor,
    draws: int,
    generator: torch.Generator,
    residual: bool,
) -> tuple[torch.Tensor, ...]:
    """Unbiased importance estimate of the sum over ``indices``."""
    if draws <= 0 or indices.numel() == 0:
        return tuple(f[:, :0].clone() for f in factors)

    all_scores = torch.ones(
        factors[0].shape[1], dtype=factors[0].dtype, device=factors[0].device
    )
    for factor in factors:
        all_scores *= torch.linalg.vector_norm(factor, dim=0)
    scores = all_scores[indices]
    positive = scores > 0
    indices = indices[positive]
    scores = scores[positive]
    if indices.numel() == 0:
        return tuple(f[:, :0].clone() for f in factors)

    probs = scores / scores.sum()
    if residual:
        expected = draws * probs
        counts = torch.floor(expected).to(torch.int64)
        remaining = draws - int(counts.sum().item())
        if remaining:
            residual_mass = expected - counts
            residual_probs = residual_mass / residual_mass.sum()
            sampled = torch.multinomial(
                residual_probs,
                remaining,
                replacement=True,
                generator=generator,
            )
            counts.scatter_add_(
                0,
                sampled,
                torch.ones_like(sampled, dtype=counts.dtype),
            )
    else:
        sampled = torch.multinomial(
            probs, draws, replacement=True, generator=generator
        )
        counts = torch.bincount(sampled, minlength=indices.numel())

    used = counts > 0
    chosen = indices[used]
    # E[count_j / (draws * p_j)] = 1.  Put the scalar weight on one
    # factor; scaling any one leg scales the symmetric rank-one tensor.
    coef = counts[used].to(factors[0].dtype) / (draws * probs[used])
    sampled_factors = [factor[:, chosen].clone() for factor in factors]
    sampled_factors[0] *= coef[None, :]
    return tuple(sampled_factors)


def cap_factored_tensor(
    ft: FactoredTensor,
    max_rank: int,
    generator: torch.Generator,
    scheme: str,
) -> FactoredTensor:
    """Return an unbiased carrier with rank at most ``max_rank``."""
    factors = ft.factors
    rank = factors[0].shape[1]
    if rank <= max_rank:
        return ft.clone()

    matched = scheme.endswith("_matched")
    base_scheme = scheme.removesuffix("_matched")
    correction_rank = ft.n if matched else 0
    sample_budget = max_rank - correction_rank
    if sample_budget <= 0:
        raise ValueError(
            f"max_rank={max_rank} must exceed width={ft.n} for matched schemes"
        )

    all_indices = torch.arange(rank, device=ft.device)
    if base_scheme.startswith("projected_cv_"):
        subspace_rank = int(base_scheme.rsplit("_", 1)[1])
        core_rank = subspace_rank**3
        residual_draws = sample_budget - core_rank
        if residual_draws <= 0:
            raise ValueError(
                f"projected rank {subspace_rank} needs {core_rank} exact "
                f"columns, exceeding cap {sample_budget}"
            )
        balanced, basis = _balanced_shared_basis(ft, subspace_rank)
        projected = tuple(basis @ (basis.T @ factor) for factor in balanced)
        residual = tuple(
            factor - projected_factor
            for factor, projected_factor in zip(balanced, projected)
        )

        # T - P^⊗3 T is the sum of the seven products with at least one
        # residual leg. This expansion is exact and remains a CP carrier.
        residual_factors = tuple(
            torch.cat(
                [
                    (projected[leg] if bits[leg] == 0 else residual[leg])
                    for bits in (
                        (0, 0, 1),
                        (0, 1, 0),
                        (0, 1, 1),
                        (1, 0, 0),
                        (1, 0, 1),
                        (1, 1, 0),
                        (1, 1, 1),
                    )
                ],
                dim=1,
            )
            for leg in range(3)
        )
        residual_indices = torch.arange(
            residual_factors[0].shape[1], device=ft.device
        )
        sampled_factors = _sample_tail(
            residual_factors,
            residual_indices,
            residual_draws,
            generator,
            residual=True,
        )

        coordinates = tuple(basis.T @ factor for factor in balanced)
        core = torch.einsum(
            "pr,qr,sr->pqs", coordinates[0], coordinates[1], coordinates[2]
        )
        grid = torch.cartesian_prod(
            *[
                torch.arange(subspace_rank, device=ft.device)
                for _ in range(3)
            ]
        )
        exact_projected = (
            basis[:, grid[:, 0]] * core.flatten()[None, :],
            basis[:, grid[:, 1]],
            basis[:, grid[:, 2]],
        )
        capped_factors = tuple(
            torch.cat([exact, sampled], dim=1)
            for exact, sampled in zip(exact_projected, sampled_factors)
        )
        capped = FactoredTensor(
            n=ft.n,
            d=ft.d,
            factors=capped_factors,
            device=ft.device,
            dtype=ft.dtype,
        )
        if capped.factors[0].shape[1] > max_rank:
            raise AssertionError(
                f"cap produced rank {capped.factors[0].shape[1]} > {max_rank}"
            )
        return capped
    elif base_scheme == "multinomial":
        kept_factors = tuple(f[:, :0].clone() for f in factors)
        sampled_factors = _sample_tail(
            factors, all_indices, sample_budget, generator, residual=False
        )
    elif base_scheme == "residual":
        kept_factors = tuple(f[:, :0].clone() for f in factors)
        sampled_factors = _sample_tail(
            factors, all_indices, sample_budget, generator, residual=True
        )
    elif base_scheme in {
        "top_tail",
        "top_tail_25",
        "top_tail_75",
        "top_tail_90",
        "top_auto",
    }:
        scores = _column_scores(ft)
        order = torch.argsort(scores, descending=True)
        if base_scheme == "top_auto":
            # Minimize the usual importance-sampling variance upper bound
            #   (sum_{tail} ||T_j||)^2 / draws
            # over the number of exact heavy hitters. Leave at least one draw.
            sorted_scores = scores[order].to(torch.float64)
            tail_mass = torch.flip(
                torch.cumsum(torch.flip(sorted_scores, dims=(0,)), dim=0),
                dims=(0,),
            )
            possible_keep = torch.arange(
                min(sample_budget, rank), device=ft.device
            )
            draws = sample_budget - possible_keep
            objectives = (
                tail_mass[possible_keep].square()
                / draws.to(torch.float64)
            )
            keep_count = int(torch.argmin(objectives).item())
        else:
            fraction = {
                "top_tail": 0.5,
                "top_tail_25": 0.25,
                "top_tail_75": 0.75,
                "top_tail_90": 0.9,
            }[base_scheme]
            keep_count = min(int(sample_budget * fraction), rank)
        keep = order[:keep_count]
        tail = order[keep_count:]
        kept_factors = tuple(f[:, keep].clone() for f in factors)
        sampled_factors = _sample_tail(
            factors,
            tail,
            sample_budget - keep_count,
            generator,
            residual=True,
        )
    elif base_scheme.startswith("subspace_head_"):
        subspace_rank = int(base_scheme.rsplit("_", 1)[1])
        keep_count = min(int(sample_budget * 0.75), rank)
        scores = _shared_subspace_scores(ft, subspace_rank)
        order = torch.argsort(scores, descending=True)
        keep = order[:keep_count]
        tail = order[keep_count:]
        kept_factors = tuple(f[:, keep].clone() for f in factors)
        sampled_factors = _sample_tail(
            factors,
            tail,
            sample_budget - keep_count,
            generator,
            residual=True,
        )
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    capped_factors = tuple(
        torch.cat([keep, sample], dim=1)
        for keep, sample in zip(kept_factors, sampled_factors)
    )
    capped = FactoredTensor(
        n=ft.n,
        d=ft.d,
        factors=capped_factors,
        device=ft.device,
        dtype=ft.dtype,
    )

    if matched:
        exact_repeated = ft.get_repeated()
        capped_repeated = capped.get_repeated()
        delta = DSTensor(
            {
                part: exact_repeated.slices[part]
                - capped_repeated.slices[part]
                for part in exact_repeated.slices
            },
            n=ft.n,
            d=ft.d,
            device=ft.device,
            dtype=ft.dtype,
        )
        capped = capped + FactoredTensor.from_dstensor(delta)

    if capped.factors[0].shape[1] > max_rank:
        raise AssertionError(
            f"cap produced rank {capped.factors[0].shape[1]} > {max_rank}"
        )
    return capped


def propagate(
    weights: Iterable[torch.Tensor],
    *,
    cap: int | None,
    scheme: str,
    seed: int,
) -> RunResult:
    weights = list(weights)
    width = weights[0].shape[0]
    dtype = weights[0].dtype
    device = weights[0].device
    tower = coerce_input(
        {
            1: torch.zeros(width, dtype=dtype, device=device),
            2: torch.eye(width, dtype=dtype, device=device),
        },
        k_max=3,
        kind=SIMPLE,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    ranks: list[int] = []
    start = time.perf_counter()
    for layer, weight in enumerate(weights):
        tower = linear_kprop(
            tower,
            weight,
            k_max=3,
            set_metric=torch.full(
                (width,), 2.0, dtype=dtype, device=device
            ),
        )
        tower = nonlin_kprop(
            tower,
            nonlin_wick_coef=relu_wick_coef,
            k_max=3,
            kind=SIMPLE,
            factor=True,
        )
        assert isinstance(tower[3], FactoredTensor)
        # Capping after the final activation cannot affect its already-computed
        # mean, but doing so makes final carrier comparisons apples-to-apples.
        if cap is not None:
            tower[3] = cap_factored_tensor(
                tower[3], max_rank=cap, generator=generator, scheme=scheme
            )
        ranks.append(tower[3].factors[0].shape[1])
    elapsed = time.perf_counter() - start
    return RunResult(
        prediction=tower[1].core.detach().clone(),
        carrier=tower[3],
        ranks=ranks,
        seconds=elapsed,
    )


def _relative_carrier_error(
    approx: FactoredTensor, exact: FactoredTensor
) -> tuple[float, float]:
    if exact.n > 64:
        return math.nan, math.nan
    approx_dense = approx.to_tensor()
    exact_dense = exact.to_tensor()
    rel_dense = float(
        torch.linalg.vector_norm(approx_dense - exact_dense)
        / torch.linalg.vector_norm(exact_dense)
    )
    rep_num = torch.zeros((), dtype=exact.dtype, device=exact.device)
    rep_den = torch.zeros((), dtype=exact.dtype, device=exact.device)
    for part in ((3,), (2, 1)):
        delta = approx.get_dslice(part) - exact.get_dslice(part)
        rep_num += delta.square().sum()
        rep_den += exact.get_dslice(part).square().sum()
    rel_rep = float(torch.sqrt(rep_num / rep_den))
    return rel_dense, rel_rep


def _random_weights(
    width: int, depth: int, dtype: torch.dtype, seed: int
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(width, width, dtype=dtype, generator=generator)
        * math.sqrt(2.0 / width)
        for _ in range(depth)
    ]


def _load_official(path: str, dtype: torch.dtype) -> tuple[list[torch.Tensor], np.ndarray]:
    data = np.load(path)
    # Challenge data uses row-vector activations h @ W; the reference
    # propagator uses W @ h.
    weights = [
        torch.as_tensor(weight.T, dtype=dtype)
        for weight in np.asarray(data["weights"])
    ]
    return weights, np.asarray(data["means"])[-1]


def summarize(values: list[float]) -> dict[str, float | int]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {
            "mean": math.nan,
            "sd": math.nan,
            "min": math.nan,
            "max": math.nan,
            "finite": 0,
        }
    return {
        "mean": statistics.fmean(finite),
        "sd": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
        "finite": len(finite),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-npz")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--network-seed", type=int, default=1234)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--caps", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=[
            "multinomial",
            "residual",
            "top_tail",
            "residual_matched",
            "top_tail_matched",
        ],
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.set_default_dtype(dtype)
    if args.official_npz:
        weights, target = _load_official(args.official_npz, dtype)
    else:
        weights = _random_weights(
            args.width, args.depth, dtype=dtype, seed=args.network_seed
        )
        target = None

    exact = propagate(weights, cap=None, scheme="residual", seed=0)
    exact_record: dict[str, object] = {
        "type": "exact",
        "width": weights[0].shape[0],
        "depth": len(weights),
        "seconds": exact.seconds,
        "ranks": exact.ranks,
        "prediction_norm": float(torch.linalg.vector_norm(exact.prediction)),
    }
    if target is not None:
        exact_record["target_mse"] = float(
            np.mean((exact.prediction.cpu().numpy() - target) ** 2)
        )
    print(json.dumps(exact_record), flush=True)

    for cap in args.caps:
        for scheme in args.schemes:
            if scheme.endswith("_matched") and cap <= weights[0].shape[0]:
                continue
            runs: list[RunResult] = []
            for seed in range(args.seeds):
                runs.append(
                    propagate(weights, cap=cap, scheme=scheme, seed=seed)
                )
            predictions = torch.stack([run.prediction for run in runs])
            mean_prediction = predictions.mean(dim=0)
            bias2 = float((mean_prediction - exact.prediction).square().mean())
            variance = float(
                (predictions - mean_prediction[None, :]).square().mean()
            )
            per_seed_mse = [
                float((run.prediction - exact.prediction).square().mean())
                for run in runs
            ]
            dense_rel, rep_rel = zip(
                *[
                    _relative_carrier_error(run.carrier, exact.carrier)
                    for run in runs
                ]
            )
            record: dict[str, object] = {
                "type": "capped",
                "scheme": scheme,
                "cap": cap,
                "seeds": args.seeds,
                "mean_seconds": statistics.fmean(run.seconds for run in runs),
                "final_rank": runs[0].ranks[-1],
                "prediction_mse_vs_exact": summarize(per_seed_mse),
                "prediction_bias2_vs_exact": bias2,
                "prediction_variance_vs_exact": variance,
                "carrier_dense_relative_error": summarize(list(dense_rel)),
                "carrier_repeated_relative_error": summarize(list(rep_rel)),
            }
            if target is not None:
                record["target_mse"] = summarize(
                    [
                        float(
                            np.mean(
                                (run.prediction.cpu().numpy() - target) ** 2
                            )
                        )
                        for run in runs
                    ]
                )
                record["ensemble_target_mse"] = float(
                    np.mean((mean_prediction.cpu().numpy() - target) ** 2)
                )
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
