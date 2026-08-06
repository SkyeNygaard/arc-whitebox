"""Fuse K3 soft shrinkage into the next linear contraction.

For a shared projector P=UU^T and residual scale lambda,

    f' = [lambda I + (1-lambda) P] f = A f
    W f' = (W A) f.

This avoids applying A separately to every CP column.  The cheap proxy derives
U from the already-present preactivation K2 covariance rather than from an
O(n^2 * carrier_rank) covariance over K3 factors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
VENDOR = HERE.parents[0] / "vendor" / "mlp_cumulant_propagation" / "src"
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(HERE))

from factor_k3_subspace_ablation import (  # noqa: E402
    KINDS,
    initial_tower,
    load_official,
    shared_projection,
)
from mlp_kprop.factor_k3 import FactoredTensor  # noqa: E402
from mlp_kprop.kprop_harmonic import nonlin_kprop  # noqa: E402
from mlp_kprop.wick import relu_wick_coef  # noqa: E402


def fused_weight(
    weight: torch.Tensor,
    basis: torch.Tensor | None,
    residual_scale: float,
) -> torch.Tensor:
    if basis is None:
        return weight
    return (
        residual_scale * weight
        + (1.0 - residual_scale) * (weight @ basis) @ basis.T
    )


def linear_kprop_split_k3(
    tower: dict,
    weight: torch.Tensor,
    k3_weight: torch.Tensor,
) -> dict:
    """Reference linear_kprop with a distinct weight only for Factored K3."""
    metric = torch.full(
        (weight.shape[0],),
        2.0,
        dtype=weight.dtype,
        device=weight.device,
    )
    output = {}
    for degree, carrier in tower.items():
        if degree == 3:
            assert isinstance(carrier, FactoredTensor)
            output[degree] = carrier.contract_W(k3_weight)
        else:
            assert carrier.has_identity_metric()
            output[degree] = carrier.contract_W(weight, set_metric=metric)
    return output


def top_k2_basis(preactivation_tower: dict, rank: int) -> torch.Tensor:
    covariance = preactivation_tower[2].core
    covariance = (covariance + covariance.T) * 0.5
    _, eigenvectors = torch.linalg.eigh(covariance)
    return eigenvectors[:, -rank:]


def sketched_k3_basis(
    carrier: FactoredTensor,
    *,
    rank: int,
    sample_columns: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Importance-sketched gauge-balanced shared covariance.

    A CP column's three legs are rescaled to have equal norm g without changing
    their tensor product. Sampling probability is proportional to g^2, the
    trace contribution of each balanced leg. The count/(m p) correction makes
    the covariance estimate unbiased.
    """
    factors = carrier.factors
    norms = torch.stack(
        [torch.linalg.vector_norm(factor, dim=0) for factor in factors]
    )
    tiny = torch.finfo(norms.dtype).tiny
    geometric = torch.exp(torch.log(norms.clamp_min(tiny)).mean(dim=0))
    scores = geometric.square()
    positive = scores > 0
    positive_indices = torch.nonzero(positive, as_tuple=False).flatten()
    probabilities = scores[positive]
    probabilities /= probabilities.sum()
    sampled_local = torch.multinomial(
        probabilities,
        sample_columns,
        replacement=True,
        generator=generator,
    )
    counts = torch.bincount(
        sampled_local, minlength=positive_indices.numel()
    )
    used = counts > 0
    chosen = positive_indices[used]
    chosen_probabilities = probabilities[used]
    importance_sqrt = torch.sqrt(
        counts[used].to(carrier.dtype)
        / (sample_columns * chosen_probabilities)
    )
    covariance = torch.zeros(
        (carrier.n, carrier.n),
        dtype=carrier.dtype,
        device=carrier.device,
    )
    for leg, factor in enumerate(factors):
        gauge_scale = (
            geometric[chosen]
            / norms[leg, chosen].clamp_min(tiny)
        )
        sketched = factor[:, chosen] * (
            gauge_scale * importance_sqrt
        )[None, :]
        covariance += sketched @ sketched.T
    covariance = (covariance + covariance.T) * 0.5
    _, eigenvectors = torch.linalg.eigh(covariance)
    return eigenvectors[:, -rank:]


def run_fused(
    weights: list[torch.Tensor],
    *,
    rank: int,
    residual_scale: float,
    basis_source: str,
    kind,
    sketch_columns: int = 128,
    sketch_seed: int = 2026,
) -> tuple[torch.Tensor, float]:
    """Run exact, explicit, K3-basis-fused, or K2-proxy-fused propagation."""
    width = weights[0].shape[0]
    tower = initial_tower(width, weights[0].dtype, kind)
    pending_basis: torch.Tensor | None = None
    generator = torch.Generator(device=weights[0].device)
    generator.manual_seed(sketch_seed)
    start = time.perf_counter()
    for layer, weight in enumerate(weights):
        if basis_source in ("fused_k3", "fused_k2", "fused_k3_sketch"):
            k3_weight = fused_weight(
                weight, pending_basis, residual_scale
            )
        else:
            k3_weight = weight
        tower = linear_kprop_split_k3(tower, weight, k3_weight)

        if basis_source == "fused_k2" and layer < len(weights) - 1:
            # This is the preactivation covariance for the current layer. Its
            # basis will shrink postactivation K3 at the following linear step.
            next_basis = top_k2_basis(tower, rank)
        else:
            next_basis = None

        tower = nonlin_kprop(
            tower,
            nonlin_wick_coef=relu_wick_coef,
            k_max=3,
            kind=kind,
            factor=True,
        )
        carrier = tower[3]
        assert isinstance(carrier, FactoredTensor)
        if basis_source == "explicit_k3":
            projection = shared_projection(
                carrier, rank, residual_scale=residual_scale
            )
            tower[3] = projection.carrier
        elif basis_source == "fused_k3" and layer < len(weights) - 1:
            pending_basis = shared_projection(
                carrier, rank, residual_scale=0.0
            ).basis
        elif (
            basis_source == "fused_k3_sketch"
            and layer < len(weights) - 1
        ):
            pending_basis = sketched_k3_basis(
                carrier,
                rank=rank,
                sample_columns=sketch_columns,
                generator=generator,
            )
        elif basis_source == "fused_k2":
            pending_basis = next_basis
    return tower[1].core.detach().clone(), time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("official_npz")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["exact", "explicit_k3", "fused_k3", "fused_k2"],
        choices=[
            "exact",
            "explicit_k3",
            "fused_k3",
            "fused_k2",
            "fused_k3_sketch",
        ],
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--residual-scale", type=float, default=0.75)
    parser.add_argument("--kind", choices=KINDS, default="simple")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--sketch-columns", type=int, default=128)
    parser.add_argument("--sketch-seed", type=int, default=2026)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float32)
    weights, target = load_official(args.official_npz, torch.float32)
    kind = KINDS[args.kind]
    predictions: dict[str, torch.Tensor] = {}
    for source in args.sources:
        prediction, seconds = run_fused(
            weights,
            rank=args.rank,
            residual_scale=args.residual_scale,
            basis_source=source,
            kind=kind,
            sketch_columns=args.sketch_columns,
            sketch_seed=args.sketch_seed,
        )
        predictions[source] = prediction
        record = {
            "source": source,
            "seconds": seconds,
            "target_mse": float(
                np.mean((prediction.cpu().numpy() - target) ** 2)
            ),
        }
        if "explicit_k3" in predictions and source == "fused_k3":
            delta = prediction - predictions["explicit_k3"]
            record["mse_vs_explicit"] = float(delta.square().mean())
            record["max_abs_vs_explicit"] = float(delta.abs().max())
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
