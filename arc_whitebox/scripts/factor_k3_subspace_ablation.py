"""Test shared neuron-subspace structure in the factorized K3 carrier.

This is a deterministic companion to ``factor_k3_resampling_ablation.py``.
It deliberately leaves the reference package untouched.

The CP representation has a gauge freedom: a column can be multiplied on one
leg and divided on another without changing the tensor.  Before estimating a
shared covariance, this script balances every CP column so all three leg norms
equal the geometric mean.  The resulting covariance and its eigenspectrum are
therefore invariant to that arbitrary per-leg scaling.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


VENDOR = (
    Path(__file__).resolve().parents[1]
    / "vendor"
    / "mlp_cumulant_propagation"
    / "src"
)
sys.path.insert(0, str(VENDOR))

from mlp_kprop.factor_k3 import FactoredTensor  # noqa: E402
from mlp_kprop.kprop_harmonic import (  # noqa: E402
    AUGMENT,
    SIMPLE,
    coerce_input,
    linear_kprop,
    nonlin_kprop,
)
from mlp_kprop.wick import relu_wick_coef  # noqa: E402


@dataclass
class Projection:
    carrier: FactoredTensor
    basis: torch.Tensor
    eigenvalues: torch.Tensor

    @property
    def energy_fraction(self) -> float:
        return float(self.eigenvalues[: self.basis.shape[1]].sum() / self.eigenvalues.sum())


def balanced_factors(ft: FactoredTensor) -> tuple[torch.Tensor, ...]:
    """Gauge-balance CP columns without changing their represented tensor."""
    factors = ft.factors
    norms = torch.stack(
        [torch.linalg.vector_norm(factor, dim=0) for factor in factors]
    )
    tiny = torch.finfo(norms.dtype).tiny
    geometric = torch.exp(torch.log(norms.clamp_min(tiny)).mean(dim=0))
    balanced = []
    for leg, factor in enumerate(factors):
        scale = torch.where(
            norms[leg] > 0,
            geometric / norms[leg].clamp_min(tiny),
            torch.zeros_like(geometric),
        )
        balanced.append(factor * scale[None, :])
    return tuple(balanced)


def shared_projection(
    ft: FactoredTensor, rank: int, residual_scale: float = 0.0
) -> Projection:
    """Project all three legs onto a gauge-invariant shared top eigenspace."""
    factors = balanced_factors(ft)
    covariance = sum(factor @ factor.T for factor in factors)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    basis = eigenvectors[:, order[:rank]]
    projected = []
    for factor in factors:
        in_subspace = basis @ (basis.T @ factor)
        projected.append(
            in_subspace + residual_scale * (factor - in_subspace)
        )
    return Projection(
        carrier=FactoredTensor(
            n=ft.n,
            d=ft.d,
            factors=tuple(projected),
            device=ft.device,
            dtype=ft.dtype,
        ),
        basis=basis,
        eigenvalues=eigenvalues,
    )


def cubic_probe_error(
    exact: FactoredTensor,
    approx: FactoredTensor,
    *,
    probes: int,
    seed: int,
) -> float:
    """Relative RMS error of random cubic tensor contractions."""
    generator = torch.Generator(device=exact.device).manual_seed(seed)
    directions = torch.randn(
        exact.n,
        probes,
        generator=generator,
        dtype=exact.dtype,
        device=exact.device,
    ) / math.sqrt(exact.n)

    def contract(ft: FactoredTensor) -> torch.Tensor:
        a, b, c = ft.factors
        # Symmetrization is irrelevant when all three external vectors agree.
        return ((a.T @ directions) * (b.T @ directions) * (c.T @ directions)).sum(dim=0)

    truth = contract(exact)
    estimate = contract(approx)
    return float(
        torch.sqrt((estimate - truth).square().mean() / truth.square().mean())
    )


def load_official(path: str, dtype: torch.dtype) -> tuple[list[torch.Tensor], np.ndarray]:
    data = np.load(path)
    weights = [
        torch.as_tensor(weight.T, dtype=dtype)
        for weight in np.asarray(data["weights"])
    ]
    return weights, np.asarray(data["means"])[-1]


KINDS = {"simple": SIMPLE, "augment": AUGMENT}


def initial_tower(width: int, dtype: torch.dtype, kind) -> dict:
    return coerce_input(
        {
            1: torch.zeros(width, dtype=dtype),
            2: torch.eye(width, dtype=dtype),
        },
        k_max=3,
        kind=kind,
    )


def run(
    weights: list[torch.Tensor],
    *,
    projection_rank: int | None,
    residual_scale: float = 0.0,
    kind=SIMPLE,
) -> tuple[torch.Tensor, FactoredTensor, list[float], list[int], float]:
    width = weights[0].shape[0]
    tower = initial_tower(width, weights[0].dtype, kind)
    captures: list[float] = []
    ranks: list[int] = []
    start = time.perf_counter()
    for weight in weights:
        tower = linear_kprop(
            tower,
            weight,
            k_max=3,
            set_metric=torch.full((width,), 2.0, dtype=weight.dtype),
        )
        tower = nonlin_kprop(
            tower,
            nonlin_wick_coef=relu_wick_coef,
            k_max=3,
            kind=kind,
            factor=True,
        )
        carrier = tower[3]
        assert isinstance(carrier, FactoredTensor)
        ranks.append(carrier.factors[0].shape[1])
        if projection_rank is not None:
            projection = shared_projection(
                carrier, projection_rank, residual_scale=residual_scale
            )
            captures.append(projection.energy_fraction)
            tower[3] = projection.carrier
    elapsed = time.perf_counter() - start
    return tower[1].core.detach().clone(), tower[3], captures, ranks, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("official_npz")
    parser.add_argument("--ranks", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument(
        "--residual-scales",
        nargs="+",
        type=float,
        default=[0.0],
        help="Multiplier on each leg's component outside the top subspace.",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--kind", choices=KINDS, default="simple")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--probes", type=int, default=256)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.set_default_dtype(dtype)
    weights, target = load_official(args.official_npz, dtype)
    kind = KINDS[args.kind]

    exact_mean, exact_carrier, _, exact_ranks, exact_seconds = run(
        weights, projection_rank=None, kind=kind
    )
    print(
        json.dumps(
            {
                "type": "exact",
                "seconds": exact_seconds,
                "final_carrier_rank": exact_ranks[-1],
                "target_mse": float(
                    np.mean((exact_mean.cpu().numpy() - target) ** 2)
                ),
            }
        ),
        flush=True,
    )

    # First inspect the exact final carrier without compounding projection
    # errors across layers.
    for rank in args.ranks:
        start = time.perf_counter()
        final_projection = shared_projection(exact_carrier, rank)
        diagnostic_seconds = time.perf_counter() - start
        print(
            json.dumps(
                {
                    "type": "exact_final_carrier_projection",
                    "subspace_rank": rank,
                    "shared_factor_energy_fraction": final_projection.energy_fraction,
                    "cubic_probe_relative_rmse": cubic_probe_error(
                        exact_carrier,
                        final_projection.carrier,
                        probes=args.probes,
                        seed=2026,
                    ),
                    "projection_seconds": diagnostic_seconds,
                }
            ),
            flush=True,
        )

    # Then project after every nonlinearity and measure the quantity that
    # matters: the final activation mean.
    for rank in args.ranks:
        for residual_scale in args.residual_scales:
            mean, carrier, captures, carrier_ranks, seconds = run(
                weights,
                projection_rank=rank,
                residual_scale=residual_scale,
                kind=kind,
            )
            print(
                json.dumps(
                    {
                        "type": "project_every_layer",
                        "subspace_rank": rank,
                        "residual_scale": residual_scale,
                        "seconds": seconds,
                        "mse_vs_exact_mean": float(
                            (mean - exact_mean).square().mean()
                        ),
                        "target_mse": float(
                            np.mean((mean.cpu().numpy() - target) ** 2)
                        ),
                        "capture_min": min(captures),
                        "capture_mean": float(np.mean(captures)),
                        "capture_final": captures[-1],
                        "final_cp_columns": carrier_ranks[-1],
                        "final_cubic_probe_relative_rmse_vs_exact_path": cubic_probe_error(
                            exact_carrier,
                            carrier,
                            probes=args.probes,
                            seed=2026,
                        ),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
