"""Near-full two-stream spherical randomized-QMC estimator.

Frozen design from ``results/two_nearfull_rqmc.json``:

* A: 16,384 base Sobol directions (seed 101) plus antipodes = 32,768;
* D: 15,000 base Sobol directions (seed 404) plus antipodes = 30,000;
* output: 0.4922222558500433 * A + 0.5077777441499567 * D.

Every direction has radius E[chi_256], exactly integrating Gaussian radius by
positive homogeneity. All inference numerical work uses flopscope.numpy. There
are no NumPy imports, target/MLP lookups, accounting bypasses, or manual FLOP
deductions.
"""

from __future__ import annotations

import math
from pathlib import Path

import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP


_CHUNK_BASE_ROWS = 1 << 11
_WEIGHT_A = 0.4922222558500433
_WEIGHT_D = 0.5077777441499567
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._directions_a = None
        self._directions_d = None

    def setup(self, ctx: SetupContext) -> None:
        root = (
            Path(__file__).resolve().parent
            if ctx.submission_dir is None
            else Path(ctx.submission_dir)
        )
        archive = fnp.load(str(root / "sobol_sphere_a101_d404.npz"))
        self._directions_a = archive["directions_a"]
        self._directions_d = archive["directions_d"]

    @staticmethod
    def _mean_gaussian_radius(width: int) -> float:
        return math.sqrt(2.0) * math.exp(
            math.lgamma((width + 1.0) / 2.0) - math.lgamma(width / 2.0)
        )

    def _stream_mean(
        self,
        directions: fnp.ndarray,
        weights: list[fnp.ndarray],
        width: int,
    ) -> fnp.ndarray:
        total = fnp.zeros(width, dtype=fnp.float64)
        n_base = directions.shape[0]
        for start in range(0, n_base, _CHUNK_BASE_ROWS):
            stop = min(start + _CHUNK_BASE_ROWS, n_base)
            direction = directions[start:stop, :width]
            if width != directions.shape[1]:
                norm = fnp.sqrt(
                    fnp.sum(direction * direction, axis=1, keepdims=True)
                )
                direction = direction * (
                    self._mean_gaussian_radius(width) / norm
                )
            # Interleave antipodes to reproduce the research stream order.
            activation = fnp.stack((direction, -direction), axis=1).reshape(
                (-1, width)
            )
            for weight in weights:
                activation = fnp.maximum(activation @ weight, 0.0)
            total = total + fnp.sum(
                activation.astype(fnp.float64), axis=0
            )
        return total / float(2 * n_base)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        if self._directions_a is None or self._directions_d is None:
            raise RuntimeError("setup() did not load the A/D Sobol asset")
        if mlp.width > self._directions_a.shape[1]:
            raise ValueError(
                f"asset supports width <= {self._directions_a.shape[1]}, "
                f"got {mlp.width}"
            )

        weights = [weight.astype(fnp.float32) for weight in mlp.weights]
        mean_a = self._stream_mean(
            self._directions_a, weights, mlp.width
        )
        mean_d = self._stream_mean(
            self._directions_d, weights, mlp.width
        )
        final_mean = _WEIGHT_A * mean_a + _WEIGHT_D * mean_d

        first_weight = weights[0]
        first_mean = (
            fnp.sqrt(fnp.sum(first_weight * first_weight, axis=0))
            * _INV_SQRT_2PI
        )
        rows = [fnp.zeros(mlp.width) for _ in range(mlp.depth)]
        rows[0] = first_mean
        rows[-1] = final_mean
        return fnp.stack(rows, axis=0)
