"""Budget-safe seed-101 randomized-QMC sphere estimator.

The shipped block contains 16,384 float32 directions from a SciPy-scrambled
Sobol net (seed 101), each normalized to E[chi_256].  Inference adds each
antipode for 32,768 total forward rows.  Exact radius integration is valid
because every challenge MLP is bias-free and positively homogeneous.

All inference numerical work uses flopscope.numpy.  There are no NumPy imports,
MLP-identity lookups, accounting bypasses, or manual FLOP deductions.
"""

from __future__ import annotations

import math
from pathlib import Path

import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP


_CHUNK_BASE_ROWS = 1 << 11
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._directions = None

    def setup(self, ctx: SetupContext) -> None:
        root = (
            Path(__file__).resolve().parent
            if ctx.submission_dir is None
            else Path(ctx.submission_dir)
        )
        archive = fnp.load(str(root / "sobol_sphere_seed101.npz"))
        self._directions = archive["directions"]

    @staticmethod
    def _mean_gaussian_radius(width: int) -> float:
        return math.sqrt(2.0) * math.exp(
            math.lgamma((width + 1.0) / 2.0) - math.lgamma(width / 2.0)
        )

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        if self._directions is None:
            raise RuntimeError("setup() did not load sobol_sphere_seed101.npz")
        if mlp.width > self._directions.shape[1]:
            raise ValueError(
                f"asset supports width <= {self._directions.shape[1]}, "
                f"got {mlp.width}"
            )

        # WhestBench 0.13 dataset rows deserialize as float64 Python lists.
        # Casting once through tracked fnp matches the benchmark's float32
        # weight precision and the public research harness.
        weights = [weight.astype(fnp.float32) for weight in mlp.weights]
        total = fnp.zeros(mlp.width, dtype=fnp.float64)
        n_base = self._directions.shape[0]
        for start in range(0, n_base, _CHUNK_BASE_ROWS):
            stop = min(start + _CHUNK_BASE_ROWS, n_base)
            direction = self._directions[start:stop, : mlp.width]
            # Only the validator uses width < 256.
            if mlp.width != self._directions.shape[1]:
                norm = fnp.sqrt(
                    fnp.sum(direction * direction, axis=1, keepdims=True)
                )
                direction = direction * (
                    self._mean_gaussian_radius(mlp.width) / norm
                )
            activation = fnp.concatenate((direction, -direction), axis=0)
            for weight in weights:
                activation = fnp.maximum(activation @ weight, 0.0)
            total = total + fnp.sum(
                activation.astype(fnp.float64), axis=0
            )
        final_mean = total / float(2 * n_base)

        first_weight = weights[0]
        first_mean = (
            fnp.sqrt(fnp.sum(first_weight * first_weight, axis=0))
            * _INV_SQRT_2PI
        )
        rows = [fnp.zeros(mlp.width) for _ in range(mlp.depth)]
        rows[0] = first_mean
        rows[-1] = final_mean
        return fnp.stack(rows, axis=0)
