"""Smoke-safe 90,624-row Kerdock multifidelity estimator.

This version avoids importing NumPy directly. The large fixed direction matrix
and row weights are precomputed in the bundled NPZ file. Runtime matrix
products use ndarray methods on arrays returned by flopscope.numpy.load; all
other operations use flopscope.numpy.
"""
from __future__ import annotations

import math
from pathlib import Path

import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_WIDTH = 256
_DEPTH = 32
_TOTAL_POS = 45_312
_TOTAL_ROWS = 90_624
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_ASSET = "kerdock_mf_90624_precomputed.npz"


def _mean_chi(n: int) -> float:
    return math.sqrt(2.0) * math.exp(
        math.lgamma((n + 1.0) / 2.0) - math.lgamma(n / 2.0)
    )


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._positive_directions = None
        self._row_weights = None
        self._radius = _mean_chi(_WIDTH)

    def setup(self, ctx: SetupContext) -> None:
        root = (
            Path(__file__).resolve().parent
            if ctx.submission_dir is None
            else Path(ctx.submission_dir)
        )
        asset = fnp.load(str(root / _ASSET))
        self._positive_directions = asset["positive_directions"].astype(
            "float32", copy=False
        )
        self._row_weights = asset["row_weights"].astype("float32", copy=False)

    def predict(self, mlp: MLP, budget: int):
        del budget
        if mlp.width != _WIDTH or mlp.depth != _DEPTH:
            return fnp.zeros((mlp.depth, mlp.width), dtype="float64")
        if self._positive_directions is None or self._row_weights is None:
            raise RuntimeError("setup() was not called")

        # Keep right operands Fortran-contiguous for the large GEMMs.
        weights = [
            weight.astype("float32", order="F", copy=True)
            for weight in mlp.weights
        ]

        positive = fnp.empty((_TOTAL_POS, _WIDTH), dtype="float32")
        positive[...] = self._positive_directions.dot(weights[0])
        positive *= self._radius

        activation = fnp.empty((_TOTAL_ROWS, _WIDTH), dtype="float32")
        activation[:_TOTAL_POS] = positive
        activation[_TOTAL_POS:] = -positive
        fnp.maximum(activation, 0.0, out=activation)
        scratch = fnp.empty_like(activation)

        for weight in weights[1:]:
            scratch[...] = activation.dot(weight)
            fnp.maximum(scratch, 0.0, out=scratch)
            activation, scratch = scratch, activation

        final_mean = activation.T.dot(self._row_weights).astype(
            "float64", copy=False
        )
        first_mean = fnp.sqrt(
            fnp.sum(
                weights[0] * weights[0],
                axis=0,
                dtype="float64",
            )
        ) * _INV_SQRT_2PI

        rows = fnp.zeros((_DEPTH, _WIDTH), dtype="float64")
        rows[0] = first_mean
        rows[-1] = final_mean
        return rows
