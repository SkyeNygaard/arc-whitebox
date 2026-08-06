"""T0 local-only floor diagnostic: 16 complete prefix bases."""
from __future__ import annotations
import math
from pathlib import Path
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP
from fast_matmul import (
    encoded_chunk_to_final_sum,
    encoded_chunk_to_relu_encoding,
    first_layer_chunk_to_relu_encoding,
    prepare_right_p3_d5,
)

_WIDTH = 256
_KERDOCK_BASES = 16
_TOTAL_ROWS = 8192
_STREAM_ROWS = 512
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._chirps = None
        self._rotation = None

    def setup(self, ctx: SetupContext) -> None:
        root = Path(__file__).resolve().parent if ctx.submission_dir is None else Path(ctx.submission_dir)
        asset = fnp.load(str(root / "kerdock_mub5_seed3.npz"))
        self._chirps = asset["chirps"]
        self._rotation = asset["rotation"]

    @staticmethod
    def _mean_gaussian_radius(width: int) -> float:
        return math.sqrt(2.0) * math.exp(math.lgamma((width + 1.0) / 2.0) - math.lgamma(width / 2.0))

    @staticmethod
    def _fwht_axis_one(values: fnp.ndarray) -> fnp.ndarray:
        span = 1
        while span < _WIDTH:
            grouped = values.reshape((_KERDOCK_BASES, _WIDTH // (2 * span), 2, span, _WIDTH))
            left = grouped[:, :, 0, :, :]
            right = grouped[:, :, 1, :, :]
            values = fnp.stack((left + right, left - right), axis=2).reshape((_KERDOCK_BASES, _WIDTH, _WIDTH))
            span *= 2
        return values

    def _first_layer_design(self, first_weight: fnp.ndarray) -> fnp.ndarray:
        if self._chirps is None or self._rotation is None:
            raise RuntimeError("setup() did not load the Kerdock asset")
        effective_weight = self._rotation @ first_weight
        radius = self._mean_gaussian_radius(_WIDTH)
        weighted = self._chirps[:_KERDOCK_BASES, :, None] * effective_weight[None, :, :]
        preactivation = self._fwht_axis_one(weighted) * (radius / math.sqrt(_WIDTH))
        kerdock_rows = fnp.stack((preactivation, -preactivation), axis=2).reshape((-1, _WIDTH))
        return fnp.maximum(kerdock_rows, 0.0)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        if mlp.width != _WIDTH or mlp.depth != 32:
            return fnp.zeros((mlp.depth, mlp.width))

        first_weight = mlp.weights[0].astype(fnp.float32)
        prepared_weights = tuple(
            prepare_right_p3_d5(weight.astype(fnp.float32))
            for weight in mlp.weights[1:]
        )
        first_activation = self._first_layer_design(first_weight)

        final_total = None
        for start in range(0, _TOTAL_ROWS, _STREAM_ROWS):
            encoded = first_layer_chunk_to_relu_encoding(
                first_activation[start : start + _STREAM_ROWS],
                prepared_weights[0],
            )
            for prepared in prepared_weights[1:-1]:
                encoded = encoded_chunk_to_relu_encoding(encoded, prepared)
            chunk_sum = encoded_chunk_to_final_sum(encoded, prepared_weights[-1])
            final_total = chunk_sum if final_total is None else final_total + chunk_sum
        if final_total is None:
            raise RuntimeError("empty Kerdock design")
        final_mean = final_total / _TOTAL_ROWS

        first_mean = fnp.sqrt(fnp.sum(first_weight * first_weight, axis=0)) * _INV_SQRT_2PI
        rows = [fnp.zeros(_WIDTH) for _ in range(mlp.depth)]
        rows[0] = first_mean
        rows[-1] = final_mean
        return fnp.stack(rows, axis=0)
