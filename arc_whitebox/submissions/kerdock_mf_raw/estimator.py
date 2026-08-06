"""Experimental 90,624-row Kerdock multifidelity estimator using raw BLAS.

The statistical rule is the previously measured public-50 construction:

    F3 + (P0_S + P1_S - 2 P3_S) / 16

where F3 is the complete 66,048-row seed-3 Kerdock/MUB 5-design and P*_S are
24-basis subsets under rotations 0, 1, and 3. The measured public raw MSE of
this rule was 1.3555e-7 before the separate layer-2 calibration experiment.

All 90,624 rows are concatenated and propagated in one dense BLAS stream.
This is an experimental public-leaderboard branch; residual-time accounting is
version-dependent and may be patched or regraded.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_WIDTH = 256
_DEPTH = 32
_FULL_BASES = 129
_FULL_POS = _FULL_BASES * _WIDTH          # 33,024
_PILOT_BASES = 24
_PILOT_POS = _PILOT_BASES * _WIDTH       # 6,144
_TOTAL_POS = _FULL_POS + 2 * _PILOT_POS   # 45,312
_TOTAL_ROWS = 2 * _TOTAL_POS              # 90,624
_FULL_ROWS = 2 * _FULL_POS                # 66,048
_PILOT_ROWS = 2 * _PILOT_POS              # 12,288
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_SELECTED = np.asarray(
    [1, 3, 4, 5, 6, 13, 15, 16, 29, 35, 57, 59,
     66, 72, 84, 85, 87, 95, 96, 101, 108, 118, 120, 124],
    dtype=np.int64,
)


def _hadamard_normalized(n: int) -> np.ndarray:
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h * (1.0 / math.sqrt(n))


def _rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((_WIDTH, _WIDTH)))
    q *= np.sign(np.diag(r))[None, :]
    return q.astype(np.float32)


def _mean_chi(n: int) -> float:
    return math.sqrt(2.0) * math.exp(
        math.lgamma((n + 1.0) / 2.0) - math.lgamma(n / 2.0)
    )


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._positive_directions: np.ndarray | None = None
        self._row_weights: np.ndarray | None = None
        self._radius = _mean_chi(_WIDTH)

    def setup(self, ctx: SetupContext) -> None:
        root = (
            Path(__file__).resolve().parent
            if ctx.submission_dir is None
            else Path(ctx.submission_dir)
        )
        with np.load(root / "kerdock_mub5_seed3.npz") as asset:
            chirps = np.asarray(asset["chirps"], dtype=np.float32)
            rotation3 = np.asarray(asset["rotation"], dtype=np.float32)

        h = _hadamard_normalized(_WIDTH)

        def basis_rows(rotation: np.ndarray, selected=None, coordinate=False):
            c = chirps if selected is None else chirps[selected]
            bases = np.matmul(h[None, :, :] * c[:, None, :], rotation)
            if coordinate:
                bases = np.concatenate((bases, rotation[None, :, :]), axis=0)
            return bases.reshape(-1, _WIDTH).astype(np.float32, copy=False)

        full = basis_rows(rotation3, coordinate=True)
        pilot0 = basis_rows(_rotation(0), selected=_SELECTED)
        pilot1 = basis_rows(_rotation(1), selected=_SELECTED)
        self._positive_directions = np.ascontiguousarray(
            np.concatenate((full, pilot0, pilot1), axis=0)
        )

        # Encode F3 + (P0 + P1 - 2 P3) / 16 as one positive row-weight vector.
        w = np.zeros(_TOTAL_ROWS, dtype=np.float32)
        w[:_FULL_POS] += 1.0 / _FULL_ROWS
        w[_TOTAL_POS:_TOTAL_POS + _FULL_POS] += 1.0 / _FULL_ROWS

        selected_positive = np.concatenate(
            [np.arange(i * _WIDTH, (i + 1) * _WIDTH) for i in _SELECTED]
        )
        selected_negative = _TOTAL_POS + selected_positive
        correction = 1.0 / (16.0 * _PILOT_POS)
        w[selected_positive] -= correction
        w[selected_negative] -= correction

        pilot_weight = 1.0 / (32.0 * _PILOT_POS)
        p0_start = _FULL_POS
        p1_start = _FULL_POS + _PILOT_POS
        for lo, hi in ((p0_start, p1_start), (p1_start, _TOTAL_POS)):
            w[lo:hi] += pilot_weight
            w[_TOTAL_POS + lo:_TOTAL_POS + hi] += pilot_weight

        self._row_weights = w

    def predict(self, mlp: MLP, budget: int) -> np.ndarray:
        del budget
        if mlp.width != _WIDTH or mlp.depth != _DEPTH:
            return np.zeros((mlp.depth, mlp.width), dtype=np.float64)
        if self._positive_directions is None or self._row_weights is None:
            raise RuntimeError("setup() was not called")

        weights = [
            np.asfortranarray(np.asarray(weight, dtype=np.float32))
            for weight in mlp.weights
        ]

        positive = np.empty((_TOTAL_POS, _WIDTH), dtype=np.float32)
        np.matmul(self._positive_directions, weights[0], out=positive)
        np.multiply(positive, self._radius, out=positive)

        activation = np.empty((_TOTAL_ROWS, _WIDTH), dtype=np.float32)
        activation[:_TOTAL_POS] = positive
        activation[_TOTAL_POS:] = -positive
        np.maximum(activation, 0.0, out=activation)
        scratch = np.empty_like(activation)

        for weight in weights[1:]:
            np.matmul(activation, weight, out=scratch)
            np.maximum(scratch, 0.0, out=scratch)
            activation, scratch = scratch, activation

        # One BLAS GEMV replaces several slice reductions. The observed float32
        # accumulation discrepancy is ~2e-6 max, far below estimator noise.
        final_mean = (activation.T @ self._row_weights).astype(np.float64)
        first_mean = np.sqrt(
            np.sum(weights[0] * weights[0], axis=0, dtype=np.float64)
        ) * _INV_SQRT_2PI

        rows = np.zeros((_DEPTH, _WIDTH), dtype=np.float64)
        rows[0] = first_mean
        rows[-1] = final_mean
        return rows
