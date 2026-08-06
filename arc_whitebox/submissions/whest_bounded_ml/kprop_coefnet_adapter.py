#!/usr/bin/env python3
"""Apply the portable CoefNet closure to ARC's factorized K3 state."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from coefnet_numpy_runtime import NumpyCoefNet
from kprop_x1_adapter import PreStateFeatures, features_from_kprop_tower

class FactorizedK3CoefNetClosure:
    def __init__(self, model_path: str | Path, depth: int = 32):
        self.model = NumpyCoefNet(model_path)
        self.depth = depth

    def predict_residual(self, state: PreStateFeatures) -> np.ndarray:
        rows = state.invariant_rows(self.depth)
        d = state.a[state.iu] - state.a[state.ju]
        normalized = self.model.predict_invariant(
            rows[:, :5], d.astype(np.float32),
            state.x1.astype(np.float32), state.x1a.astype(np.float32),
        )
        scale = state.sigma[state.iu] * state.sigma[state.ju]
        residual = np.zeros_like(state.covariance, dtype=np.float64)
        residual[state.iu, state.ju] = normalized * scale
        residual[state.ju, state.iu] = normalized * scale
        return residual

    def correct(self, kprop_tower: dict[int, Any], layer: int,
                baseline_post_covariance: Any) -> np.ndarray:
        state = features_from_kprop_tower(kprop_tower, layer)
        baseline = np.asarray(baseline_post_covariance, dtype=np.float64).copy()
        if baseline.shape != state.covariance.shape:
            raise ValueError("baseline covariance shape mismatch")
        return baseline + self.predict_residual(state)
