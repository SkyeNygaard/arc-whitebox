"""Numerically stable pure-NumPy runtime for the compact x1/x1a CoefNet."""
from __future__ import annotations
from pathlib import Path
from typing import Union
import numpy as np


class NumpyCoefNet:
    def __init__(self, path: Union[str, Path], feature_clip: float = 30.0):
        with np.load(path) as d:
            self.mean = np.asarray(d["mean"], np.float64)
            self.std = np.maximum(np.asarray(d["std"], np.float64), 1e-8)
            self.W: list[np.ndarray] = []
            self.b: list[np.ndarray] = []
            i = 0
            while f"W{i}" in d:
                self.W.append(np.asarray(d[f"W{i}"], np.float64))
                self.b.append(np.asarray(d[f"b{i}"], np.float64))
                i += 1
        if len(self.W) != 3:
            raise ValueError(f"expected 3 linear layers, found {len(self.W)}")
        self.feature_clip = float(feature_clip)
        self.last_diagnostics: dict[str, Union[int, float]] = {}

    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        z = np.clip(x, -60.0, 60.0)
        return x / (1.0 + np.exp(-z))

    def predict_invariant(
        self,
        base: np.ndarray,
        difference: np.ndarray,
        x1: np.ndarray,
        x1a: np.ndarray,
    ) -> np.ndarray:
        base = np.asarray(base, np.float64)
        difference = np.asarray(difference, np.float64)
        x1 = np.asarray(x1, np.float64)
        x1a = np.asarray(x1a, np.float64)

        normalized = (base - self.mean) / self.std
        nonfinite = int(np.size(normalized) - np.count_nonzero(np.isfinite(normalized)))
        clipped = int(np.count_nonzero(np.abs(np.nan_to_num(normalized)) > self.feature_clip))
        normalized = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=self.feature_clip,
            neginf=-self.feature_clip,
        )
        normalized = np.clip(normalized, -self.feature_clip, self.feature_clip)

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            h = self.silu(normalized @ self.W[0].T + self.b[0])
            h = self.silu(h @ self.W[1].T + self.b[1])
            coefficients = h @ self.W[2].T + self.b[2]
            prediction = coefficients[:, 0] * x1 + difference * coefficients[:, 1] * x1a

        prediction_nonfinite = int(np.size(prediction) - np.count_nonzero(np.isfinite(prediction)))
        prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        self.last_diagnostics = {
            "nonfinite_normalized_features": nonfinite,
            "clipped_normalized_features": clipped,
            "nonfinite_predictions_replaced": prediction_nonfinite,
            "rows": int(len(prediction)),
        }
        return prediction

    def predict(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, np.float64)
        layer, ai, aj, rho, x1, x1a = [rows[:, k] for k in range(6)]
        difference = ai - aj
        base = np.column_stack([
            layer,
            ai + aj,
            ai * aj,
            np.abs(difference),
            rho,
        ])
        return self.predict_invariant(base, difference, x1, x1a)
