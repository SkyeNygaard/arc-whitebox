"""Submission-safe CoefNet runtime.

Runtime imports only flopscope. Model parameters are loaded once in Estimator.setup
through flops.Module.from_file, which the Phase-1 packaging mechanism supports.
"""
from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp


class CoefNet(flops.Module):
    def __init__(self, n_in: int = 5, hidden: int = 64, n_out: int = 2,
                 feature_clip: float = 30.0):
        self.n_in = int(n_in)
        self.hidden = int(hidden)
        self.n_out = int(n_out)
        self.feature_clip = float(feature_clip)
        self.mean = fnp.zeros((self.n_in,), dtype=fnp.float32)
        self.std = fnp.ones((self.n_in,), dtype=fnp.float32)
        self.W0 = fnp.zeros((self.hidden, self.n_in), dtype=fnp.float32)
        self.b0 = fnp.zeros((self.hidden,), dtype=fnp.float32)
        self.W1 = fnp.zeros((self.hidden, self.hidden), dtype=fnp.float32)
        self.b1 = fnp.zeros((self.hidden,), dtype=fnp.float32)
        self.W2 = fnp.zeros((self.n_out, self.hidden), dtype=fnp.float32)
        self.b2 = fnp.zeros((self.n_out,), dtype=fnp.float32)

    def config(self):
        return {"n_in": self.n_in, "hidden": self.hidden, "n_out": self.n_out,
                "feature_clip": self.feature_clip}

    @staticmethod
    def _silu(x):
        return x / (1.0 + fnp.exp(-x))

    def __call__(self, features):
        z = (features - self.mean) / fnp.maximum(self.std, 1e-8)
        z = fnp.clip(z, -self.feature_clip, self.feature_clip)
        z = self._silu(fnp.einsum("bi,oi->bo", z, self.W0) + self.b0)
        z = self._silu(fnp.einsum("bi,oi->bo", z, self.W1) + self.b1)
        return fnp.einsum("bi,oi->bo", z, self.W2) + self.b2
