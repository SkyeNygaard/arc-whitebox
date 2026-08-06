#!/usr/bin/env python3
"""Adapter between ARC's factorized K3 state and the learned x1/x1a closure.

This module does not modify the upstream kprop implementation. Call
`features_from_pre_state` at each pre-activation state produced by factorized
K3 propagation, then add the predicted normalized residual to the Mehler-8
(or other baseline) post-ReLU covariance.

Upstream objects expected:
  * K[1].to_tensor() -> mean vector
  * K[2].to_tensor() -> covariance matrix
  * K[3].get_dslice((2, 1)) -> k21 matrix

The upstream FactoredTensor API exposes get_dslice and computes the K3 nonlinear
step in O(n^3), so the full K3 tensor never has to be materialized.
"""
from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _as_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _to_tensor(obj: Any) -> np.ndarray:
    return _as_numpy(obj.to_tensor() if hasattr(obj, "to_tensor") else obj)


def upper_triangle(width: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(width, 1)


@dataclass
class PreStateFeatures:
    layer: int
    mean: np.ndarray
    covariance: np.ndarray
    sigma: np.ndarray
    a: np.ndarray
    rho: np.ndarray
    k21: np.ndarray
    x1: np.ndarray
    x1a: np.ndarray
    iu: np.ndarray
    ju: np.ndarray

    def invariant_rows(self, depth: int = 32) -> np.ndarray:
        ai = self.a[self.iu]
        aj = self.a[self.ju]
        d = ai - aj
        return np.column_stack((
            np.full(self.iu.size, (self.layer + 1) / depth, dtype=np.float32),
            ai + aj,
            ai * aj,
            np.abs(d),
            self.rho[self.iu, self.ju],
            self.x1,
            d * self.x1a,
        )).astype(np.float32, copy=False)


def features_from_pre_state(mean: Any, covariance: Any, k3: Any, layer: int,
                            eps: float = 1e-12) -> PreStateFeatures:
    """Extract the exact minimal feature family used by the 10k corpus.

    `k3` can be the upstream FactoredTensor or a dense ndarray. For a dense tensor,
    k21[i,j] = K3[i,i,j].
    """
    mu = _to_tensor(mean).astype(np.float64, copy=False)
    cov = _to_tensor(covariance).astype(np.float64, copy=False)
    if mu.ndim != 1 or cov.shape != (mu.size, mu.size):
        raise ValueError(f"bad mean/covariance shapes: {mu.shape}, {cov.shape}")
    n = mu.size
    sigma = np.sqrt(np.maximum(np.diag(cov), eps))
    denom2 = np.maximum(np.outer(sigma, sigma), eps)
    rho = np.clip(cov / denom2, -1.0, 1.0)
    np.fill_diagonal(rho, 1.0)
    a = mu / sigma

    if hasattr(k3, "get_dslice"):
        k21 = _as_numpy(k3.get_dslice((2, 1))).astype(np.float64, copy=False)
    else:
        dense = _to_tensor(k3).astype(np.float64, copy=False)
        if dense.shape != (n, n, n):
            raise ValueError(f"dense K3 must have shape {(n,n,n)}, got {dense.shape}")
        idx = np.arange(n)
        k21 = dense[idx[:, None], idx[:, None], idx[None, :]]
    if k21.shape != (n, n):
        raise ValueError(f"k21 slice has shape {k21.shape}, expected {(n,n)}")

    # Dataset normalization: (k21 +/- k21.T) / (sigma_i^3 + sigma_j^3).
    d3 = np.maximum(sigma[:, None] ** 3 + sigma[None, :] ** 3, eps)
    x1_full = (k21 + k21.T) / d3
    x1a_full = (k21 - k21.T) / d3
    iu, ju = upper_triangle(n)
    return PreStateFeatures(
        layer=layer, mean=mu, covariance=cov, sigma=sigma, a=a, rho=rho,
        k21=k21, x1=x1_full[iu, ju], x1a=x1a_full[iu, ju], iu=iu, ju=ju,
    )


def features_from_kprop_tower(K: dict[int, Any], layer: int) -> PreStateFeatures:
    if not all(k in K for k in (1, 2, 3)):
        raise KeyError("factorized K3 tower must contain orders 1, 2, and 3")
    return features_from_pre_state(K[1], K[2], K[3], layer)


class X1Closure:
    def __init__(self, model: Any, depth: int = 32):
        self.model = model
        self.depth = depth

    @classmethod
    def load(cls, path: str | Path, depth: int = 32) -> "X1Closure":
        with open(path, "rb") as f:
            return cls(pickle.load(f), depth=depth)

    def predict_normalized_pairs(self, state: PreStateFeatures) -> np.ndarray:
        pred = np.asarray(self.model.predict(state.invariant_rows(self.depth)), dtype=np.float64)
        if pred.shape != (state.iu.size,):
            raise ValueError(f"model returned shape {pred.shape}, expected {(state.iu.size,)}")
        return pred

    def predict_residual_covariance(self, state: PreStateFeatures) -> np.ndarray:
        """Return covariance residual in original activation units."""
        q = self.predict_normalized_pairs(state)
        scale = state.sigma[state.iu] * state.sigma[state.ju]
        M = np.zeros_like(state.covariance, dtype=np.float64)
        M[state.iu, state.ju] = q * scale
        M[state.ju, state.iu] = q * scale
        return M

    def correct_covariance(self, baseline_post_covariance: Any,
                           state: PreStateFeatures) -> np.ndarray:
        base = _to_tensor(baseline_post_covariance).astype(np.float64, copy=True)
        if base.shape != state.covariance.shape:
            raise ValueError("baseline covariance shape mismatch")
        return base + self.predict_residual_covariance(state)


def _self_test() -> None:
    class FakeK3:
        def __init__(self, k21): self.k21 = k21
        def get_dslice(self, part):
            assert tuple(part) == (2, 1)
            return self.k21
    class ZeroModel:
        def predict(self, X): return np.zeros(len(X), dtype=np.float32)
    rng = np.random.default_rng(0)
    n = 8
    A = rng.standard_normal((n, n)); cov = A @ A.T + np.eye(n)
    mean = rng.standard_normal(n)
    k21 = rng.standard_normal((n, n))
    state = features_from_pre_state(mean, cov, FakeK3(k21), layer=3)
    assert state.invariant_rows().shape == (n * (n - 1) // 2, 7)
    out = X1Closure(ZeroModel()).correct_covariance(np.eye(n), state)
    assert np.allclose(out, np.eye(n))
    print("self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
