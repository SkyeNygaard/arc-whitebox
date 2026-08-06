#!/usr/bin/env python3
"""Torch-native x1/x1a closure hook for ARC's factorized K3 propagation.

Usage inside a layer-by-layer factorized-K3 loop:

    K_pre = linear_kprop(...)
    K_post = factored_nonlin_kprop_k3(K_pre, relu_wick_coef, ...)
    patch.apply_(K_pre, K_post, layer=layer)

The patch changes only K_post[2] (post-ReLU covariance). K_post[1] and K_post[3]
remain those produced by upstream kprop. At the following layer, the corrected K2
is contracted through the next weight matrix and participates in the next
nonlinear cumulant update.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch


def _tensor(value: Any) -> torch.Tensor:
    if hasattr(value, "to_tensor"):
        value = value.to_tensor()
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value


class TorchCoefNet:
    def __init__(self, model_path: Union[str, Path], device: Union[str, torch.device],
                 dtype: torch.dtype = torch.float64, feature_clip: float = 30.0):
        device = torch.device(device)
        with np.load(model_path) as data:
            self.mean = torch.as_tensor(data["mean"], device=device, dtype=dtype)
            self.std = torch.clamp(torch.as_tensor(data["std"], device=device, dtype=dtype), min=1e-8)
            self.W = []
            self.b = []
            index = 0
            while f"W{index}" in data:
                self.W.append(torch.as_tensor(data[f"W{index}"], device=device, dtype=dtype))
                self.b.append(torch.as_tensor(data[f"b{index}"], device=device, dtype=dtype))
                index += 1
        if len(self.W) != 3:
            raise ValueError(f"expected three linear layers, got {len(self.W)}")
        self.device = device
        self.dtype = dtype
        self.feature_clip = float(feature_clip)

    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x)

    def coefficients(self, base: torch.Tensor) -> torch.Tensor:
        normalized = (base - self.mean) / self.std
        normalized = torch.nan_to_num(
            normalized, nan=0.0, posinf=self.feature_clip, neginf=-self.feature_clip
        ).clamp(-self.feature_clip, self.feature_clip)
        hidden = self.silu(normalized @ self.W[0].T + self.b[0])
        hidden = self.silu(hidden @ self.W[1].T + self.b[1])
        return hidden @ self.W[2].T + self.b[2]


class KPropCoefNetPatch:
    def __init__(self, model_path: Union[str, Path], alpha: float = 0.65,
                 depth: int = 32, device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float64,
                 psd_mode: str = "none", feature_clip: float = 30.0):
        self.model = TorchCoefNet(model_path, device, dtype, feature_clip)
        self.alpha = float(alpha)
        self.depth = int(depth)
        if psd_mode not in {"none", "clip"}:
            raise ValueError(psd_mode)
        self.psd_mode = psd_mode
        self.last_diagnostics: dict[str, Union[float, int]] = {}

    def residual(self, K_pre: dict[int, Any], layer: int) -> torch.Tensor:
        mean = _tensor(K_pre[1]).to(device=self.model.device, dtype=self.model.dtype)
        covariance = _tensor(K_pre[2]).to(device=self.model.device, dtype=self.model.dtype)
        covariance = (covariance + covariance.T) * 0.5
        k21 = _tensor(K_pre[3].get_dslice((2, 1))).to(
            device=self.model.device, dtype=self.model.dtype
        )
        width = mean.numel()
        variance = covariance.diagonal().clamp_min(1e-12)
        sigma = variance.sqrt()
        rho = covariance / torch.outer(sigma, sigma).clamp_min(1e-12)
        rho = rho.clamp(-1.0, 1.0)
        a = mean / sigma
        iu, ju = torch.triu_indices(width, width, offset=1, device=mean.device)
        difference = a[iu] - a[ju]
        denominator = (sigma[iu] ** 3 + sigma[ju] ** 3).clamp_min(1e-12)
        x1 = (k21[iu, ju] + k21[ju, iu]) / denominator
        x1a = (k21[iu, ju] - k21[ju, iu]) / denominator
        base = torch.stack([
            torch.full_like(difference, (layer + 1) / self.depth),
            a[iu] + a[ju],
            a[iu] * a[ju],
            difference.abs(),
            rho[iu, ju],
        ], dim=1)
        coefficients = self.model.coefficients(base)
        normalized = coefficients[:, 0] * x1 + difference * coefficients[:, 1] * x1a
        values = normalized * sigma[iu] * sigma[ju]
        output = torch.zeros_like(covariance)
        output[iu, ju] = values
        output[ju, iu] = values
        self.last_diagnostics = {
            "pairs": int(iu.numel()),
            "max_abs_normalized_residual": float(normalized.abs().max().item()),
            "max_abs_covariance_residual": float(values.abs().max().item()),
        }
        return output

    @staticmethod
    def _nearest_psd_preserve_diagonal(covariance: torch.Tensor,
                                       eps: float = 1e-8) -> tuple[torch.Tensor, float]:
        covariance = (covariance + covariance.T) * 0.5
        diagonal = covariance.diagonal().clamp_min(1e-12)
        scale = diagonal.sqrt()
        correlation = covariance / torch.outer(scale, scale).clamp_min(1e-30)
        correlation = (correlation + correlation.T) * 0.5
        correlation.diagonal().fill_(1.0)
        eigenvalues, eigenvectors = torch.linalg.eigh(correlation)
        minimum = float(eigenvalues[0].item())
        if minimum >= eps:
            return covariance, minimum
        eigenvalues = eigenvalues.clamp_min(eps)
        repaired = (eigenvectors * eigenvalues) @ eigenvectors.T
        normalizer = repaired.diagonal().clamp_min(eps).sqrt()
        repaired = repaired / torch.outer(normalizer, normalizer)
        repaired.diagonal().fill_(1.0)
        output = repaired * torch.outer(scale, scale)
        output.diagonal().copy_(diagonal)
        return (output + output.T) * 0.5, minimum

    def apply_(self, K_pre: dict[int, Any], K_post: dict[int, Any],
               layer: int) -> dict[int, Any]:
        baseline = _tensor(K_post[2]).to(device=self.model.device, dtype=self.model.dtype)
        corrected = (baseline + baseline.T) * 0.5 + self.alpha * self.residual(K_pre, layer)
        minimum = float("nan")
        repaired = False
        if self.psd_mode == "clip":
            corrected, minimum = self._nearest_psd_preserve_diagonal(corrected)
            repaired = minimum < 1e-8
        target = K_post[2]
        if hasattr(target, "core"):
            if getattr(target, "r", 0) != 0:
                raise ValueError("K_post[2] must be an r=0 HTensor")
            target.core = corrected.to(device=target.core.device, dtype=target.core.dtype)
        else:
            K_post[2] = corrected
        self.last_diagnostics.update({
            "alpha": self.alpha,
            "minimum_pre_repair_eigenvalue": minimum,
            "psd_repaired": int(repaired),
        })
        return K_post


def _self_test(model_path: Path) -> None:
    class Wrapper:
        def __init__(self, core): self.core = core; self.r = 0
        def to_tensor(self): return self.core
    class K3:
        def __init__(self, value): self.value = value
        def get_dslice(self, part):
            assert tuple(part) == (2, 1)
            return self.value
    torch.manual_seed(0)
    n = 16
    matrix = torch.randn(n, n, dtype=torch.float64)
    covariance = matrix @ matrix.T + torch.eye(n, dtype=torch.float64)
    pre = {1: Wrapper(torch.randn(n, dtype=torch.float64)), 2: Wrapper(covariance),
           3: K3(torch.randn(n, n, dtype=torch.float64))}
    post = {2: Wrapper(covariance.clone())}
    patch = KPropCoefNetPatch(model_path, alpha=0.65, depth=32, psd_mode="clip")
    patch.apply_(pre, post, layer=4)
    assert post[2].core.shape == (n, n)
    assert torch.isfinite(post[2].core).all()
    assert torch.allclose(post[2].core, post[2].core.T)
    print({"self_test": "passed", **patch.last_diagnostics})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", type=Path)
    args = parser.parse_args()
    if args.self_test:
        _self_test(args.self_test)
