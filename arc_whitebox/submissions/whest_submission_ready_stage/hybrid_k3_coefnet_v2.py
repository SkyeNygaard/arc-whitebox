#!/usr/bin/env python3
"""Hardened hybrid factorized-K3 + learned x1/x1a covariance closure.

General validation-tunable family:
  mean_out = G_mean + gamma * (K3_mean - G_mean)
  cov_out  = G_cov  + beta  * (K3_cov  - G_cov) + alpha * ML_residual

The original replacement model is beta=gamma=0. Keeping beta/gamma tunable lets
validation determine whether ARC's native K3 estimates contain complementary
signal. Predicted x1/x1a may also be calibrated to exact public-data features.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
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


@dataclass(frozen=True)
class HybridConfig:
    alpha: float = 0.5
    beta: float = 0.0
    gamma: float = 0.0
    corr_cap: float = 0.999
    x_clip: float = 20.0
    residual_clip: float = 0.5
    next_variance_guard: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HybridConfig":
        fields = cls.__dataclass_fields__
        return cls(**{k: value[k] for k in fields if k in value})


class FeatureCalibration:
    def __init__(self, depth: int, path: Union[str, Path, None] = None):
        self.depth = int(depth)
        self.x1_scale = np.ones(depth, np.float64)
        self.x1a_scale = np.ones(depth, np.float64)
        if path is not None:
            data = json.loads(Path(path).read_text())
            self.x1_scale = self._expand(data.get("x1_scale", 1.0))
            self.x1a_scale = self._expand(data.get("x1a_scale", 1.0))

    def _expand(self, value: Any) -> np.ndarray:
        a = np.asarray(value, np.float64)
        if a.ndim == 0:
            return np.full(self.depth, float(a), np.float64)
        if len(a) != self.depth:
            raise ValueError(f"calibration length {len(a)} != depth {self.depth}")
        return a


class TorchCoefNet:
    def __init__(self, model_path: Union[str, Path], device: Union[str, torch.device],
                 dtype: torch.dtype = torch.float64, feature_clip: float = 30.0):
        device = torch.device(device)
        with np.load(model_path) as data:
            self.mean = torch.as_tensor(data["mean"], device=device, dtype=dtype)
            self.std = torch.clamp(torch.as_tensor(data["std"], device=device, dtype=dtype), min=1e-8)
            self.W, self.b = [], []
            i = 0
            while f"W{i}" in data:
                self.W.append(torch.as_tensor(data[f"W{i}"], device=device, dtype=dtype))
                self.b.append(torch.as_tensor(data[f"b{i}"], device=device, dtype=dtype))
                i += 1
        if len(self.W) != 3:
            raise ValueError(f"expected 3 linear layers, found {len(self.W)}")
        self.device = device
        self.dtype = dtype
        self.feature_clip = float(feature_clip)

    @staticmethod
    def _silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def coefficients(self, base: torch.Tensor) -> torch.Tensor:
        z = (base - self.mean) / self.std
        z = torch.nan_to_num(z, nan=0.0, posinf=self.feature_clip,
                             neginf=-self.feature_clip).clamp(-self.feature_clip, self.feature_clip)
        z = self._silu(z @ self.W[0].T + self.b[0])
        z = self._silu(z @ self.W[1].T + self.b[1])
        return z @ self.W[2].T + self.b[2]


class HybridK3CoefNetV2:
    def __init__(self, model_path: Union[str, Path], config: HybridConfig,
                 depth: int = 32, device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float64, feature_clip: float = 30.0,
                 quadrature_nodes: int = 20,
                 calibration_path: Union[str, Path, None] = None):
        self.model = TorchCoefNet(model_path, device, dtype, feature_clip)
        self.config = config
        self.depth = int(depth)
        self.calibration = FeatureCalibration(depth, calibration_path)
        x, w = np.polynomial.legendre.leggauss(quadrature_nodes)
        self.quad_x = torch.as_tensor(x, device=self.model.device, dtype=dtype)
        self.quad_w = torch.as_tensor(w, device=self.model.device, dtype=dtype)
        self.last_diagnostics: dict[str, float | int] = {}

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * x.square()) / math.sqrt(2.0 * math.pi)

    def _bvn_cdf(self, a: torch.Tensor, b: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        # Plackett integral with t=sin(theta). The substitution analytically
        # removes the 1/sqrt(1-t^2) endpoint singularity, so modest fixed-order
        # quadrature remains accurate even for correlations close to +/-1.
        rho = rho.clamp(-0.999999999, 0.999999999)
        angle = torch.asin(rho)
        theta = 0.5 * angle[..., None] * (self.quad_x + 1.0)
        st = torch.sin(theta)
        ct2 = torch.cos(theta).square().clamp_min(1e-18)
        exponent = -(a[..., None].square() - 2.0 * st * a[..., None] * b[..., None]
                     + b[..., None].square()) / (2.0 * ct2)
        integrand = torch.exp(exponent) / (2.0 * math.pi)
        integral = 0.5 * angle * torch.sum(integrand * self.quad_w, dim=-1)
        return (torch.special.ndtr(a) * torch.special.ndtr(b) + integral).clamp(0.0, 1.0)

    def gaussian_relu_mean(self, mean: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
        variance = covariance.diagonal().clamp_min(1e-12)
        sigma = variance.sqrt()
        a = mean / sigma
        return mean * torch.special.ndtr(a) + sigma * self._phi(a)

    def gaussian_relu_moments(self, mean: torch.Tensor, covariance: torch.Tensor
                              ) -> tuple[torch.Tensor, torch.Tensor]:
        covariance = (covariance + covariance.T) * 0.5
        variance = covariance.diagonal().clamp_min(1e-12)
        sigma = variance.sqrt()
        a = mean / sigma
        rho = (covariance / torch.outer(sigma, sigma).clamp_min(1e-12)).clamp(-0.999999, 0.999999)
        rho.diagonal().fill_(1.0)
        ai, aj = a[:, None], a[None, :]
        si, sj = sigma[:, None], sigma[None, :]
        mui, muj = mean[:, None], mean[None, :]
        one_minus = (1.0 - rho.square()).clamp_min(1e-14)
        root = one_minus.sqrt()
        probability = self._bvn_cdf(ai, aj, rho)
        density2 = torch.exp(-(ai.square() - 2.0 * rho * ai * aj + aj.square())
                             / (2.0 * one_minus)) / (2.0 * math.pi * root)
        second = (mui * muj * probability
                  + mui * sj * self._phi(aj) * torch.special.ndtr((ai - rho * aj) / root)
                  + muj * si * self._phi(ai) * torch.special.ndtr((aj - rho * ai) / root)
                  + si * sj * (rho * probability + one_minus * density2))
        post_mean = mean * torch.special.ndtr(a) + sigma * self._phi(a)
        post_second_diag = ((mean.square() + variance) * torch.special.ndtr(a)
                            + mean * sigma * self._phi(a))
        second.diagonal().copy_(post_second_diag)
        post_cov = second - torch.outer(post_mean, post_mean)
        return post_mean, (post_cov + post_cov.T) * 0.5

    def residual(self, K_pre: dict[int, Any], layer: int) -> torch.Tensor:
        mean = _tensor(K_pre[1]).to(self.model.device, self.model.dtype)
        covariance = _tensor(K_pre[2]).to(self.model.device, self.model.dtype)
        covariance = (covariance + covariance.T) * 0.5
        # At layer 0 the pre-activation is a linear map of the Gaussian input, so its
        # third cumulant is exactly zero and linear_kprop carries no K3 term. The
        # closure features x1/x1a are then identically zero, i.e. no correction --
        # which is right, because the Gaussian moments are exact for that layer.
        if 3 not in K_pre:
            n0 = mean.numel()
            k21 = torch.zeros((n0, n0), device=self.model.device, dtype=self.model.dtype)
        else:
            k21 = _tensor(K_pre[3].get_dslice((2, 1))).to(self.model.device, self.model.dtype)
        variance = covariance.diagonal().clamp_min(1e-12)
        sigma = variance.sqrt()
        rho = (covariance / torch.outer(sigma, sigma).clamp_min(1e-12)).clamp(-1.0, 1.0)
        a = mean / sigma
        n = mean.numel()
        iu, ju = torch.triu_indices(n, n, offset=1, device=mean.device)
        d = a[iu] - a[ju]
        denominator = (sigma[iu].pow(3) + sigma[ju].pow(3)).clamp_min(1e-12)
        x1 = (k21[iu, ju] + k21[ju, iu]) / denominator
        x1a = (k21[iu, ju] - k21[ju, iu]) / denominator
        x1 = x1 * float(self.calibration.x1_scale[layer])
        x1a = x1a * float(self.calibration.x1a_scale[layer])
        x1 = torch.nan_to_num(x1, nan=0.0, posinf=0.0, neginf=0.0).clamp(-self.config.x_clip, self.config.x_clip)
        x1a = torch.nan_to_num(x1a, nan=0.0, posinf=0.0, neginf=0.0).clamp(-self.config.x_clip, self.config.x_clip)
        base = torch.stack([torch.full_like(d, (layer + 1) / self.depth),
                            a[iu] + a[ju], a[iu] * a[ju], d.abs(), rho[iu, ju]], dim=1)
        coef = self.model.coefficients(base)
        normalized = coef[:, 0] * x1 + d * coef[:, 1] * x1a
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        normalized = normalized.clamp(-self.config.residual_clip, self.config.residual_clip)
        values = normalized * sigma[iu] * sigma[ju]
        out = torch.zeros_like(covariance)
        out[iu, ju] = values
        out[ju, iu] = values
        self.last_diagnostics = {
            "pairs": int(iu.numel()), "max_abs_x1": float(x1.abs().max().item()),
            "max_abs_x1a": float(x1a.abs().max().item()),
            "max_abs_normalized_residual": float(normalized.abs().max().item()),
        }
        return out

    @staticmethod
    def _set_core(target: Any, value: torch.Tensor) -> None:
        if not hasattr(target, "core") or getattr(target, "r", 0) != 0:
            raise TypeError(f"expected r=0 HTensor, got {type(target)!r}")
        target.core = value.to(device=target.core.device, dtype=target.core.dtype)

    @staticmethod
    def _next_variances(cov: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return torch.sum(W * (W @ cov), dim=1)

    def _guard_delta(self, base_cov: torch.Tensor, delta_cov: torch.Tensor,
                     next_weights: torch.Tensor | None) -> tuple[torch.Tensor, float]:
        # Pairwise Cauchy-Schwarz cap. This does not guarantee PSD, but prevents
        # impossible individual correlations and is much less destructive than eig-clipping.
        diag = base_cov.diagonal().clamp_min(1e-12)
        limit = self.config.corr_cap * torch.sqrt(torch.outer(diag, diag))
        lower, upper = -limit - base_cov, limit - base_cov
        delta_cov = torch.maximum(torch.minimum(delta_cov, upper), lower)
        delta_cov.diagonal().zero_()
        scale = 1.0
        if self.config.next_variance_guard and next_weights is not None:
            base_v = self._next_variances(base_cov, next_weights).clamp_min(1e-14)
            delta_v = self._next_variances(delta_cov, next_weights)
            bad = delta_v < 0
            if bad.any():
                allowed = 0.99 * base_v[bad] / (-delta_v[bad]).clamp_min(1e-30)
                scale = min(1.0, float(allowed.min().item()))
                delta_cov = delta_cov * scale
        return delta_cov, scale

    def apply_(self, K_pre: dict[int, Any], K_post: dict[int, Any], layer: int,
               next_weights: torch.Tensor | None = None) -> dict[int, Any]:
        mean = _tensor(K_pre[1]).to(self.model.device, self.model.dtype)
        covariance = _tensor(K_pre[2]).to(self.model.device, self.model.dtype)
        upstream_mean = _tensor(K_post[1]).to(self.model.device, self.model.dtype)
        upstream_cov = _tensor(K_post[2]).to(self.model.device, self.model.dtype)
        need_gaussian_cov = abs(self.config.beta - 1.0) > 1e-15
        need_gaussian_mean = abs(self.config.gamma - 1.0) > 1e-15
        gaussian_mean = gaussian_cov = None
        if need_gaussian_cov:
            gaussian_mean, gaussian_cov = self.gaussian_relu_moments(mean, covariance)
        elif need_gaussian_mean:
            gaussian_mean = self.gaussian_relu_mean(mean, covariance)
        output_mean = upstream_mean if not need_gaussian_mean else (
            gaussian_mean + self.config.gamma * (upstream_mean - gaussian_mean)
        )
        base_cov = upstream_cov if not need_gaussian_cov else (
            gaussian_cov + self.config.beta * (upstream_cov - gaussian_cov)
        )
        delta_cov = self.config.alpha * self.residual(K_pre, layer)
        delta_cov, safety_scale = self._guard_delta(base_cov, delta_cov, next_weights)
        corrected_cov = (base_cov + delta_cov + (base_cov + delta_cov).T) * 0.5
        self._set_core(K_post[1], output_mean)
        self._set_core(K_post[2], corrected_cov)
        diag = corrected_cov.diagonal()
        self.last_diagnostics.update({
            "alpha": self.config.alpha, "beta": self.config.beta, "gamma": self.config.gamma,
            "safety_scale": safety_scale, "minimum_variance": float(diag.min().item()),
            "nonfinite_mean": int((~torch.isfinite(output_mean)).sum().item()),
            "nonfinite_covariance": int((~torch.isfinite(corrected_cov)).sum().item()),
        })
        return K_post
