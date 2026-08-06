"""Exact scalar interface for the sparse radial-Hermite anchor.

The module deliberately does not implement an unvalidated moment propagator.
It freezes the algebra that any deployable joint-scalar estimator must satisfy.

For probe p=(i_p, v_p) and the observable Kerdock center m, define

    z_p = E[v_p^T h]
    s_p = E[h_{i_p}^2]
    u_p = E[h_{i_p} (v_p^T h)]
    r_p = E[h_{i_p}^2 (v_p^T h)]

Then the exact radially homogenized anchor is

    a_p = (r_p - (m^T v_p)s_p - 2m_{i_p}u_p
                 + 2m_{i_p}^2 z_p) / (d+1).

Thus K probes require exactly 4K scalar slots before sharing/deduplication.
The cubic quantity may also be represented as a true-centered connected c21
contraction plus the lower-order recentering terms; both forms are tested here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class ProbeBatch:
    """Network-specific observable probe support."""

    indices: Array  # shape [K], integer i_p
    directions: Array  # shape [D, K], columns v_p

    def validate(self, width: int | None = None) -> None:
        indices = np.asarray(self.indices)
        directions = np.asarray(self.directions)
        if indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if directions.ndim != 2 or directions.shape[1] != len(indices):
            raise ValueError("directions must have shape [D, K]")
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("indices must be integers")
        if width is None:
            width = directions.shape[0]
        if directions.shape[0] != width:
            raise ValueError("direction width mismatch")
        if np.any(indices < 0) or np.any(indices >= width):
            raise ValueError("probe index outside width")
        if not np.all(np.isfinite(directions)):
            raise ValueError("directions contain non-finite values")

    @property
    def count(self) -> int:
        return int(len(self.indices))


@dataclass(frozen=True)
class JointScalars:
    """The four scalar families required by the exact anchor."""

    target_mean: Array  # z_p = E[v_p^T h]
    marginal_second: Array  # s_p = E[h_i^2]
    row_direction_second: Array  # u_p = E[h_i (v_p^T h)]
    cubic_contraction: Array  # r_p = E[h_i^2 (v_p^T h)]

    def validate(self, count: int) -> None:
        for name, value in self.as_mapping().items():
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape != (count,):
                raise ValueError(f"{name} must have shape ({count},)")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values")

    def as_mapping(self) -> Mapping[str, Array]:
        return {
            "target_mean": self.target_mean,
            "marginal_second": self.marginal_second,
            "row_direction_second": self.row_direction_second,
            "cubic_contraction": self.cubic_contraction,
        }


def scalarize_moments(
    mean: Array,
    second: Array,
    raw_m21: Array,
    probes: ProbeBatch,
) -> JointScalars:
    """Extract the four target scalar families from full reference moments.

    This is for oracle validation only. A deployable implementation must target
    these quantities directly and must not construct the full matrices/tensor.
    """

    mean = np.asarray(mean, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    raw_m21 = np.asarray(raw_m21, dtype=np.float64)
    width = len(mean)
    probes.validate(width)
    if second.shape != (width, width) or raw_m21.shape != (width, width):
        raise ValueError("second and raw_m21 must have shape [D, D]")
    idx = np.asarray(probes.indices, dtype=np.int64)
    v = np.asarray(probes.directions, dtype=np.float64)
    return JointScalars(
        target_mean=mean @ v,
        marginal_second=np.diag(second)[idx],
        row_direction_second=np.einsum("ki,ik->k", second[idx], v),
        cubic_contraction=np.einsum("ki,ik->k", raw_m21[idx], v),
    )


def compose_anchor(
    scalars: JointScalars,
    probes: ProbeBatch,
    observable_center: Array,
    *,
    width: int,
) -> Array:
    """Compose the exact K-vector radial-Hermite anchor."""

    probes.validate(width)
    scalars.validate(probes.count)
    center = np.asarray(observable_center, dtype=np.float64)
    if center.shape != (width,):
        raise ValueError(f"observable_center must have shape ({width},)")
    idx = np.asarray(probes.indices, dtype=np.int64)
    v = np.asarray(probes.directions, dtype=np.float64)
    center_i = center[idx]
    center_direction = center @ v
    return (
        np.asarray(scalars.cubic_contraction, dtype=np.float64)
        - center_direction * np.asarray(scalars.marginal_second, dtype=np.float64)
        - 2.0 * center_i * np.asarray(scalars.row_direction_second, dtype=np.float64)
        + 2.0 * np.square(center_i) * np.asarray(scalars.target_mean, dtype=np.float64)
    ) / (width + 1.0)


def exact_anchor_matrix(
    mean: Array,
    second: Array,
    raw_m21: Array,
    observable_center: Array,
    *,
    width: int,
) -> Array:
    """Reference full-matrix form used only for equivalence tests."""

    mean = np.asarray(mean, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    raw_m21 = np.asarray(raw_m21, dtype=np.float64)
    center = np.asarray(observable_center, dtype=np.float64)
    if mean.shape != (width,) or center.shape != (width,):
        raise ValueError("mean/center shape mismatch")
    if second.shape != (width, width) or raw_m21.shape != (width, width):
        raise ValueError("moment matrix shape mismatch")
    return (
        raw_m21
        - np.diag(second)[:, None] * center[None, :]
        - 2.0 * center[:, None] * second
        + 2.0 * np.square(center)[:, None] * mean[None, :]
    ) / (width + 1.0)


def contract_matrix(matrix: Array, probes: ProbeBatch) -> Array:
    matrix = np.asarray(matrix, dtype=np.float64)
    probes.validate(matrix.shape[0])
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    idx = np.asarray(probes.indices, dtype=np.int64)
    v = np.asarray(probes.directions, dtype=np.float64)
    return np.einsum("ki,ik->k", matrix[idx], v)


def connected_cubic_contractions(
    mean: Array,
    second: Array,
    raw_m21: Array,
    probes: ProbeBatch,
) -> Array:
    """Return true-centered connected c21 row-direction contractions."""

    mean = np.asarray(mean, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    raw_m21 = np.asarray(raw_m21, dtype=np.float64)
    probes.validate(len(mean))
    idx = np.asarray(probes.indices, dtype=np.int64)
    v = np.asarray(probes.directions, dtype=np.float64)
    mean_i = mean[idx]
    mean_v = mean @ v
    marginal = np.diag(second)[idx]
    row_second = np.einsum("ki,ik->k", second[idx], v)
    raw = np.einsum("ki,ik->k", raw_m21[idx], v)
    return raw - 2.0 * mean_i * row_second - marginal * mean_v + 2.0 * np.square(mean_i) * mean_v


def raw_from_connected(
    connected: Array,
    mean: Array,
    second: Array,
    probes: ProbeBatch,
) -> Array:
    """Reconstruct r_p from true-centered c21 plus lower-order scalars."""

    connected = np.asarray(connected, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    probes.validate(len(mean))
    idx = np.asarray(probes.indices, dtype=np.int64)
    v = np.asarray(probes.directions, dtype=np.float64)
    mean_i = mean[idx]
    mean_v = mean @ v
    marginal = np.diag(second)[idx]
    row_second = np.einsum("ki,ik->k", second[idx], v)
    return connected + 2.0 * mean_i * row_second + marginal * mean_v - 2.0 * np.square(mean_i) * mean_v


def retained_oracle_improvement(candidate_ratio: float, exact_ratio: float) -> float:
    """Fraction of exact-anchor MSE reduction retained by a candidate."""

    denom = 1.0 - float(exact_ratio)
    if denom <= 0.0:
        raise ValueError("exact_ratio must be below 1")
    return (1.0 - float(candidate_ratio)) / denom


def ratio_for_retention(exact_ratio: float, retention: float) -> float:
    """Maximum candidate/base ratio allowed for a requested retention."""

    if not 0.0 <= retention <= 1.0:
        raise ValueError("retention must lie in [0, 1]")
    return 1.0 - retention * (1.0 - float(exact_ratio))
