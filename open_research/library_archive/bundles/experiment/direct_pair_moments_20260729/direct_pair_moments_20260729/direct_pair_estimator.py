"""Direct selected pair contractions for the radial-Hermite lower anchor.

This module deliberately never constructs a 256x256 covariance matrix.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

Array = np.ndarray

@dataclass(frozen=True)
class PairContractions:
    marginal_second: Array  # s_p = M[i_p,i_p]
    row_direction: Array    # t_p = M[i_p,:] @ v_p


def _validate(indices: Array, directions: Array, dimension: int) -> tuple[Array, Array]:
    idx = np.asarray(indices, dtype=np.int64)
    v = np.asarray(directions, dtype=np.float64)
    if idx.ndim != 1 or v.shape != (idx.size, dimension):
        raise ValueError(f"expected indices (p,) and directions (p,{dimension}), got {idx.shape}, {v.shape}")
    if np.any(idx < 0) or np.any(idx >= dimension):
        raise ValueError("probe index out of range")
    return idx, v


def accumulate_selected_pair_moments(
    activations: Array,
    indices: Array,
    directions: Array,
    *,
    second_moment_scale: float = 1.0,
    block_rows: int = 4096,
) -> PairContractions:
    """Accumulate s_p and t_p directly, with bounded memory.

    M is defined as ``second_moment_scale * mean(h h^T)``. When ``H @ V.T``
    already exists for radial features, fuse the two reductions in that pass;
    this standalone implementation recomputes the projection blockwise.
    """
    h = np.asarray(activations)
    if h.ndim != 2:
        raise ValueError("activations must have shape (n, d)")
    n, d = h.shape
    if n == 0:
        raise ValueError("activations must be nonempty")
    idx, v = _validate(indices, directions, d)
    s = np.zeros(idx.size, dtype=np.float64)
    t = np.zeros(idx.size, dtype=np.float64)
    for start in range(0, n, block_rows):
        hb = np.asarray(h[start:start + block_rows], dtype=np.float64)
        hi = hb[:, idx]
        hv = hb @ v.T
        s += np.sum(hi * hi, axis=0)
        t += np.sum(hi * hv, axis=0)
    factor = float(second_moment_scale) / n
    return PairContractions(factor * s, factor * t)


def lower_anchor_defect(
    estimated_mean: Array,
    sample_center: Array,
    indices: Array,
    directions: Array,
    pair: PairContractions,
) -> Array:
    """Return the selected lower-order recentering defect.

    For d = mu-m, a_p=v_p^T d, z_p=v_p^T mu, s_p=M[i,i],
    t_p=M[i,:]v_p:

      ell_p = [s_p a_p + 2 d_i t_p + 2(m_i^2-mu_i^2) z_p] / (D+1).
    """
    mu = np.asarray(estimated_mean, dtype=np.float64)
    m = np.asarray(sample_center, dtype=np.float64)
    if mu.shape != m.shape or mu.ndim != 1:
        raise ValueError("means must be matching one-dimensional arrays")
    idx, v = _validate(indices, directions, mu.size)
    s = np.asarray(pair.marginal_second, dtype=np.float64)
    t = np.asarray(pair.row_direction, dtype=np.float64)
    if s.shape != idx.shape or t.shape != idx.shape:
        raise ValueError("pair contraction shapes must match probe count")
    delta = mu - m
    a = v @ delta
    z = v @ mu
    di = delta[idx]
    return (s * a + 2.0 * di * t + 2.0 * (m[idx] ** 2 - mu[idx] ** 2) * z) / (mu.size + 1.0)


def local_scalar_sensitivities(
    mean: Array,
    sample_center: Array,
    indices: Array,
    directions: Array,
    pair: PairContractions,
) -> dict[str, Array]:
    """Jacobian coefficients mapping scalar errors to anchor error.

    Interactions among simultaneous errors remain; use this map for local
    downstream weighting and the exact plug-in formula for final evaluation.
    """
    mu = np.asarray(mean, dtype=np.float64)
    m = np.asarray(sample_center, dtype=np.float64)
    idx, v = _validate(indices, directions, mu.size)
    delta = mu - m
    a = v @ delta
    z = v @ mu
    di = delta[idx]
    s = np.asarray(pair.marginal_second, dtype=np.float64)
    t = np.asarray(pair.row_direction, dtype=np.float64)
    den = mu.size + 1.0
    return {
        "projected_defect": s / den,
        "marginal_second": a / den,
        "selected_center": (2.0 * t - 4.0 * mu[idx] * z) / den,
        "row_direction_pair": 2.0 * di / den,
        "projected_mean": 2.0 * (m[idx] ** 2 - mu[idx] ** 2) / den,
    }


def final_output_correction(anchor_defect: Array, beta: Array) -> Array:
    """Map p scalar anchor defects through the frozen p x 256 coefficient map."""
    e = np.asarray(anchor_defect, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    if b.ndim != 2 or b.shape[0] != e.size:
        raise ValueError("beta must have shape (probe_count, output_dimension)")
    return e @ b


def output_metric(base_error: Array, correction: Array) -> dict[str, float]:
    """Authoritative quadratic output metric (unnormalized by output width)."""
    e = np.asarray(base_error, dtype=np.float64)
    c = np.asarray(correction, dtype=np.float64)
    if e.shape != c.shape:
        raise ValueError("base_error and correction must match")
    ec = float(np.dot(e, c)); cc = float(np.dot(c, c)); ee = float(np.dot(e, e))
    return {
        "error_norm_squared": ee,
        "error_correction_inner_product": ec,
        "correction_norm_squared": cc,
        "corrected_error_norm_squared": ee + 2.0 * ec + cc,
        "signed_correction_cosine": float(-ec / max(np.sqrt(ee * cc), 1e-300)),
    }
