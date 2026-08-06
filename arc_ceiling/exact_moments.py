"""Exact layer-1 and layer-2 moments of a ReLU network from the weights alone.

At the fixed Kerdock radius r = E[chi_d] the first preactivation is

    h1 = W1^T (r u),   u uniform on S^{d-1},

and the radial factor is decided by *homogeneity degree*, not by a single
rescaling of the law.  Because the network is positively homogeneous, a1 is
degree 1 and a1 a1^T is degree 2, so for x ~ N(0, I),

    E_N[a1]      = E[chi_d] * E_u[a1]          -> radius E[chi_d] is EXACT
    E_N[a1 a1^T] = E[chi_d^2] * E_u[..] = d * E_u[..]

Evaluating on the sphere of radius r = E[chi_d] therefore reproduces the
Gaussian mean with no correction at all, while the second moment is short by
r^2/d = 0.99805.  That asymmetry is the entire reason the fixed radius is
chosen this way:

    E[a1]      =            sigma * (1/sqrt(2*pi))
    E[a1 a1^T] = (r^2/d) * (sigma_i sigma_j / 2pi)(sin t + (pi - t) cos t)

with sigma = sqrt(diag(W1^T W1)) and t = arccos(rho).  `verify_against_monte_carlo`
checks both conventions against sampling; the factor-free mean is the correct one.

Because the second preactivation is *linear* in a1, its mean and full
covariance are then exact as well:

    mu_2    = W2^T E[a1]
    Sigma_2 = W2^T Cov(a1) W2

This is the deepest point in the network where exact moments are available
with no closure approximation, and Sigma_2 is exact as a full matrix -- not
merely on its diagonal.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def sphere_radius_mean(d: int) -> float:
    """E[chi_d] = sqrt(2) * Gamma((d+1)/2) / Gamma(d/2)."""
    return float(
        math.sqrt(2.0) * math.exp(gammaln((d + 1) / 2.0) - gammaln(d / 2.0))
    )


def exact_layer1_moments(
    first_weight: np.ndarray, radial: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Exact mean and centered covariance of a1 = ReLU(W1^T x), |x| = E[chi_d].

    The default (`radial=False`) is the correct convention: by homogeneity the
    degree-1 mean needs no radial factor.  `radial=True` is kept only so the
    incorrect alternative can be exhibited against ground truth.
    """
    weight = np.asarray(first_weight, dtype=np.float64)
    d = weight.shape[0]
    gram = weight.T @ weight
    sigma = np.sqrt(np.maximum(np.diag(gram), 0.0))
    scale = np.outer(sigma, sigma)
    rho = np.divide(gram, scale, out=np.zeros_like(gram), where=scale > 0.0)
    theta = np.arccos(np.clip(rho, -1.0, 1.0))
    gaussian_second = (
        scale * (np.sin(theta) + (math.pi - theta) * np.cos(theta)) / (2.0 * math.pi)
    )

    r = sphere_radius_mean(d)
    var_factor = r * r / d           # applies to second moments
    mean_factor = math.sqrt(var_factor)  # applies once to the mean

    second = var_factor * gaussian_second
    mean = sigma * INV_SQRT_2PI * (mean_factor if radial else 1.0)
    cov = second - np.outer(mean, mean)
    return mean, 0.5 * (cov + cov.T)


def exact_layer2_moments(
    weights, radial: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Exact mean and FULL covariance of the second preactivation."""
    mean_a1, cov_a1 = exact_layer1_moments(weights[0], radial=radial)
    w2 = np.asarray(weights[1], dtype=np.float64)
    return mean_a1 @ w2, w2.T @ cov_a1 @ w2


def verify_against_monte_carlo(first_weight, n_samples=4_000_000, seed=0, chunk=65536):
    """Direct check of both conventions against sampling on the sphere."""
    w = np.asarray(first_weight, dtype=np.float64)
    d = w.shape[0]
    r = sphere_radius_mean(d)
    rng = np.random.default_rng(seed)
    s1 = np.zeros(w.shape[1])
    s2 = np.zeros((w.shape[1], w.shape[1]))
    done = 0
    while done < n_samples:
        b = min(chunk, n_samples - done)
        z = rng.standard_normal((b, d))
        z *= r / np.linalg.norm(z, axis=1, keepdims=True)
        a = np.maximum(z @ w, 0.0)
        s1 += a.sum(0)
        s2 += a.T @ a
        done += b
    mean = s1 / n_samples
    cov = s2 / n_samples - np.outer(mean, mean)
    return mean, cov


if __name__ == "__main__":
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1] / "arc_whitebox"
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    import os

    os.chdir(ROOT.parent)
    from eval_sampling_official import DEFAULT_DATA, _load_rows

    rows = _load_rows(DEFAULT_DATA, [0, 1])
    N = 6_000_000
    print(f"Validating the exact layer-1 moment formulas against {N:,} sphere samples")
    print(f"  radius E[chi_256] = {sphere_radius_mean(256):.10f}")
    print(f"  radial factor r^2/d = {sphere_radius_mean(256)**2/256:.10f}\n")
    for idx, (name, W, _) in enumerate(rows):
        mc_mean, mc_cov = verify_against_monte_carlo(W[0], n_samples=N, seed=100 + idx)
        noise = 1.0 / math.sqrt(N)
        print(f"MLP {idx} ({name})   MC noise floor ~ {noise*100:.4f}%")
        for label, radial in (("with radial factor", True), ("without (their code)", False)):
            m, c = exact_layer1_moments(W[0], radial=radial)
            em = np.sqrt(np.mean((m / mc_mean - 1.0) ** 2))
            sc = np.sqrt(np.outer(np.diag(mc_cov), np.diag(mc_cov)))
            ec = np.sqrt(np.mean(((c - mc_cov) / sc) ** 2))
            print(
                f"    {label:<22} mean rel.err {em*100:8.4f}%   "
                f"cov rel.err {ec*100:8.4f}%"
            )
        print()
