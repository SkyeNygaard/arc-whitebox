"""Exact Gaussian ReLU moments, including the bivariate case with nonzero means.

The bivariate second moment is what makes closed-form covariance propagation
possible.  With zero means it is the Cho-Saul arc-cosine kernel; with nonzero
means it needs the bivariate normal CDF, which we get from the
Drezner-Wesolowsky one-dimensional integral representation.
"""

from __future__ import annotations

import numpy as np

SQRT2PI = np.sqrt(2.0 * np.pi)
INV_SQRT2PI = 1.0 / SQRT2PI


def phi(x):
    return INV_SQRT2PI * np.exp(-0.5 * x * x)


def Phi(x):
    """Standard normal CDF via erf (scipy-free, vectorised)."""
    from scipy.special import ndtr

    return ndtr(x)


def relu_mean(mu, sigma):
    """E[ReLU(X)], X ~ N(mu, sigma^2)."""
    t = mu / sigma
    return mu * Phi(t) + sigma * phi(t)


def relu_second(mu, sigma):
    """E[ReLU(X)^2]."""
    t = mu / sigma
    return (mu * mu + sigma * sigma) * Phi(t) + mu * sigma * phi(t)


def relu_neg_mean(mu, sigma):
    """E[ReLU(-X)] = E[ReLU(X)] - mu  (the 'negative part' correction)."""
    return relu_mean(mu, sigma) - mu


# ---------------------------------------------------------------------------
# bivariate normal CDF  (Drezner & Wesolowsky 1990)
# ---------------------------------------------------------------------------
_GL_NODES: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gauss_legendre(k: int):
    if k not in _GL_NODES:
        _GL_NODES[k] = np.polynomial.legendre.leggauss(k)
    return _GL_NODES[k]


def bvn_cdf(h, k, rho, n_nodes: int = 12):
    """Phi_2(h, k; rho) = P(X <= h, Y <= k) for standard bivariate normal.

    Uses the Drezner-Wesolowsky representation *after* the substitution
    r = sin(theta), which removes the (1-r^2)^{-1/2} endpoint singularity that
    otherwise wrecks Gauss-Legendre convergence as |rho| -> 1:

        Phi_2 = Phi(h)Phi(k)
              + (1/2pi) int_0^{arcsin rho} exp(-(h^2 - 2 h k sin t + k^2)
                                               / (2 cos^2 t)) dt

    The integrand is now smooth and bounded on the whole range, so 12 nodes
    give ~1e-12 uniformly in rho.
    """
    h = np.asarray(h, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    rho = np.clip(np.asarray(rho, dtype=np.float64), -1 + 1e-12, 1 - 1e-12)

    x, w = _gauss_legendre(n_nodes)
    asr = np.arcsin(rho)
    half = 0.5 * asr
    theta = half[..., None] * (x + 1.0)  # (..., n_nodes), range [0, arcsin rho]
    s = np.sin(theta)
    c2 = 1.0 - s * s
    num = h[..., None] ** 2 - 2.0 * s * h[..., None] * k[..., None] + k[..., None] ** 2
    integ = np.exp(-0.5 * num / c2)
    val = (integ * w).sum(-1) * half / (2.0 * np.pi)
    return Phi(h) * Phi(k) + val


def relu_cross_moment(mx, my, sx, sy, rho, n_nodes: int = 12):
    """E[ReLU(X) ReLU(Y)] for jointly Gaussian (X, Y).

    X ~ N(mx, sx^2), Y ~ N(my, sy^2), Corr = rho.

    Writing X = sx (U + tx), Y = sy (V + ty) with (U,V) standard bivariate
    normal of correlation rho and tx = mx/sx, ty = my/sy, and alpha = -tx,
    beta = -ty:

        E = sx sy ( E[UV 1] - beta E[U 1] - alpha E[V 1] + alpha beta P )

    with 1 = 1{U > alpha, V > beta}.
    """
    tx = mx / sx
    ty = my / sy
    a = -tx
    b = -ty
    rho = np.clip(rho, -1 + 1e-9, 1 - 1e-9)
    q = np.sqrt(1.0 - rho * rho)

    P = bvn_cdf(-a, -b, rho, n_nodes)  # P(U>a, V>b)
    ga = Phi((rho * a - b) / q)
    gb = Phi((rho * b - a) / q)
    pa, pb = phi(a), phi(b)

    EU = pa * ga + rho * pb * gb
    EV = pb * gb + rho * pa * ga
    phi2 = np.exp(-0.5 * (a * a - 2 * rho * a * b + b * b) / (q * q)) / (2 * np.pi * q)
    EUV = rho * P + rho * (a * pa * ga + b * pb * gb) + q * q * phi2

    return sx * sy * (EUV - b * EU - a * EV + a * b * P)


def relu_cov_from_gauss(mu, Sigma, n_nodes: int = 12, chunk: int = 64):
    """Cov(ReLU(H)) where H ~ N(mu, Sigma).  Returns (mu_a, Sigma_a)."""
    sd = np.sqrt(np.maximum(np.diag(Sigma), 1e-30))
    mu_a = relu_mean(mu, sd)
    n = len(mu)
    R = Sigma / np.outer(sd, sd)
    out = np.empty((n, n), dtype=np.float64)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        out[i0:i1] = relu_cross_moment(
            mu[i0:i1, None], mu[None, :], sd[i0:i1, None], sd[None, :],
            R[i0:i1], n_nodes,
        )
    Sigma_a = out - np.outer(mu_a, mu_a)
    # exact diagonal (no quadrature error)
    np.fill_diagonal(Sigma_a, relu_second(mu, sd) - mu_a * mu_a)
    Sigma_a = 0.5 * (Sigma_a + Sigma_a.T)
    return mu_a, Sigma_a


def relu_cov_linearized(mu, Sigma):
    """Statistical linearisation: a ~= Phi(t)(h - mu) + eps, eps iid.

    Needs only marginals -- no bivariate integrals -- but assumes the ReLU
    residuals are uncorrelated across neurons, which fails as rho -> 1.
    """
    sd = np.sqrt(np.maximum(np.diag(Sigma), 1e-30))
    t = mu / sd
    beta = Phi(t)
    mu_a = mu * beta + sd * phi(t)
    var_a = relu_second(mu, sd) - mu_a * mu_a
    Sigma_a = (beta[:, None] * beta[None, :]) * Sigma
    resid = var_a - beta * beta * sd * sd
    np.fill_diagonal(Sigma_a, np.diag(Sigma_a) + resid)
    return mu_a, Sigma_a
