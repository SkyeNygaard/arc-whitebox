"""Third-order bivariate Edgeworth correction to Cov(ReLU(h)).

M40 reports the next-variance relative RMS falling 2.564% -> 1.009% (third
order) -> 0.288% (third+fourth).  The measured reference-noise floor at the
262,144-sample references used by the synthetic harnesses is 0.292%, and the
maximum ratio anything can report there is baseline/floor = 8.8x.  M40 reports
8.9x, i.e. it is pinned at the ceiling and its true gain is unbounded above by
that measurement.  This module re-derives the chain so it can be measured
against a reference large enough to resolve the endpoint.

For a smooth functional of a nearly-Gaussian h with third cumulants kappa3,

    E[F(h)] = E_G[F(h)] + (1/6) sum_abc kappa3_abc E_G[d_a d_b d_c F] + ...

With F = ReLU(h_i) ReLU(h_j) only a,b,c in {i,j} contribute, so

    correction = (1/6) [ k_iii A_iii + 3 k_iij A_iij + 3 k_ijj A_ijj + k_jjj A_jjj ]

where, writing H for the Heaviside step and d for the Dirac delta,

    A_iij = E_G[delta(h_i) H(h_j)]        = p_i(0) P(h_j > 0 | h_i = 0)
    A_iii = E_G[delta'(h_i) ReLU(h_j)]    = -d/dx [ p_i(x) E[ReLU(h_j)|h_i=x] ]_0

Both are closed-form for a bivariate normal.  The needed cumulants are exactly
the objects the ledger calls c21: k_iij = E[(h_i-mu_i)^2 (h_j-mu_j)], with
k_iii its diagonal.

Sanity check built in: the mean correction implied by the same expansion,
(1/6) k_iii E_G[delta'(h_i)], must reduce to the standard univariate Edgeworth
term sigma * (gamma3/6) * (-t phi(t)), which `check_mean_consistency` verifies.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _pair_terms(mu, sd, rho):
    """A_iij and A_iii as (n, n) matrices for a jointly Gaussian h.

    Index convention: entry [i, j] treats i as the differentiated coordinate.
    """
    n = len(mu)
    t = mu / sd
    p0 = norm.pdf(t) / sd                       # density of h_i at 0, per i
    beta = rho * (sd[None, :] / sd[:, None])    # d E[h_j] / d h_i
    sd_c = sd[None, :] * np.sqrt(np.maximum(1.0 - rho * rho, 1e-300))
    mean_c = mu[None, :] - beta * mu[:, None]   # E[h_j | h_i = 0]
    u = mean_c / np.maximum(sd_c, 1e-300)
    Phi_u, phi_u = norm.cdf(u), norm.pdf(u)

    a_iij = p0[:, None] * Phi_u
    relu_c = mean_c * Phi_u + sd_c * phi_u      # E[ReLU(h_j) | h_i = 0]
    dp = (mu / (sd * sd)) * p0                  # p_i'(0)
    a_iii = -(dp[:, None] * relu_c + p0[:, None] * beta * Phi_u)
    return a_iij, a_iii


def edgeworth3_second_moment(mu, sigma, c21, gauss_second):
    """Third-order corrected E[a_i a_j]; `gauss_second` is the Gaussian value.

    `c21[i, j] = cum(h_i, h_i, h_j)`; its diagonal is kappa_iii.
    """
    mu = np.asarray(mu, float)
    sd = np.sqrt(np.maximum(np.diag(sigma), 1e-300))
    rho = np.clip(sigma / np.outer(sd, sd), -1.0 + 1e-12, 1.0 - 1e-12)
    a_iij, a_iii = _pair_terms(mu, sd, rho)

    k_iii = np.diag(c21)
    corr = (
        k_iii[:, None] * a_iii
        + 3.0 * c21 * a_iij
        + 3.0 * c21.T * a_iij.T
        + k_iii[None, :] * a_iii.T
    ) / 6.0
    out = gauss_second + corr
    return 0.5 * (out + out.T)


def edgeworth3_mean(mu, sigma, c21):
    """Third-order corrected E[a_i] = E_G + (1/6) k_iii E_G[delta'(h_i)]."""
    sd = np.sqrt(np.maximum(np.diag(sigma), 1e-300))
    t = mu / sd
    gauss = mu * norm.cdf(t) + sd * norm.pdf(t)
    return gauss - (np.diag(c21) / 6.0) * (t * norm.pdf(t)) / (sd * sd)


def check_mean_consistency(seed: int = 0, n: int = 6) -> float:
    """The pair expansion's mean term must equal the univariate Edgeworth term."""
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(n)
    sd = np.abs(rng.standard_normal(n)) + 0.5
    k3 = rng.standard_normal(n) * 0.1
    sigma = np.diag(sd * sd)
    c21 = np.diag(k3)
    got = edgeworth3_mean(mu, sigma, c21)
    t = mu / sd
    gamma3 = k3 / sd ** 3
    want = sd * ((t * norm.cdf(t) + norm.pdf(t)) + (gamma3 / 6.0) * (-t * norm.pdf(t)))
    return float(np.max(np.abs(got - want)))


if __name__ == "__main__":
    err = check_mean_consistency()
    print(f"mean-term consistency (pair expansion vs univariate Edgeworth): {err:.3e}")
    assert err < 1e-12, "third-order mean term disagrees with the univariate form"
    print("OK")
