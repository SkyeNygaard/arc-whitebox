"""Two falsification gates for the cumulant-propagation route.

GATE 1 -- can the third-cumulant slice be computed from the weights at layer 1?
    h1 = W1^T x is EXACTLY Gaussian, so kappa3(h1) = 0 and everything about
    a1 = ReLU(h1) is a function of Sigma1 = W1^T W1 alone.  The slice we need,
    c21(a1)[p,q] = cum(a_p, a_p, a_q), reduces to the bivariate moment
    E[ReLU(X)^2 ReLU(Y)].  Conditioning on X = t > 0 gives a ONE-dimensional
    integral,

        M(rho) = int_0^inf t^2 [ rho t Phi(rho t/s) + s phi(rho t/s) ] phi(t) dt,
        s = sqrt(1 - rho^2),

    so it is closed to machine precision by quadrature -- no trivariate orthant
    probabilities, no sampling.  Checks: M(1) = int_0^inf t^3 phi = 2/sqrt(2pi),
    M(-1) = 0.  If the analytic slice does not match Monte Carlo to well under
    1%, nothing downstream can work.

GATE 2 -- is the third cumulant INHERITED or GENERATED?
    This decides whether propagation is needed at all.  Compare, at layer l+1:
      * truth              c21(h_{l+1}) on the real network;
      * "generated"        resample z ~ N(mu_l, Sigma_l) with the TRUE first two
                           moments, push through ReLU and W_{l+1}.  This has
                           kappa3(h_l) = 0 by construction, so whatever c21 it
                           produces was manufactured by the ReLU acting on a
                           Gaussian.
    If generated ~= truth, the slice is a function of (mu, Sigma) at every layer
    and needs no cumulant state -- the route collapses to something easy.  If
    generated is far from truth, the cumulant is genuinely inherited and must be
    propagated, which is the hard version.  Either answer is decisive.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

SQRT_2PI = np.sqrt(2.0 * np.pi)


def _nodes(n_nodes: int, hi: float = 12.0):
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    return 0.5 * hi * (x + 1.0), 0.5 * hi * w


def relu2_relu_moment(rho: np.ndarray, n_nodes: int = 240, chunk: int = 4096):
    """E[ReLU(X)^2 ReLU(Y)] for standard bivariate normal with correlation rho."""
    rho = np.clip(np.asarray(rho, dtype=np.float64), -1.0, 1.0)
    t, w = _nodes(n_nodes)
    t2phi = (t * t) * np.exp(-0.5 * t * t) / SQRT_2PI * w      # t^2 phi(t) dt
    flat = rho.ravel()
    out = np.empty_like(flat)
    for i0 in range(0, flat.size, chunk):
        r = flat[i0 : i0 + chunk][:, None]
        s = np.sqrt(np.maximum(1.0 - r * r, 0.0))
        arg = np.divide(r * t[None, :], s, out=np.full((r.size, t.size), np.inf),
                        where=s > 0)
        arg = np.where(s > 0, arg, np.where(r > 0, np.inf, -np.inf))
        inner = r * t[None, :] * norm.cdf(arg) + s * norm.pdf(arg)
        out[i0 : i0 + chunk] = inner @ t2phi
    return out.reshape(rho.shape)


def analytic_c21_layer1(first_weight: np.ndarray):
    """c21(a1)[p,q] = cum(a_p, a_p, a_q) for a1 = ReLU(W1^T x), x ~ N(0, I)."""
    w = np.asarray(first_weight, dtype=np.float64)
    gram = w.T @ w
    sd = np.sqrt(np.maximum(np.diag(gram), 0.0))
    scale = np.outer(sd, sd)
    rho = np.divide(gram, scale, out=np.zeros_like(gram), where=scale > 0)

    m = sd / SQRT_2PI                        # E[ReLU]
    second = 0.5 * sd * sd                   # E[ReLU^2]
    theta = np.arccos(np.clip(rho, -1.0, 1.0))
    cross = scale * (np.sin(theta) + (np.pi - theta) * np.cos(theta)) / (2 * np.pi)

    # E[a_p^2 a_q] = sd_p^2 sd_q * M(rho_pq)
    raw = (sd * sd)[:, None] * sd[None, :] * relu2_relu_moment(rho)
    # cum(a_p,a_p,a_q) = E[a_p^2 a_q] - m_q E[a_p^2] - 2 m_p E[a_p a_q] + 2 m_p^2 m_q
    return (raw - m[None, :] * second[:, None] - 2.0 * m[:, None] * cross
            + 2.0 * (m * m)[:, None] * m[None, :])


def mc_c21(H: np.ndarray) -> np.ndarray:
    Z = H - H.mean(0)
    return (Z * Z).T @ Z / Z.shape[0]
