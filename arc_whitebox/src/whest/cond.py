"""Conditionally-independent latent propagation (CIL).

Factor the pre-activation covariance as

    Sigma_l  =  F F^T  +  D,        F: n x q,   D diagonal

and write  h_l = mu + F z + sqrt(D) . e.  **Conditional on the q-dimensional
latent z, the coordinates of h_l are exactly independent**, so the ReLU
factorises and every quantity we need becomes a one-dimensional closed form
inside a q-dimensional outer integral:

    alpha_i(z) = E[ReLU(h_i) | z]          (closed form)
    v_i(z)     = Var(ReLU(h_i) | z)        (closed form)

    E[a_i]        = E_z alpha_i
    Cov(a_i,a_j)  = Cov_z(alpha_i, alpha_j)          for i != j   (exact!)
    Var(a_i)      = Var_z(alpha_i) + E_z v_i
    kappa_p(h_{l+1,i}) = law of total cumulance over z, using
                         sum_j W_ij^p * (conditional cumulants)

Three things fall out that the bivariate-Gaussian route could not give:

1. No bivariate normal CDF anywhere -- the off-diagonal covariance is an exact
   consequence of conditional independence, not a quadrature over Owen's T.
2. The third and fourth cumulants of the *next* layer come from the same pass,
   which is what the Edgeworth marginal needs and what sampling cannot afford
   (kappa_3 needs absolute accuracy ~0.005, i.e. ~240k Monte-Carlo samples).
3. The latent z need not stay Gaussian.  Carrying it as weighted particles
   captures the low-rank non-Gaussian structure exactly -- which is precisely
   the structure the measured rank collapse (effective rank 165 -> 2.7) says
   dominates at depth -- while the diagonal residual stays Gaussian, where the
   CLT genuinely applies.

This is the representation the measurements have been pointing at all along:
non-perturbative in the few collapsed directions, Gaussian in the other ~250.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from . import gaussmath as gm
from .budget import CHEAP, Budget
from .estimators import COST_PHI
from .nets import MLP


def _latent_design(K: int, q: int, kind: str = "sobol", seed: int = 0) -> np.ndarray:
    if kind == "iid":
        return np.random.default_rng(seed).standard_normal((K, q))
    m = int(np.ceil(np.log2(max(K, 2))))
    u = qmc.Sobol(d=q, scramble=True, seed=seed).random_base2(m)[:K]
    from scipy.special import ndtri
    return ndtri(np.clip(u, 1e-12, 1 - 1e-12))


def _relu_cond_moments(m, s):
    """Closed-form E[ReLU], E[ReLU^2], E[ReLU^3], E[ReLU^4] for N(m, s^2).

    Uses the recursion  M_p = m M_{p-1} + (p-1) s^2 M_{p-2}  for
    M_p = E[X^p 1{X>0}], with M_0 = Phi(t), M_1 = m Phi + s phi.
    """
    t = m / s
    P = gm.Phi(t)
    p = gm.phi(t)
    M0 = P
    M1 = m * P + s * p
    s2 = s * s
    M2 = m * M1 + s2 * M0
    M3 = m * M2 + 2 * s2 * M1
    M4 = m * M3 + 3 * s2 * M2
    return M1, M2, M3, M4


def _factor(Sig, q, bud: Budget | None = None):
    """Sigma ~= F F^T + D with D diagonal and non-negative."""
    n = Sig.shape[0]
    w, V = np.linalg.eigh(Sig)
    if bud:
        bud._add("eigh", 2.0 * 9 * n**3)
    idx = np.argsort(w)[::-1][:q]
    lam = np.maximum(w[idx], 0.0)
    F = V[:, idx] * np.sqrt(lam)
    D = np.maximum(np.diag(Sig) - (F * F).sum(1), 1e-12)
    return F, D


def cil(
    mlp: MLP,
    q: int = 16,
    K: int = 2048,
    design: str = "sobol",
    seed: int = 0,
    order: int = 4,
    dtype=np.float32,
    gaussian_latent: bool = False,
):
    """Conditionally-independent latent propagation.  Returns (Yhat, Budget)."""
    n, L = mlp.n, mlp.L
    bud = Budget(dtype=dtype)

    W1 = mlp.Ws[0].astype(np.float64)
    Sig = W1 @ W1.T
    bud.matmul(n, n, n, symmetric=True, op="Sigma_1")
    mu = np.zeros(n)

    F, D = _factor(Sig, q, bud)
    Z = _latent_design(K, q, design, seed)
    bud.randn(K * q)

    Y = np.zeros((L, n))
    for li in range(L):
        sd = np.sqrt(D)
        m = mu + Z @ F.T                       # (K, n) conditional means
        bud.matmul(K, q, n, op="latent_reconstruct")
        M1, M2, M3, M4 = _relu_cond_moments(m, sd)
        bud.elementwise(K * n, COST_PHI, "cond_Phi")
        bud.elementwise(12 * K * n, CHEAP, "cond_moments")

        alpha = M1
        Y[li] = alpha.mean(0)
        bud.elementwise(K * n, CHEAP, "reduce_alpha")
        if li + 1 == L:
            break

        # ---- covariance of a, exactly, via conditional independence --------
        A = alpha - Y[li]
        Cov = (A.T @ A) / K
        bud.matmul(n, K, n, symmetric=True, op="latent_cov")
        vbar = (M2 - M1 * M1).mean(0)
        np.fill_diagonal(Cov, np.diag(Cov) + vbar)
        bud.elementwise(3 * K * n, CHEAP, "cond_var")

        W = mlp.Ws[li + 1].astype(np.float64)
        mu = W @ Y[li]
        bud.matmul(n, n, 1, op="mu_prop")
        Sig = W @ Cov @ W.T
        bud.matmul(n, n, n, op="Sigma_prop")
        bud.matmul(n, n, n, symmetric=True, op="Sigma_prop")
        Sig = 0.5 * (Sig + Sig.T)
        bud.elementwise(2 * n * n, CHEAP, "sym")

        # ---- refactor and carry the particles forward ----------------------
        F, D = _factor(Sig, q, bud)
        if gaussian_latent:
            Z = _latent_design(K, q, design, seed + li + 1)
            bud.randn(K * q)
        else:
            # project each particle's own next-layer mean onto the new latent
            # basis: this is what keeps z non-Gaussian, which is the point.
            Anew = (alpha - Y[li]) @ W.T
            bud.matmul(K, n, n, op="particle_prop")
            G = F.T @ F
            bud.matmul(q, n, q, symmetric=True, op="gram")
            Z = np.linalg.solve(G + 1e-12 * np.eye(q), (Anew @ F).T).T
            bud.matmul(K, n, q, op="project")
            bud._add("solve_q", 2.0 * q**3 / 3.0 + 2.0 * q * q * K)
            Z = Z - Z.mean(0)
            zs = Z.std(0)
            Z = Z / np.maximum(zs, 1e-12)
            bud.elementwise(4 * K * q, CHEAP, "renorm")

    return Y, bud
