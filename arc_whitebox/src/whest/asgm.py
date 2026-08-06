"""Active-Subspace Gaussian Mixture propagation (ASGM).

Represent the law of the pre-activations at layer l as a Gaussian mixture with
K components sharing one covariance:

    h_l  ~  (1/K) sum_k  N( c_l^(k), Sigma_l )

The component means (`particles`) carry the low-rank, strongly non-Gaussian
structure that develops with depth; the shared covariance is integrated exactly
through every ReLU using the closed forms in `gaussmath`.  Only `r` directions
are ever sampled, so the particle design can be quasi-Monte-Carlo rather than
i.i.d.

Degenerate limits:  r = 0 (K = 1) is exactly GaussProp;  Sigma -> 0 is exactly
plain Monte Carlo.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from . import gaussmath as gm
from .budget import CHEAP, TRANSCENDENTAL, Budget
from .estimators import COST_PHI, _relu_cross_cost, gauss_prop
from .nets import MLP

COST_PHIPAIR = TRANSCENDENTAL + 15.0  # one exp -> both Phi and phi


# ---------------------------------------------------------------------------
# particle designs in r dimensions
# ---------------------------------------------------------------------------
def design_points(K: int, r: int, kind: str = "sobol", seed: int = 0) -> np.ndarray:
    """K points in R^r that integrate N(0, I_r) as accurately as possible."""
    if kind == "iid":
        return np.random.default_rng(seed).standard_normal((K, r))
    if kind == "iid_anti":
        h = K // 2
        Z = np.random.default_rng(seed).standard_normal((h, r))
        return np.concatenate([Z, -Z])[:K]
    if kind == "sobol":
        m = int(np.ceil(np.log2(max(K, 2))))
        u = qmc.Sobol(d=r, scramble=True, seed=seed).random_base2(m)[:K]
        u = np.clip(u, 1e-12, 1 - 1e-12)
        return _ppf(u)
    if kind == "sobol_anti":
        h = K // 2
        m = int(np.ceil(np.log2(max(h, 2))))
        u = qmc.Sobol(d=r, scramble=True, seed=seed).random_base2(m)[:h]
        u = np.clip(u, 1e-12, 1 - 1e-12)
        Z = _ppf(u)
        return np.concatenate([Z, -Z])[:K]
    raise ValueError(kind)


def _ppf(u):
    from scipy.special import ndtri

    return ndtri(u)


# ---------------------------------------------------------------------------
# active subspace
# ---------------------------------------------------------------------------
def active_subspace(mlp: MLP, stats, r: int, bud: Budget) -> np.ndarray:
    """Top-r right singular vectors of the mean end-to-end Jacobian d a_L / d h_1.

    J = diag(p_L) W_L diag(p_{L-1}) W_{L-1} ... W_2 diag(p_1),  p_l = Phi(mu_l/sd_l).
    Directions in h_1-space that most influence the final layer.
    """
    n, L = mlp.n, mlp.L
    p = []
    for li in range(L):
        mu, Sig = stats[li]
        sd = np.sqrt(np.maximum(np.diag(Sig), 1e-30))
        p.append(gm.Phi(mu / sd))
    bud.elementwise(L * n, COST_PHI, "jac_Phi")

    J = np.diag(p[0])
    for li in range(1, L):
        J = (mlp.Ws[li].astype(np.float64) @ J) * p[li][:, None]
        bud.matmul(n, n, n, op="jacobian_chain")
        bud.elementwise(n * n, CHEAP, "jacobian_scale")
    # top-r right singular vectors
    G = J.T @ J
    bud.matmul(n, n, n, symmetric=True, op="jacobian_gram")
    w, V = np.linalg.eigh(G)
    bud._add("eigh", 2.0 * 9 * n**3)  # flopscope prices eigh(256) at ~9 n^3 (fp32)
    return V[:, ::-1][:, :r]


# ---------------------------------------------------------------------------
# main estimator
# ---------------------------------------------------------------------------
def asgm(
    mlp: MLP,
    r: int = 8,
    K: int = 4096,
    design: str = "sobol_anti",
    seed: int = 0,
    subspace: str = "jacobian",
    residual_mode: str = "linearized",
    nodes: int = 8,
    dtype=np.float32,
    fold_truncation: bool = True,
):
    """Returns (Yhat (L,n), Budget)."""
    n, L = mlp.n, mlp.L
    bud = Budget(dtype=dtype)

    # ---- pass 1: cheap GaussProp for the active subspace -------------------
    if r > 0:
        _, gbud, stats = gauss_prop(mlp, mode="linearized", return_stats=True, dtype=dtype)
        for k, v in gbud.by_op.items():
            bud._add("gaussprop:" + k, v)
        if subspace == "jacobian":
            P = active_subspace(mlp, stats, r, bud)
        elif subspace == "sigma1":
            Sig1 = mlp.Ws[0].astype(np.float64) @ mlp.Ws[0].astype(np.float64).T
            w, V = np.linalg.eigh(Sig1)
            P = V[:, ::-1][:, :r]
        else:
            raise ValueError(subspace)

    # ---- exact split of the (exactly Gaussian) layer-1 law -----------------
    W1 = mlp.Ws[0].astype(np.float64)
    Sigma = W1 @ W1.T
    bud.matmul(n, n, n, symmetric=True, op="Sigma_1")

    if r > 0:
        M = Sigma @ P                     # (n, r)
        bud.matmul(n, n, r, op="Sigma_P")
        G = P.T @ M                       # (r, r)
        bud.matmul(r, n, r, symmetric=True, op="PtSigmaP")
        Lg = np.linalg.cholesky(G)
        bud._add("chol_r", 2.0 * r**3 / 3.0)
        Z = design_points(K, r, design, seed) @ Lg.T   # (K, r) ~ N(0, G)
        bud.randn(K * r)
        bud.matmul(K, r, r, op="design_scale")
        A = np.linalg.solve(G, M.T)       # (r, n) = G^{-1} M^T
        bud._add("solve_r", 2.0 * r * r * n)
        C = Z @ A                         # (K, n) component means
        bud.matmul(K, r, n, op="particle_means")
        Sigma = Sigma - M @ A
        bud.matmul(n, r, n, symmetric=True, op="residual_cov")
    else:
        C = np.zeros((1, n))
        K = 1

    Y = np.zeros((L, n))

    for li in range(L):
        sd = np.sqrt(np.maximum(np.diag(Sigma), 1e-30))
        bud.elementwise(n, CHEAP, "sd")
        T = C / sd
        Phi_T = gm.Phi(T)
        phi_T = gm.phi(T)
        bud.elementwise(K * n, COST_PHIPAIR, "component_Phi")
        Acomp = C * Phi_T + sd * phi_T           # (K, n) component E[ReLU]
        bud.elementwise(K * n * 4, CHEAP, "component_mean")
        Y[li] = Acomp.mean(0)
        bud.elementwise(K * n, CHEAP, "particle_average")

        if li + 1 == L:
            break
        W = mlp.Ws[li + 1].astype(np.float64)

        # ---- within-component covariance (shared) -------------------------
        second = (C * C + sd * sd) * Phi_T + C * sd * phi_T
        vcomp = second - Acomp * Acomp           # (K, n) within-component var
        bud.elementwise(K * n * 8, CHEAP, "component_var")
        vbar = vcomp.mean(0)
        beta = Phi_T.mean(0)
        bud.elementwise(2 * K * n, CHEAP, "component_reduce")

        if residual_mode == "linearized":
            Psi = (beta[:, None] * beta[None, :]) * Sigma
            bud.elementwise(n * n * 2, CHEAP, "psi_scale")
        elif residual_mode == "exact":
            _, Psi = gm.relu_cov_from_gauss(C.mean(0), Sigma, n_nodes=nodes)
            bud._add("relu_cross", _relu_cross_cost(n * (n + 1) // 2, nodes))
        else:
            raise ValueError(residual_mode)
        np.fill_diagonal(Psi, vbar)

        # ---- propagate ----------------------------------------------------
        Cn = Acomp @ W.T
        bud.matmul(K, n, n, op="particle_prop")
        Sigma = W @ Psi @ W.T
        bud.matmul(n, n, n, op="Sigma_prop")
        bud.matmul(n, n, n, symmetric=True, op="Sigma_prop")
        Sigma = 0.5 * (Sigma + Sigma.T)
        bud.elementwise(n * n * 2, CHEAP, "symmetrize")
        C = Cn

    return Y, bud
