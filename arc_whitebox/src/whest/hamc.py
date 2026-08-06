"""Hermite-Anchored Monte Carlo (HAMC).

Write ReLU in the Hermite basis of the standardised pre-activation
u = (h - mu)/sigma:

    ReLU(h) = sigma * sum_j  c_j(t) He_j(u) / j!,      t = mu/sigma
    c_1 = Phi(t),   c_j = (-1)^j He_{j-2}(t) phi(t)  for j >= 2.

Monte-Carlo noise in `mean_k ReLU(h_k)` is the sum of the noises in
`mean_k He_j(u_k)`.  Each one can be *annihilated* by a control variate --
but only if the true value of `E[He_j(u)]` is known more accurately than the
sample can measure it.  Removing j=1 alone caps out at 3.8x when t ~ 0;
removing j=1 and j=2 reaches 36x; through j=4, 118x.

j=1 needs E[h] and j=2 needs E[h^2] to better than sampling precision.  Both
are obtained by propagating *corrections*, not values:

    d_l    = Y_l^true - mean_k a_l               (mean correction)
    DM_l   = E[a_l a_l^T] - mean_k a_l a_l^T     (second-moment correction)

Both are **exactly known at layer 1** -- h_1 = W_1 x is exactly Gaussian, so
E[a_1] is closed form and E[a_1 a_1^T] is the Cho-Saul arc-cosine kernel -- and
both propagate through the linearised layer map.  Crucially they are *small*
(the size of Monte-Carlo noise), so errors in propagating them are second
order.  That is what makes 0.04%-accurate variances reachable when the sample
itself only gives 0.79%.

Cost over plain Monte Carlo: ~6% (one n x n empirical second moment at layer 1,
two n^3 matmuls per layer to push DM forward, seven cheap reductions per layer).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from . import gaussmath as gm
from .budget import CHEAP, Budget
from .nets import MLP


def _cho_saul_second_moment(K: np.ndarray) -> np.ndarray:
    """E[ReLU(h) ReLU(h)^T] for h ~ N(0, K).  Exact (arc-cosine kernel)."""
    sd = np.sqrt(np.maximum(np.diag(K), 1e-300))
    rho = np.clip(K / np.outer(sd, sd), -1.0, 1.0)
    val = (np.sqrt(np.maximum(1.0 - rho * rho, 0.0)) + rho * (np.pi - np.arccos(rho)))
    return np.outer(sd, sd) * val / (2.0 * np.pi)


def hamc(
    mlp: MLP,
    n_samples: int,
    seed: int = 0,
    order: int = 2,
    sphere: bool = True,
    chunk: int = 8192,
    dtype=np.float32,
    propagate_mean_terms: bool = True,
) -> tuple[np.ndarray, Budget]:
    """order=0 plain MC, order=1 mean anchor only, order=2 mean + second moment."""
    n, L = mlp.n, mlp.L
    bud = Budget(dtype=dtype)
    rng = np.random.default_rng(seed)
    Wl = [W.astype(np.float64) for W in mlp.Ws]

    ER = float(np.sqrt(2.0) * np.exp(gammaln((n + 1) / 2) - gammaln(n / 2)))

    # ---- one pass: per-layer scalar moments + layer-1 second moment matrix ----
    S = {k: np.zeros((L, n)) for k in
         ("h", "h2", "h3", "h4", "a", "ah", "ah2")}
    M1 = np.zeros((n, n))
    done = 0
    while done < n_samples:
        b = min(chunk, n_samples - done)
        X = rng.standard_normal((b, n)).astype(dtype)
        bud.randn(b * n)
        if sphere:
            X = X * (ER / np.linalg.norm(X, axis=1, keepdims=True)).astype(dtype)
            bud.elementwise(3 * b * n, CHEAP, "sphere")
        A = X
        for li, W in enumerate(mlp.Ws):
            H = A @ W.T
            bud.matmul(b, n, n)
            A = np.maximum(H, 0.0)
            bud.elementwise(b * n, CHEAP, "relu")
            Hd = H.astype(np.float64)
            Ad = A.astype(np.float64)
            H2 = Hd * Hd
            S["h"][li] += Hd.sum(0)
            S["h2"][li] += H2.sum(0)
            S["h3"][li] += (H2 * Hd).sum(0)
            S["h4"][li] += (H2 * H2).sum(0)
            S["a"][li] += Ad.sum(0)
            S["ah"][li] += (Ad * Ad).sum(0)          # ReLU(h)*h = ReLU(h)^2
            S["ah2"][li] += (Ad * Ad * Hd).sum(0)    # ReLU(h)*h^2
            bud.elementwise(11 * b * n, CHEAP, "moment_accumulate")
            if li == 0 and order >= 2:
                M1 += Ad.T @ Ad
                bud.matmul(n, b, n, symmetric=True, op="layer1_second_moment")
        done += b

    N = float(n_samples)
    m = {k: v / N for k, v in S.items()}
    bud.elementwise(7 * L * n, CHEAP, "reduce")

    Y = np.zeros((L, n))
    Y[0] = np.linalg.norm(mlp.Ws[0], axis=1) / gm.SQRT2PI     # exact
    bud.elementwise(2 * n * n, CHEAP, "layer1_exact")
    if order == 0:
        return m["a"], bud

    d = Y[0] - m["a"][0]                                       # mean correction
    if order >= 2:
        K1 = Wl[0] @ Wl[0].T
        bud.matmul(n, n, n, symmetric=True, op="K1")
        DM = _cho_saul_second_moment(K1) - M1 / N               # exact - empirical
        bud.transcendental(n * n, "cho_saul_arccos")
        bud.elementwise(8 * n * n, CHEAP, "cho_saul_rest")

    for li in range(1, L):
        W = Wl[li]
        c1 = W @ d                                             # correction to E[h]
        bud.matmul(n, n, 1, op="mean_correction")

        feats = [c1]
        if order >= 2:
            WD = W @ DM @ W.T
            bud.matmul(n, n, n, op="DM_prop")
            bud.matmul(n, n, n, symmetric=True, op="DM_prop")
            # h = W a exactly, so mean_k h^2 = (W M_emp W^T)_ii exactly and the
            # correction to E[h^2] is exactly diag(W DM W^T) -- no mean term.
            c2 = np.diag(WD)
            bud.elementwise(n, CHEAP, "second_moment_correction")
            feats.append(c2)

        # ---- optimal linear control variate on (h, h^2) --------------------
        mh, mh2, mh3, mh4 = m["h"][li], m["h2"][li], m["h3"][li], m["h4"][li]
        ma, mah, mah2 = m["a"][li], m["ah"][li], m["ah2"][li]
        v11 = mh2 - mh * mh
        cy1 = mah - mh * ma
        if order == 1:
            lam1 = cy1 / np.maximum(v11, 1e-30)
            corr = lam1 * c1
            bud.elementwise(6 * n, CHEAP, "cv_solve1")
        else:
            v12 = mh3 - mh * mh2
            v22 = mh4 - mh2 * mh2
            cy2 = mah2 - mh2 * ma
            det = v11 * v22 - v12 * v12
            det = np.where(np.abs(det) < 1e-24, 1e-24, det)
            lam1 = (v22 * cy1 - v12 * cy2) / det
            lam2 = (v11 * cy2 - v12 * cy1) / det
            corr = lam1 * c1 + lam2 * c2
            bud.elementwise(20 * n, CHEAP, "cv_solve2")

        Y[li] = ma + corr
        bud.elementwise(2 * n, CHEAP, "apply_cv")
        d = Y[li] - ma

        if order >= 2 and li + 1 < L:
            # push the second-moment correction through the linearised layer map:
            #   M_ij ~= alpha_i alpha_j + beta_i beta_j (R_ij - mu_i mu_j)
            # statistical-linearisation slope (NOT the partial regression coeff)
            beta = cy1 / np.maximum(v11, 1e-30)
            mu_h = mh + c1
            DR = WD
            if propagate_mean_terms:
                DR = DR - (c1[:, None] * mu_h[None, :] + mu_h[:, None] * c1[None, :])
                bud.elementwise(4 * n * n, CHEAP, "DM_mean_terms")
            DM = (beta[:, None] * beta[None, :]) * DR
            bud.elementwise(2 * n * n, CHEAP, "DM_scale")
            if propagate_mean_terms:
                DM = DM + (d[:, None] * Y[li][None, :] + Y[li][:, None] * d[None, :])
                bud.elementwise(4 * n * n, CHEAP, "DM_alpha_terms")

    return Y, bud
