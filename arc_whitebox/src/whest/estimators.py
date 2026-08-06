"""Candidate estimators for E[ReLU(h_l)] under x ~ N(0, I).

Every estimator returns `(Yhat, budget)` where `Yhat` is (L, n) and `budget` is
a `whest.budget.Budget` carrying the analytic FLOP count.
"""

from __future__ import annotations

import numpy as np

from . import gaussmath as gm
from .budget import CHEAP, TRANSCENDENTAL, Budget
from .nets import MLP

# ---------------------------------------------------------------------------
# FLOP models for the composite kernels
# ---------------------------------------------------------------------------
# our fast Phi (one exp + polynomial) and phi (one exp + 3 ops), in fp32 units
COST_PHI = TRANSCENDENTAL + 12.0
COST_PHIPDF = TRANSCENDENTAL + 3.0


def _bvn_cost(n_pairs: int, nodes: int) -> float:
    # arcsin + per-node (sin, exp, ~6 cheap) + 2 Phi + reduction
    per = TRANSCENDENTAL + nodes * (2 * TRANSCENDENTAL + 8.0) + 2 * COST_PHI + 6.0
    return per * n_pairs


def _relu_cross_cost(n_pairs: int, nodes: int) -> float:
    # bvn + 2 Phi + 2 phi + 1 exp + ~35 cheap ops
    return _bvn_cost(n_pairs, nodes) + n_pairs * (2 * COST_PHI + 2 * COST_PHIPDF + TRANSCENDENTAL + 35.0)


# ---------------------------------------------------------------------------
# Monte Carlo family
# ---------------------------------------------------------------------------
def monte_carlo(
    mlp: MLP,
    n_samples: int,
    seed: int = 0,
    antithetic: bool = False,
    chunk: int = 4096,
    dtype=np.float32,
    exact_layer1: bool = True,
) -> tuple[np.ndarray, Budget]:
    """Plain (optionally antithetic) batched Monte Carlo.

    `exact_layer1` replaces the layer-1 row with its closed form
    Y[1,i] = ||W_1,i|| / sqrt(2 pi), which is exact and costs ~nothing.
    """
    n, L = mlp.n, mlp.L
    bud = Budget(dtype=dtype)
    rng = np.random.default_rng(seed)
    acc = np.zeros((L, n), dtype=np.float64)
    done = 0
    while done < n_samples:
        b = min(chunk, n_samples - done)
        if antithetic:
            half = (b + 1) // 2
            Z = rng.standard_normal((half, n)).astype(dtype)
            X = np.concatenate([Z, -Z])[:b]
            bud.randn(half * n)
        else:
            X = rng.standard_normal((b, n)).astype(dtype)
            bud.randn(b * n)
        H = X
        for W in mlp.Ws:
            H = H @ W.T
            bud.matmul(b, n, n)
            H = np.maximum(H, 0.0)
            bud.elementwise(b * n, CHEAP, "relu")
        # accumulate all layers (re-run to keep memory small)
        H = X
        rows = []
        for W in mlp.Ws:
            H = np.maximum(H @ W.T, 0.0)
            rows.append(H.sum(0))
        acc += np.stack(rows).astype(np.float64)
        bud.elementwise(b * n * L, CHEAP, "accumulate")
        done += b
    Y = acc / n_samples
    bud.elementwise(L * n, CHEAP, "final_divide")
    if exact_layer1:
        Y[0] = np.linalg.norm(mlp.Ws[0], axis=1) / gm.SQRT2PI
        bud.elementwise(n * n * 2, CHEAP, "layer1_exact")
    return Y, bud


# ---------------------------------------------------------------------------
# Gaussian moment propagation
# ---------------------------------------------------------------------------
def gauss_prop(
    mlp: MLP,
    mode: str = "exact",
    nodes: int = 8,
    dtype=np.float32,
    oracle_stats: dict | None = None,
    return_stats: bool = False,
):
    """Closed-form propagation of (mu_l, Sigma_l) assuming h_l is jointly Gaussian.

    mode = 'exact'       -> full bivariate ReLU second moment (Drezner-Wesolowsky)
    mode = 'linearized'  -> statistical linearisation (marginals only, cheap)
    mode = 'diag'        -> ignore off-diagonal covariance entirely

    `oracle_stats` (from `oracle_moments`) substitutes the *true* (mu, Sigma)
    at each layer, isolating the error contributed by the Gaussian marginal
    assumption from the error contributed by covariance propagation.
    """
    n, L = mlp.n, mlp.L
    bud = Budget(dtype=dtype)
    W1 = mlp.Ws[0].astype(np.float64)
    mu = np.zeros(n)
    Sig = W1 @ W1.T
    bud.matmul(n, n, n, symmetric=True, op="Sigma_1")

    Y = np.zeros((L, n))
    stats = []
    n_pairs = n * (n + 1) // 2

    for li in range(L):
        if oracle_stats is not None:
            mu, Sig = oracle_stats["mu_h"][li], oracle_stats["Sigma_h"][li]
        sd = np.sqrt(np.maximum(np.diag(Sig), 1e-30))
        if return_stats:
            stats.append((mu.copy(), Sig.copy()))

        if mode == "exact":
            mu_a, Sig_a = gm.relu_cov_from_gauss(mu, Sig, n_nodes=nodes)
            bud.elementwise(n, CHEAP, "sd")
            bud.elementwise(3 * n, COST_PHI, "marginal_Phi")
            bud._add("relu_cross", _relu_cross_cost(n_pairs, nodes))
        elif mode == "linearized":
            mu_a, Sig_a = gm.relu_cov_linearized(mu, Sig)
            bud.elementwise(4 * n, COST_PHI, "marginal_Phi")
            bud.elementwise(n * n, CHEAP * 3, "outer_scale")
        elif mode == "diag":
            mu_a = gm.relu_mean(mu, sd)
            Sig_a = np.diag(gm.relu_second(mu, sd) - mu_a**2)
            bud.elementwise(4 * n, COST_PHI, "marginal_Phi")
        else:
            raise ValueError(mode)

        Y[li] = mu_a
        if li + 1 < L:
            W = mlp.Ws[li + 1].astype(np.float64)
            mu = W @ mu_a
            bud.matmul(n, n, 1, op="mu_prop")
            M = W @ Sig_a
            Sig = M @ W.T
            bud.matmul(n, n, n, op="Sigma_prop")
            bud.matmul(n, n, n, symmetric=True, op="Sigma_prop")
            Sig = 0.5 * (Sig + Sig.T)
            bud.elementwise(n * n, CHEAP * 2, "symmetrize")

    if return_stats:
        return Y, bud, stats
    return Y, bud


def oracle_moments(mlp: MLP, n_samples: int = 200_000, seed: int = 7, chunk: int = 8192):
    """True (mu, Sigma) of the pre-activations at every layer, by brute force.

    Diagnostic only -- not an estimator (it costs far more than the budget).
    """
    n, L = mlp.n, mlp.L
    rng = np.random.default_rng(seed)
    s1 = np.zeros((L, n))
    s2 = np.zeros((L, n, n))
    m = 0
    while m < n_samples:
        b = min(chunk, n_samples - m)
        X = rng.standard_normal((b, n)).astype(np.float32)
        H = X
        for li, W in enumerate(mlp.Ws):
            H = H @ W.T
            Hd = H.astype(np.float64)
            s1[li] += Hd.sum(0)
            s2[li] += Hd.T @ Hd
            H = np.maximum(H, 0.0)
        m += b
    mu = s1 / m
    Sig = [s2[li] / m - np.outer(mu[li], mu[li]) for li in range(L)]
    return {"mu_h": mu, "Sigma_h": Sig, "n_samples": m}


# ---------------------------------------------------------------------------
# Anchored Monte Carlo
# ---------------------------------------------------------------------------
def anchored_mc(
    mlp: MLP,
    n_samples: int,
    seed: int = 0,
    antithetic: bool = True,
    sphere: bool = True,
    chunk: int = 4096,
    dtype=np.float32,
    anchor: bool = True,
) -> tuple[np.ndarray, Budget]:
    """Monte Carlo anchored to the exactly-known layer-1 answer.

    Layer 1 is exact: h_1 = W_1 x is exactly Gaussian, so
    Y[1,i] = ||W_1,i|| / sqrt(2 pi).  Propagate that exactness forward with a
    per-neuron linear control variate.  Because a_l = ReLU(h_l) exactly and
    h_l = W_l a_{l-1} exactly, the empirical mean of h_l is W_l times the
    empirical mean of a_{l-1}, so

        Yhat_l = mean_k a_l  +  diag(beta_l) W_l ( Yhat_{l-1} - mean_k a_{l-1} )

    is an exact control variate with the Stein-optimal coefficient
    beta_l = E[ReLU'(h_l)] = P(h_l > 0), estimated for free from the same pass.
    The residual noise is that of ReLU(h) - beta h, which vanishes as |mu/sd|
    grows -- and |mu/sd| ~ 2.9 in these networks.

    The error recursion is  eps_l = residual_l + diag(beta_l) W_l eps_{l-1},
    whose propagation operator is exactly the (measured, contracting)
    sensitivity operator -- so anchor error does not blow up with depth.

    `sphere` exploits exact positive homogeneity of a bias-free ReLU net:
    a_L(x) = ||x|| a_L(x/||x||) with ||x|| independent of the direction, so
    sampling on the sphere and multiplying by the closed-form E||x|| removes the
    radial component of the variance at zero cost.
    """
    from scipy.special import gammaln

    n, L = mlp.n, mlp.L
    bud = Budget(dtype=dtype)
    rng = np.random.default_rng(seed)

    S_a = np.zeros((L, n))   # sum of post-ReLU activations
    S_h = np.zeros((L, n))   # sum of pre-activations
    S_p = np.zeros((L, n))   # count of h > 0

    ER = float(np.sqrt(2.0) * np.exp(gammaln((n + 1) / 2) - gammaln(n / 2)))

    done = 0
    while done < n_samples:
        b = min(chunk, n_samples - done)
        if antithetic:
            half = (b + 1) // 2
            Z = rng.standard_normal((half, n)).astype(dtype)
            bud.randn(half * n)
            X = np.concatenate([Z, -Z])[:b]
        else:
            X = rng.standard_normal((b, n)).astype(dtype)
            bud.randn(b * n)
        if sphere:
            X = X * (ER / np.linalg.norm(X, axis=1, keepdims=True)).astype(dtype)
            bud.elementwise(b * n * 3, CHEAP, "sphere_normalise")

        A = X
        for li, W in enumerate(mlp.Ws):
            H = A @ W.T
            bud.matmul(b, n, n)
            A = np.maximum(H, 0.0)
            bud.elementwise(b * n, CHEAP, "relu")
            S_a[li] += A.sum(0, dtype=np.float64)
            S_h[li] += H.sum(0, dtype=np.float64)
            S_p[li] += (H > 0).sum(0, dtype=np.float64)
            bud.elementwise(3 * b * n, CHEAP, "accumulate")
        done += b

    N = float(n_samples)
    mean_a = S_a / N
    beta = S_p / N            # Stein-optimal linear coefficient E[ReLU'(h)] = P(h>0)
    bud.elementwise(3 * L * n, CHEAP, "reduce")

    Y = np.zeros((L, n))
    # layer 1 in closed form: h_1 = W_1 x is exactly Gaussian
    Y[0] = np.linalg.norm(mlp.Ws[0], axis=1) / gm.SQRT2PI
    bud.elementwise(2 * n * n, CHEAP, "layer1_exact")
    if not anchor:
        return mean_a, bud

    for li in range(1, L):
        W = mlp.Ws[li].astype(np.float64)
        corr = W @ (Y[li - 1] - mean_a[li - 1])
        bud.matmul(n, n, 1, op="anchor_prop")
        Y[li] = mean_a[li] + beta[li] * corr
        bud.elementwise(3 * n, CHEAP, "anchor_apply")
    return Y, bud
