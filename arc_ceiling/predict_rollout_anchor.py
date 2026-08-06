"""Deployable anchor: propagate (mean, covariance, c21) from weights alone.

Writes `mlp_{index:05d}.npz` with keys `mean`, `covariance`, `c21`, matching the
interface `eval_crossfit_cumulant_control.py --factorized-k3-dir` already reads,
so this plugs into the validated control harness without modifying it.

It fills the one empty cell in that harness's anchor table:

    oracle pre-moments        0.5441x  8/8      works
    same-cloud (Kerdock)      0.9606x           fails
    one-Gaussian closure      anchor err 2-34x  fails
    factorized K3             0.992x holdout    fails
    THIS (analytic rollout)   untested

Nothing here touches the Kerdock cloud or any oracle.  Starting from the one
exact fact -- h_1 is Gaussian, so kappa3(h_1) = 0 and Sigma_1 = W1^T W1 -- the
state is carried forward as

    c21(h_l) = sum_{k<l} c21( ReLU(z_k) @ Q_k ),   z_k ~ N(mu_k, Sigma_k)
    Q_k updated in place by Q_k <- Q_k @ M_k,      M_k = diag(Phi(t_k)) W_{k+1}

so each source cloud is sampled ONCE from a Gaussian whose moments the state
already knows, and thereafter only linear maps are updated.  The mean and
covariance are propagated with the third-order bivariate Edgeworth correction
driven by the same c21 -- jointly, never separately: the ablation showed the
mean correction alone is worse than no correction at all (3.70% vs 3.35%) while
both together give 6.4x.

Measured accuracy of this state (gate4_rollout.py): c21 relative error ~25%,
next-layer variance error 0.92% against 3.35% for the Gaussian closure.  Whether
that suffices for the anchor is exactly what the control harness decides -- the
same-cloud failure at 0.9606x shows the requirement is not forgiving.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import whest.gaussmath as gm  # noqa: E402

from bivariate_edgeworth import edgeworth3_mean, edgeworth3_second_moment  # noqa: E402

WIDTH = 256


def _c21_of(cloud):
    """c21[i,j] = E[(y_i - m_i)^2 (y_j - m_j)] for a sample cloud."""
    z = cloud - cloud.mean(0)
    return (z * z).T @ z / z.shape[0]


def propagate(weights, target_layer, n_cloud=2048, seed=0, edgeworth=True):
    """POST-ReLU (mean, covariance, c21) of a_{target_layer}, from weights only.

    The consuming harness expects post-activation state, not the preactivation
    moments.  The post-ReLU third cumulant follows from the same expected-gate
    split used throughout: the linearisation contributes a gate-scaled copy of
    the preactivation cumulant, and the ReLU manufactures a fresh term,

        c21(a_l)[i,j] ~= D_i D_i D_j c21(h_l)[i,j] + c21(ReLU(z_l))[i,j]

    with z_l ~ N(mu_l, Sigma_l) drawn from the state's own moments.
    """
    rng = np.random.default_rng(seed)
    w0 = weights[0].astype(np.float64)

    # layer 1 is exactly Gaussian: kappa3 = 0, Sigma = W1^T W1
    mu = np.zeros(WIDTH)
    sigma = w0.T @ w0
    clouds: list[np.ndarray] = []   # ReLU(z_k) samples, fixed once drawn
    maps: list[np.ndarray] = []     # Q_k, updated in place each layer

    for layer in range(target_layer):
        sd = np.sqrt(np.maximum(np.diag(sigma), 1e-300))
        # c21 of the current preactivation = sum of transported generated terms
        c21 = np.zeros((WIDTH, WIDTH))
        for cloud, q in zip(clouds, maps):
            c21 += _c21_of(cloud @ q)

        # ReLU moments, corrected jointly by the same c21
        gmean, gcov = gm.relu_cov_from_gauss(mu, sigma, n_nodes=8)
        if edgeworth and clouds:
            second = edgeworth3_second_moment(
                mu, sigma, c21, gcov + np.outer(gmean, gmean))
            amean = edgeworth3_mean(mu, sigma, c21)
            acov = second - np.outer(amean, amean)
        else:
            amean, acov = gmean, gcov

        nxt = weights[layer + 1].astype(np.float64)
        gate = norm.cdf(mu / sd)
        step = gate[:, None] * nxt                 # M_l, the expected-gate map

        # new generated source at this layer, drawn from the state's own Gaussian
        ev, V = np.linalg.eigh(sigma)
        root = V * np.sqrt(np.maximum(ev, 0.0))
        z = mu + rng.standard_normal((n_cloud, WIDTH)) @ root.T
        clouds.append(np.maximum(z, 0.0))
        maps.append(nxt.copy())
        # transport every earlier source one more layer
        for i in range(len(maps) - 1):
            maps[i] = maps[i] @ step

        mu = amean @ nxt
        sigma = nxt.T @ acov @ nxt

    # preactivation cumulant at the target layer
    c21_pre = np.zeros((WIDTH, WIDTH))
    for cloud, q in zip(clouds, maps):
        c21_pre += _c21_of(cloud @ q)

    sd = np.sqrt(np.maximum(np.diag(sigma), 1e-300))
    gate = norm.cdf(mu / sd)
    gmean, gcov = gm.relu_cov_from_gauss(mu, sigma, n_nodes=8)
    if edgeworth and clouds:
        second = edgeworth3_second_moment(
            mu, sigma, c21_pre, gcov + np.outer(gmean, gmean))
        amean = edgeworth3_mean(mu, sigma, c21_pre)
        acov = second - np.outer(amean, amean)
    else:
        amean, acov = gmean, gcov

    ev, V = np.linalg.eigh(sigma)
    root = V * np.sqrt(np.maximum(ev, 0.0))
    z = mu + rng.standard_normal((n_cloud, WIDTH)) @ root.T
    generated = _c21_of(np.maximum(z, 0.0))
    inherited = (gate * gate)[:, None] * gate[None, :] * c21_pre
    return amean, acov, inherited + generated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", type=int, nargs="+", required=True)
    ap.add_argument("--weights-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=29)
    ap.add_argument("--n-cloud", type=int, default=2048)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--no-edgeworth", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for index in args.indices:
        path = args.weights_dir / f"mlp_{index:05d}.npy"
        weights = list(np.load(path))
        mu, sigma, c21 = propagate(
            weights, args.layer, n_cloud=args.n_cloud, seed=1000 + index,
            edgeworth=not args.no_edgeworth)
        out = args.out_dir / f"mlp_{index:05d}.npz"
        np.savez_compressed(out, mean=mu, covariance=sigma, c21=c21)
        print(f"[{index:>5}] wrote {out.name}  "
              f"|mu| {np.linalg.norm(mu):.4e}  |c21| {np.linalg.norm(c21):.4e}",
              flush=True)


if __name__ == "__main__":
    main()
