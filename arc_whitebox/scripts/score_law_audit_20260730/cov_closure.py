"""Is COVARIANCE propagation the blocker in the analytic route?

The marginal closure is fine (closure.py: 1.9e-4/layer with Edgeworth).  The
other half of the recursion is

    Sigma_{l+1} = W^T Cov(relu(z_l)) W

evaluated under the Gaussian closure "z_l ~ N(mu_l, Sigma_l)".  notes/03 §4
identified this as problem 1 ("a few-% error where sigma needs 3e-3") but never
measured it end-to-end.

Measured here, one step at a time with EXACT input moments taken from the 66,048
propagated design rows, so this is pure closure error:

  truth      Var(z_{l+1,i}) directly from the propagated rows
  closure    Var(w_i . relu(g)),  g ~ N(mu_l, Sigma_l) fitted to those same rows

Sampling the fitted Gaussian avoids needing a closed form for the bivariate
rectified-Gaussian moment with non-zero means, and its own MC error is
sqrt(2/M) = 0.32% at M = 200,000 -- below the effect size we are looking for.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, ASSET, DATA, first_layer_design

N_NETS = 3
M = 200_000
LAYERS = [1, 2, 4, 8, 16, 24, 30]


def gaussian_sampler(mu, Sigma, m, rng):
    """Draw m samples from N(mu, Sigma); Sigma is rank-deficient at depth."""
    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.maximum(evals, 0.0)
    L = evecs * np.sqrt(evals)
    return mu + rng.standard_normal((m, mu.size)).astype(np.float32) @ L.T.astype(np.float32)


def main():
    asset = np.load(ASSET)
    chirps = asset["chirps"].astype(np.float32)
    rotation = asset["rotation"].astype(np.float32)
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)
    rng = np.random.default_rng(5)

    errs = {li: [] for li in LAYERS}
    mus = {li: [] for li in LAYERS}
    for net in range(N_NETS):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        act = first_layer_design(weights[0], chirps, rotation)
        for li, w in enumerate(weights[1:], start=1):
            Z = act @ w
            if li in LAYERS and li + 1 <= DEPTH - 1:
                wn = weights[li + 1]
                # truth, from the propagated design rows
                A = np.maximum(Z, 0.0, dtype=np.float32)
                zt = A @ wn
                var_true = zt.var(axis=0, dtype=np.float64)
                mu_true = zt.mean(axis=0, dtype=np.float64)
                # Gaussian closure on z_l, using its exact empirical moments
                Z64 = Z.astype(np.float64)
                mu = Z64.mean(0)
                Sig = np.cov(Z64, rowvar=False, bias=True)
                g = gaussian_sampler(mu.astype(np.float32), Sig, M, rng)
                zg = np.maximum(g, 0.0, dtype=np.float32) @ wn
                var_cl = zg.var(axis=0, dtype=np.float64)
                mu_cl = zg.mean(axis=0, dtype=np.float64)
                # error in sigma (what the recursion consumes)
                rel_sd = np.sqrt(var_cl) / np.sqrt(var_true) - 1.0
                errs[li].append(np.sqrt(np.mean(rel_sd ** 2)))
                scale = np.sqrt(np.mean(mu_true ** 2))
                mus[li].append(np.sqrt(np.mean((mu_cl - mu_true) ** 2)) / scale)
            act = np.maximum(Z, 0.0, dtype=np.float32)

    print("One-step Gaussian-closure error, exact input moments, "
          f"{N_NETS} networks, M={M:,}\n")
    print(f"{'layer':>6} {'RMS sigma err':>15} {'RMS mu err':>13} "
          f"{'sigma req (4.34x)':>18}")
    for li in LAYERS:
        if errs[li]:
            print(f"{li:>6} {np.mean(errs[li]):>15.3e} {np.mean(mus[li]):>13.3e} "
                  f"{'1e-2':>18}")
    allsd = np.mean([np.mean(errs[li]) for li in LAYERS if errs[li]])
    print(f"\n  mean sigma closure error per layer : {allsd:.3e}")
    print(f"  MC noise floor of this measurement : {np.sqrt(2/M)/2:.3e}")
    print("\n  requirement: ~3e-3 for the 11x ceiling, ~1e-2 for a 4.34x win")


if __name__ == "__main__":
    main()
