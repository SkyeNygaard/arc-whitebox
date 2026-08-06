"""Is the non-Gaussianity that breaks covariance propagation LOCAL or ACCUMULATED?

cov_closure.py showed the Gaussian closure mis-states sigma by ~1.1% per layer
(0.44% early, 1.66% deep) against a 0.3% requirement.  The decisive question for
the whole analytic route is where that non-Gaussianity comes from.

Gaussianize at layer  L - lag  instead of at  L, then let the true relu dynamics
run forward for `lag+1` layers before reading off sigma at L+1:

  lag = 0   the standard one-layer closure
  lag = k   a k-step delayed closure -- the distribution is allowed to develop
            its own non-Gaussian shape before being re-Gaussianized

If the error collapses with small lag, the non-Gaussianity is created locally by
the previous relu and a short-memory closure fixes it.  If it decays slowly, the
activation law carries long-range structure and no finite-memory analytic
closure can reach the target.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, ASSET, DATA, first_layer_design

N_NETS = 3
M = 200_000
TARGET = 30            # predict Var(z_{TARGET+1})
LAGS = [0, 1, 2, 3, 4, 6, 8, 12]


def gaussian_sample(mu, Sigma, m, rng):
    evals, evecs = np.linalg.eigh(Sigma)
    L = (evecs * np.sqrt(np.maximum(evals, 0.0))).astype(np.float32)
    return mu.astype(np.float32) + rng.standard_normal((m, mu.size), dtype=np.float32) @ L.T


def main():
    asset = np.load(ASSET)
    chirps = asset["chirps"].astype(np.float32)
    rotation = asset["rotation"].astype(np.float32)
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)
    rng = np.random.default_rng(17)

    need = sorted({TARGET - lag for lag in LAGS})
    out = {lag: [] for lag in LAGS}

    for net in range(N_NETS):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        act = first_layer_design(weights[0], chirps, rotation)
        stash = {}
        var_true = None
        for li, w in enumerate(weights[1:], start=1):
            Z = act @ w
            if li in need:
                stash[li] = Z.astype(np.float64)
            if li == TARGET + 1:
                var_true = Z.var(axis=0, dtype=np.float64)
            act = np.maximum(Z, 0.0, dtype=np.float32)

        for lag in LAGS:
            L0 = TARGET - lag
            Z0 = stash[L0]
            g = gaussian_sample(Z0.mean(0), np.cov(Z0, rowvar=False, bias=True), M, rng)
            for w in weights[L0 + 1: TARGET + 2]:
                g = np.maximum(g, 0.0, dtype=np.float32) @ w
            rel = np.sqrt(g.var(axis=0, dtype=np.float64)) / np.sqrt(var_true) - 1.0
            out[lag].append(float(np.sqrt(np.mean(rel ** 2))))

    print(f"sigma error at layer {TARGET+1}, Gaussianized `lag` layers earlier")
    print(f"{N_NETS} networks, M={M:,}, MC floor {np.sqrt(2/M)/2:.2e}\n")
    print(f"{'lag':>4} {'RMS sigma err':>15} {'vs lag 0':>10}")
    base = np.mean(out[0])
    for lag in LAGS:
        v = np.mean(out[lag])
        print(f"{lag:>4} {v:>15.3e} {base/v:>9.2f}x")
    print("\n  requirement: 3e-3 for the 11x ceiling, ~1e-2 for a 4.34x win")


if __name__ == "__main__":
    main()
