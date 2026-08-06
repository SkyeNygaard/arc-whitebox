"""THE HYBRID: sample the moments, propagate the mean analytically.

Gate 1 asked whether Sigma can be propagated ANALYTICALLY. It cannot (copula).
But that is the wrong question. Sigma and kappa need only ~3e-3 and ~3% relative
accuracy (notes/03 section 3), where the MEAN needs 1e-4. So:

    sigma_l, kappa_l   <- sampled from the propagated design rows
    mu_l               <- propagated ANALYTICALLY through the Edgeworth marginal
                          E[relu(p)] = sigma[ a0(t) + a3(t) k3/6 + a4(t) k4/24 ]
    mu_{l+1} = W^T E[relu(p_l)]

The covariance-closure error never enters, because Sigma is never closed.  Only
the MARGINAL closure enters, measured at 1.9e-4/layer and stable across depth.

And mu_{l+1,i} = sum_j W_ij E[relu(p_l,j)] contracts 256 independently noisy
sigma estimates, averaging their noise down by sqrt(256) = 16x every layer.
That is the mechanism -- the samples buy moments, not points.

Baseline for comparison at identical cost: the direct empirical mean of
relu(p_31) over the same rows, which is the shipped estimator.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, ASSET, DATA, first_layer_design

ROWS_PER_BASIS = 512
BASIS_SETS = [8, 16, 32, 64, 129]        # -> 4,096 .. 66,048 rows
N_NETS = 12
SQRT2PI = math.sqrt(2.0 * math.pi)


def phi(t):
    return np.exp(-0.5 * t * t) / SQRT2PI


def edgeworth_relu_mean(mu, sd, k3, k4, order):
    """E[relu(p)] from marginal moments. order=0 Gaussian, 3 adds k3, 4 adds k4."""
    t = mu / sd
    ph, PH = phi(t), ndtr(t)
    out = sd * (t * PH + ph)
    if order >= 3:
        out = out + sd * (-t * ph * k3 / 6.0)
    if order >= 4:
        out = out + sd * ((t * t - 1.0) * ph * k4 / 24.0)
    return out


def run(weights, rows, order):
    """Return (direct estimate, hybrid estimate)."""
    act = rows                                  # a_0, post-ReLU layer 0
    mu = act.mean(0, dtype=np.float64)          # E[a_0]: layer 0 is exact
    for li, w in enumerate(weights[1:], start=1):
        p = (act @ w).astype(np.float64)        # true rows at this layer
        # --- sampled marginal moments (this is what the rows are spent on)
        m = p.mean(0)
        c = p - m
        v = (c ** 2).mean(0)
        sd = np.sqrt(np.maximum(v, 1e-300))
        k3 = (c ** 3).mean(0) / sd ** 3
        k4 = (c ** 4).mean(0) / sd ** 4 - 3.0
        # --- analytic mean propagation: mu carried forward, never sampled
        mu_l = mu @ w
        Erelu = edgeworth_relu_mean(mu_l, sd, k3, k4, order)
        mu = Erelu
        act = np.maximum(p, 0.0).astype(np.float32)
    direct = act.mean(0, dtype=np.float64)      # shipped estimator
    return direct, mu


def main():
    asset = np.load(ASSET)
    chirps = asset["chirps"].astype(np.float32)
    rotation = asset["rotation"].astype(np.float32)
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)
    Y_all = np.asarray(table.column("final_means").to_pylist(), dtype=np.float64)

    res = {(nb, o): [] for nb in BASIS_SETS for o in (0, 3, 4)}
    dir_res = {nb: [] for nb in BASIS_SETS}
    for net in range(N_NETS):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        full = first_layer_design(weights[0], chirps, rotation)
        for nb in BASIS_SETS:
            rows = full[: nb * ROWS_PER_BASIS]
            for o in (0, 3, 4):
                d, h = run(weights, rows, o)
                res[(nb, o)].append(np.mean((h - Y_all[net]) ** 2))
                if o == 0:
                    dir_res[nb].append(np.mean((d - Y_all[net]) ** 2))
        print(f"  net {net} done", flush=True)

    FLOPS_PER_ROW = 170_875_096_064 / 66_048
    B = 272_000_000_000
    print(f"\nHYBRID: sampled moments + analytic mean ({N_NETS} networks)\n")
    print(f"{'rows':>7} {'C/B':>6} {'direct MSE':>12} {'hybrid G':>11} "
          f"{'hybrid +k3':>11} {'hybrid +k3k4':>13} {'best score':>12} {'vs direct':>10}")
    for nb in BASIS_SETS:
        n = nb * ROWS_PER_BASIS
        mult = max(0.1, n * FLOPS_PER_ROW / B)
        d = np.mean(dir_res[nb])
        hs = [np.mean(res[(nb, o)]) for o in (0, 3, 4)]
        best = min(hs)
        print(f"{n:>7} {mult:>6.3f} {d:>12.4e} {hs[0]:>11.4e} {hs[1]:>11.4e} "
              f"{hs[2]:>13.4e} {best*mult:>12.4e} {d*mult/(best*mult):>9.2f}x")
    print(f"\n  shipped estimator reference score: {np.mean(dir_res[129])*0.6488:.4e}")
    print("  notes/03 oracle-moment EMP ceiling : 1.30e-7 MSE")


if __name__ == "__main__":
    main()
