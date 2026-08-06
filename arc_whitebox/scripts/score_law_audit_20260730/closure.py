"""Where exactly does analytic moment propagation break?

The analytic route is the only one the harmonic obstruction does not touch, and
its whole cost is O(depth * width^3) -- below the 0.1 multiplier floor, so it
scores 0.1 * MSE with every further FLOP free.  Its measured ceiling with oracle
moments is 1.30e-7 (score 1.30e-8, 11x better than shipped).

The recursion is
    mu_{l+1} = W_l^T E[relu(z_l)],   Sigma_{l+1} = W_l^T Cov(relu(z_l)) W_l
so everything turns on the CLOSURE: how accurately can E[relu(z)] be recovered
from a few moments of z?  This isolates closure error from accumulation error by
feeding the closure the EXACT empirical moments at every layer, measured from
66,048 propagated design rows.

Measured per layer:
  * Gaussian closure     E[relu] = sigma (t Phi(t) + phi(t))
  * Edgeworth closure    + sigma [ -t phi(t) k3/6 + (t^2-1) phi(t) k4/24 ]
and, critically, whether the residual is UNIVERSAL across networks -- because a
universal per-layer constant is free at test time (notes/03 §5.2 found the sigma
bias universal to 3e-4, but tested the fix with 6k-sample cumulants and it
failed for that reason; it has never been retested with accurate cumulants).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, ASSET, DATA, first_layer_design

N_NETS = 8
SQRT2 = math.sqrt(2.0)


def phi(t):
    return np.exp(-0.5 * t * t) / math.sqrt(2 * math.pi)


def Phi(t):
    from math import erf
    return 0.5 * (1.0 + np.vectorize(erf)(t / SQRT2))


def closures(Z):
    """Empirical E[relu(z)] and the two closure predictions, from exact moments."""
    Z = Z.astype(np.float64)
    mu = Z.mean(0)
    c = Z - mu
    var = (c ** 2).mean(0)
    sd = np.sqrt(var)
    k3 = (c ** 3).mean(0) / sd ** 3
    k4 = (c ** 4).mean(0) / sd ** 4 - 3.0
    t = mu / sd
    ph, PH = phi(t), Phi(t)
    gauss = sd * (t * PH + ph)
    edge = gauss + sd * (-t * ph * k3 / 6.0 + (t * t - 1.0) * ph * k4 / 24.0)
    truth = np.maximum(Z, 0.0).mean(0)
    return truth, gauss, edge, t, k3, k4


def main():
    asset = np.load(ASSET)
    chirps = asset["chirps"].astype(np.float32)
    rotation = asset["rotation"].astype(np.float32)
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)

    eg, ee, ratios, dead = [], [], [], []
    for net in range(N_NETS):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        act = first_layer_design(weights[0], chirps, rotation)
        g_l, e_l, r_l, d_l = [], [], [], []
        for li, w in enumerate(weights[1:], start=1):
            Z = act @ w
            truth, gauss, edge, t, k3, k4 = closures(Z)
            scale = np.sqrt(np.mean(truth ** 2))          # layer activation scale
            g_l.append(np.sqrt(np.mean((gauss - truth) ** 2)) / scale)
            e_l.append(np.sqrt(np.mean((edge - truth) ** 2)) / scale)
            # signed relative bias -- the part a universal constant could fix
            r_l.append(float(np.sum(gauss - truth) / np.sum(truth)))
            d_l.append(float(np.mean(truth <= 0)))
            act = np.maximum(Z, 0.0, dtype=np.float32)
        eg.append(g_l)
        ee.append(e_l)
        ratios.append(r_l)
        dead.append(d_l)

    eg, ee, ratios, dead = np.array(eg), np.array(ee), np.array(ratios), np.array(dead)
    print("RMS relative error of the one-step closure for E[relu(z)],")
    print("given EXACT moments at that layer (so this is pure closure error):\n")
    print(f"{'layer':>6} {'Gaussian':>11} {'Edgeworth':>11} {'gain':>7} "
          f"{'signed bias':>13} {'spread/nets':>13} {'dead':>7}")
    for li in [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 31]:
        k = li - 1
        print(f"{li:>6} {eg[:,k].mean():>11.3e} {ee[:,k].mean():>11.3e} "
              f"{eg[:,k].mean()/ee[:,k].mean():>7.2f}x {ratios[:,k].mean():>13.3e} "
              f"{ratios[:,k].std():>13.3e} {dead[:,k].mean():>7.1%}")

    print("\nWhat matters is the per-layer error that survives a universal correction:")
    resid_g = eg.mean(0) ** 2 - ratios.mean(0) ** 2
    resid_g = np.sqrt(np.maximum(resid_g, 0))
    print(f"  mean RMS Gaussian closure error, layers 2-31          : {eg[:,1:].mean():.3e}")
    print(f"  mean RMS Edgeworth closure error, layers 2-31         : {ee[:,1:].mean():.3e}")
    print(f"  after removing a universal per-layer constant (Gauss) : {resid_g[1:].mean():.3e}")
    print(f"  universality of the bias (std/mean across nets)       : "
          f"{np.mean(ratios[:,1:].std(0)/np.abs(ratios[:,1:].mean(0))):.3f}")
    print("\nRequirement: the final mean needs ~1e-3 relative for a 4.34x win,")
    print("~1e-4 for the 11x ceiling. Errors compound over 31 layers.")


if __name__ == "__main__":
    main()
