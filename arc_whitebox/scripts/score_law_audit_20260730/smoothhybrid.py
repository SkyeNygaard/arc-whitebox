"""THE CANDIDATE: analytic smoothed-network anchor + structured residual blocks.

    E[f] = E[g_alpha]  +  E[f - g_alpha]
           \_ analytic _/    \_ R complete Kerdock blocks _/

g_alpha is f with every ReLU replaced by its Gaussian-smoothed version at scale
s = alpha * (layer std).  Two properties make it the right anchor:

  * for Gaussian p, E[smoothed-relu(p)] = E[relu(N(mu, sigma^2 + s^2))] EXACTLY,
    so the analytic chain for g is the ordinary ReLU chain with an inflated
    variance -- no new machinery, and smoothing damps the sensitivity to the
    joint non-Gaussianity that broke Gate 1;
  * it keeps f's kinks, so the residual f - g inherits the design's cancellation.
    Gate A: blockwise S_r/S_f = 0.242 at alpha = 0.5 (against ~1.0 for every
    rank anchor).

The trade-off is explicit: alpha -> 0 shrinks the residual but makes E[g] as hard
as E[f]; large alpha makes E[g] easy but the residual big.  Measured here:

  bias(alpha) = | analytic E[g] - design E[g] |    (anchor quality)
  S_r(alpha)  = Var_b( Q_b(f - g) )                (residual economics)
  score(alpha, R) = [bias^2 + S_r/R] * max(0.1, C(R)/B)

Cost model: analytic chain ~3e9 FLOPs; each residual block needs BOTH f and g on
its 512 rows, so 2 x 512 x 2.587e6 per block.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, ASSET, DATA, first_layer_design

ROWS_PER_BASIS = 512
NB = 129
N_NETS = 8
ALPHAS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
SQRT2PI = np.sqrt(2 * np.pi)
FPR = 170_875_096_064 / 66_048
B = 272_000_000_000
ANALYTIC_FLOPS = 3.0e9


def smooth_relu(p, s):
    t = p / s
    return p * ndtr(t) + s * np.exp(-0.5 * t * t) / SQRT2PI


def propagate(weights, rows, alpha):
    """Return final activations of g_alpha (alpha=0 gives f exactly)."""
    act = rows
    for w in weights:
        p = (act @ w).astype(np.float64)
        if alpha == 0.0:
            act = np.maximum(p, 0.0).astype(np.float32)
        else:
            s = alpha * p.std(0) + 1e-30
            act = np.maximum(smooth_relu(p, s), 0.0).astype(np.float32)
    return act


def analytic_anchor(weights, rows, alpha):
    """Analytic E[g_alpha]: Edgeworth chain with variance inflated by s^2.

    Sigma and kappa are read from the SAME rows (they are produced anyway while
    evaluating the residual blocks); mu is propagated analytically.
    """
    act = rows
    mu = act.mean(0, dtype=np.float64)
    for w in weights:
        p = (act @ w).astype(np.float64)
        m = p.mean(0)
        c = p - m
        sd = np.sqrt(np.maximum((c ** 2).mean(0), 1e-300))
        k3 = (c ** 3).mean(0) / sd ** 3
        k4 = (c ** 4).mean(0) / sd ** 4 - 3.0
        s = alpha * sd
        eff = np.sqrt(sd ** 2 + s ** 2)          # exact for the smoothed ReLU
        mu_l = mu @ w
        t = mu_l / eff
        ph = np.exp(-0.5 * t * t) / SQRT2PI
        mu = eff * (t * ndtr(t) + ph
                    - t * ph * k3 / 6.0 + (t * t - 1.0) * ph * k4 / 24.0)
        if alpha == 0.0:
            act = np.maximum(p, 0.0).astype(np.float32)
        else:
            act = np.maximum(smooth_relu(p, alpha * p.std(0) + 1e-30), 0.0).astype(np.float32)
    return mu


def blocks(act):
    return act.reshape(NB, ROWS_PER_BASIS, WIDTH).mean(1, dtype=np.float64)


def main():
    asset = np.load(ASSET)
    chirps = asset["chirps"].astype(np.float32)
    rotation = asset["rotation"].astype(np.float32)
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)
    Y_all = np.asarray(table.column("final_means").to_pylist(), dtype=np.float64)

    bias, Sr, Sf = {a: [] for a in ALPHAS}, {a: [] for a in ALPHAS}, []
    for net in range(N_NETS):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        rows = first_layer_design(weights[0], chirps, rotation)
        F = propagate(weights[1:], rows, 0.0)
        Bf = blocks(F)
        Sf.append(np.mean(Bf.var(0)))
        for a in ALPHAS:
            G = F if a == 0.0 else propagate(weights[1:], rows, a)
            Eg_design = G.mean(0, dtype=np.float64)
            Eg_analytic = analytic_anchor(weights[1:], rows, a)
            bias[a].append(np.mean((Eg_analytic - Eg_design) ** 2))
            D = (F.astype(np.float64) - G.astype(np.float64)).astype(np.float32)
            Sr[a].append(np.mean(blocks(D).var(0)))
        print(f"  net {net} done", flush=True)

    sf = np.mean(Sf)
    direct_mse = sf / NB
    direct_score = direct_mse * max(0.1, NB * ROWS_PER_BASIS * FPR / B)
    print(f"\nSMOOTHED-ANCHOR HYBRID ({N_NETS} networks)\n")
    print(f"  direct: MSE {direct_mse:.4e}  score {direct_score:.4e}\n")
    print(f"{'alpha':>7} {'anchor bias MSE':>17} {'S_r/S_f':>9} "
          f"{'best R':>7} {'best MSE':>11} {'best score':>12} {'vs direct':>10}")
    for a in ALPHAS:
        b = np.mean(bias[a])
        sr = np.mean(Sr[a])
        best = None
        for R in range(1, NB + 1):
            C = ANALYTIC_FLOPS + R * ROWS_PER_BASIS * 2 * FPR
            mse = b + sr / R
            sc = mse * max(0.1, C / B)
            if best is None or sc < best[0]:
                best = (sc, R, mse)
        print(f"{a:>7.2f} {b:>17.4e} {sr/sf:>9.4f} {best[1]:>7d} "
              f"{best[2]:>11.4e} {best[0]:>12.4e} {direct_score/best[0]:>9.2f}x")
    print("\n  alpha=0 is the pure analytic chain (no smoothing, residual zero by")
    print("  construction) -- its bias column is the analytic chain's own error.")


if __name__ == "__main__":
    main()
