"""HONEST hybrid accounting: every ingredient paid for out of the same R blocks.

The alpha-scan's score column was invalid because the analytic anchor drew its
moments from all 66,048 rows while only R blocks were charged.  Here the anchor's
sigma and kappa come from the SAME R blocks that carry the residual, so nothing
is free:

    estimate = E[g]_analytic(moments from R blocks) + mean_R(f - g)
    cost     = 2R blocks   (f and g both propagated on those rows)

Baseline at matched cost: the direct estimator on 2R blocks.

Both are scored against the baked ground truth, not against each other.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, ASSET, DATA, first_layer_design

RPB = 512
NB = 129
N_NETS = 10
RS = [4, 8, 16, 32, 64]
ALPHAS = [0.0, 0.1, 0.2, 0.35]
SQRT2PI = np.sqrt(2 * np.pi)


def smooth(p, s):
    t = p / s
    return p * ndtr(t) + s * np.exp(-0.5 * t * t) / SQRT2PI


def hybrid(weights, rows, alpha):
    """Propagate g_alpha on `rows`, returning (analytic anchor, g's final mean)."""
    act = rows
    mu = act.mean(0, dtype=np.float64)
    for w in weights:
        p = (act @ w).astype(np.float64)
        m = p.mean(0)
        c = p - m
        sd = np.sqrt(np.maximum((c ** 2).mean(0), 1e-300))
        k3 = (c ** 3).mean(0) / sd ** 3
        k4 = (c ** 4).mean(0) / sd ** 4 - 3.0
        eff = np.sqrt(sd ** 2 + (alpha * sd) ** 2)
        t = (mu @ w) / eff
        ph = np.exp(-0.5 * t * t) / SQRT2PI
        mu = eff * (t * ndtr(t) + ph - t * ph * k3 / 6.0
                    + (t * t - 1.0) * ph * k4 / 24.0)
        act = (np.maximum(p, 0.0) if alpha == 0.0
               else np.maximum(smooth(p, alpha * sd + 1e-300), 0.0)).astype(np.float32)
    return mu, act.mean(0, dtype=np.float64)


def main():
    asset = np.load(ASSET)
    chirps = asset["chirps"].astype(np.float32)
    rotation = asset["rotation"].astype(np.float32)
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)
    Y_all = np.asarray(table.column("final_means").to_pylist(), dtype=np.float64)

    hy = {(R, a): [] for R in RS for a in ALPHAS}
    di = {R: [] for R in RS}
    for net in range(N_NETS):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        full = first_layer_design(weights[0], chirps, rotation)
        act = full
        for w in weights[1:]:
            act = np.maximum(act @ w, 0.0, dtype=np.float32)
        Fb = act.reshape(NB, RPB, WIDTH).mean(1, dtype=np.float64)   # per-block f
        for R in RS:
            # matched-cost baseline: direct on 2R blocks
            di[R].append(np.mean((Fb[:min(2 * R, NB)].mean(0) - Y_all[net]) ** 2))
            rows = full[: R * RPB]
            fR = Fb[:R].mean(0)
            for a in ALPHAS:
                anchor, gR = hybrid(weights[1:], rows, a)
                est = anchor + (fR - gR)
                hy[(R, a)].append(np.mean((est - Y_all[net]) ** 2))
        print(f"  net {net} done", flush=True)

    print(f"\nHONEST HYBRID -- all ingredients from the same R blocks "
          f"({N_NETS} networks)\n")
    print(f"{'R':>4} {'direct @2R blocks':>19} " +
          " ".join(f"{'hybrid a='+str(a):>16}" for a in ALPHAS))
    for R in RS:
        d = np.mean(di[R])
        print(f"{R:>4} {d:>19.4e} " +
              " ".join(f"{np.mean(hy[(R,a)]):>16.4e}" for a in ALPHAS))
    print("\n  Both columns are MSE against the baked ground truth at matched cost.")
    print("  A win requires a hybrid cell below the direct column in its row.")


if __name__ == "__main__":
    main()
