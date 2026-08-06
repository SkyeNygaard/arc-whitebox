"""Minimise the hybrid objective  c * (S_r / S_f).

For E[f] = E[g] + E[f-g] with the residual on R complete blocks,

    score_hybrid / score_direct = c * (S_r / S_f)       (independent of R)

where c is the cost factor per residual block.  Full smoothing forces c = 2,
because f and g diverge from layer 0 and need two separate propagations.

If g smooths only layers L..31 and uses the exact ReLU below L, then f and g
SHARE THE TRUNK: propagate once to L-1, then branch.

    c = (L + 2*(32 - L)) / 32 = (64 - L) / 32

L=24 -> c=1.25, L=28 -> c=1.125.  The residual also shrinks with L, but so does
the anchor's tractability, so the objective has an interior optimum.

Reported: S_r/S_f, c, the objective, and the implied improvement 1/objective.
This assumes a free, unbiased analytic anchor -- the anchor bias is measured
separately and must be added before any deployment claim.
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
N_NETS = 6
SQRT2PI = np.sqrt(2 * np.pi)
LS = [0, 8, 16, 20, 24, 28, 30]          # first smoothed layer
ALPHAS = [0.25, 0.5, 1.0, 2.0]


def blocks(act):
    return act.reshape(NB, ROWS_PER_BASIS, WIDTH).mean(1, dtype=np.float64)


def main():
    asset = np.load(ASSET)
    chirps = asset["chirps"].astype(np.float32)
    rotation = asset["rotation"].astype(np.float32)
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)

    Sr = {(L, a): [] for L in LS for a in ALPHAS}
    Sf = []
    for net in range(N_NETS):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        rows = first_layer_design(weights[0], chirps, rotation)
        # exact trunk, cached at each branch point
        trunk = {}
        act = rows
        for li, w in enumerate(weights[1:], start=1):
            if li in [L + 1 for L in LS] or li == 1:
                trunk[li - 1] = act
            act = np.maximum(act @ w, 0.0, dtype=np.float32)
        F = act
        Bf = blocks(F)
        Sf.append(np.mean(Bf.var(0)))
        for L in LS:
            start = max(L, 0)
            base = trunk.get(start, rows)
            for a in ALPHAS:
                g = base
                for w in weights[start + 1:]:
                    p = (g @ w).astype(np.float64)
                    s = a * p.std(0) + 1e-30
                    t = p / s
                    g = np.maximum(p * ndtr(t) + s * np.exp(-0.5 * t * t) / SQRT2PI,
                                   0.0).astype(np.float32)
                D = (F.astype(np.float64) - g.astype(np.float64)).astype(np.float32)
                Sr[(L, a)].append(np.mean(blocks(D).var(0)))
        print(f"  net {net} done", flush=True)

    sf = np.mean(Sf)
    print(f"\nPARTIAL SMOOTHING -- objective c * (S_r/S_f), {N_NETS} networks\n")
    print(f"{'first smoothed layer':>21} {'c':>6} " +
          " ".join(f"{'a='+str(a):>16}" for a in ALPHAS))
    best = None
    for L in LS:
        c = (64 - L) / 32.0
        cells = []
        for a in ALPHAS:
            r = np.mean(Sr[(L, a)]) / sf
            obj = c * r
            cells.append(f"{r:.3f}/{1/obj:>5.2f}x")
            if best is None or obj < best[0]:
                best = (obj, L, a, r)
        print(f"{L:>21} {c:>6.3f} " + " ".join(f"{x:>16}" for x in cells))
    print("\n  cell = S_r/S_f  /  implied improvement 1/(c * S_r/S_f)")
    print(f"\n  best: L={best[1]}, alpha={best[2]}, S_r/S_f={best[3]:.3f}, "
          f"improvement {1/best[0]:.2f}x")
    print(f"  4.34x needs c*(S_r/S_f) <= 0.115; 5.00x needs <= 0.100")
    print("  (assumes a free, unbiased analytic anchor -- bias measured separately)")


if __name__ == "__main__":
    main()
