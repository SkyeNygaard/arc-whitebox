"""Where in depth is the blockwise variance created?

Anchor at layer k with the best linear readout,  g_k = a_k C_k,  fitted by least
squares.  g_k reproduces every kink up to layer k and linearises everything
after it, so

    S_r(k)/S_f  =  the fraction of blockwise variance NOT explained by the first
                   k layers' geometry

k = 0 is the exactly-integrable anchor (measured 0.97).  k = 31 is exact.  The
shape of the curve says how deep an anchor must reach to have blockwise power --
and therefore how much of the unknown deep distribution any viable anchor must
carry.

Cost of g_k: propagate to layer k plus one matmul, c = 1 + (k+1)/31.
Against it, the anchor's mean needs E[a_k], which is exact only at k = 0 and
otherwise costs the analytic chain's accumulated error (~1.6e-2 relative by the
final layer, i.e. ~8e-5 MSE) or sampled moments.

Fitted on 16 blocks (honest) and on all rows (ceiling).
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, ASSET, DATA, first_layer_design

RPB = 512
NB = 129
N_NETS = 6
KS = [0, 2, 4, 8, 16, 22, 26, 28, 30]
R_FIT = 16
RIDGE = 1e-6


def blocks(a):
    return a.reshape(NB, RPB, WIDTH).mean(1, dtype=np.float64)


def main():
    asset = np.load(ASSET)
    chirps = asset["chirps"].astype(np.float32)
    rotation = asset["rotation"].astype(np.float32)
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)

    res = {(k, m): [] for k in KS for m in ("all", "fit")}
    for net in range(N_NETS):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        a = first_layer_design(weights[0], chirps, rotation)
        cache = {0: a}
        for li, w in enumerate(weights[1:], start=1):
            a = np.maximum(a @ w, 0.0, dtype=np.float32)
            if li in KS:
                cache[li] = a
        F = a.astype(np.float64)
        Sf = np.mean(blocks(a).var(0))
        for k in KS:
            X = cache[k].astype(np.float64)
            for mode in ("all", "fit"):
                n = len(X) if mode == "all" else R_FIT * RPB
                G = X[:n].T @ X[:n] + RIDGE * n * np.eye(WIDTH)
                C = np.linalg.solve(G, X[:n].T @ F[:n])
                Rr = (F - X @ C).astype(np.float32)
                res[(k, mode)].append(np.mean(blocks(Rr).var(0)) / Sf)
        print(f"  net {net} done", flush=True)

    print(f"\nBLOCKWISE VARIANCE BY ANCHOR DEPTH ({N_NETS} networks)\n")
    print(f"{'anchor layer k':>15} {'cost c':>8} {'S_r/S_f (all)':>15} "
          f"{'S_r/S_f (honest)':>18} {'improvement':>13} {'E[a_k] exact?':>15}")
    for k in KS:
        c = 1.0 + (k + 1) / 31.0
        a_ = np.mean(res[(k, "all")])
        h = np.mean(res[(k, "fit")])
        print(f"{k:>15} {c:>8.3f} {a_:>15.4f} {h:>18.4f} {1/(c*h):>12.2f}x "
              f"{('YES' if k == 0 else 'no'):>15}")
    print("\n  improvement is the zero-bias bound; only k=0 has an exact mean.")
    print("  4.34x needs c * S_r/S_f <= 0.115.")


if __name__ == "__main__":
    main()
