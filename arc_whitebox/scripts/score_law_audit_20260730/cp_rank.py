"""Can the third-cumulant tensor be contracted cheaply?

The covariance closure is the blocker (calib.py), and fixing it needs bivariate
third-order information: contractions of T_abc = E[c_a c_b c_c] with the weight
matrix.  Done densely that is an n^4 contraction, 4.3e9 FLOPs/layer = 1.3e11
total -- about half the budget, which forfeits the 10x multiplier floor and caps
the whole route near 2x.

With a Tucker/CP factorisation T ~ G x1 U x2 U x3 U at rank R the cost becomes
n^2 R (project) + n R^3 (contract):

    R = 32   ~1.0e7 /layer   3.2e8  total    comfortably under the floor
    R = 64   ~6.9e7 /layer   2.1e9  total    comfortably under the floor
    R = 128  ~5.4e8 /layer   1.7e10 total    62% of the floor, tight

So the route is affordable iff a rank <= 64 factorisation captures T.

Measured by randomized range finding: v_k = T x2 g_k x3 h_k for random Gaussian
g,h.  Since E[vec(g⊗h)vec(g⊗h)^T] = I, the sketch is isotropic and the singular
values of [v_1..v_K] track the mode-1 spectrum of T.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, ASSET, DATA, first_layer_design

N_NETS = 3
K = 400                       # sketch columns
LAYERS = [1, 4, 8, 16, 24, 31]
RS = [8, 16, 32, 64, 128]


def main():
    asset = np.load(ASSET)
    chirps = asset["chirps"].astype(np.float32)
    rotation = asset["rotation"].astype(np.float32)
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)
    rng = np.random.default_rng(31)

    cap = {li: [] for li in LAYERS}
    for net in range(N_NETS):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        act = first_layer_design(weights[0], chirps, rotation)
        for li, w in enumerate(weights[1:], start=1):
            Z = act @ w
            A = np.maximum(Z, 0.0, dtype=np.float32)
            if li in LAYERS:
                C = (A - A.mean(0, dtype=np.float64)).astype(np.float32)
                G = rng.standard_normal((WIDTH, K), dtype=np.float32)
                H = rng.standard_normal((WIDTH, K), dtype=np.float32)
                V = C.T @ ((C @ G) * (C @ H)) / C.shape[0]      # (n, K)
                s = np.linalg.svd(V, compute_uv=False)
                e = s ** 2
                cap[li].append([e[:r].sum() / e.sum() for r in RS])
            act = A

    print(f"mode-1 energy of the third-cumulant tensor captured by the top-r "
          f"subspace\n({N_NETS} networks, sketch K={K})\n")
    print(f"{'layer':>6} " + " ".join(f"{'r='+str(r):>9}" for r in RS))
    for li in LAYERS:
        m = np.mean(cap[li], axis=0)
        print(f"{li:>6} " + " ".join(f"{v:>9.4f}" for v in m))
    print("\nA cheap bivariate closure needs ~1-1e-3 capture at r <= 64.")


if __name__ == "__main__":
    main()
