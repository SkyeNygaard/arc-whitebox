"""Does the activation law collapse to low dimension?

Two claims in the corpus are in tension:
  * notes/03: "participation ratio ~ 2.7" at the last layers (rank collapse),
    which would make an EXACT low-dimensional quadrature possible -- no Gaussian
    closure, no moment truncation, cost K*2n^2 per layer, far under the floor.
  * mlmc.py (this session): a rank-128 projection still loses 51% of the
    pointwise output variance.

Both can hold if the spectrum is one huge eigenvalue plus a long flat tail:
the participation ratio is dominated by the top eigenvalue, while the tail
carries the variance that actually matters.  Measured here directly.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, ASSET, DATA, first_layer_design

N_NETS = 4
LAYERS = [1, 2, 4, 8, 16, 24, 31]
RS = [1, 2, 4, 8, 16, 32, 64, 128]


def main():
    asset = np.load(ASSET)
    chirps = asset["chirps"].astype(np.float32)
    rotation = asset["rotation"].astype(np.float32)
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)

    cap = {li: [] for li in LAYERS}
    pr = {li: [] for li in LAYERS}
    for net in range(N_NETS):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        act = first_layer_design(weights[0], chirps, rotation)
        for li, w in enumerate(weights[1:], start=1):
            Z = act @ w
            if li in LAYERS:
                Z64 = Z.astype(np.float64)
                C = np.cov(Z64, rowvar=False, bias=True)
                ev = np.sort(np.linalg.eigvalsh(C))[::-1]
                ev = np.maximum(ev, 0.0)
                tot = ev.sum()
                cap[li].append([ev[:r].sum() / tot for r in RS])
                pr[li].append(tot ** 2 / (ev ** 2).sum())
            act = np.maximum(Z, 0.0, dtype=np.float32)

    print("fraction of pre-activation variance captured by the top-r eigenspace")
    print(f"({N_NETS} networks, exact covariance of 66,048 propagated design rows)\n")
    print(f"{'layer':>6} " + " ".join(f"{'r='+str(r):>8}" for r in RS) + f" {'part.ratio':>11}")
    for li in LAYERS:
        m = np.mean(cap[li], axis=0)
        print(f"{li:>6} " + " ".join(f"{v:>8.4f}" for v in m) + f" {np.mean(pr[li]):>11.2f}")

    print("\nFor an exact low-dimensional quadrature to replace moment closure,")
    print("the top-r capture must reach ~1-1e-3 at an r whose cost fits the floor.")


if __name__ == "__main__":
    main()
