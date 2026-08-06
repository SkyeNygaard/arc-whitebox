"""Can the MARGINAL closure error be corrected universally?

The analytic chain's structural floor (2.5e-7) is set entirely by the Edgeworth
marginal closure, 1.9e-4/layer.  That is a ONE-DIMENSIONAL problem: E[relu(p)]
from the marginal law of a scalar p.  Unlike the joint/copula problem it lives in
a low-dimensional, smooth parameter space.

Key structural point: the closure error is a DETERMINISTIC FUNCTIONAL of the
marginal shape.  Two neurons with the same standardised (t, k3, k4) should incur
nearly the same error, differing only through higher cumulants which are small.
If so the error is a smooth universal function of a few standardised parameters,
fittable OFFLINE on training networks and applied for free at test time -- a
distributional prior over He-initialised networks, not per-network fitting.

This is the opposite regime from the sigma calibration that failed in section 3d:
there the error was per-neuron scatter with no shared structure.  Here it should
be shared by construction.

Measured on (network, layer, neuron) triples with leave-one-network-out.
The floor scales as the SQUARE of the per-layer error, so removing a fraction R2
of the error variance drops the floor by 1/(1-R2).
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, ASSET, DATA, first_layer_design

SQRT2PI = np.sqrt(2 * np.pi)
N_NETS = 14
LAYERS = list(range(1, 31))


def main():
    asset = np.load(ASSET)
    chirps = asset["chirps"].astype(np.float32)
    rotation = asset["rotation"].astype(np.float32)
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)

    F, T, G = [], [], []
    for net in range(N_NETS):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        act = first_layer_design(weights[0], chirps, rotation)
        for li, w in enumerate(weights[1:], start=1):
            p = (act @ w).astype(np.float64)
            m = p.mean(0)
            c = p - m
            sd = np.sqrt(np.maximum((c ** 2).mean(0), 1e-300))
            k3 = (c ** 3).mean(0) / sd ** 3
            k4 = (c ** 4).mean(0) / sd ** 4 - 3.0
            k5 = (c ** 5).mean(0) / sd ** 5
            k6 = (c ** 6).mean(0) / sd ** 6 - 15.0
            t = m / sd
            ph = np.exp(-0.5 * t * t) / SQRT2PI
            edge = sd * (t * ndtr(t) + ph - t * ph * k3 / 6.0
                         + (t * t - 1.0) * ph * k4 / 24.0)
            truth = np.maximum(p, 0.0).mean(0)
            live = sd > 1e-8 * sd.max()
            # standardised closure residual: the thing we want to predict
            r = (edge - truth)[live] / sd[live]
            F.append(np.stack([t[live], k3[live], k4[live], k5[live], k6[live],
                               np.full(live.sum(), li / 31.0)], 1))
            T.append(r)
            G.append(np.full(live.sum(), net))
            act = np.maximum(p, 0.0).astype(np.float32)
        print(f"  net {net} done", flush=True)

    X0 = np.concatenate(F); y = np.concatenate(T); g = np.concatenate(G)
    print(f"\nMARGINAL CLOSURE CORRECTION  ({N_NETS} networks, {len(y):,} neurons)\n")
    print(f"  RMS standardised residual (uncorrected): {np.sqrt(np.mean(y**2)):.4e}")

    def design(X, deg, nfeat):
        Z = X[:, :nfeat]
        cols = [np.ones(len(Z))]
        for d in range(1, deg + 1):
            for j in range(nfeat):
                cols.append(Z[:, j] ** d)
        for i in range(nfeat):
            for j in range(i + 1, nfeat):
                cols.append(Z[:, i] * Z[:, j])
                cols.append(Z[:, i] ** 2 * Z[:, j])
                cols.append(Z[:, i] * Z[:, j] ** 2)
        return np.stack(cols, 1)

    for name, nfeat, deg in [("t,k3,k4", 3, 4), ("t,k3,k4,k5,k6", 5, 4),
                             ("t,k3,k4,k5,k6,depth", 6, 4)]:
        Xd = design(X0, deg, nfeat)
        pred = np.zeros_like(y)
        for m in np.unique(g):
            tr = g != m
            A = Xd[tr].T @ Xd[tr] + 1e-8 * len(y[tr]) * np.eye(Xd.shape[1])
            pred[g == m] = Xd[g == m] @ np.linalg.solve(A, Xd[tr].T @ y[tr])
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum(y ** 2)
        rms = np.sqrt(np.mean((y - pred) ** 2))
        print(f"  features [{name:<22}] LORO R^2 = {r2:+.4f}   "
              f"residual RMS = {rms:.4e}   floor drops {1/max(1-r2,1e-9):.1f}x")

    print("\n  A universal offline fit is free at test time (a prior over")
    print("  He-initialised networks). Floor scales as the SQUARE of this error.")


if __name__ == "__main__":
    main()
