"""Multilevel (telescoping) estimator: the one construction that can beat score = V*f/B.

    E[f] = E[f_coarse]  +  E[f - f_coarse]
           \___ N0 rows at cheap cost f0 ___/   \___ N1 rows at f0+f1 ___/

Optimal allocation gives   score = (sqrt(V0*f0) + sqrt(Vd*f1))^2 / B,
so the ceiling as Vd -> 0 is exactly the cost ratio f/f0.  The whole question is
how fast the *difference* variance Vd falls -- and the earlier rank-truncation
work measured the coarse level's BIAS, which a telescope cancels and therefore
does not care about.

Coarse level = rank-r subspace tracking of the activation batch.
Per row per layer:  2*r*n (propagate coefficients) + 2*n*r (reproject) + n
against the exact 2*n*n, i.e. a cost ratio of about n/(2r).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, ASSET, DATA, first_layer_design

RANKS = [8, 16, 32, 64, 128]
N_PROJ = 8192          # rows used to estimate each layer's subspace


def exact_chain(rows, weights):
    act = rows
    for w in weights:
        act = np.maximum(act @ w, 0.0, dtype=np.float32)
    return act


def coarse_chain(rows, weights, r, rng):
    """Propagate, projecting the centred activation onto a rank-r subspace each layer."""
    act = rows
    idx = rng.choice(act.shape[0], size=min(N_PROJ, act.shape[0]), replace=False)
    for w in weights:
        act = np.maximum(act @ w, 0.0, dtype=np.float32)
        mean = act.mean(0, dtype=np.float64).astype(np.float32)
        sample = act[idx] - mean
        # top-r right singular vectors via the 256x256 Gram matrix
        gram = (sample.T.astype(np.float64) @ sample.astype(np.float64))
        evals, evecs = np.linalg.eigh(gram)
        B = evecs[:, -r:].astype(np.float32)
        act = mean + (act - mean) @ B @ B.T
    return act


def cost_ratio(r, n=WIDTH):
    """exact FLOPs per row-layer / coarse FLOPs per row-layer."""
    return (2.0 * n * n) / (2.0 * r * n + 2.0 * n * r + n)


def main():
    asset = np.load(ASSET)
    chirps = asset["chirps"].astype(np.float32)
    rotation = asset["rotation"].astype(np.float32)
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)

    rng = np.random.default_rng(11)
    n_nets = 3
    res = {r: [] for r in RANKS}

    for net in range(n_nets):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        rows = first_layer_design(weights[0], chirps, rotation)
        t0 = time.time()
        f_exact = exact_chain(rows, weights[1:])
        v0 = f_exact.var(axis=0, dtype=np.float64).mean()
        print(f"net {net}: exact pass {time.time()-t0:.1f}s   row variance V0 = {v0:.5f}")
        for r in RANKS:
            f_coarse = coarse_chain(rows, weights[1:], r, rng)
            d = f_exact.astype(np.float64) - f_coarse.astype(np.float64)
            vd = d.var(axis=0).mean()
            res[r].append(vd / v0)
            print(f"    rank {r:>3}:  Vd/V0 = {vd/v0:.4f}   "
                  f"mean-shift(bias) = {np.abs(d.mean(0)).mean():.3e}")

    print("\n" + "=" * 74)
    print(f"{'rank':>5} {'Vd/V0':>9} {'cost ratio f/f0':>17} {'2-level score gain':>20}")
    print("=" * 74)
    for r in RANKS:
        ratio = float(np.mean(res[r]))
        cr = cost_ratio(r)
        # score_ml / score_direct = (sqrt(1/cr) + sqrt(ratio*(1+1/cr)))^2
        gain = 1.0 / (np.sqrt(1.0 / cr) + np.sqrt(ratio * (1.0 + 1.0 / cr))) ** 2
        print(f"{r:>5} {ratio:>9.4f} {cr:>17.2f}x {gain:>19.3f}x")
    print("\nceiling as Vd->0 is the cost ratio; 4.34x is needed to take the lead.")


if __name__ == "__main__":
    main()
