"""GATE A -- blockwise residual variance of a hybrid anchor.

For a hybrid  E[f] = E[g] + E[f-g]  evaluated on the Kerdock design, the object
that matters is NOT the pointwise variance of f-g.  Each complete antipodal
Kerdock basis Q_b already annihilates the low degrees, so what governs the
residual sampler is the variance ACROSS complete blocks:

    S_r = Var_b( Q_b(f-g) )      vs      S_f = Var_b( Q_b f )

My earlier MLMC result measured the pointwise ratio (0.974 / 0.812 / 0.508 at
rank 8 / 32 / 128) and concluded the class was dead.  That was the wrong metric:
it mixes in the low-degree content the design already kills.  Re-measured here
blockwise.

Anchors tested (all legal, all pointwise-evaluable on the same rows so the
residual inherits the design's cancellation):
  rank-r   activation-subspace propagation, r = 8, 32, 128
  smooth   same kinks, each ReLU replaced by its Gaussian-smoothed version at
           the layer's own measured scale -- matches f's coarse structure but
           deliberately attenuates the fine structure

Reported: S_r/S_f, and the resulting hybrid MSE if the residual is estimated
from R complete blocks with the anchor integrated over all 129.
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
N_PROJ = 8192


def blocks(act):
    return act.reshape(NB, ROWS_PER_BASIS, WIDTH).mean(1, dtype=np.float64)


def exact(weights, rows):
    act = rows
    for w in weights:
        act = np.maximum(act @ w, 0.0, dtype=np.float32)
    return act


def coarse_rank(weights, rows, r, rng):
    act = rows
    idx = rng.choice(act.shape[0], size=min(N_PROJ, act.shape[0]), replace=False)
    for w in weights:
        act = np.maximum(act @ w, 0.0, dtype=np.float32)
        mean = act.mean(0, dtype=np.float64).astype(np.float32)
        s = act[idx] - mean
        g = s.T.astype(np.float64) @ s.astype(np.float64)
        B = np.linalg.eigh(g)[1][:, -r:].astype(np.float32)
        act = mean + (act - mean) @ B @ B.T
    return act


def coarse_smooth(weights, rows, alpha):
    """Same kinks, ReLU -> Gaussian-smoothed ReLU at scale alpha * layer std."""
    act = rows
    for w in weights:
        p = act @ w
        s = alpha * p.std(0, dtype=np.float64).astype(np.float32) + 1e-20
        t = p / s
        act = (p * ndtr(t).astype(np.float32)
               + s * np.exp(-0.5 * t * t).astype(np.float32) / SQRT2PI)
        act = np.maximum(act, 0.0, dtype=np.float32)
    return act


def main():
    asset = np.load(ASSET)
    chirps = asset["chirps"].astype(np.float32)
    rotation = asset["rotation"].astype(np.float32)
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)
    Y_all = np.asarray(table.column("final_means").to_pylist(), dtype=np.float64)

    names = ["rank8", "rank32", "rank128", "smooth0.5", "smooth1.0"]
    Sr = {k: [] for k in names}
    point = {k: [] for k in names}
    Sf = []
    for net in range(N_NETS):
        rng = np.random.default_rng(900 + net)
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        rows = first_layer_design(weights[0], chirps, rotation)
        F = exact(weights[1:], rows)
        Bf = blocks(F)
        Sf.append(np.mean(Bf.var(0)))
        for k in names:
            if k.startswith("rank"):
                G = coarse_rank(weights[1:], rows, int(k[4:]), rng)
            else:
                G = coarse_smooth(weights[1:], rows, float(k[6:]))
            D = F.astype(np.float64) - G.astype(np.float64)
            Sr[k].append(np.mean(blocks(D.astype(np.float32)).var(0)))
            point[k].append(D.var(0).mean() / F.var(0, dtype=np.float64).mean())
        print(f"  net {net} done", flush=True)

    sf = np.mean(Sf)
    print(f"\nGATE A -- blockwise residual variance ({N_NETS} networks)\n")
    print(f"  S_f = Var_b(Q_b f) = {sf:.4e}\n")
    print(f"{'anchor':>11} {'S_r':>12} {'S_r/S_f (blockwise)':>21} "
          f"{'pointwise ratio':>17} {'blocks for parity':>18}")
    for k in names:
        sr = np.mean(Sr[k])
        print(f"{k:>11} {sr:>12.4e} {sr/sf:>21.4f} {np.mean(point[k]):>17.4f} "
              f"{129*sr/sf:>18.1f}")
    print("\n  'blocks for parity' = residual blocks R needed for the hybrid to")
    print("  match the full 129-block direct estimator. R < 129 means a win,")
    print("  but only if the anchor itself is cheap (Gate C).")


if __name__ == "__main__":
    main()
