"""EXACTLY-INTEGRABLE ANCHORS -- no per-layer moments at all.

My hybrid closure claimed "every hybrid needs per-layer moments, and moments cost
what the answer costs".  That is false for anchors whose Gaussian expectation is
known in closed form.  Because p_0 = x W_0 is EXACTLY Gaussian:

    E[a_0]      = ||w_j|| / sqrt(2 pi)                     exact
    E[a_0 a_0^T] = Cho-Saul pair moment of a rectified      exact
                   Gaussian with covariance W_0^T W_0

so both of these anchors have exactly computable means and need no sampling:

    linear      g = a_0 C
    quadratic   g = a_0 C + sum_p lambda_p (u_p . a_0)^2

and a_0 IS the design rows, so evaluating the anchor is nearly free:
n^2 + 2Pn per row against f's 2.587e6, i.e. cost factor c ~ 1.08 rather than 2.

Scored with the Gate A metric, since the design already annihilates low degrees:
    S_r / S_f = Var_b(Q_b(f-g)) / Var_b(Q_b f)
    improvement = 1 / (c * S_r/S_f)

Reported both fitted-on-all-rows (the ceiling) and fitted-on-R-blocks (honest).
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
PS = [0, 64, 256]            # rank of the quadratic part (0 = linear only)
R_FIT = 16                   # blocks used for the honest fit
RIDGE = 1e-6


def blocks(a):
    return a.reshape(NB, RPB, WIDTH).mean(1, dtype=np.float64)


def features(a0, U):
    """[a_0 , (U^T a_0)^2] -- the anchor's basis, evaluated on rows."""
    if U is None or U.shape[1] == 0:
        return a0
    q = (a0 @ U) ** 2
    return np.concatenate([a0, q], axis=1)


def main():
    asset = np.load(ASSET)
    chirps = asset["chirps"].astype(np.float32)
    rotation = asset["rotation"].astype(np.float32)
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)

    out = {(P, k): [] for P in PS for k in ("all", "fit")}
    for net in range(N_NETS):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        a0 = first_layer_design(weights[0], chirps, rotation)
        act = a0
        for w in weights[1:]:
            act = np.maximum(act @ w, 0.0, dtype=np.float32)
        F = act.astype(np.float64)
        Sf = np.mean(blocks(act).var(0))

        # principal directions of a_0 for the quadratic part
        A = a0.astype(np.float64)
        Am = A - A.mean(0)
        C0 = (Am.T @ Am) / A.shape[0]
        Uall = np.linalg.eigh(C0)[1][:, ::-1]

        for P in PS:
            U = Uall[:, :P].astype(np.float32) if P else None
            X = features(a0, U).astype(np.float64)
            for mode in ("all", "fit"):
                if mode == "all":
                    Xf, Ff = X, F
                else:
                    n = R_FIT * RPB
                    Xf, Ff = X[:n], F[:n]
                G = Xf.T @ Xf + RIDGE * len(Xf) * np.eye(X.shape[1])
                Cfit = np.linalg.solve(G, Xf.T @ Ff)
                Rres = (F - X @ Cfit).astype(np.float32)
                out[(P, mode)].append(np.mean(blocks(Rres).var(0)) / Sf)
        print(f"  net {net} done", flush=True)

    fpr = 2.5871e6
    print(f"\nEXACTLY-INTEGRABLE ANCHOR ({N_NETS} networks)\n")
    print(f"{'quad rank P':>12} {'cost factor c':>14} "
          f"{'S_r/S_f (fit all)':>19} {'S_r/S_f (fit 16 blk)':>21} {'improvement':>13}")
    for P in PS:
        c = 1.0 + (WIDTH * WIDTH + 2 * P * WIDTH) / fpr
        a = np.mean(out[(P, "all")])
        h = np.mean(out[(P, "fit")])
        print(f"{P:>12} {c:>14.4f} {a:>19.4f} {h:>21.4f} {1/(c*h):>12.2f}x")
    print("\n  improvement uses the HONEST (fit-on-16-blocks) column.")
    print("  4.34x needs c * S_r/S_f <= 0.115.  The anchor needs no moments,")
    print("  so its expectation is exact and contributes zero bias.")


if __name__ == "__main__":
    main()
