"""Cost floor for evaluating diag(W^T Cov_q(relu) W) WITHOUT forming Cov_q.

Via Mehler/Hermite, v_k = sum_d (1/d!) beta_d^T R^od beta_d, and the d=1 term is
gamma^T S gamma with gamma = c1(t) . w_k.  For all 256 k that costs n^3 unless S
is low rank, in which case 2 n^2 r.

Budget for K=1536 within the multiplier floor is 5.7e5 FLOPs per component per
layer, so 2 n^2 r <= 5.7e5 gives r <= 4.4 -- and that is only the FIRST Hermite
term.

So: what rank of S is actually needed for the next-layer variances?  Truncate S
to rank r, compute the variances exactly from the truncated S, compare.
"""
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, DATA
from gate1_update import cov_relu

N, CHUNK = 180_000, 20_000
LAYERS = [16, 29]
RANKS = [2, 4, 8, 16, 32, 64, 128]
N_NETS = 2


def main():
    tb = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(tb.column("weights").to_pylist(), dtype=np.float32)
    res = {(l, r): [] for l in LAYERS for r in RANKS}
    for net in range(N_NETS):
        rng = np.random.default_rng(7700 + net)
        wts = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        for tgt in LAYERS:
            buf, done = [], 0
            while done < N:
                m = min(CHUNK, N - done)
                h = rng.standard_normal((m, WIDTH), dtype=np.float32)
                for w in wts[:tgt + 1]:
                    z = h @ w
                    h = np.maximum(z, 0.0, dtype=np.float32)
                buf.append(z); done += m
            Z = np.concatenate(buf, 0).astype(np.float64); del buf
            wn = wts[tgt + 1].astype(np.float64)
            mu = Z.mean(0); Cc = Z - mu; S = (Cc.T @ Cc) / len(Z); del Cc
            Cex, _, _ = cov_relu(mu, S)
            vex = np.einsum("ij,ik,jk->k", Cex, wn, wn)
            ev, U = np.linalg.eigh(S)
            o = np.argsort(-ev); ev, U = ev[o], U[:, o]
            for r in RANKS:
                Sr = (U[:, :r] * np.maximum(ev[:r], 0)) @ U[:, :r].T
                # keep the true marginal variances: truncation must not change sigma
                d = np.sqrt(np.maximum(np.diag(S), 1e-300) /
                            np.maximum(np.diag(Sr), 1e-300))
                Sr = Sr * np.outer(d, d)
                Ca, _, _ = cov_relu(mu, Sr)
                va = np.einsum("ij,ik,jk->k", Ca, wn, wn)
                rr = np.sqrt(np.maximum(va, 0) / np.maximum(vex, 1e-300)) - 1.0
                res[(tgt, r)].append(float(np.sqrt(np.mean(rr ** 2))))
            del Z
        print(f"  net {net} done", flush=True)

    print("\nRANK OF S NEEDED FOR THE NEXT-LAYER VARIANCES\n")
    print(f"{'layer':>6} " + " ".join(f"{'r='+str(r):>10}" for r in RANKS))
    for l in LAYERS:
        print(f"{l:>6} " + " ".join(f"{np.mean(res[(l,r)]):>10.3e}" for r in RANKS))
    print("\n  budget: sigma error <= 1.5e-3, and the affordable rank is r <= 4.4")
    print("  (2 n^2 r <= 5.7e5), for the FIRST Hermite term alone.")


if __name__ == "__main__":
    main()
