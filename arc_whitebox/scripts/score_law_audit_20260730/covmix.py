"""M187 -- can a conditional COVARIANCE mixture represent the deep copula?

v28's primary hypothesis:   z | H=h ~ N(m_h, S_h),   p(z) ~ sum_q pi_q N(m_q,S_q)

This is a different class from what I tested before:
  * my latent test was a LOCATION mixture (mean shifted by B xi, covariance FIXED
    at Sigma_eps).  It needed r = 64-128 to reach 3e-3, i.e. an r-dimensional
    copula, and died on representation cost.
  * my scale-mixture test was the K=1 degenerate case (one global radial scale).
    It failed at 0.40x, which says nothing about K > 1.

A covariance mixture with K prototypes is O(K n^3) per layer to propagate:
K=8 gives ~4e9 FLOPs total, comfortably under the 2.72e10 multiplier floor.
So if this class reaches sigma <= 3e-3 at small K, the analytic route is live and
scores ~1.3e-8.

ORACLE-CLOSURE test: the mixture is fitted to reference samples of z_l, then used
to predict Var(z_{l+1}).  It never uses the answer -- it asks only whether the
MODEL CLASS can represent the truth.  K=1 is exactly the Gaussian closure.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, DATA

N = 400_000
CHUNK = 20_000
LAYERS = [4, 8, 16, 24, 29]
KS = [1, 2, 4, 8, 16, 32]
N_NETS = 3
PC = 8            # cluster in the top-PC subspace (cheap, and legally derivable)


def kmeans(X, K, rng, iters=25):
    if K == 1:
        return np.zeros(X.shape[0], dtype=np.int64)
    c = X[rng.choice(X.shape[0], K, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for q in range(K):
            m = lab == q
            if m.any():
                c[q] = X[m].mean(0)
    return lab


def main():
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)
    res = {(l, K): [] for l in LAYERS for K in KS}

    for net in range(N_NETS):
        rng = np.random.default_rng(3100 + net)
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        for target in LAYERS:
            buf, done = [], 0
            while done < N:
                m = min(CHUNK, N - done)
                h = rng.standard_normal((m, WIDTH), dtype=np.float32)
                for w in weights[:target + 1]:
                    z = h @ w
                    h = np.maximum(z, 0.0, dtype=np.float32)
                buf.append(z)
                done += m
            Z = np.concatenate(buf, 0); del buf
            wn = weights[target + 1]
            var_true = ((np.maximum(Z, 0.0, dtype=np.float32) @ wn)
                        .var(axis=0, dtype=np.float64))

            Z64 = Z.astype(np.float64)
            mu = Z64.mean(0)
            C = Z64 - mu
            Sig = (C.T @ C) / Z.shape[0]
            V = np.linalg.eigh(Sig)[1][:, -PC:]
            proj = (C @ V)
            proj /= proj.std(0) + 1e-30
            sub = proj[rng.choice(proj.shape[0], 40_000, replace=False)]

            for K in KS:
                lab_sub = kmeans(sub, K, np.random.default_rng(7))
                if K == 1:
                    lab = np.zeros(Z.shape[0], dtype=np.int64)
                else:
                    cent = np.stack([sub[lab_sub == q].mean(0) if (lab_sub == q).any()
                                     else sub[0] for q in range(K)])
                    lab = ((proj[:, None, :] - cent[None]) ** 2).sum(-1).argmin(1)
                synth = np.empty_like(Z)
                for q in range(K):
                    m_ = lab == q
                    nq = int(m_.sum())
                    if nq < WIDTH + 2:
                        synth[m_] = Z[m_]
                        continue
                    Zq = Z64[m_]
                    mq = Zq.mean(0)
                    Cq = Zq - mq
                    Sq = (Cq.T @ Cq) / nq
                    ev, U = np.linalg.eigh(Sq)
                    L = (U * np.sqrt(np.maximum(ev, 0.0))).astype(np.float32)
                    synth[m_] = (mq.astype(np.float32)
                                 + rng.standard_normal((nq, WIDTH), dtype=np.float32) @ L.T)
                v = (np.maximum(synth, 0.0, dtype=np.float32) @ wn).var(axis=0, dtype=np.float64)
                ok = var_true > 0
                res[(target, K)].append(float(np.sqrt(np.mean(
                    (np.sqrt(np.maximum(v[ok], 0) / var_true[ok]) - 1.0) ** 2))))
            del Z, Z64, C
        print(f"  net {net} done", flush=True)

    print(f"\nM187 -- conditional covariance mixture ({N_NETS} nets, N={N:,}, "
          f"floor {1/np.sqrt(2*N):.1e})\n")
    print(f"{'layer':>6} " + " ".join(f"{'K='+str(k):>10}" for k in KS) + f" {'gain':>8}")
    for l in LAYERS:
        m = [np.mean(res[(l, K)]) for K in KS]
        print(f"{l:>6} " + " ".join(f"{v:>10.3e}" for v in m) + f" {m[0]/min(m):>7.2f}x")
    print("\n  K=1 is the plain Gaussian closure. Requirement: <= 3e-3.")
    print("  K prototypes cost O(K n^3)/layer: K=8 -> ~4e9 FLOPs, under the floor.")


if __name__ == "__main__":
    main()
