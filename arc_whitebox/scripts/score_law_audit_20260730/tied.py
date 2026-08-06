"""TIED-COVARIANCE MIXTURE at large K -- the rung I closed by a wrong argument.

I dismissed tied covariance as dominated by full covariance at equal K. True, but
the wrong axis: at equal COST tied buys far more components.

    full covariance : O(K n^3)          -> K <= ~100 within budget
    tied covariance : O(n^3 + K n^2)    -> one shared Sigma, K means

so tied affords K in the thousands. The accuracy question is therefore open:
how many tied components are needed to reach the calibrated requirement
delta <= 1.5e-3?

Model: cluster the reference into K groups; each component gets its own mean and
the POOLED within-cluster covariance. Total covariance is preserved exactly
(Sigma = between + pooled-within), so this is a pure copula compression -- and it
is deterministic, i.e. exactly the object v30 asks for.
"""
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, DATA

N, CHUNK = 250_000, 25_000
LAYERS = [16, 29]
KS = [1, 128, 1024, 4096]
N_NETS, PC = 2, 16


def kmeans(X, K, rng, iters=8):
    if K == 1:
        return np.zeros(len(X), dtype=np.int64)
    c = X[rng.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - c[None]) ** 2).sum(-1), 1) if K <= 64 else \
              np.argmin((X @ c.T) * (-2) + (c ** 2).sum(1)[None], 1)
        for q in range(K):
            m = lab == q
            if m.any():
                c[q] = X[m].mean(0)
    return lab


tb = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
W_all = np.asarray(tb.column("weights").to_pylist(), dtype=np.float32)
res = {(l, K): [] for l in LAYERS for K in KS}
for net in range(N_NETS):
    rng = np.random.default_rng(8100 + net)
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
        Z = np.concatenate(buf, 0); del buf
        wn = wts[tgt + 1]
        vt = (np.maximum(Z, 0.0, dtype=np.float32) @ wn).var(axis=0, dtype=np.float64)
        Z64 = Z.astype(np.float64); mu = Z64.mean(0); C = Z64 - mu
        Sig = (C.T @ C) / len(Z)
        V = np.linalg.eigh(Sig)[1][:, -PC:]
        proj = np.ascontiguousarray((C @ V).astype(np.float32))
        for K in KS:
            lab = kmeans(proj, K, np.random.default_rng(11))
            cnt = np.bincount(lab, minlength=K).astype(np.float64)
            M = np.stack([np.bincount(lab, weights=Z64[:, j], minlength=K)
                          for j in range(WIDTH)], 1)
            M /= np.maximum(cnt, 1)[:, None]
            pi = cnt / cnt.sum()
            d = M - mu
            between = (d * pi[:, None]).T @ d
            S = Sig - between                      # pooled within, tied
            ev, U = np.linalg.eigh(S)
            L = (U * np.sqrt(np.maximum(ev, 0.0))).astype(np.float32)
            synth = M[lab].astype(np.float32) + rng.standard_normal(
                (len(Z), WIDTH), dtype=np.float32) @ L.T
            v = (np.maximum(synth, 0.0, dtype=np.float32) @ wn).var(axis=0, dtype=np.float64)
            ok = vt > 0
            res[(tgt, K)].append(float(np.sqrt(np.mean(
                (np.sqrt(np.maximum(v[ok], 0) / vt[ok]) - 1.0) ** 2))))
        del Z, Z64, C
    print(f"  net {net} done", flush=True)

print(f"\nTIED-COVARIANCE MIXTURE ({N_NETS} nets, N={N:,})\n")
print(f"{'layer':>6} " + " ".join(f"{'K='+str(k):>10}" for k in KS))
for l in LAYERS:
    print(f"{l:>6} " + " ".join(f"{np.mean(res[(l,K)]):>10.3e}" for K in KS))
avg = {K: np.mean([np.mean(res[(l, K)]) for l in LAYERS]) for K in KS}
print("\n  layer-averaged:")
for K in KS:
    cost = (6.7e7 + K * 6.6e4) * 31
    print(f"    K={K:>5}: delta={avg[K]:.3e}   cost={cost/2.72e11:.4f}xB   "
          f"{'PASSES 1.5e-3' if avg[K] <= 1.5e-3 else ''}")
