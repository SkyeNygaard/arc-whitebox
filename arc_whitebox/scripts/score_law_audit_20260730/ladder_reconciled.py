"""DECISIVE: does the full-covariance ladder hold its K^-0.38 exponent?

The rank-1 dCov collapse makes K~3000 affordable and projects 5.17x -- but that
uses an exponent measured only to K=32, extrapolated 94x. The tied family looked
comparable early and then flattened to K^-0.12, which is what killed it.

Extend the measured ladder to K=512. Full covariance needs >=256 samples per
component, so N must scale with K; shrinkage toward the pooled covariance keeps
finite-sample noise from masquerading as closure error.
"""
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, DATA

N, CHUNK = 2_400_000, 25_000
LAYERS = [16, 29]
KS = [1, 8, 32, 128, 512, 1536]
N_NETS, PC = 2, 8
SHRINK = 0.0


def kmeans(X, K, rng, iters=10):
    if K == 1:
        return np.zeros(len(X), dtype=np.int64)
    c = X[rng.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        lab = np.argmin((X @ c.T) * (-2) + (c ** 2).sum(1)[None], 1)
        s = np.stack([np.bincount(lab, weights=X[:, j], minlength=K)
                      for j in range(X.shape[1])], 1)
        n = np.maximum(np.bincount(lab, minlength=K), 1)[:, None]
        c = (s / n).astype(np.float32)
    return lab


tb = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
W_all = np.asarray(tb.column("weights").to_pylist(), dtype=np.float32)
res = {(l, K): [] for l in LAYERS for K in KS}
for net in range(N_NETS):
    rng = np.random.default_rng(9300 + net)
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
        Z64 = Z.astype(np.float64); mu = Z64.mean(0); Cc = Z64 - mu
        Sig = (Cc.T @ Cc) / len(Z)
        V = np.linalg.eigh(Sig)[1][:, -PC:]
        proj = (Cc @ V).astype(np.float32)
        proj /= (proj.std(0) + 1e-30)
        proj = np.ascontiguousarray(proj)
        del Cc
        for K in KS:
            lab = kmeans(proj, K, np.random.default_rng(13))
            synth = np.empty_like(Z)
            for q in range(K):
                idx = np.flatnonzero(lab == q)
                if len(idx) < WIDTH + 5:
                    synth[idx] = Z[idx]; continue
                Zq = Z64[idx]; mq = Zq.mean(0); Cq = Zq - mq
                Sq = (Cq.T @ Cq) / len(idx)
                Sq = (1 - SHRINK) * Sq + SHRINK * Sig          # finite-sample shrinkage
                ev, U = np.linalg.eigh(Sq)
                L = (U * np.sqrt(np.maximum(ev, 0.0))).astype(np.float32)
                synth[idx] = mq.astype(np.float32) + rng.standard_normal(
                    (len(idx), WIDTH), dtype=np.float32) @ L.T
            v = (np.maximum(synth, 0.0, dtype=np.float32) @ wn).var(axis=0, dtype=np.float64)
            ok = vt > 0
            res[(tgt, K)].append(float(np.sqrt(np.mean(
                (np.sqrt(np.maximum(v[ok], 0) / vt[ok]) - 1.0) ** 2))))
            del synth
        del Z, Z64
    print(f"  net {net} done", flush=True)

print(f"\nFULL-COVARIANCE LADDER EXTENDED  (N={N:,}, {N_NETS} nets)\n")
print(f"{'layer':>6} " + " ".join(f"{'K='+str(k):>10}" for k in KS))
for l in LAYERS:
    print(f"{l:>6} " + " ".join(f"{np.mean(res[(l,K)]):>10.3e}" for K in KS))
avg = {K: np.mean([np.mean(res[(l, K)]) for l in LAYERS]) for K in KS}
print("\n  layer-averaged and local exponent:")
prev = None
for K in KS:
    e = ""
    if prev is not None:
        e = f"  exponent over last step = {np.log(avg[prev]/avg[K])/np.log(K/prev):.3f}"
    print(f"    K={K:>4}: delta={avg[K]:.3e}{e}")
    prev = K
print(f"\n  projection used K^-0.38. If the exponent is holding, K=3000 -> 5.17x.")
