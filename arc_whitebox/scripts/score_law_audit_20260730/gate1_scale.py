"""Is the first-order update BUGGY or just outside its radius of convergence?

Scale the component offset by t: (m0 + t*dm, S0 + t*dS).
  relative error constant in t  -> the formula is wrong
  relative error ~ t            -> formula correct, second-order truncation
"""
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, DATA
from closures import relu_mean_gauss, ncdf, phi
from gate1_update import cov_relu, derivatives

N, CHUNK, K, PC = 180_000, 20_000, 16, 8
TS = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
tb = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
W_all = np.asarray(tb.column("weights").to_pylist(), dtype=np.float32)
rng = np.random.default_rng(77)
wts = [np.ascontiguousarray(W_all[0][i]) for i in range(DEPTH)]
tgt = 16
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
mu0 = Z.mean(0); Cc = Z - mu0; S0 = (Cc.T @ Cc) / len(Z)
V = np.linalg.eigh(S0)[1][:, -PC:]
pr = (Cc @ V); pr /= pr.std(0) + 1e-30; del Cc
c = pr[rng.choice(len(pr), K, replace=False)].copy()
for _ in range(10):
    lab = np.argmin((pr @ c.T) * (-2) + (c ** 2).sum(1)[None], 1)
    for q in range(K):
        mm = lab == q
        if mm.any():
            c[q] = pr[mm].mean(0)
C0, E10, sd0 = cov_relu(mu0, S0)
dS, dm, dSii = derivatives(mu0, S0, sd0)
a0 = mu0 / sd0
v0 = np.einsum("ij,ik,jk->k", C0, wn, wn)
idx = np.flatnonzero(lab == np.bincount(lab).argmax())
Zq = Z[idx]; mq = Zq.mean(0); Cq = Zq - mq; Sq = (Cq.T @ Cq) / len(idx)
DM, DS = mq - mu0, Sq - S0
print(f"component offsets: ||dm||/||sd|| = {np.linalg.norm(DM)/np.linalg.norm(sd0):.3f}   "
      f"||dS||/||S0|| = {np.linalg.norm(DS)/np.linalg.norm(S0):.3f}\n")
print(f"{'t':>7} {'|dv|/|v0|':>11} {'rel err':>10} {'err/t':>10}")
for t in TS:
    dm_, dSm = t * DM, t * DS
    Cex, _, _ = cov_relu(mu0 + dm_, S0 + dSm)
    vex = np.einsum("ij,ik,jk->k", Cex, wn, wn) - v0
    dv = np.diag(dSm)
    dE2 = (dS * dSm + dm * dm_[:, None] + dm.T * dm_[None, :]
           + dSii * dv[:, None] + dSii.T * dv[None, :])
    np.fill_diagonal(dE2, ncdf(a0) * dv + 2.0 * E10 * dm_)
    dE1 = ncdf(a0) * dm_ + (phi(a0) / (2.0 * sd0)) * dv
    dC = dE2 - (np.outer(dE1, E10) + np.outer(E10, dE1))
    dC = 0.5 * (dC + dC.T)
    va = np.einsum("ij,ik,jk->k", dC, wn, wn)
    e = np.linalg.norm(va - vex) / max(np.linalg.norm(vex), 1e-300)
    print(f"{t:>7.3f} {np.linalg.norm(vex)/np.linalg.norm(v0):>11.4f} {e:>10.4f} {e/t:>10.3f}")
print("\n  err/t roughly constant => formula CORRECT, second-order truncation.")
print("  rel err roughly constant => formula still wrong.")
