"""GATE: is Delta Cov(relu) low-rank when Delta Sigma is low-rank?

The full-covariance mixture has the accuracy (K=32 -> 4.8e-3) but costs
O(K n^3): each component needs its own W^T Cov_q(relu) W contraction, 4n^3.
It is short by ~65x. If

    Delta C_q = Cov(relu | m_q, S_0 + low-rank) - Cov(relu | m_q, S_0)

is itself low-rank, that contraction becomes (W^T U) D (U^T W) = 2 n^2 r,
which is exactly a ~65x collapse at r=8.

Reason to doubt: to first order Delta C picks up a Hadamard term D_u B D_u with
B_ij = s_i s_j (1/4 + arcsin(rho_ij)/2pi), and Hadamard products do not preserve
low rank. But arcsin(R) expands in Hadamard POWERS of R, and R is strongly
rank-collapsed at depth, so the numerical rank may still be small.

Measured exactly (no sampling) with the validated Psi pair moment.
"""
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, DATA
from closures import relu_pair_gauss, relu_mean_gauss

N, CHUNK = 120_000, 20_000
LAYERS = [8, 16, 24, 29]
N_NETS = 2
NPERT = 4
RELS = [0.10, 0.30]


def cov_relu(mu, Sig):
    sd = np.sqrt(np.maximum(np.diag(Sig), 1e-300))
    E2 = relu_pair_gauss(mu, sd, Sig)
    E1 = relu_mean_gauss(mu, sd)
    return E2 - np.outer(E1, E1)


tb = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
W_all = np.asarray(tb.column("weights").to_pylist(), dtype=np.float32)
out = {(l, r): [] for l in LAYERS for r in RELS}
wout = {(l, r): [] for l in LAYERS for r in RELS}

for net in range(N_NETS):
    rng = np.random.default_rng(4700 + net)
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
        mu = Z.mean(0); C = Z - mu
        Sig = (C.T @ C) / len(Z)
        C0 = cov_relu(mu, Sig)
        wn = wts[tgt + 1].astype(np.float64)
        v0 = np.einsum("ij,ik,jk->k", C0, wn, wn)
        ev, V = np.linalg.eigh(Sig)
        for rel in RELS:
            ranks, wranks = [], []
            for _ in range(NPERT):
                # rank-1 modulation along a randomly weighted principal direction
                c = rng.standard_normal(WIDTH) * np.sqrt(np.maximum(ev, 0))
                u = V @ c
                u *= np.sqrt(rel * np.trace(Sig) / max(u @ u, 1e-300))
                S1 = Sig + np.outer(u, u)
                dC = cov_relu(mu, S1) - C0
                s = np.linalg.svd(dC, compute_uv=False)
                cum = np.cumsum(s ** 2) / np.sum(s ** 2)
                ranks.append([np.searchsorted(cum, t) + 1 for t in (0.9, 0.99, 0.999)])
                # downstream-weighted: rank needed to reproduce the next-layer variances
                dv = np.einsum("ij,ik,jk->k", dC, wn, wn)
                U, sv, Vt = np.linalg.svd(dC)
                errs = []
                for r in (4, 8, 16, 32, 64):
                    A = (U[:, :r] * sv[:r]) @ Vt[:r]
                    dvr = np.einsum("ij,ik,jk->k", A, wn, wn)
                    errs.append(np.linalg.norm(dvr - dv) / max(np.linalg.norm(dv), 1e-300))
                wranks.append(errs)
            out[(tgt, rel)].append(np.mean(ranks, 0))
            wout[(tgt, rel)].append(np.mean(wranks, 0))
        del Z, C
    print(f"  net {net} done", flush=True)

print("\nRANK OF Delta Cov(relu) UNDER A RANK-1 COVARIANCE MODULATION\n")
print(f"{'layer':>6} {'||dS||/tr':>10} {'rank 90%':>9} {'rank 99%':>9} {'rank 99.9%':>11}")
for l in LAYERS:
    for r in RELS:
        m = np.mean(out[(l, r)], 0)
        print(f"{l:>6} {r:>10.2f} {m[0]:>9.0f} {m[1]:>9.0f} {m[2]:>11.0f}")
print("\nDOWNSTREAM-WEIGHTED: rel. error in next-layer variance change vs rank kept\n")
print(f"{'layer':>6} {'||dS||/tr':>10} " + " ".join(f"{'r='+str(r):>9}" for r in (4,8,16,32,64)))
for l in LAYERS:
    for r in RELS:
        m = np.mean(wout[(l, r)], 0)
        print(f"{l:>6} {r:>10.2f} " + " ".join(f"{x:>9.3f}" for x in m))
print("\n  a ~65x cost collapse needs r<=8 to reproduce the weighted change.")
