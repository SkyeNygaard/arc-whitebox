"""Would a SECOND-ORDER update rescue Gate 2?

I dismissed second order on cost, wrongly: by Price's theorem the derivatives of
E[relu_i relu_j] are nonzero only for indices in {i,j}, so the second-order term
is a closed-form expression in a few entries -- O(n^2), same as first order.

This measures the CEILING of any second-order scheme without deriving it, by
taking the exact directional derivatives numerically:

    C(t) ~ C0 + t C' + (t^2/2) C''      evaluated at t = 1 (the real offset)

C' and C'' from central differences of the exact Psi computation at small t.
If the second-order residual is still >> 1.5e-3, the route is closed for good.
"""
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, DATA
from gate1_update import cov_relu

N, CHUNK, PC, H = 180_000, 20_000, 8, 0.15
LAYERS = [16, 29]
KS = [4, 16, 32]
N_NETS = 2


def cluster(pr, K, rng):
    c = pr[rng.choice(len(pr), K, replace=False)].copy()
    for _ in range(12):
        lab = np.argmin((pr @ c.T) * (-2) + (c ** 2).sum(1)[None], 1)
        for q in range(K):
            m = lab == q
            if m.any():
                c[q] = pr[m].mean(0)
    return lab


def main():
    tb = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(tb.column("weights").to_pylist(), dtype=np.float32)
    r1 = {(l, K): [] for l in LAYERS for K in KS}
    r2 = {(l, K): [] for l in LAYERS for K in KS}
    for net in range(N_NETS):
        rng = np.random.default_rng(3300 + net)
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
            mu0 = Z.mean(0); Cc = Z - mu0; S0 = (Cc.T @ Cc) / len(Z)
            V = np.linalg.eigh(S0)[1][:, -PC:]
            pr = (Cc @ V); pr /= pr.std(0) + 1e-30; del Cc
            C0, E10, _ = cov_relu(mu0, S0)
            for K in KS:
                lab = cluster(pr, K, np.random.default_rng(23))
                pis, Ee, E1o, E2o, Ce, C1o, C2o = [], [], [], [], [], [], []
                for q in range(K):
                    idx = np.flatnonzero(lab == q)
                    if len(idx) < WIDTH + 5:
                        continue
                    Zq = Z[idx]; mq = Zq.mean(0); Cq = Zq - mq
                    Sq = (Cq.T @ Cq) / len(idx)
                    dm, dS = mq - mu0, Sq - S0
                    ce, e1, _ = cov_relu(mq, Sq)
                    cp, ep, _ = cov_relu(mu0 + H * dm, S0 + H * dS)
                    cm, em, _ = cov_relu(mu0 - H * dm, S0 - H * dS)
                    Cd1 = (cp - cm) / (2 * H)                 # exact directional C'
                    Cd2 = (cp - 2 * C0 + cm) / (H * H)        # exact directional C''
                    Ed1 = (ep - em) / (2 * H)
                    Ed2 = (ep - 2 * E10 + em) / (H * H)
                    pis.append(len(idx) / len(Z))
                    Ee.append(e1); Ce.append(ce)
                    E1o.append(E10 + Ed1); C1o.append(C0 + Cd1)
                    E2o.append(E10 + Ed1 + 0.5 * Ed2); C2o.append(C0 + Cd1 + 0.5 * Cd2)
                pis = np.array(pis); pis /= pis.sum()

                def mixvar(Es, Cs):
                    Eb = sum(p * e for p, e in zip(pis, Es))
                    M = sum(p * (c + np.outer(e, e)) for p, c, e in zip(pis, Cs, Es))
                    return np.einsum("ij,ik,jk->k", M - np.outer(Eb, Eb), wn, wn)

                ve = mixvar(Ee, Ce)
                for tag, store in ((1, r1), (2, r2)):
                    va = mixvar(E1o if tag == 1 else E2o, C1o if tag == 1 else C2o)
                    rr = np.sqrt(np.maximum(va, 0) / np.maximum(ve, 1e-300)) - 1.0
                    store[(tgt, K)].append(float(np.sqrt(np.mean(rr ** 2))))
            del Z
        print(f"  net {net} done", flush=True)

    print(f"\nSECOND-ORDER CEILING (exact directional derivatives)\n")
    print(f"{'layer':>6} {'K':>5} {'first order':>13} {'second order':>14} {'gain':>7}")
    for l in LAYERS:
        for K in KS:
            a, b = np.mean(r1[(l, K)]), np.mean(r2[(l, K)])
            print(f"{l:>6} {K:>5} {a:>13.4e} {b:>14.4e} {a/b:>6.2f}x")
    print("\n  budget is 1.5e-3 at the deployed K (~1536), and the error GROWS")
    print("  with K as ~K^+0.14, so it must be well under 1.5e-3 already at K=32.")


if __name__ == "__main__":
    main()
