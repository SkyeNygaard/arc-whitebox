"""Expand around the POOLED-WITHIN covariance, not the global one.

Gates 1-2 expanded every component around the global (mu0, S0). But
S0 = Sbar + between, with Sbar = sum_q pi_q S_q the pooled within covariance.
Components cluster around Sbar, NOT around S0, so those offsets were inflated by
the whole between-component spread -- the part that grows with K. That alone
could produce the observed "error grows with K".

Error goes as offset^2, so re-centring should help quadratically. Measured here
for first and second order, against the same exact mixture quantity as Gate 2.
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
KS = [4, 16, 32, 64]
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
    out = {(l, K, r, o): [] for l in LAYERS for K in KS
           for r in ("global", "pooled") for o in (1, 2)}
    off = {(l, K, r): [] for l in LAYERS for K in KS for r in ("global", "pooled")}
    for net in range(N_NETS):
        rng = np.random.default_rng(6600 + net)
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
            for K in KS:
                lab = cluster(pr, K, np.random.default_rng(29))
                comp = []
                for q in range(K):
                    idx = np.flatnonzero(lab == q)
                    if len(idx) < WIDTH + 5:
                        continue
                    Zq = Z[idx]; mq = Zq.mean(0); Cq = Zq - mq
                    comp.append((len(idx) / len(Z), mq, (Cq.T @ Cq) / len(idx)))
                pis = np.array([c[0] for c in comp]); pis /= pis.sum()
                Sbar = sum(p * c[2] for p, c in zip(pis, comp))
                for refname, Sref in (("global", S0), ("pooled", Sbar)):
                    Cref, Eref, _ = cov_relu(mu0, Sref)
                    Ee, Ce = [], []
                    E1, C1, E2, C2 = [], [], [], []
                    offs = []
                    for (p, mq, Sq) in comp:
                        dm, dS = mq - mu0, Sq - Sref
                        offs.append(np.linalg.norm(dS) / np.linalg.norm(Sref))
                        ce, e1, _ = cov_relu(mq, Sq)
                        cp, ep, _ = cov_relu(mu0 + H * dm, Sref + H * dS)
                        cm, em, _ = cov_relu(mu0 - H * dm, Sref - H * dS)
                        d1C, d2C = (cp - cm) / (2 * H), (cp - 2 * Cref + cm) / (H * H)
                        d1E, d2E = (ep - em) / (2 * H), (ep - 2 * Eref + em) / (H * H)
                        Ee.append(e1); Ce.append(ce)
                        E1.append(Eref + d1E); C1.append(Cref + d1C)
                        E2.append(Eref + d1E + 0.5 * d2E); C2.append(Cref + d1C + 0.5 * d2C)
                    off[(tgt, K, refname)].append(np.mean(offs))

                    def mixvar(Es, Cs):
                        Eb = sum(p * e for p, e in zip(pis, Es))
                        M = sum(p * (c + np.outer(e, e)) for p, c, e in zip(pis, Cs, Es))
                        return np.einsum("ij,ik,jk->k", M - np.outer(Eb, Eb), wn, wn)

                    ve = mixvar(Ee, Ce)
                    for o, (Es, Cs) in ((1, (E1, C1)), (2, (E2, C2))):
                        va = mixvar(Es, Cs)
                        rr = np.sqrt(np.maximum(va, 0) / np.maximum(ve, 1e-300)) - 1.0
                        out[(tgt, K, refname, o)].append(float(np.sqrt(np.mean(rr ** 2))))
            del Z
        print(f"  net {net} done", flush=True)

    print("\nEXPANSION REFERENCE: GLOBAL vs POOLED-WITHIN\n")
    print(f"{'layer':>6} {'K':>4} {'||dS||/||S|| g':>15} {'p':>7} "
          f"{'1st global':>11} {'1st pooled':>11} {'2nd global':>11} {'2nd pooled':>11}")
    for l in LAYERS:
        for K in KS:
            og = np.mean(off[(l, K, "global")]); op = np.mean(off[(l, K, "pooled")])
            print(f"{l:>6} {K:>4} {og:>15.3f} {op:>7.3f} "
                  f"{np.mean(out[(l,K,'global',1)]):>11.3e} {np.mean(out[(l,K,'pooled',1)]):>11.3e} "
                  f"{np.mean(out[(l,K,'global',2)]):>11.3e} {np.mean(out[(l,K,'pooled',2)]):>11.3e}")
    print("\n  budget 1.5e-3 at deployed K. Watch whether the pooled columns still grow with K.")


if __name__ == "__main__":
    main()
