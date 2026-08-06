"""GATE 2: does the first-order truncation error average across components?

Quantified in Gate 1: ~12% error on each component's variance CHANGE, which is
~3.6% of the variance. Fatal at 1.8e-2 in sigma if systematic; fine at 4.6e-4 if
it falls as sqrt(K).

Measured directly on the MIXTURE quantity that the estimator actually uses:

    E[relu]   = sum_q pi_q E_q
    Cov(relu) = sum_q pi_q [C_q + E_q E_q^T] - E E^T
    v_next    = diag(W^T Cov(relu) W)

computed twice -- with exact per-component C_q (Psi) and with the O(n^2)
first-order C_q -- and compared as a function of K.
"""
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, DATA
from closures import relu_mean_gauss, ncdf, phi
from gate1_update import cov_relu, derivatives

N, CHUNK, PC = 180_000, 20_000, 8
LAYERS = [16, 29]
KS = [4, 8, 16, 32, 64]
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
    res = {(l, K): [] for l in LAYERS for K in KS}
    for net in range(N_NETS):
        rng = np.random.default_rng(2400 + net)
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
            C0, E10, sd0 = cov_relu(mu0, S0)
            dS, dm, dSii = derivatives(mu0, S0, sd0)
            a0 = mu0 / sd0
            for K in KS:
                lab = cluster(pr, K, np.random.default_rng(19))
                pis, Ee, Ea, Ce, Ca = [], [], [], [], []
                for q in range(K):
                    idx = np.flatnonzero(lab == q)
                    if len(idx) < WIDTH + 5:
                        continue
                    Zq = Z[idx]; mq = Zq.mean(0); Cq = Zq - mq
                    Sq = (Cq.T @ Cq) / len(idx)
                    ce, e1, _ = cov_relu(mq, Sq)                # exact
                    dm_, dSm = mq - mu0, Sq - S0
                    dv = np.diag(dSm)
                    dE2 = (dS * dSm + dm * dm_[:, None] + dm.T * dm_[None, :]
                           + dSii * dv[:, None] + dSii.T * dv[None, :])
                    np.fill_diagonal(dE2, ncdf(a0) * dv + 2.0 * E10 * dm_)
                    dE1 = ncdf(a0) * dm_ + (phi(a0) / (2.0 * sd0)) * dv
                    ca = C0 + 0.5 * ((dE2 - (np.outer(dE1, E10) + np.outer(E10, dE1)))
                                     + (dE2 - (np.outer(dE1, E10) + np.outer(E10, dE1))).T)
                    pis.append(len(idx) / len(Z))
                    Ee.append(e1); Ea.append(E10 + dE1); Ce.append(ce); Ca.append(ca)
                pis = np.array(pis); pis /= pis.sum()

                def mixvar(Es, Cs):
                    Eb = sum(p * e for p, e in zip(pis, Es))
                    M = sum(p * (c + np.outer(e, e)) for p, c, e in zip(pis, Cs, Es))
                    M -= np.outer(Eb, Eb)
                    return np.einsum("ij,ik,jk->k", M, wn, wn)

                ve, va = mixvar(Ee, Ce), mixvar(Ea, Ca)
                # error in sigma, the quantity the calibration is in
                r = np.sqrt(np.maximum(va, 0) / np.maximum(ve, 1e-300)) - 1.0
                res[(tgt, K)].append(float(np.sqrt(np.mean(r ** 2))))
            del Z
        print(f"  net {net} done", flush=True)

    print(f"\nGATE 2 -- does the first-order error average over components?\n")
    print(f"{'layer':>6} " + " ".join(f"{'K='+str(k):>11}" for k in KS))
    for l in LAYERS:
        print(f"{l:>6} " + " ".join(f"{np.mean(res[(l,K)]):>11.4e}" for K in KS))
    avg = {K: np.mean([np.mean(res[(l, K)]) for l in LAYERS]) for K in KS}
    print("\n  layer-averaged sigma error from the O(n^2) approximation, and slope:")
    prev = None
    for K in KS:
        sl = "" if prev is None else f"   slope = {np.log(avg[prev]/avg[K])/np.log(K/prev):+.3f}"
        print(f"    K={K:>3}: {avg[K]:.4e}{sl}")
        prev = K
    print("\n  slope ~ -0.5 => averages as sqrt(K) (fine).  slope ~ 0 => systematic (fatal).")
    print("  budget: this must stay well under 1.5e-3 at the deployed K.")


if __name__ == "__main__":
    main()
