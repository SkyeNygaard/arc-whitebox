"""GATE 1: the O(n^2) per-component update, validated against exact Psi.

The whole cost model rests on replacing the per-component 4n^3 contraction
W^T Cov_q(relu) W with a first-order update plus a rank-1 contraction.

Price's theorem gives the derivatives in closed form, evaluated ONCE at the
reference component and reused for all K:

    i != j :  dE[relu_i relu_j]/dS_ij = Phi2(a_i, a_j; rho_ij)
              dE[relu_i relu_j]/dm_i  = E[1{z_i>0} relu(z_j)]
              dE[relu_i relu_j]/dS_ii = (1/2) E[delta(z_i) relu(z_j)]
    i == j :  dE[relu_i^2]/dS_ii = Phi(a_i),   dE[relu_i^2]/dm_i = 2 E[relu(z_i)]

so each component costs O(n^2) with no special-function evaluations, and the
rank-1 structure of Delta C makes the W-contraction 2n^2.

Validated by comparing the resulting next-layer variances against the exact
Psi computation, on real mixture components obtained by clustering.
"""
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, DATA
from closures import relu_pair_gauss, relu_mean_gauss, Phi2, phi, ncdf

N, CHUNK, K = 180_000, 20_000, 16
LAYERS = [16, 29]
N_NETS, PC = 2, 8
RANKS = [1, 2, 4, 8, 16, 32, 256]


def cov_relu(mu, Sig):
    sd = np.sqrt(np.maximum(np.diag(Sig), 1e-300))
    E1 = relu_mean_gauss(mu, sd)
    return relu_pair_gauss(mu, sd, Sig) - np.outer(E1, E1), E1, sd


def derivatives(mu, Sig, sd):
    """Price derivatives at the reference component. All O(n^2), computed once."""
    a = mu / sd
    rho = np.clip(Sig / np.outer(sd, sd), -0.999999, 0.999999)
    A = a[:, None] + np.zeros_like(rho)
    Bb = a[None, :] + np.zeros_like(rho)
    s = np.sqrt(1 - rho * rho)
    dS = Phi2(A, Bb, rho)                       # dE/dS_ij  (i != j)
    np.fill_diagonal(dS, ncdf(a))               # dE/dS_ii  for i == j
    P = ncdf((Bb - rho * A) / s)
    Q = ncdf((A - rho * Bb) / s)
    # dE/dm_i = E[1{z_i>0} relu(z_j)] = sd_j (b Phi2 + phi(b) Q + rho phi(a) P)
    dm = sd[None, :] * (Bb * dS + phi(Bb) * Q + rho * phi(A) * P)
    np.fill_diagonal(dm, 2.0 * relu_mean_gauss(mu, sd))
    # dE/dS_ii for i != j : (1/2) f_i(0) E[relu(z_j) | z_i = 0]
    beta = rho * sd[None, :] / sd[:, None]
    tau = sd[None, :] * s
    m0 = mu[None, :] - beta * mu[:, None]
    r = m0 / tau
    dSii = 0.5 * (phi(A) / sd[:, None]) * (m0 * ncdf(r) + tau * phi(r))
    return dS, dm, dSii


def main():
    print("starting", flush=True)
    tb = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(tb.column("weights").to_pylist(), dtype=np.float32)
    errs = {(l, r): [] for l in LAYERS for r in RANKS}
    base = {l: [] for l in LAYERS}
    for net in range(N_NETS):
        rng = np.random.default_rng(5500 + net)
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
            print(f'    layer {tgt}: propagated', flush=True)
            wn = wts[tgt + 1].astype(np.float64)
            mu0 = Z.mean(0); Cc = Z - mu0
            S0 = (Cc.T @ Cc) / len(Z)
            V = np.linalg.eigh(S0)[1][:, -PC:]
            pr = (Cc @ V); pr /= pr.std(0) + 1e-30
            del Cc
            c = pr[rng.choice(len(pr), K, replace=False)].copy()
            for _ in range(10):
                lab = np.argmin((pr @ c.T) * (-2) + (c ** 2).sum(1)[None], 1)
                for q in range(K):
                    mm = lab == q
                    if mm.any():
                        c[q] = pr[mm].mean(0)
            C0, E10, sd0 = cov_relu(mu0, S0)
            dS, dm, dSii = derivatives(mu0, S0, sd0)
            v0 = np.einsum("ij,ik,jk->k", C0, wn, wn)
            for q in range(K):
                idx = np.flatnonzero(lab == q)
                if len(idx) < WIDTH + 5:
                    continue
                Zq = Z[idx]; mq = Zq.mean(0); Cq = Zq - mq
                Sq = (Cq.T @ Cq) / len(idx)
                Cex, _, _ = cov_relu(mq, Sq)
                vex = np.einsum("ij,ik,jk->k", Cex, wn, wn) - v0
                # ---- O(n^2) first-order update
                dm_ = mq - mu0
                dSm = Sq - S0
                dv = np.diag(dSm)
                a0 = mu0 / sd0
                # d(second moment) -- off-diagonal formula
                dE2 = (dS * dSm
                       + dm * dm_[:, None] + dm.T * dm_[None, :]
                       + dSii * dv[:, None] + dSii.T * dv[None, :])
                # diagonal handled explicitly (dm/dm.T would double-count it)
                np.fill_diagonal(dE2, ncdf(a0) * dv + 2.0 * E10 * dm_)
                # d(mean) and the mean-product term of the covariance
                dE1 = ncdf(a0) * dm_ + (phi(a0) / (2.0 * sd0)) * dv
                dC = dE2 - (np.outer(dE1, E10) + np.outer(E10, dE1))
                dC = 0.5 * (dC + dC.T)
                # dC is a DIFFERENCE of covariances -> indefinite. Use eigh and
                # keep the largest-|lambda| modes WITH their signs (an SVD-based
                # U S U^T silently flips the negative ones).
                lam, U = np.linalg.eigh(dC)
                order = np.argsort(-np.abs(lam))
                for r in RANKS:
                    sel = order[:r]
                    Ar = (U[:, sel] * lam[sel]) @ U[:, sel].T
                    va = np.einsum("ij,ik,jk->k", Ar, wn, wn)
                    errs[(tgt, r)].append(np.linalg.norm(va - vex) / max(np.linalg.norm(vex), 1e-300))
                base[tgt].append(np.linalg.norm(vex) / max(np.linalg.norm(v0), 1e-300))
            del Z
        print(f"  net {net} done", flush=True)

    print(f"\nGATE 1 -- O(n^2) FIRST-ORDER UPDATE vs EXACT Psi  "
          f"({N_NETS} nets, K={K} components)\n")
    print(f"{'layer':>6} {'|dv|/|v0|':>11} " + " ".join(f"{'rank '+str(r):>10}" for r in RANKS))
    for l in LAYERS:
        print(f"{l:>6} {np.mean(base[l]):>11.3f} " +
              " ".join(f"{np.mean(errs[(l,r)]):>10.4f}" for r in RANKS))
    print("\n  columns = relative error in the next-layer variance CHANGE.")
    print("  |dv|/|v0| is how large that change is, for scale.")


if __name__ == "__main__":
    main()
