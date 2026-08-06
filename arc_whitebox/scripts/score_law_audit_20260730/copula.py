"""Where does the latent closure's gain live: marginals or copula?

This decides whether the class is deployable at all.

The latent xi = B^T (z - mu) is built from eigenvectors, so its covariance is
DIAGONAL.  That makes a clean decomposition possible:

  true      xi as it is                      -- full joint law (the oracle gain)
  marg      each COLUMN of xi permuted
            independently                    -- mean, covariance and every
                                                marginal preserved EXACTLY;
                                                only the copula destroyed
  gauss     xi replaced by a Gaussian with
            the same diagonal covariance     -- equals the r=0 closure

If `marg` recovers most of `true`, the latent law is r independent 1-D
distributions: O(r) parameters, parametrisable, propagatable without particles,
and the 9.4x win is live.
If `marg` collapses to `gauss`, the gain is a genuine r-dimensional copula, no
finite parametrisation is available, and the class is closed.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, DATA

N = 800_000
CHUNK = 20_000
LAYERS = [8, 16, 24, 29]
RANKS = [32, 64, 128]
N_NETS = 3
VARIANTS = ["true", "marg", "gauss"]


def build(Z, wn, rng):
    n = Z.shape[1]
    mu = Z.mean(0, dtype=np.float64)
    C = Z - mu
    Sig = (C.T @ C) / Z.shape[0]
    var_true = ((np.maximum(Z, 0.0, dtype=np.float32) @ wn)
                .var(axis=0, dtype=np.float64))
    ev, V = np.linalg.eigh(Sig)
    o = np.argsort(ev)[::-1]
    ev, V = np.maximum(ev[o], 0.0), V[:, o]
    out = {}
    for r in RANKS:
        B = V[:, :r]
        xi = C @ B                                        # (N, r), diagonal cov
        Sig_eps = Sig - (B * ev[:r]) @ B.T
        ee, VE = np.linalg.eigh(Sig_eps)
        L = (VE * np.sqrt(np.maximum(ee, 0.0))).astype(np.float32)
        eps = rng.standard_normal((Z.shape[0], n), dtype=np.float32) @ L.T
        for v in VARIANTS:
            if v == "true":
                X = xi[rng.permutation(Z.shape[0])]
            elif v == "marg":
                # permute each column independently: marginals + covariance kept,
                # copula destroyed
                X = np.empty_like(xi)
                for k in range(r):
                    X[:, k] = xi[rng.permutation(Z.shape[0]), k]
            else:
                X = rng.standard_normal((Z.shape[0], r)) * np.sqrt(ev[:r])
            synth = mu.astype(np.float32) + eps + (X @ B.T).astype(np.float32)
            vv = (np.maximum(synth, 0.0, dtype=np.float32) @ wn).var(axis=0, dtype=np.float64)
            ok = var_true > 0
            out[(r, v)] = float(np.sqrt(np.mean(
                (np.sqrt(np.maximum(vv[ok], 0) / var_true[ok]) - 1.0) ** 2)))
    return out


def main():
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)
    res = {(l, r, v): [] for l in LAYERS for r in RANKS for v in VARIANTS}

    for net in range(N_NETS):
        rng = np.random.default_rng(1300 + net)
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
            o = build(Z, weights[target + 1], rng)
            for k, val in o.items():
                res[(target,) + k].append(val)
            del Z
        print(f"  net {net} done", flush=True)

    print(f"\nMARGINALS vs COPULA ({N_NETS} nets, N={N:,}, floor "
          f"{1/np.sqrt(2*N):.1e})\n")
    print(f"{'layer':>6} {'r':>5} {'gauss (r=0)':>12} {'marginals only':>15} "
          f"{'full joint':>12} {'gain recovered':>15}")
    for l in LAYERS:
        for r in RANKS:
            g = np.mean(res[(l, r, 'gauss')])
            m = np.mean(res[(l, r, 'marg')])
            t = np.mean(res[(l, r, 'true')])
            frac = (g - m) / (g - t) if g > t else float('nan')
            print(f"{l:>6} {r:>5} {g:>12.3e} {m:>15.3e} {t:>12.3e} {frac:>14.1%}")
    print("\n  'gain recovered' = how much of the full-joint improvement the")
    print("  marginals alone deliver. High => deployable. Low => copula, dead.")


if __name__ == "__main__":
    main()
