"""M192 rung: is the r=64 latent copula a LINEAR MIXTURE of independent sources?

The r=64 location latent passes the closure gate (1.96e-3) but has no known
compact representation: destroying its dependence in the PCA basis loses 89-95%
of the gain, so the copula is essential.

But that only shows the components are not independent IN THAT BASIS.  If the
latent is a linear mixture of independent non-Gaussian sources -- xi = A s with s
independent -- then in the ICA basis the components ARE independent, and the
whole state collapses to

    64 one-dimensional marginal laws  +  a 64x64 rotation

which is O(n^2) parameters, deterministic, and propagatable at O(n^3)/layer.
That is exactly the object v30 asks for.

Three reconstructions of the latent, all preserving mean and covariance exactly:
    full      true joint law                (the 1.96e-3 baseline)
    pca-marg  independence forced in the PCA basis   (measured: recovers 5-11%)
    ica-marg  independence forced in the ICA basis   (the new test)

If ica-marg approaches full, the copula is linearly separable and the
representation problem is solved.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, DATA

N = 400_000
CHUNK = 20_000
LAYERS = [8, 16, 24, 29]
R = 64
N_NETS = 3
N_ICA = 60_000
ICA_ITERS = 120


def fastica(Xw, rng, iters=ICA_ITERS):
    """Symmetric FastICA with tanh contrast on whitened data (N, r)."""
    r = Xw.shape[1]
    W = np.linalg.qr(rng.standard_normal((r, r)))[0]
    for _ in range(iters):
        S = Xw @ W
        G = np.tanh(S)
        Gp = 1.0 - G ** 2
        Wn = (Xw.T @ G) / Xw.shape[0] - W * Gp.mean(0)
        # symmetric orthogonalisation
        u, _, vt = np.linalg.svd(Wn, full_matrices=False)
        Wn = u @ vt
        if np.max(np.abs(np.abs(np.sum(Wn * W, 0)) - 1.0)) < 1e-8:
            W = Wn
            break
        W = Wn
    return W


def main():
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)
    modes = ["full", "pca-marg", "ica-marg"]
    res = {(l, m): [] for l in LAYERS for m in modes}

    for net in range(N_NETS):
        rng = np.random.default_rng(6100 + net)
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
            ev, V = np.linalg.eigh(Sig)
            o = np.argsort(ev)[::-1]
            ev, V = np.maximum(ev[o], 0.0), V[:, o]
            B = V[:, :R]
            xi = C @ B                                   # (N, R), diagonal cov
            sd = np.sqrt(np.maximum(ev[:R], 1e-300))
            Xw = xi / sd                                 # whitened

            Sig_eps = Sig - (B * ev[:R]) @ B.T
            ee, VE = np.linalg.eigh(Sig_eps)
            L = (VE * np.sqrt(np.maximum(ee, 0.0))).astype(np.float32)
            eps = rng.standard_normal((Z.shape[0], WIDTH), dtype=np.float32) @ L.T

            sub = Xw[rng.choice(Xw.shape[0], N_ICA, replace=False)]
            Wica = fastica(sub, rng)
            S_ica = Xw @ Wica

            for mode in modes:
                if mode == "full":
                    Xn = Xw[rng.permutation(Xw.shape[0])]
                elif mode == "pca-marg":
                    Xn = np.empty_like(Xw)
                    for k in range(R):
                        Xn[:, k] = Xw[rng.permutation(Xw.shape[0]), k]
                else:
                    Sn = np.empty_like(S_ica)
                    for k in range(R):
                        Sn[:, k] = S_ica[rng.permutation(S_ica.shape[0]), k]
                    Xn = Sn @ Wica.T
                synth = (mu.astype(np.float32) + eps
                         + ((Xn * sd) @ B.T).astype(np.float32))
                v = (np.maximum(synth, 0.0, dtype=np.float32) @ wn).var(axis=0,
                                                                       dtype=np.float64)
                ok = var_true > 0
                res[(target, mode)].append(float(np.sqrt(np.mean(
                    (np.sqrt(np.maximum(v[ok], 0) / var_true[ok]) - 1.0) ** 2))))
            del Z, Z64, C, Xw, eps
        print(f"  net {net} done", flush=True)

    print(f"\nIS THE LATENT COPULA LINEARLY SEPARABLE?  r={R}, {N_NETS} nets, "
          f"N={N:,}\n")
    print(f"{'layer':>6} {'full joint':>12} {'pca-marg':>11} {'ica-marg':>11} "
          f"{'ICA recovers':>13}")
    for l in LAYERS:
        f = np.mean(res[(l, "full")])
        p = np.mean(res[(l, "pca-marg")])
        i = np.mean(res[(l, "ica-marg")])
        frac = (p - i) / (p - f) if p > f else float("nan")
        print(f"{l:>6} {f:>12.4e} {p:>11.4e} {i:>11.4e} {frac:>12.1%}")
    print("\n  'ICA recovers' = fraction of the full-joint gain that independence")
    print("  in the ICA basis retains. High => the copula is a linear mixture and")
    print("  compresses to 64 1-D laws + a rotation. Low => genuinely entangled.")


if __name__ == "__main__":
    main()
