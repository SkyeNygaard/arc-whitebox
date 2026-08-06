"""NEW CLASS: a latent-geometry closure, not a moment closure.

Gate 1 established that the covariance-closure failure is NOT a cumulant effect
(marginal skew 0.40, |kappa3|/6 = 0.067) but a geometry effect: the joint law
collapses to participation ratio 2.2 while each marginal stays near-Gaussian.
A moment expansion cannot see that.  A latent-variable model can:

    z = mu + B xi + eps ,   xi in R^r with its TRUE (non-Gaussian) law,
                            eps ~ N(0, Sigma_eps) independent of xi

r = 0 is exactly the Gaussian closure.  Larger r hands the model the actual
low-dimensional geometry and Gaussianises only the high-dimensional remainder --
which is a sum of many small contributions and so genuinely near-Gaussian.

This is an ORACLE-CLOSURE test, not an oracle-capacity test: it never uses the
answer, only an intermediate state.  It asks whether the MODEL CLASS can
represent the truth.  If it cannot, the class is dead regardless of how the
latent law would be propagated.  If it can, the next question is whether the
latent law is legally computable.

Independence of xi and eps is imposed by permuting the empirical latent against
fresh Gaussian residuals -- exactly the model assumption, nothing more.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, DATA

N = 800_000
CHUNK = 20_000
LAYERS = [4, 8, 16, 24, 29]
RANKS = [0, 8, 16, 32, 64, 128]
N_NETS = 3


def sweep(Z, wn, rng):
    """Z: (N, n) reference pre-activations. Returns sigma error per rank."""
    n = Z.shape[1]
    mu = Z.mean(0, dtype=np.float64)
    C = Z - mu
    Sig = (C.T @ C) / Z.shape[0]
    var_true = ((np.maximum(Z, 0.0, dtype=np.float32) @ wn)
                .var(axis=0, dtype=np.float64))
    ev, V = np.linalg.eigh(Sig)
    order = np.argsort(ev)[::-1]
    ev, V = np.maximum(ev[order], 0.0), V[:, order]

    out = {}
    for r in RANKS:
        B = V[:, :r]                                   # (n, r)
        xi = (C @ B) if r else np.zeros((Z.shape[0], 0))
        Sig_eps = Sig - (B * ev[:r]) @ B.T if r else Sig
        ee, VE = np.linalg.eigh(Sig_eps)
        L = (VE * np.sqrt(np.maximum(ee, 0.0))).astype(np.float32)
        # permute the true latent against fresh residuals: imposes independence
        perm = rng.permutation(Z.shape[0])
        eps = rng.standard_normal((Z.shape[0], n), dtype=np.float32) @ L.T
        synth = mu.astype(np.float32) + eps
        if r:
            synth = synth + (xi[perm] @ B.T).astype(np.float32)
        v = (np.maximum(synth, 0.0, dtype=np.float32) @ wn).var(axis=0, dtype=np.float64)
        ok = var_true > 0
        out[r] = float(np.sqrt(np.mean(
            (np.sqrt(np.maximum(v[ok], 0) / var_true[ok]) - 1.0) ** 2)))
    return out, ev


def main():
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)
    res = {l: {r: [] for r in RANKS} for l in LAYERS}

    for net in range(N_NETS):
        rng = np.random.default_rng(700 + net)
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        for target in LAYERS:
            buf = []
            done = 0
            while done < N:
                m = min(CHUNK, N - done)
                h = rng.standard_normal((m, WIDTH), dtype=np.float32)
                for l, w in enumerate(weights[:target + 1]):
                    z = h @ w
                    h = np.maximum(z, 0.0, dtype=np.float32)
                buf.append(z)
                done += m
            Z = np.concatenate(buf, 0)
            del buf
            o, ev = sweep(Z, weights[target + 1], rng)
            for r in RANKS:
                res[target][r].append(o[r])
            del Z
        print(f"  net {net} done", flush=True)

    print(f"\nLATENT-GEOMETRY CLOSURE -- oracle-closure test "
          f"({N_NETS} nets, N={N:,}, floor {1/np.sqrt(2*N):.1e})\n")
    print(f"{'layer':>6} " + " ".join(f"{'r='+str(r):>10}" for r in RANKS)
          + f" {'best gain':>10}")
    for l in LAYERS:
        m = [np.mean(res[l][r]) for r in RANKS]
        print(f"{l:>6} " + " ".join(f"{v:>10.3e}" for v in m)
              + f" {m[0]/min(m):>9.2f}x")
    print("\n  r=0 is the plain Gaussian closure. Requirement: <= 3e-3.")


if __name__ == "__main__":
    main()
