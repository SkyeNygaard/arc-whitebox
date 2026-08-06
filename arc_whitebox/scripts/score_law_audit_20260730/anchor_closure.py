"""Is the Gaussian closure accurate for the SMOOTHED network?

This is the other half of the smoothed-anchor hybrid.  The residual economics
(Gate A / partial.py) are useless unless E[g] can be computed analytically with
small bias.

Why it should work: the smoothed ReLU is  E_eps[relu(p + s eps)],  so for a
bivariate Gaussian pair the smoothed pair moment is EXACTLY the ordinary Psi
formula with inflated variances

    sigma_i^2 -> sigma_i^2 + s^2 ,   rho -> rho sigma_i sigma_j /
                                            (sqrt(si^2+s^2) sqrt(sj^2+s^2))

so smoothing shrinks the effective correlation toward 0 -- exactly where the
Gaussian closure is exact and where the copula sensitivity that broke Gate 1
lives.  The question is how fast the closure error falls with alpha.

Measured as in cov_closure.py: one step, EXACT input moments from the reference,
predicting the next layer's sigma for the smoothed dynamics.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, DATA

N = 300_000
CHUNK = 20_000
LAYERS = [4, 8, 16, 24, 29]
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0]
N_NETS = 3
SQRT2PI = np.sqrt(2 * np.pi)


def act_fn(p, alpha):
    if alpha == 0.0:
        return np.maximum(p, 0.0, dtype=np.float32)
    s = alpha * p.std(0, dtype=np.float64).astype(np.float32) + 1e-30
    t = p / s
    v = p * ndtr(t).astype(np.float32) + s * np.exp(-0.5 * t * t).astype(np.float32) / SQRT2PI
    return np.maximum(v, 0.0, dtype=np.float32)


def gsample(mu, Sig, m, rng):
    ev, V = np.linalg.eigh(Sig)
    L = (V * np.sqrt(np.maximum(ev, 0.0))).astype(np.float32)
    return mu.astype(np.float32) + rng.standard_normal((m, mu.size), dtype=np.float32) @ L.T


def main():
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)
    res = {(l, a): [] for l in LAYERS for a in ALPHAS}

    for net in range(N_NETS):
        rng = np.random.default_rng(4400 + net)
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        for a in ALPHAS:
            for target in LAYERS:
                buf, done = [], 0
                while done < N:
                    m = min(CHUNK, N - done)
                    h = rng.standard_normal((m, WIDTH), dtype=np.float32)
                    for w in weights[:target + 1]:
                        z = h @ w
                        h = act_fn(z, a)
                    buf.append(z)
                    done += m
                Z = np.concatenate(buf, 0); del buf
                wn = weights[target + 1]
                var_true = (act_fn(Z, a) @ wn).var(axis=0, dtype=np.float64)
                Z64 = Z.astype(np.float64)
                g = gsample(Z64.mean(0), np.cov(Z64, rowvar=False, bias=True), N, rng)
                var_cl = (act_fn(g, a) @ wn).var(axis=0, dtype=np.float64)
                ok = var_true > 0
                res[(target, a)].append(float(np.sqrt(np.mean(
                    (np.sqrt(np.maximum(var_cl[ok], 0) / var_true[ok]) - 1.0) ** 2))))
                del Z, Z64, g
        print(f"  net {net} done", flush=True)

    print(f"\nGAUSSIAN-CLOSURE ERROR FOR THE SMOOTHED NETWORK "
          f"({N_NETS} nets, N={N:,}, floor {1/np.sqrt(2*N):.1e})\n")
    print(f"{'layer':>6} " + " ".join(f"{'a='+str(a):>11}" for a in ALPHAS))
    for l in LAYERS:
        print(f"{l:>6} " + " ".join(f"{np.mean(res[(l,a)]):>11.3e}" for a in ALPHAS))
    print("\n  a=0 is the raw ReLU network (the Gate 1 blocker, ~1.4e-2).")
    print("  The anchor is tractable if smoothing drops this toward ~1e-3.")


if __name__ == "__main__":
    main()
