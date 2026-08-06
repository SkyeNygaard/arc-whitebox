"""How much variance is left for a BETTER cubature design?

The shipped design is a spherical 5-design with antipodes, so it integrates
degrees 1..5 exactly and its residual error is the energy in even degrees >= 6.
The question that decides the whole cubature paradigm is therefore:

    what fraction of the even-harmonic energy of the network sits in degree 6?

If most of it does, a 7-design wins outright.  If the spectrum is flat/heavy
tailed, no design of any strength helps and the paradigm is finished.

Method: for the antipodally symmetrised output g(u) = (f(u)+f(-u))/2, the
two-point function on the sphere expands in Gegenbauer polynomials,

    K(t) = E[<g(u), g(v)>]_{u.v = t}  =  sum_l  a_l  P_l^(d)(t),

with a_l >= 0 the degree-l energy.  Sampling K on a grid of t and solving a
nonnegative least squares problem recovers the spectrum.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, DATA

D = WIDTH
LAMBDA = (D - 2) / 2.0
TS = np.array([0.20, 0.35, 0.50, 0.60, 0.70, 0.78, 0.85, 0.90, 0.94, 0.97, 0.99])
DEGREES = np.arange(0, 21, 2)          # even degrees only (g is even)
M = 6000                               # base directions
N_NETS = 6


def gegenbauer_normalised(l, t):
    """C_l^lambda(t) / C_l^lambda(1)  -- the addition-theorem polynomial on S^{d-1}."""
    t = np.asarray(t, dtype=np.float64)
    cm1, c0 = np.zeros_like(t), np.ones_like(t)          # C_{-1}, C_0
    v1 = 0.0, 1.0
    if l == 0:
        return np.ones_like(t)
    c1 = 2 * LAMBDA * t
    d1 = 2 * LAMBDA * 1.0
    cm1, c0 = c0, c1
    dm1, d0 = 1.0, d1
    for n in range(1, l):
        cn = (2 * (n + LAMBDA) * t * c0 - (n + 2 * LAMBDA - 1) * cm1) / (n + 1)
        dn = (2 * (n + LAMBDA) * 1.0 * d0 - (n + 2 * LAMBDA - 1) * dm1) / (n + 1)
        cm1, c0 = c0, cn
        dm1, d0 = d0, dn
    return c0 / d0


def net_output(rows, weights):
    act = np.maximum(rows @ weights[0], 0.0, dtype=np.float32)
    for w in weights[1:]:
        act = np.maximum(act @ w, 0.0, dtype=np.float32)
    return act


def sym_output(dirs, weights):
    """g(u) = (f(u) + f(-u)) / 2 for unit directions `dirs`."""
    both = np.concatenate([dirs, -dirs], axis=0).astype(np.float32)
    out = net_output(both, weights)
    n = dirs.shape[0]
    return 0.5 * (out[:n].astype(np.float64) + out[n:].astype(np.float64))


def nnls(A, b, iters=20000):
    """Projected-gradient nonnegative least squares (no scipy dependency)."""
    x = np.zeros(A.shape[1])
    step = 1.0 / np.linalg.norm(A, 2) ** 2
    for _ in range(iters):
        x = np.maximum(x - step * (A.T @ (A @ x - b)), 0.0)
    return x


def main():
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)
    rng = np.random.default_rng(3)

    spectra = []
    for net in range(N_NETS):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]

        u = rng.standard_normal((M, D))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        gu = sym_output(u, weights)
        gbar = gu.mean(axis=0)
        gu_c = gu - gbar

        K = []
        for t in TS:
            w = rng.standard_normal((M, D))
            w -= (w * u).sum(1, keepdims=True) * u          # orthogonalise
            w /= np.linalg.norm(w, axis=1, keepdims=True)
            v = t * u + np.sqrt(1 - t * t) * w
            gv_c = sym_output(v, weights) - gbar
            K.append(float(np.mean(np.sum(gu_c * gv_c, axis=1))))
        # anchor at t=1: K(1) is exactly the measured variance of g, no model needed
        K1 = float(np.mean(np.sum(gu_c * gu_c, axis=1)))
        ts_fit = np.append(TS, 1.0)
        K = np.array(K + [K1])

        # K(t) = sum_{l>=2 even} a_l P_l(t)   (the l=0 term is removed by centring)
        degs = DEGREES[1:]
        A = np.stack([gegenbauer_normalised(int(l), ts_fit) for l in degs], axis=1)
        a = nnls(A, K)
        total = a.sum()
        spectra.append(a / total)
        resid = np.linalg.norm(A @ a - K) / np.linalg.norm(K)
        print(f"net {net}: fit residual {resid:.2%}  K(1)={K1:.5f}  "
              f"explained by fit {total/K1:.3f}")
        print("   " + "  ".join(f"deg{int(l)}={f:.3f}" for l, f in zip(degs, a / total)))
        # model-free check: how much of K(1) survives at each t
        print("   K(t)/K(1): " + " ".join(f"{t:.2f}:{k/K1:.4f}" for t, k in zip(TS, K[:-1])))

    S = np.mean(spectra, axis=0)
    degs = DEGREES[1:]
    print("\n" + "=" * 70)
    print("mean normalised even-harmonic spectrum (fraction of degree>=2 energy)")
    print("=" * 70)
    for l, f in zip(degs, S):
        print(f"  degree {int(l):>2}: {f:>7.4f}   {'#' * int(round(f * 60))}")

    tail = {int(l): S[degs >= l].sum() for l in degs}
    print("\nresidual energy above each design strength, and the gain over the 5-design:")
    print(f"{'design':>10} {'kills degrees':>16} {'residual':>10} {'gain vs 5-design':>18}")
    base = tail[6]
    for strength, first in [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]:
        r = tail.get(first, 0.0)
        print(f"{strength:>9}-design {'<= ' + str(strength):>16} {r:>10.4f} "
              f"{(base / r if r > 0 else float('inf')):>17.2f}x")
    print("\n(the shipped Kerdock design is a 5-design: 4.34x is needed to take the lead)")


if __name__ == "__main__":
    main()
