"""Can the final-layer fluctuation be predicted -- and therefore controlled?

MC's error lives in the (very low-rank) fluctuation of a_L.  If that fluctuation
is a low-degree polynomial in a handful of linear projections of x, then Hermite
control variates (whose Gaussian means are known exactly) kill most of the MC
variance and the whole problem changes character.

Measures R^2 of predicting a_L from:
  * degree-1..3 polynomials in the top-r active-subspace projections of x
  * ||x||  (the exactly-known radial factor -- the net is positively homogeneous)
  * an intermediate layer's activations (linear)
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from whest.asgm import active_subspace  # noqa: E402
from whest.budget import Budget  # noqa: E402
from whest.estimators import gauss_prop  # noqa: E402
from whest.nets import make_mlp  # noqa: E402


def poly_features(S, degree):
    """Hermite-ish polynomial features of S (N, r), all mean-zero under N(0,I)."""
    feats = [S]
    if degree >= 2:
        r = S.shape[1]
        iu = np.triu_indices(r)
        Q = (S[:, :, None] * S[:, None, :])[:, iu[0], iu[1]]
        Q = Q - (iu[0] == iu[1]).astype(float)  # He_2
        feats.append(Q)
    if degree >= 3:
        feats.append(S**3 - 3 * S)  # He_3, diagonal only
        feats.append((S**2 - 1) * np.roll(S, 1, axis=1))
    return np.concatenate(feats, axis=1)


def r2_of(F, y):
    """R^2 of the least-squares fit of y (N,m) on F (N,p) plus intercept."""
    F = np.concatenate([np.ones((len(F), 1)), F], 1)
    beta, *_ = np.linalg.lstsq(F, y, rcond=None)
    resid = y - F @ beta
    return 1.0 - resid.var(0).sum() / y.var(0).sum()


def main(width=256, depth=32, seed=0, N=60000, r=16):
    mlp = make_mlp(width, depth, seed)
    _, _, stats = gauss_prop(mlp, mode="linearized", return_stats=True)
    V = active_subspace(mlp, stats, r, Budget())  # (n, r) in h_1 space

    rng = np.random.default_rng(3)
    X = rng.standard_normal((N, width)).astype(np.float32)
    H = X @ mlp.Ws[0].T
    S = (H @ V).astype(np.float64)  # active-subspace projections
    S = S / S.std(0)
    mids = {}
    A = np.maximum(H, 0)
    for li in range(1, depth):
        A = np.maximum(A @ mlp.Ws[li].T, 0)
        if li + 1 in (8, 16, 24, 28, 30, 31):
            mids[li + 1] = A.astype(np.float64).copy()
    aL = A.astype(np.float64)

    C = np.cov(aL.T)
    w, U = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    print(f"final-layer Cov: top-5 eigenvalue shares = "
          f"{np.round(w[order][:5]/w.sum(), 3)}   eff_rank={w.sum()**2/(w**2).sum():.2f}")
    xi = aL @ U[:, order[:3]]  # top-3 fluctuation coordinates

    out = {}
    rad = np.linalg.norm(X, axis=1).astype(np.float64)[:, None]
    print(f"\n{'predictor':>40} {'R2(top-3 xi)':>14} {'R2(all a_L)':>13}")

    def report(name, F):
        a = r2_of(F, xi)
        b = r2_of(F, aL)
        out[name] = (float(a), float(b))
        print(f"{name:>40} {a:>14.4f} {b:>13.4f}")

    report("||x|| only (exact, free)", rad - rad.mean())
    for rr in (2, 4, 8, 16):
        report(f"linear in top-{rr} active dirs", S[:, :rr])
    for deg in (2, 3):
        for rr in (4, 8, 16):
            report(f"degree-{deg} poly, top-{rr} active dirs", poly_features(S[:, :rr], deg))
    report("linear in all 256 x-coords", X.astype(np.float64))
    for k, M in sorted(mids.items()):
        report(f"linear in a_{k} (all 256)", M)

    with open(os.path.join(os.path.dirname(__file__), "..", "results",
                           f"predictability_s{seed}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
