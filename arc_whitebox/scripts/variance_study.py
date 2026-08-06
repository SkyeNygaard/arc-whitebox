"""Precise variance comparison of MC-family estimators, from a single pass.

Anchored MC is *linear* in the sample averages (with beta held fixed), so it is
exactly the sample mean of a per-sample influence vector

    c_1(x) = Y_1            (constant -- layer 1 is known exactly)
    c_l(x) = a_l(x) + beta_l * ( W_l ( c_{l-1}(x) - a_{l-1}(x) ) )

and Var(Yhat_l) = Var(c_l)/N.  Estimating Var(c_l) from one pass gives a far
more precise comparison than repeated MSE measurements, which are chi^2_1-ish
because the final-layer fluctuation is rank-1 dominated.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from whest.gaussmath import SQRT2PI  # noqa: E402
from whest.nets import make_mlp  # noqa: E402


def influence_variance(mlp, N=40000, seed=0, chunk=4096, antithetic=False, sphere=False):
    """Returns dict of mean-over-neurons Var(c_L) for several estimator variants."""
    from scipy.special import gammaln

    n, L = mlp.n, mlp.L
    rng = np.random.default_rng(seed)
    ER = float(np.sqrt(2.0) * np.exp(gammaln((n + 1) / 2) - gammaln(n / 2)))
    Y1 = np.linalg.norm(mlp.Ws[0], axis=1) / SQRT2PI

    # first pass: beta_l = P(h_l > 0)
    S_p = np.zeros((L, n))
    m = 0
    while m < N:
        b = min(chunk, N - m)
        X = _draw(rng, b, n, antithetic, sphere, ER)
        A = X
        for li, W in enumerate(mlp.Ws):
            H = A @ W.T
            S_p[li] += (H > 0).sum(0)
            A = np.maximum(H, 0.0)
        m += b
    beta = S_p / N

    # second pass: accumulate mean/var of the influence vectors
    rng = np.random.default_rng(seed)
    keys = ["plain", "anchored"]
    s1 = {k: np.zeros((L, n)) for k in keys}
    s2 = {k: np.zeros((L, n)) for k in keys}
    m = 0
    while m < N:
        b = min(chunk, N - m)
        X = _draw(rng, b, n, antithetic, sphere, ER)
        A = X
        acts = []
        for W in mlp.Ws:
            A = np.maximum(A @ W.T, 0.0)
            acts.append(A.astype(np.float64))
        # plain
        for li in range(L):
            s1["plain"][li] += acts[li].sum(0)
            s2["plain"][li] += (acts[li] ** 2).sum(0)
        # anchored
        c = np.broadcast_to(Y1, (b, n)).astype(np.float64)
        s1["anchored"][0] += c.sum(0)
        s2["anchored"][0] += (c**2).sum(0)
        for li in range(1, L):
            c = acts[li] + beta[li] * ((c - acts[li - 1]) @ mlp.Ws[li].astype(np.float64).T)
            s1["anchored"][li] += c.sum(0)
            s2["anchored"][li] += (c**2).sum(0)
        m += b

    out = {}
    for k in keys:
        mu = s1[k] / N
        var = s2[k] / N - mu**2
        out[k] = var
    return out, beta


def _draw(rng, b, n, antithetic, sphere, ER):
    if antithetic:
        h = (b + 1) // 2
        Z = rng.standard_normal((h, n)).astype(np.float32)
        X = np.concatenate([Z, -Z])[:b]
    else:
        X = rng.standard_normal((b, n)).astype(np.float32)
    if sphere:
        X = X * (ER / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--samples", type=int, default=40000)
    a = ap.parse_args()

    rows = {}
    for seed in a.seeds:
        mlp = make_mlp(256, 32, seed)
        res = {}
        for tag, kw in [("iid", {}), ("anti", dict(antithetic=True)),
                        ("anti+sphere", dict(antithetic=True, sphere=True))]:
            v, beta = influence_variance(mlp, a.samples, seed=1234, **kw)
            res[tag] = {k: float(v[k][-1].mean()) for k in v}
            res[tag + "_perlayer"] = {k: v[k].mean(1).tolist() for k in v}
        base = res["iid"]["plain"]
        print(f"\n--- seed {seed} --- Var(a_L) = {base:.5f}   (mean over neurons)")
        print(f"{'sampling':>14} {'plain Var':>12} {'anchored Var':>14} {'reduction':>11}")
        for tag in ("iid", "anti", "anti+sphere"):
            p, an = res[tag]["plain"], res[tag]["anchored"]
            print(f"{tag:>14} {p:12.5f} {an:14.5f} {base/an:11.2f}x")
        rows[seed] = res

    with open(os.path.join(os.path.dirname(__file__), "..", "results",
                           "variance_study.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
