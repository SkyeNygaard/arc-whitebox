"""Offline calibration of the moment-propagation drift.

Every evaluation MLP is an i.i.d. draw from one known distribution (He init,
256x32), so anything *universal* about the propagation error can be measured
offline on training MLPs and applied at zero test-time FLOP cost.  This is a
distributional prior, not seed memorisation -- verified here with strict
train/test separation over disjoint MLP seeds.

Measured: the ratio (propagated sigma)/(true sigma) at layer 4 is
0.99740/0.99805/0.99718/0.99737 across four independent MLPs -- a spread of
3e-4, against a 3e-3 accuracy requirement.  The bias is systematic; only its
*accumulation* makes the late layers diverge.  So we fit one scalar per layer,
applied inside the recursion so drift never compounds.

The cumulants are NOT universal (kappa3 per (layer,|t|) bin swings from -0.20 to
+0.15 across seeds), so those still have to be bought per-MLP.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import whest.gaussmath as gm  # noqa: E402
from whest.budget import Budget, budget_for, score  # noqa: E402
from whest.edgeworth import relu_mean_edgeworth, sample_cumulants  # noqa: E402
from whest.estimators import COST_PHI, _relu_cross_cost  # noqa: E402
from whest.nets import load_or_build_reference, make_mlp, unbiased_mse  # noqa: E402
from whest.budget import CHEAP  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "refs")
N, L = 256, 32


def true_moments(mlp, Ns=400_000, chunk=32768, seed=11):
    M = [np.zeros((L, N)) for _ in range(5)]
    rng = np.random.default_rng(seed)
    done = 0
    while done < Ns:
        b = min(chunk, Ns - done)
        X = rng.standard_normal((b, N)).astype(np.float32)
        A = X
        for li, W in enumerate(mlp.Ws):
            H = A @ W.T
            A = np.maximum(H, 0.0)
            Hd = H.astype(np.float64)
            P = np.ones_like(Hd)
            for k in range(5):
                M[k][li] += P.sum(0)
                P = P * Hd
        done += b
    M = [m / Ns for m in M]
    mu = M[1]
    c2 = M[2] - mu**2
    c3 = M[3] - 3 * mu * M[2] + 2 * mu**3
    c4 = M[4] - 4 * mu * M[3] + 6 * mu**2 * M[2] - 3 * mu**4
    sd = np.sqrt(c2)
    return mu, sd, c3 / sd**3, c4 / sd**4 - 3.0


def propagate(mlp, k3, k4, cal=None, bud: Budget | None = None, nodes=8):
    """EMP with an optional per-layer sigma calibration applied in-loop."""
    W1 = mlp.Ws[0].astype(np.float64)
    mu = np.zeros(N)
    Sig = W1 @ W1.T
    if bud:
        bud.matmul(N, N, N, symmetric=True, op="Sigma_1")
    Y = np.zeros((L, N))
    SD = np.zeros((L, N))
    for li in range(L):
        sd = np.sqrt(np.maximum(np.diag(Sig), 1e-30))
        if cal is not None:
            sd = sd / cal[li]
            if bud:
                bud.elementwise(N, CHEAP, "calibration")
        SD[li] = sd
        Y[li] = relu_mean_edgeworth(mu, sd, k3[li], k4[li])
        if bud:
            bud.elementwise(3 * N, COST_PHI, "marginal_Phi")
            bud.elementwise(12 * N, CHEAP, "edgeworth")
        if li + 1 == L:
            break
        # rescale the covariance consistently with the calibrated sd
        if cal is not None:
            Sig = Sig / (cal[li] ** 2)
            if bud:
                bud.elementwise(N * N, CHEAP, "cal_cov")
        _, Sa = gm.relu_cov_from_gauss(mu, Sig, n_nodes=nodes)
        if bud:
            bud._add("relu_cross", _relu_cross_cost(N * (N + 1) // 2, nodes))
        d = Y[li] - gm.relu_mean(mu, sd)
        Sa = Sa - np.outer(d, d)
        W = mlp.Ws[li + 1].astype(np.float64)
        mu = W @ Y[li]
        Sig = W @ Sa @ W.T
        Sig = 0.5 * (Sig + Sig.T)
        if bud:
            bud.matmul(N, N, 1, op="mu_prop")
            bud.matmul(N, N, N, op="Sigma_prop")
            bud.matmul(N, N, N, symmetric=True, op="Sigma_prop")
            bud.elementwise(3 * N * N, CHEAP, "sym")
    return Y, SD


def fit_calibration(train_seeds, use_true_cumulants=True, cum_samples=6000):
    """Sequential per-layer fit: correct layer l using MLPs already corrected 1..l-1."""
    data = []
    for s in train_seeds:
        mlp = make_mlp(256, 32, s)
        mu_t, sd_t, k3_t, k4_t = true_moments(mlp)
        k3, k4 = (k3_t, k4_t) if use_true_cumulants else sample_cumulants(
            mlp, cum_samples, seed=90 + s, shrink=0.3)
        data.append((mlp, sd_t, k3, k4))
        print(f"  trained on seed {s}", flush=True)

    cal = np.ones(L)
    for li in range(L):
        ratios = []
        for mlp, sd_t, k3, k4 in data:
            _, SD = propagate(mlp, k3, k4, cal=cal)
            ratios.append(np.median(SD[li] / sd_t[li]))
        cal[li] = cal[li] * float(np.median(ratios))
    return cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, nargs="+", default=[10, 11, 12, 13, 14, 15])
    ap.add_argument("--test", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--cum-samples", type=int, default=6000)
    a = ap.parse_args()

    print(f"fitting calibration on held-out MLP seeds {a.train} ...", flush=True)
    cal = fit_calibration(a.train)
    print("\nper-layer sigma calibration factor:")
    print("  " + " ".join(f"{c:.4f}" for c in cal))

    B = budget_for(256, 32)
    print(f"\n{'seed':>5} {'EMP':>12} {'EMP+cal':>12} {'gain':>7} {'score(cal)':>12}")
    tot = []
    for s in a.test:
        mlp, ref = load_or_build_reference(256, 32, s, 20_000_000, ROOT)
        bud = Budget(dtype=np.float32)
        k3, k4 = sample_cumulants(mlp, a.cum_samples, seed=1234 + s, bud=bud, shrink=0.3)
        Y0, _ = propagate(mlp, k3, k4, cal=None)
        Y1, _ = propagate(mlp, k3, k4, cal=cal, bud=bud)
        m0, m1 = unbiased_mse(Y0, ref), unbiased_mse(Y1, ref)
        sc = score(max(m1, 0), bud.total, B)
        tot.append(sc)
        print(f"{s:>5} {m0:12.4e} {m1:12.4e} {m0/max(m1,1e-30):6.2f}x {sc:12.4e}")
    print(f"\nmean score over held-out MLPs: {np.mean(tot):.4e}   "
          f"(AIcrowd #1 = 1.24e-8, plain MC = 7.7e-7)")

    with open(os.path.join(os.path.dirname(__file__), "..", "results",
                           "calibration.json"), "w") as f:
        json.dump({"cal": cal.tolist(), "train": a.train, "test": a.test,
                   "scores": [float(v) for v in tot]}, f, indent=1)


if __name__ == "__main__":
    main()
