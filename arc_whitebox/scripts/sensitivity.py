"""Where along the depth does the final-layer error actually get injected?

Two measurements:

1. Analytic sensitivity  ||d Y_L / d Y_l||  =  || prod_{k>l} diag(p_k) W_k ||_F / sqrt(n).
   If this decays, errors made early are damped and only late layers matter.

2. Hybrid-oracle sweep: take the TRUE (mu, Sigma) up to layer k, then run
   GaussProp from k+1 to L.  Final-layer MSE as a function of k says exactly how
   much of GaussProp's error is injected after layer k.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from whest.estimators import gauss_prop, oracle_moments  # noqa: E402
from whest.gaussmath import Phi  # noqa: E402
from whest.nets import load_or_build_reference, unbiased_mse  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "refs")
ORC = os.path.join(os.path.dirname(__file__), "..", "data", "oracle")


def get_oracle(mlp, seed, samples):
    os.makedirs(ORC, exist_ok=True)
    p = os.path.join(ORC, f"orc_s{seed}_m{samples}.npz")
    if os.path.exists(p):
        z = np.load(p)
        return {"mu_h": z["mu"], "Sigma_h": list(z["Sig"])}
    o = oracle_moments(mlp, samples)
    np.savez(p, mu=o["mu_h"], Sig=np.stack(o["Sigma_h"]))
    return o


def main(seed=0, width=256, depth=32, oracle_samples=1_000_000):
    mlp, ref = load_or_build_reference(width, depth, seed, 20_000_000, ROOT)
    orc = get_oracle(mlp, seed, oracle_samples)

    # ---- 1. analytic sensitivity ----
    _, _, stats = gauss_prop(mlp, mode="exact", return_stats=True)
    p = [Phi(mu / np.sqrt(np.maximum(np.diag(S), 1e-30))) for mu, S in stats]
    n = width
    sens = np.zeros(depth)
    Jac = np.eye(n)
    sens[depth - 1] = 1.0
    for l in range(depth - 2, -1, -1):
        Jac = Jac @ (mlp.Ws[l + 1].astype(np.float64) * p[l][None, :])
        sens[l] = np.linalg.norm(Jac, "fro") / np.sqrt(n)

    # ---- 2. hybrid-oracle sweep ----
    hyb = {}
    for k in [0, 1, 2, 4, 8, 12, 16, 20, 24, 26, 28, 30, 31, 32]:
        st = {"mu_h": orc["mu_h"], "Sigma_h": orc["Sigma_h"]}
        Y, _ = gauss_prop_hybrid(mlp, st, k)
        hyb[k] = unbiased_mse(Y, ref)

    print(f"{'layer l':>8} {'sensitivity |dY_L/dY_l|':>24}")
    for l in range(depth):
        print(f"{l+1:>8} {sens[l]:>24.4f}")
    print(f"\n{'oracle up to k':>15} {'final MSE':>14}   (k=0 is pure GaussProp, k=32 is pure oracle)")
    for k, v in hyb.items():
        print(f"{k:>15} {v:>14.4e}")

    with open(os.path.join(os.path.dirname(__file__), "..", "results",
                           f"sensitivity_s{seed}.json"), "w") as f:
        json.dump({"sensitivity": sens.tolist(),
                   "hybrid_oracle": {str(k): float(v) for k, v in hyb.items()}}, f, indent=1)


def gauss_prop_hybrid(mlp, orc, k):
    """Oracle (mu, Sigma) for layers 1..k, then analytic propagation."""
    import whest.gaussmath as gm
    from whest.budget import Budget

    n, L = mlp.n, mlp.L
    bud = Budget()
    W1 = mlp.Ws[0].astype(np.float64)
    mu, Sig = np.zeros(n), W1 @ W1.T
    Y = np.zeros((L, n))
    for li in range(L):
        if li < k:
            mu, Sig = orc["mu_h"][li], orc["Sigma_h"][li]
        mu_a, Sig_a = gm.relu_cov_from_gauss(mu, Sig)
        Y[li] = mu_a
        if li + 1 < L:
            W = mlp.Ws[li + 1].astype(np.float64)
            mu = W @ mu_a
            Sig = W @ Sig_a @ W.T
            Sig = 0.5 * (Sig + Sig.T)
    return Y, bud


if __name__ == "__main__":
    main(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
