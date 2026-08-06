"""Where does GaussProp's error come from?

Three variants, run against the same reference:
  A. full GaussProp                     -> marginal-shape error + covariance-propagation error
  B. GaussProp with ORACLE (mu, Sigma)  -> marginal-shape error only
  C. oracle mu + oracle sigma, Gaussian marginal, per layer -> pure marginal error

The gap A - B is what better covariance propagation could buy; B/C is the floor
for *any* method that assumes Gaussian marginals.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from whest.estimators import gauss_prop, oracle_moments  # noqa: E402
from whest.gaussmath import relu_mean  # noqa: E402
from whest.nets import load_or_build_reference, unbiased_mse_all  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "refs")


def main(seed=0, width=256, depth=32, oracle_samples=1_000_000):
    mlp, ref = load_or_build_reference(width, depth, seed, 20_000_000, ROOT)
    print(f"computing oracle moments with {oracle_samples:,} samples ...", flush=True)
    orc = oracle_moments(mlp, oracle_samples)

    Ya, _ = gauss_prop(mlp, mode="exact")
    Yb, _ = gauss_prop(mlp, mode="exact", oracle_stats=orc)

    mse_a = unbiased_mse_all(Ya, ref)
    mse_b = unbiased_mse_all(Yb, ref)

    Yc = np.stack([
        relu_mean(orc["mu_h"][li], np.sqrt(np.maximum(np.diag(orc["Sigma_h"][li]), 1e-30)))
        for li in range(depth)
    ])
    mse_c = unbiased_mse_all(Yc, ref)

    print(f"\n{'layer':>5} {'A: full':>12} {'B: oracle mu,Sig':>18} {'C: oracle marginals':>21}")
    for li in range(depth):
        print(f"{li+1:>5} {mse_a[li]:>12.3e} {mse_b[li]:>18.3e} {mse_c[li]:>21.3e}")

    out = dict(seed=seed, full=[float(v) for v in mse_a],
               oracle_moments=[float(v) for v in mse_b],
               oracle_marginals=[float(v) for v in mse_c])
    with open(os.path.join(os.path.dirname(__file__), "..", "results",
                           f"decompose_s{seed}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
