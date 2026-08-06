"""CANDIDATE: a scale-mixture (radial-conditioned) closure.

The blocker is the covariance closure: Gaussianizing z_l mis-states sigma by
1.34e-2 against a ~3e-3 requirement.

Structural idea.  These networks are positively homogeneous (no biases), so
writing a_{l-1} = rho * ahat with rho = ||a_{l-1}||,

    z_l = a_{l-1} W = rho * (ahat W) = rho * zhat

and because relu is positively homogeneous,

    E[relu(z_i)]            = E[rho]   E[relu(zhat_i)]
    E[relu(z_i) relu(z_j)]  = E[rho^2] E[relu(zhat_i) relu(zhat_j)]

exactly, whenever rho is independent of the direction ahat.  The hypothesis is
that the dominant non-Gaussianity of z is precisely this fluctuating radial
scale -- a scale mixture of Gaussians -- and that zhat is much closer to
Gaussian than z is.  If so the fix costs two extra scalars per layer, not an
n^4 contraction.

Both radial moments are available analytically in a real propagation:
E[rho^2] = sum_j E[a_j^2] comes straight from the closure.  Here they are taken
from the rows so the test isolates the closure question.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, ASSET, DATA, first_layer_design

N_NETS = 4
M = 200_000
LAYERS = [2, 4, 8, 12, 16, 20, 24, 28, 30]


def gsample(mu, Sigma, m, rng):
    ev, V = np.linalg.eigh(Sigma)
    L = (V * np.sqrt(np.maximum(ev, 0.0))).astype(np.float32)
    return mu.astype(np.float32) + rng.standard_normal((m, mu.size), dtype=np.float32) @ L.T


def main():
    asset = np.load(ASSET)
    chirps = asset["chirps"].astype(np.float32)
    rotation = asset["rotation"].astype(np.float32)
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)
    rng = np.random.default_rng(41)

    plain, mixed = {l: [] for l in LAYERS}, {l: [] for l in LAYERS}
    for net in range(N_NETS):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        act = first_layer_design(weights[0], chirps, rotation)
        for li, w in enumerate(weights[1:], start=1):
            Z = act @ w
            if li in LAYERS and li + 1 <= DEPTH - 1:
                wn = weights[li + 1]
                A = np.maximum(Z, 0.0, dtype=np.float32)
                var_true = (A @ wn).var(axis=0, dtype=np.float64)

                # --- plain Gaussian closure on z
                Z64 = Z.astype(np.float64)
                g = gsample(Z64.mean(0), np.cov(Z64, rowvar=False, bias=True), M, rng)
                v_plain = (np.maximum(g, 0.0, dtype=np.float32) @ wn).var(axis=0, dtype=np.float64)

                # --- scale-mixture closure: Gaussianize zhat = z / ||a_{l-1}||
                rho = np.linalg.norm(act, axis=1).astype(np.float64)
                Zh = Z64 / rho[:, None]
                gh = gsample(Zh.mean(0), np.cov(Zh, rowvar=False, bias=True), M, rng)
                rh = np.maximum(gh, 0.0, dtype=np.float32) @ wn      # relu(zhat) . w
                m1 = rh.mean(axis=0, dtype=np.float64)
                m2 = (rh.astype(np.float64) ** 2).mean(axis=0)
                Er, Er2 = rho.mean(), (rho ** 2).mean()
                v_mixed = Er2 * m2 - (Er * m1) ** 2

                plain[li].append(np.sqrt(np.mean((np.sqrt(v_plain / var_true) - 1) ** 2)))
                mixed[li].append(np.sqrt(np.mean((np.sqrt(np.maximum(v_mixed, 0) / var_true) - 1) ** 2)))
            act = np.maximum(Z, 0.0, dtype=np.float32)

    print(f"RMS relative sigma error of the next-layer variance, {N_NETS} networks\n")
    print(f"{'layer':>6} {'Gaussian':>12} {'scale-mixture':>15} {'gain':>8}")
    for li in LAYERS:
        p, m = np.mean(plain[li]), np.mean(mixed[li])
        print(f"{li:>6} {p:>12.3e} {m:>15.3e} {p/m:>7.2f}x")
    P = np.mean([np.mean(plain[l]) for l in LAYERS])
    Q = np.mean([np.mean(mixed[l]) for l in LAYERS])
    print(f"\n  mean  Gaussian {P:.3e}   scale-mixture {Q:.3e}   gain {P/Q:.2f}x")
    print("  requirement ~3e-3 for the 11x ceiling")


if __name__ == "__main__":
    main()
