"""How accurate must sigma be?  Calibrating the analytic route's requirement.

There is a 5.5x unresolved gap in the literature on this project:
  * notes/03 section 3 says sigma needs ~3e-3 relative to reach MSE 1.30e-7;
  * anchoring MSE ~ (sigma error)^2 on covariance_propagation (1.4e-2 -> 8.4e-5)
    implies ~5.5e-4.

It decides which problem is open.  If the requirement is 3e-3, the location
latent at r=64 (measured 1.96e-3) ALREADY clears it and what remains is purely a
compression problem.  If it is 5.5e-4, nothing measured is close.

Method: propagate the design rows ONCE per network and store the per-layer
sigma/kappa.  Then re-run the analytic mean chain many times, injecting
multiplicative noise  sigma -> sigma (1 + eps),  eps ~ N(0, delta^2) per neuron
per layer.  The chain itself is 31 tiny matmuls, so hundreds of runs are free.

Fit   MSE(delta) = A + C delta^2
  A = the chain's floor at the design's own moment quality
  C = the sensitivity, which converts a sigma requirement into an MSE budget
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, ASSET, DATA, first_layer_design

SQRT2PI = np.sqrt(2 * np.pi)
N_NETS = 20
DELTAS = [0.0, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
N_SEEDS = 12


def collect(weights, rows):
    """One true propagation; return per-layer (sigma, k3, k4) and E[a_0]."""
    act = rows
    mom = []
    for w in weights:
        p = (act @ w).astype(np.float64)
        m = p.mean(0)
        c = p - m
        sd = np.sqrt(np.maximum((c ** 2).mean(0), 1e-300))
        k3 = (c ** 3).mean(0) / sd ** 3
        k4 = (c ** 4).mean(0) / sd ** 4 - 3.0
        mom.append((sd, k3, k4))
        act = np.maximum(p, 0.0).astype(np.float32)
    return mom, act.mean(0, dtype=np.float64)


def chain(weights, mu0, mom, delta, rng):
    """Analytic mean chain with sigma perturbed by relative noise `delta`."""
    mu = mu0
    for w, (sd, k3, k4) in zip(weights, mom):
        s = sd if delta == 0.0 else sd * (1.0 + delta * rng.standard_normal(sd.shape))
        s = np.maximum(s, 1e-300)
        t = (mu @ w) / s
        ph = np.exp(-0.5 * t * t) / SQRT2PI
        mu = s * (t * ndtr(t) + ph - t * ph * k3 / 6.0
                  + (t * t - 1.0) * ph * k4 / 24.0)
    return mu


def main():
    asset = np.load(ASSET)
    chirps = asset["chirps"].astype(np.float32)
    rotation = asset["rotation"].astype(np.float32)
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)
    Y_all = np.asarray(table.column("final_means").to_pylist(), dtype=np.float64)

    res = {d: [] for d in DELTAS}
    direct = []
    for net in range(N_NETS):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        rows = first_layer_design(weights[0], chirps, rotation)
        mom, fin = collect(weights[1:], rows)
        direct.append(np.mean((fin - Y_all[net]) ** 2))
        mu0 = rows.mean(0, dtype=np.float64)
        for d in DELTAS:
            reps = 1 if d == 0.0 else N_SEEDS
            vals = [np.mean((chain(weights[1:], mu0, mom, d,
                                   np.random.default_rng(9000 + 17 * k)) - Y_all[net]) ** 2)
                    for k in range(reps)]
            res[d].append(np.mean(vals))
        print(f"  net {net} done", flush=True)

    print(f"\nSIGMA SENSITIVITY OF THE ANALYTIC CHAIN ({N_NETS} networks)\n")
    print(f"  direct estimator MSE (same rows): {np.mean(direct):.4e}\n")
    print(f"{'injected delta':>15} {'chain MSE':>13} {'excess over delta=0':>21}")
    A = np.mean(res[0.0])
    for d in DELTAS:
        m = np.mean(res[d])
        print(f"{d:>15.4f} {m:>13.4e} {m - A:>21.4e}")

    ds = np.array([d for d in DELTAS if d > 0])
    ex = np.array([np.mean(res[d]) - A for d in ds])
    C = float(np.sum(ex * ds ** 2) / np.sum(ds ** 4))
    print(f"\n  floor A (design's own moment quality) = {A:.4e}")
    print(f"  sensitivity C (MSE per unit delta^2)  = {C:.4e}")
    print(f"\n  sigma budget for a given MSE target (excess only):")
    for T in [1.3e-7, 3.0e-7, 1.0e-6]:
        print(f"    MSE excess <= {T:.1e}  ->  delta <= {np.sqrt(T / C):.3e}")
    print("\n  If the tolerated delta is ~3e-3, the r=64 latent (1.96e-3) clears it")
    print("  and the open problem is compression. If ~5e-4, nothing measured does.")


if __name__ == "__main__":
    main()
