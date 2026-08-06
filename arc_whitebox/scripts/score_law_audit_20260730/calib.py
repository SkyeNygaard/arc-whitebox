"""CANDIDATE: is the covariance-closure error universal enough to calibrate away?

cov_closure.py: the Gaussian closure mis-states sigma by ~1.1% per layer against
a ~3e-3 requirement.  notes/03 §5.2 found the *sigma propagation* bias
"strikingly universal" (spread 3e-4 across networks) but the fix FAILED when
tested -- because the cumulants were sampled at 6k and sigma was not then the
binding constraint.  It has never been retested now that the marginal closure is
known to be good (1.9e-4/layer with Edgeworth).

A universal per-layer constant is free at test time: it is a distributional
prior over He-initialised networks, not per-network fitting.

Honest protocol: fit one scalar per layer on all networks EXCEPT the held-out
one, apply to the held-out one, repeat.  Reported error is out-of-sample.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, ASSET, DATA, first_layer_design

N_NETS = 8
M = 200_000
LAYERS = [1, 2, 4, 8, 12, 16, 20, 24, 28, 30]


def gaussian_sample(mu, Sigma, m, rng):
    evals, evecs = np.linalg.eigh(Sigma)
    L = (evecs * np.sqrt(np.maximum(evals, 0.0))).astype(np.float32)
    return mu.astype(np.float32) + rng.standard_normal((m, mu.size), dtype=np.float32) @ L.T


def main():
    asset = np.load(ASSET)
    chirps = asset["chirps"].astype(np.float32)
    rotation = asset["rotation"].astype(np.float32)
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)
    rng = np.random.default_rng(23)

    # rel[net][layer] = per-neuron relative sigma error of the Gaussian closure
    rel = {li: [] for li in LAYERS}
    for net in range(N_NETS):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        act = first_layer_design(weights[0], chirps, rotation)
        for li, w in enumerate(weights[1:], start=1):
            Z = act @ w
            if li in LAYERS and li + 1 <= DEPTH - 1:
                wn = weights[li + 1]
                var_true = (np.maximum(Z, 0.0, dtype=np.float32) @ wn).var(axis=0, dtype=np.float64)
                Z64 = Z.astype(np.float64)
                g = gaussian_sample(Z64.mean(0), np.cov(Z64, rowvar=False, bias=True), M, rng)
                var_cl = (np.maximum(g, 0.0, dtype=np.float32) @ wn).var(axis=0, dtype=np.float64)
                rel[li].append(np.sqrt(var_cl) / np.sqrt(var_true) - 1.0)
            act = np.maximum(Z, 0.0, dtype=np.float32)

    print(f"Gaussian covariance closure, {N_NETS} networks, M={M:,}")
    print(f"MC noise floor {np.sqrt(2/M)/2:.2e}\n")
    print(f"{'layer':>6} {'RMS raw':>10} {'mean bias':>11} {'spread/nets':>12} "
          f"{'RMS after LORO calib':>22} {'gain':>7}")
    raws, cals = [], []
    for li in LAYERS:
        R = np.array(rel[li])                     # (nets, neurons)
        per_net_mean = R.mean(axis=1)
        raw = np.sqrt(np.mean(R ** 2))
        # leave-one-network-out single scalar per layer
        resid = []
        for k in range(len(R)):
            c = np.mean(np.delete(per_net_mean, k))
            resid.append((R[k] - c) ** 2)
        cal = np.sqrt(np.mean(resid))
        raws.append(raw)
        cals.append(cal)
        print(f"{li:>6} {raw:>10.3e} {per_net_mean.mean():>11.3e} "
              f"{per_net_mean.std():>12.3e} {cal:>22.3e} {raw/cal:>6.2f}x")

    print(f"\n  mean RMS sigma error, raw           : {np.mean(raws):.3e}")
    print(f"  mean RMS sigma error, LORO calibrated: {np.mean(cals):.3e}")
    print(f"  overall gain                        : {np.mean(raws)/np.mean(cals):.2f}x")
    print("\n  requirement ~3e-3 for the 11x ceiling, ~1e-2 for a 4.34x win")


if __name__ == "__main__":
    main()
