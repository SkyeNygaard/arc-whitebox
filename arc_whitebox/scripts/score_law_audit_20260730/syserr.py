"""Random vs SYSTEMATIC sigma error: which sensitivity governs a closure?

sensitivity.py injected iid noise per neuron per layer. A closure error is not
iid -- it is a deterministic function of each neuron's marginal, so it is
persistent and partially correlated, and it does NOT average down through the
256-term contraction mu_{l+1,i} = sum_j W_ij E[relu(p_l,j)].

Three injection structures at matched RMS:
  iid        fresh per neuron per layer
  persistent one draw per neuron, reused at every layer
  common     one draw per layer, shared by all neurons (fully correlated)
"""
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.special import ndtr
sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, ASSET, DATA, first_layer_design
from sensitivity import collect, SQRT2PI

N_NETS, DELTAS, N_SEEDS = 16, [0.0, 2e-3, 5e-3, 1e-2], 10

def chain(weights, mu0, mom, delta, mode, rng):
    mu = mu0
    pers = rng.standard_normal(WIDTH) if mode == "persistent" else None
    for w, (sd, k3, k4) in zip(weights, mom):
        if delta == 0.0:
            s = sd
        elif mode == "iid":
            s = sd * (1 + delta * rng.standard_normal(sd.shape))
        elif mode == "persistent":
            s = sd * (1 + delta * pers)
        else:
            s = sd * (1 + delta * rng.standard_normal())
        s = np.maximum(s, 1e-300)
        t = (mu @ w) / s
        ph = np.exp(-0.5 * t * t) / SQRT2PI
        mu = s * (t * ndtr(t) + ph - t * ph * k3 / 6 + (t * t - 1) * ph * k4 / 24)
    return mu

asset = np.load(ASSET); ch = asset["chirps"].astype(np.float32); ro = asset["rotation"].astype(np.float32)
tb = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
W_all = np.asarray(tb.column("weights").to_pylist(), dtype=np.float32)
Y_all = np.asarray(tb.column("final_means").to_pylist(), dtype=np.float64)
modes = ["iid", "persistent", "common"]
res = {(d, m): [] for d in DELTAS for m in modes}
for net in range(N_NETS):
    wts = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
    rows = first_layer_design(wts[0], ch, ro)
    mom, _ = collect(wts[1:], rows)
    mu0 = rows.mean(0, dtype=np.float64)
    for d in DELTAS:
        for m in modes:
            reps = 1 if d == 0 else N_SEEDS
            res[(d, m)].append(np.mean([np.mean((chain(wts[1:], mu0, mom, d, m,
                np.random.default_rng(500 + 31 * k)) - Y_all[net]) ** 2) for k in range(reps)]))
    print(f"  net {net} done", flush=True)
A = np.mean(res[(0.0, "iid")])
print(f"\nRANDOM vs SYSTEMATIC SIGMA ERROR ({N_NETS} networks)\n  floor A = {A:.4e}\n")
print(f"{'delta':>8} " + " ".join(f"{m:>14}" for m in modes))
for d in DELTAS:
    print(f"{d:>8.4f} " + " ".join(f"{np.mean(res[(d,m)]) - A:>14.4e}" for m in modes))
print(f"\n{'sensitivity C':>18} " + " ".join(
    f"{np.mean([ (np.mean(res[(d,m)])-A)/d**2 for d in DELTAS if d>0]):>14.4e}" for m in modes))
for m in modes:
    C = np.mean([(np.mean(res[(d, m)]) - A) / d ** 2 for d in DELTAS if d > 0])
    print(f"  {m:>11}: tolerance for MSE excess <= 8e-8  ->  delta <= {np.sqrt(8e-8/C):.3e}")
