"""Benchmark estimators against the cached MC reference."""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from whest.budget import budget_for, score  # noqa: E402
from whest.estimators import gauss_prop, monte_carlo  # noqa: E402
from whest.nets import load_or_build_reference, ref_noise_var, unbiased_mse, unbiased_mse_all  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "refs")


def evaluate(name, Yhat, bud, ref, B, extra=None):
    mse = unbiased_mse(Yhat, ref)
    per_layer = unbiased_mse_all(Yhat, ref)
    s = score(max(mse, 0.0), bud.total, B)
    row = dict(
        name=name,
        mse_final=float(mse),
        flops=float(bud.total),
        frac_budget=float(bud.total / B),
        score=float(s),
        per_layer_mse=[float(v) for v in per_layer],
    )
    if extra:
        row.update(extra)
    print(
        f"{name:34s} mse={mse:11.4e}  flops={bud.total:9.3e} ({100*bud.total/B:5.1f}% B)"
        f"  score={s:11.4e}",
        flush=True,
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--ref-samples", type=int, default=20_000_000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    B = budget_for(a.width, a.depth)
    print(f"budget B = {B:.4e} FLOPs   (half-budget operating point {B/2:.4e})\n")
    rows = []
    for seed in a.seeds:
        mlp, ref = load_or_build_reference(a.width, a.depth, seed, a.ref_samples, ROOT)
        print(f"--- seed {seed}   ref noise var={ref_noise_var(ref):.3e}  "
              f"Var(a_L)={float(np.var(ref['Y'][-1])):.4f} ---")

        # ---- Monte Carlo family, at the half-budget operating point ----
        per_sample = 2 * a.width * a.width * a.depth  # fp32
        n_half = int((B / 2) / per_sample)
        for anti in (False, True):
            t = time.time()
            Y, bud = monte_carlo(mlp, n_half, seed=1000 + seed, antithetic=anti)
            rows.append(evaluate(
                f"MC{'+antithetic' if anti else ''} (N={n_half//1000}k)", Y, bud, ref, B,
                dict(seed=seed, secs=time.time() - t)))

        # ---- Gaussian moment propagation ----
        for mode in ("diag", "linearized", "exact"):
            t = time.time()
            Y, bud = gauss_prop(mlp, mode=mode)
            rows.append(evaluate(f"GaussProp[{mode}]", Y, bud, ref, B,
                                 dict(seed=seed, secs=time.time() - t)))
        print()

    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
