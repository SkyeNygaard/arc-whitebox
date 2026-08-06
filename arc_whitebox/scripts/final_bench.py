"""Consolidated benchmark.

Monte-Carlo-family estimators are compared by the variance of the quantity they
actually average (the "unit"), which is far more precise than repeated MSE
measurement -- the final-layer fluctuation is rank-1 dominated, so single-run
MSE is essentially chi^2_1 and swings by 20x.

  * i.i.d. sampling:   unit = c(x),                   N units per N samples
  * antithetic:        unit = (c(x) + c(-x))/2,       N/2 units per N samples
  * sphere:            unit = c(E||x|| * x/||x||),    N units per N samples

so MSE = Var(unit) / n_units, and the comparison is on Var(unit)/samples_per_unit.

Deterministic (white-box) estimators are evaluated exactly against the reference.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.special import gammaln

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from whest.asgm import asgm  # noqa: E402
from whest.budget import budget_for, score  # noqa: E402
from whest.estimators import gauss_prop  # noqa: E402
from whest.gaussmath import SQRT2PI  # noqa: E402
from whest.nets import load_or_build_reference, unbiased_mse  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "refs")
PER_SAMPLE = 2 * 256 * 256 * 32  # fp32 forward pass


def _units(mlp, N, seed, mode, chunk=4096):
    """Yield (units, samples_used) where `units` is (n_units, L, n)."""
    n, L = mlp.n, mlp.L
    rng = np.random.default_rng(seed)
    ER = float(np.sqrt(2.0) * np.exp(gammaln((n + 1) / 2) - gammaln(n / 2)))
    done = 0
    while done < N:
        b = min(chunk, N - done)
        if mode == "anti":
            h = b // 2
            Z = rng.standard_normal((h, n)).astype(np.float32)
            X = np.concatenate([Z, -Z])
        else:
            X = rng.standard_normal((b, n)).astype(np.float32)
            if mode == "sphere":
                X = X * (ER / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
            elif mode == "anti_sphere":
                hh = b // 2
                Z = X[:hh]
                Z = Z * (ER / np.linalg.norm(Z, axis=1, keepdims=True)).astype(np.float32)
                X = np.concatenate([Z, -Z])
        A = X
        acts = []
        for W in mlp.Ws:
            A = np.maximum(A @ W.T, 0.0)
            acts.append(A.astype(np.float64))
        yield np.stack(acts, 1), X, b  # (b, L, n)
        done += b


def influence_stats(mlp, N=40000, seed=1234, mode="iid", chunk=4096):
    """Var of the averaged unit, for plain and anchored estimators."""
    n, L = mlp.n, mlp.L
    Y1 = np.linalg.norm(mlp.Ws[0], axis=1) / SQRT2PI
    Wl = [W.astype(np.float64) for W in mlp.Ws]

    # pass 1: beta
    S_p = np.zeros((L, n))
    tot = 0
    for acts, _, b in _units(mlp, N, seed, mode, chunk):
        S_p += (acts > 0).sum(0)
        tot += b
    beta = S_p / tot

    paired = mode in ("anti", "anti_sphere")
    s1 = {"plain": np.zeros((L, n)), "anchored": np.zeros((L, n))}
    s2 = {"plain": np.zeros((L, n)), "anchored": np.zeros((L, n))}
    nu = 0
    for acts, _, b in _units(mlp, N, seed, mode, chunk):
        c = np.empty_like(acts)
        c[:, 0] = Y1
        for li in range(1, L):
            c[:, li] = acts[:, li] + beta[li] * ((c[:, li - 1] - acts[:, li - 1]) @ Wl[li].T)
        for key, arr in (("plain", acts), ("anchored", c)):
            u = 0.5 * (arr[: b // 2] + arr[b // 2:]) if paired else arr
            s1[key] += u.sum(0)
            s2[key] += (u * u).sum(0)
        nu += (b // 2) if paired else b
    out = {}
    for k in s1:
        mu = s1[k] / nu
        out[k] = float((s2[k] / nu - mu * mu)[-1].mean())
    out["samples_per_unit"] = 2.0 if paired else 1.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--var-samples", type=int, default=60000)
    a = ap.parse_args()

    B = budget_for(256, 32)
    N_half = int((B / 2) / PER_SAMPLE)
    print(f"budget B={B:.3e}, half-budget sample count N={N_half:,}\n")

    rows = []
    for seed in a.seeds:
        mlp, ref = load_or_build_reference(256, 32, seed, 20_000_000, ROOT)
        print(f"===== seed {seed} =====")

        for mode in ("iid", "anti", "sphere", "anti_sphere"):
            st = influence_stats(mlp, a.var_samples, mode=mode)
            spu = st["samples_per_unit"]
            for est in ("plain", "anchored"):
                mse = st[est] * spu / N_half
                s = score(mse, B / 2, B)
                name = f"MC[{est}, {mode}]"
                rows.append(dict(seed=seed, name=name, mse=mse, score=s,
                                 flops=B / 2, kind="mc"))
                print(f"  {name:28s} Var(unit)={st[est]:.5f}  MSE={mse:.4e}  score={s:.4e}")

        for mode in ("diag", "linearized", "exact"):
            Y, bud = gauss_prop(mlp, mode=mode)
            mse = unbiased_mse(Y, ref)
            s = score(max(mse, 0), bud.total, B)
            rows.append(dict(seed=seed, name=f"GaussProp[{mode}]", mse=mse, score=s,
                             flops=bud.total, kind="whitebox"))
            print(f"  {'GaussProp[' + mode + ']':28s} MSE={mse:.4e}  "
                  f"flops={100*bud.total/B:.2f}%B  score={s:.4e}")

        for r, K in ((8, 4096), (32, 4096)):
            Y, bud = asgm(mlp, r=r, K=K)
            mse = unbiased_mse(Y, ref)
            s = score(max(mse, 0), bud.total, B)
            rows.append(dict(seed=seed, name=f"ASGM[r={r},K={K}]", mse=mse, score=s,
                             flops=bud.total, kind="hybrid"))
            print(f"  {'ASGM[r=' + str(r) + ']':28s} MSE={mse:.4e}  "
                  f"flops={100*bud.total/B:.1f}%B  score={s:.4e}")
        print()

    # aggregate
    names = sorted({r["name"] for r in rows})
    print("===== mean score across seeds (this is what the leaderboard averages) =====")
    agg = []
    for nm in names:
        sc = [r["score"] for r in rows if r["name"] == nm]
        agg.append((float(np.mean(sc)), nm))
    for v, nm in sorted(agg):
        print(f"  {nm:30s} {v:.4e}")

    with open(os.path.join(os.path.dirname(__file__), "..", "results",
                           "final_bench.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
