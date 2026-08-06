"""GATE 4 -- does the cumulant recursion survive a full rollout from layer 1?

Gate 1: kappa3(h_1) = 0 exactly, and c21(a_1) is closed-form from W1 (0.028%).
Gate 2: c21 is inherited, not generated (15:1 at depth), so it must propagate.
Gate 3: the expected-gate closure reconstructs one layer at correlation 0.9959
        with exact magnitude; residual 8.7% is pure shape error.

Unrolling the recursion

    kappa3(h_{l+1}) = M_l-transport of kappa3(h_l)  +  W-transport of G_l
    M_l = D_l W_{l+1},  D_l = diag(Phi(mu_l/sigma_l)),  kappa3(h_1) = 0

gives kappa3(h_L) as a SUM of generated terms, each transported forward.  Its
c21 slice therefore has a closed form needing no 256^3 tensor:

    c21(h_L) = sum_{l<L} c21( ReLU(z_l) @ Q_l ),   z_l ~ N(mu_l, Sigma_l),
    Q_l = W_{l+1} M_{l+1} ... M_{L-1}

Every term is sampled from a GAUSSIAN whose moments are known, never from the
network.  That is the whole point: it is a deployable computation, not an
oracle one.

This is the test five prior attempts failed (M30, M44, M55, M63, M79/M80): a
strong one-step closure that dies in free rollout.  True (mu_l, Sigma_l) are
supplied here so the cumulant recursion is isolated from covariance drift; if it
fails even with exact moments, the route is closed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "scripts"))

from eval_sampling_official import DEFAULT_DATA, _load_rows  # noqa: E402

WIDTH = 256
CHUNK = 32768


def _c21(sums, n):
    s1, s2, s21 = sums
    mu = s1 / n
    m2 = s2 / n
    return (s21 / n - mu[None, :] * np.diag(m2)[:, None]
            - 2.0 * mu[:, None] * m2 + 2.0 * (mu * mu)[:, None] * mu[None, :])


def layer_moments(weights, target, n_samples, seed):
    """mu_l, Sigma_l for every l < target, plus true c21(h_target)."""
    s1 = [np.zeros(WIDTH) for _ in range(target + 1)]
    s2 = [np.zeros((WIDTH, WIDTH)) for _ in range(target + 1)]
    t1 = np.zeros(WIDTH); t2 = np.zeros((WIDTH, WIDTH)); t21 = np.zeros((WIDTH, WIDTH))
    rng = np.random.default_rng(seed); done = 0
    while done < n_samples:
        b = min(CHUNK, n_samples - done)
        a = rng.standard_normal((b, WIDTH)).astype(np.float32)
        for li in range(target + 1):
            h = a @ weights[li]; a = np.maximum(h, 0.0)
            hd = h.astype(np.float64)
            s1[li] += hd.sum(0); s2[li] += hd.T @ hd
            if li == target:
                t1 += hd.sum(0); t2 += hd.T @ hd; t21 += (hd * hd).T @ hd
        done += b
    mus = [s / n_samples for s in s1]
    sigmas = [s2[i] / n_samples - np.outer(mus[i], mus[i]) for i in range(target + 1)]
    return mus, sigmas, _c21((t1, t2, t21), n_samples)


def predicted_c21(weights, mus, sigmas, target, n_samples, seed):
    """sum_{l<target} c21( ReLU(z_l) @ Q_l ), all Gaussian sampling."""
    gates = []
    for li in range(target):
        sd = np.sqrt(np.maximum(np.diag(sigmas[li]), 1e-300))
        gates.append(norm.cdf(mus[li] / sd))
    # Q_l = W_{l+1} M_{l+1} ... M_{target-1},  M_k = diag(gate_k) W_{k+1}
    q = {}
    acc = np.eye(WIDTH)
    for li in range(target - 1, -1, -1):
        q[li] = weights[li + 1].astype(np.float64) @ acc
        acc = (gates[li][:, None] * weights[li + 1].astype(np.float64)) @ acc
    total = np.zeros((WIDTH, WIDTH))
    rng = np.random.default_rng(seed)
    for li in range(target):
        ev, V = np.linalg.eigh(sigmas[li])
        root = V * np.sqrt(np.maximum(ev, 0.0))
        s1 = np.zeros(WIDTH); s2 = np.zeros((WIDTH, WIDTH)); s21 = np.zeros((WIDTH, WIDTH))
        done = 0
        while done < n_samples:
            b = min(CHUNK, n_samples - done)
            z = mus[li] + rng.standard_normal((b, WIDTH)) @ root.T
            v = np.maximum(z, 0.0) @ q[li]
            s1 += v.sum(0); s2 += v.T @ v; s21 += (v * v).T @ v
            done += b
        total += _c21((s1, s2, s21), n_samples)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--targets", type=int, nargs="+", default=[7, 15, 23])
    ap.add_argument("--n", type=int, default=400_000)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "gate4.json")
    args = ap.parse_args()

    rows = _load_rows(DEFAULT_DATA, args.indices)
    print("GATE 4: full rollout of c21 from layer 1, no network sampling")
    print("  predicted = sum_l c21(ReLU(z_l) @ Q_l), z_l ~ N(mu_l, Sigma_l)\n")
    print(f"{'mlp':>4}{'target':>8}{'|pred|/|true|':>15}{'corr':>9}{'rel err':>10}")
    out = []
    for ri, (name, W, _) in zip(args.indices, rows):
        wl = [w.astype(np.float32) for w in W]
        for target in args.targets:
            mus, sigmas, true = layer_moments(wl, target, args.n, seed=41 + ri)
            pred = predicted_c21(wl, mus, sigmas, target, args.n, seed=900 + ri)
            sc = np.sqrt(np.mean(true ** 2))
            mag = float(np.sqrt(np.mean(pred ** 2)) / sc)
            corr = float(np.corrcoef(pred.ravel(), true.ravel())[0, 1])
            rel = float(np.sqrt(np.mean((pred - true) ** 2)) / sc)
            print(f"{ri:>4}{target+1:>8}{mag:15.3f}{corr:9.4f}{rel:10.3f}", flush=True)
            out.append({"mlp": ri, "target": target + 1, "magnitude": mag,
                        "corr": corr, "rel_err": rel})

    print(f"\nmean corr {np.mean([o['corr'] for o in out]):.4f}   "
          f"mean rel err {np.mean([o['rel_err'] for o in out]):.3f}")
    print("  c21 needs ~6% relative accuracy to keep next-variance under 0.5%")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
