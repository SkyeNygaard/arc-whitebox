"""Cubic control variate anchored on c21, applied to the Kerdock residual.

Kerdock integrates every polynomial of degree <= 5 exactly, so a control can
only help if its variance sits in degrees >= 6.  A cubic form in a DEEP
preactivation qualifies: h_29 is a degree-29 composition of x, so
(u.dh)^2 (v.dh) has rich high-degree content, while its expectation is exactly
a c21 contraction:

    phi_m(x) = (u_m . (h_l - mu_l))^2 (v_m . (h_l - mu_l))
    E[phi_m] = sum_ij (u_m)_i (u_m)_i (v_m)_j c21[i,j]   -- contracted, not the tensor

The estimator is the standard exactly-anchored control

    mu_hat = Q_K(f) - sum_m beta_m ( Q_K(phi_m) - E[phi_m] )

with beta frozen offline.  Two facts govern whether it can work:

* the anchor error Delta on E[phi] enters as a bias beta^2 Delta^2, so the
  method helps only while Delta < sigma_g, the design's own error on phi;
* directions matter.  The useful ones are those the network is most sensitive
  to downstream, so u, v are taken from the top singular vectors of the
  expected-gate sensitivity S = D_l W_{l+1} D_{l+1} ... W_32.

This module measures the ORACLE headroom first -- E[phi] from a large
independent Monte Carlo -- because if the mechanism does not work with a perfect
anchor there is no point costing a propagated one.  Selection discipline: beta
is fitted on IDs 0-24 and evaluated once on 25-49.
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

from eval_kerdock_design import WIDTH, make_kerdock_design, random_rotation  # noqa: E402
from eval_sampling_official import DEFAULT_DATA, _load_rows  # noqa: E402

CHUNK = 32768


def sensitivity_directions(weights, layer, mus, sigmas, n_dirs):
    """Directions in h_layer the output is most sensitive to, and where they land.

    Returns (dirs, out_dirs): `dirs[:, m]` is a direction in h_layer space,
    STANDARDISED so that u.(h - mu) has unit variance -- without this the cubic
    feature's scale varies ~100x across networks and any fitted coefficient is
    dominated by a few of them.  `out_dirs[:, m]` is the unit output direction
    that direction feeds into, taken from the expected-gate sensitivity map so
    that the output dependence is KNOWN rather than fitted.
    """
    s = np.eye(WIDTH)
    for li in range(layer, len(weights) - 1):
        sd = np.sqrt(np.maximum(np.diag(sigmas[li]), 1e-300))
        gate = norm.cdf(mus[li] / sd)
        s = (gate[:, None] * weights[li + 1].astype(np.float64)) @ s
    # right singular vectors live in h_layer space; left ones in output space
    u_out, sv, vt = np.linalg.svd(s.T)
    dirs = vt[:n_dirs].T.copy()
    out = u_out[:, :n_dirs].copy()
    scale = np.sqrt(np.maximum(np.einsum('im,ij,jm->m', dirs, sigmas[layer], dirs), 1e-300))
    return dirs / scale, out


def design_pass(weights, points, rotation, layer):
    """h_layer at every design row, plus the Kerdock final-layer estimate."""
    a = np.maximum(points @ (rotation @ weights[0].astype(np.float64)), 0.0).astype(np.float32)
    h_l = None
    for li in range(1, len(weights)):
        h = a @ weights[li]
        a = np.maximum(h, 0.0)
        if li == layer:
            h_l = h.astype(np.float64).copy()
    return h_l, a.mean(axis=0, dtype=np.float64)


def oracle_moments(weights, layer, n_samples, seed):
    """mu_l, Sigma_l for all l, and the oracle E[phi] inputs (mu, c21)."""
    depth = len(weights)
    s1 = [np.zeros(WIDTH) for _ in range(depth)]
    s2 = [np.zeros((WIDTH, WIDTH)) for _ in range(depth)]
    t1 = np.zeros(WIDTH); t2 = np.zeros((WIDTH, WIDTH)); t21 = np.zeros((WIDTH, WIDTH))
    rng = np.random.default_rng(seed); done = 0
    while done < n_samples:
        b = min(CHUNK, n_samples - done)
        a = rng.standard_normal((b, WIDTH)).astype(np.float32)
        for li in range(depth):
            h = a @ weights[li]; a = np.maximum(h, 0.0)
            hd = h.astype(np.float64)
            s1[li] += hd.sum(0); s2[li] += hd.T @ hd
            if li == layer:
                t1 += hd.sum(0); t2 += hd.T @ hd; t21 += (hd * hd).T @ hd
        done += b
    mus = [s / n_samples for s in s1]
    sigmas = [s2[i] / n_samples - np.outer(mus[i], mus[i]) for i in range(depth)]
    mu = t1 / n_samples; m2 = t2 / n_samples
    c21 = (t21 / n_samples - mu[None, :] * np.diag(m2)[:, None]
           - 2.0 * mu[:, None] * m2 + 2.0 * (mu * mu)[:, None] * mu[None, :])
    return mus, sigmas, c21


def features(h_l, mu, dirs):
    """phi_m at every design row: (u.dh)^2 (v.dh) with v = u (symmetric form)."""
    dh = h_l - mu
    proj = dh @ dirs                       # (N, n_dirs)
    return proj * proj * proj              # cubic, exactly anchored by c21


def anchors(c21, dirs):
    """E[phi_m] = sum_ij u_i u_i u_j c21[i,j] for v = u."""
    return np.einsum('im,jm,ij->m', dirs * dirs, dirs, c21)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=int, nargs="+", default=list(range(25)))
    ap.add_argument("--test", type=int, nargs="+", default=list(range(25, 50)))
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--n-dirs", type=int, default=4)
    ap.add_argument("--n-oracle", type=int, default=400_000)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "c21_control.json")
    args = ap.parse_args()
    if max(args.fit + args.test) >= 50:
        raise ValueError("selection protocol: official IDs 0--49 only")

    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, 3)

    def collect(indices):
        rows = _load_rows(DEFAULT_DATA, indices)
        out = []
        for idx, (name, W, tg) in zip(indices, rows, strict=True):
            wl = [w.astype(np.float32) for w in W]
            mus, sigmas, c21 = oracle_moments(wl, args.layer, args.n_oracle, seed=77 + idx)
            dirs, out_dirs = sensitivity_directions(
                wl, args.layer, mus, sigmas, args.n_dirs)
            h_l, kerdock = design_pass(wl, points, rotation, args.layer)
            phi = features(h_l, mus[args.layer], dirs)
            gap = phi.mean(axis=0) - anchors(c21, dirs)   # Q_K(phi) - E[phi]
            # regressor for output i from direction m is gap_m * out_dirs[i, m]
            out.append({"err": kerdock - tg[-1], "gap": gap, "out": out_dirs,
                        "X": out_dirs * gap[None, :],
                        "base": float(np.mean((kerdock - tg[-1]) ** 2))})
            print(f"  [{idx:>3}] {name[:18]:<18} base {out[-1]['base']:.4e} "
                  f"|gap| {np.abs(gap).max():.3e}", flush=True)
        return out

    print(f"fitting set (IDs {args.fit[0]}-{args.fit[-1]}), layer {args.layer+1}, "
          f"{args.n_dirs} directions")
    fit = collect(args.fit)
    print(f"\ntest set (IDs {args.test[0]}-{args.test[-1]})")
    test = collect(args.test)

    # ONE scalar per direction: the output dependence is supplied by out_dirs,
    # not fitted.  n_dirs parameters instead of n_dirs x 256.
    X = np.concatenate([r["X"] for r in fit], axis=0)        # (n_net*256, n_dirs)
    y = np.concatenate([r["err"] for r in fit], axis=0)       # (n_net*256,)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]               # (n_dirs,)
    print(f"\nfitted beta ({args.n_dirs} parameters): "
          + " ".join(f"{b:+.4g}" for b in beta))

    def score(rs):
        b = float(np.mean([r["base"] for r in rs]))
        res = [float(np.mean((r["err"] - r["X"] @ beta) ** 2)) for r in rs]
        return b, float(np.mean(res)), int(sum(m < r["base"] for m, r in zip(res, rs)))

    bf, cf, wf = score(fit)
    bt, ct, wt = score(test)
    print(f"\n{'set':<10}{'baseline':>12}{'controlled':>13}{'ratio':>9}{'wins':>9}")
    print(f"{'fit':<10}{bf:12.4e}{cf:13.4e}{cf/bf:9.4f}{wf:8d}/{len(fit)}")
    print(f"{'TEST':<10}{bt:12.4e}{ct:13.4e}{ct/bt:9.4f}{wt:8d}/{len(test)}")
    print(f"\n  need 0.602x to reach rank 7 (9.18e-8) at the current 0.63 multiplier")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"fit_ratio": cf / bf, "test_ratio": ct / bt, "test_wins": wt,
         "n_dirs": args.n_dirs, "layer": args.layer + 1}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
