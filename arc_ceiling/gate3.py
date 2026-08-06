"""GATE 3 -- does the expected-gate closure reproduce c21 across one layer?

The propagation needed is

    c21(h_{l+1})[i,j] = sum_pqr W_pi W_qi W_rj kappa3(a_l)_pqr

so everything hinges on a closure for kappa3(a_l) given the law of h_l.  The
natural one splits it into an INHERITED and a GENERATED part:

    a_l ~= E[a_l] + D (h_l - mu_l),   D = diag(Phi(mu/sigma))     [expected gate]
    kappa3(a_l) ~= (D (x) D (x) D) kappa3(h_l)   +   G(mu_l, Sigma_l)

Gate 2 already measured the second term (resample z ~ N(mu_l, Sigma_l), push
through ReLU and W): it accounts for only ~6.5% of c21 at layer 24, growing to
~28% at layer 8.  This measures the first term and, crucially, whether the two
together reconstruct the truth.

The trick that makes this tractable at width 256 is that the linearisation is
LINEAR in h, so its transported third cumulant is just c21 of (h - mu) @ (D W)
-- computable from samples of h_l directly, with no 256^3 tensor anywhere.
That is exactly the inherited term, not an approximation of it.

Verdict logic:
  inherited + generated ~= truth  -> the closure is sound; the remaining work is
                                     engineering the state, and Gate 1 supplies
                                     the exact layer-1 start.
  inherited + generated != truth  -> the expected-gate closure is too crude and
                                     the route needs a better one before any
                                     rollout is worth building.
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
sys.path.insert(0, str(ROOT / "src"))

import whest.gaussmath as gm  # noqa: E402
from eval_sampling_official import DEFAULT_DATA, _load_rows  # noqa: E402

WIDTH = 256
CHUNK = 32768


def _c21_from_raw(s1, s2, s21, n):
    """c21[i,j] = E[(h_i-mu_i)^2 (h_j-mu_j)] from streamed raw moments."""
    mu = s1 / n
    m2 = s2 / n
    return (s21 / n
            - mu[None, :] * np.diag(m2)[:, None]
            - 2.0 * mu[:, None] * m2
            + 2.0 * (mu * mu)[:, None] * mu[None, :])


def measure(weights, layer, n_samples, seed):
    """c21 of the true h_{l+1} and of the expected-gate linearisation."""
    nxt = weights[layer + 1].astype(np.float64)

    # pass 1: mu, Sigma of h_l  -> gives the gate D and the linear map
    s1 = np.zeros(WIDTH); s2 = np.zeros((WIDTH, WIDTH))
    rng = np.random.default_rng(seed); done = 0
    while done < n_samples:
        b = min(CHUNK, n_samples - done)
        a = rng.standard_normal((b, WIDTH)).astype(np.float32)
        for li, w in enumerate(weights):
            h = a @ w; a = np.maximum(h, 0.0)
            if li == layer:
                hd = h.astype(np.float64); s1 += hd.sum(0); s2 += hd.T @ hd
                break
        done += b
    mu = s1 / n_samples
    sigma = s2 / n_samples - np.outer(mu, mu)
    sd = np.sqrt(np.maximum(np.diag(sigma), 1e-300))
    gate = norm.cdf(mu / sd)                      # E[ReLU'(h)] per unit
    lin_map = (gate[:, None] * nxt)               # D W, applied to (h - mu)

    # pass 2: raw moments of true h_{l+1} and of the linearised transport
    acc = {k: [np.zeros(WIDTH), np.zeros((WIDTH, WIDTH)), np.zeros((WIDTH, WIDTH))]
           for k in ("true", "lin")}
    rng = np.random.default_rng(seed); done = 0
    while done < n_samples:
        b = min(CHUNK, n_samples - done)
        a = rng.standard_normal((b, WIDTH)).astype(np.float32)
        for li, w in enumerate(weights):
            h = a @ w; a = np.maximum(h, 0.0)
            if li == layer:
                hd = h.astype(np.float64)
                streams = {"true": a.astype(np.float64) @ nxt,
                           "lin": (hd - mu) @ lin_map}
                for k, v in streams.items():
                    acc[k][0] += v.sum(0)
                    acc[k][1] += v.T @ v
                    acc[k][2] += (v * v).T @ v
                break
        done += b
    out = {k: _c21_from_raw(*acc[k], n_samples) for k in acc}
    return out["true"], out["lin"], mu, sigma


def generated_c21(mu, sigma, next_weight, n_samples, seed):
    """Gate 2's term: kappa3(h_l) = 0 by construction, ReLU manufactures the rest."""
    ev, V = np.linalg.eigh(sigma)
    root = V * np.sqrt(np.maximum(ev, 0.0))
    w = next_weight.astype(np.float64)
    s1 = np.zeros(WIDTH); s2 = np.zeros((WIDTH, WIDTH)); s21 = np.zeros((WIDTH, WIDTH))
    rng = np.random.default_rng(seed); done = 0
    while done < n_samples:
        b = min(CHUNK, n_samples - done)
        z = mu + rng.standard_normal((b, WIDTH)) @ root.T
        v = np.maximum(z, 0.0) @ w
        s1 += v.sum(0); s2 += v.T @ v; s21 += (v * v).T @ v
        done += b
    return _c21_from_raw(s1, s2, s21, n_samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--layers", type=int, nargs="+", default=[7, 15, 23])
    ap.add_argument("--n", type=int, default=1_500_000)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "gate3.json")
    args = ap.parse_args()

    rows = _load_rows(DEFAULT_DATA, args.indices)
    print("GATE 3: expected-gate closure for kappa3(a_l), measured on c21(h_{l+1})")
    print("  inherited = c21 of (h - mu) @ (D W)   [linear, exact transport of kappa3(h)]")
    print("  generated = Gate 2 term (Gaussian h_l, kappa3 = 0)\n")
    print(f"{'mlp':>4}{'layer':>7}{'inherited':>11}{'generated':>11}"
          f"{'in+gen':>9}{'corr':>8}{'residual':>10}")
    out = []
    for ri, (name, W, _) in zip(args.indices, rows):
        wl = [w.astype(np.float32) for w in W]
        for layer in args.layers:
            true, lin, mu, sigma = measure(wl, layer, args.n, seed=31 + ri)
            gen = generated_c21(mu, sigma, W[layer + 1], args.n, seed=707 + ri)
            sc = np.sqrt(np.mean(true ** 2))
            frac = lambda M: float(np.sqrt(np.mean(M ** 2)) / sc)
            both = lin + gen
            resid = float(np.sqrt(np.mean((both - true) ** 2)) / sc)
            corr = float(np.corrcoef(both.ravel(), true.ravel())[0, 1])
            print(f"{ri:>4}{layer+1:>7}{frac(lin):11.3f}{frac(gen):11.3f}"
                  f"{frac(both):9.3f}{corr:8.4f}{resid:10.3f}", flush=True)
            out.append({"mlp": ri, "layer": layer + 1, "inherited": frac(lin),
                        "generated": frac(gen), "sum": frac(both),
                        "corr": corr, "residual": resid})

    r = float(np.mean([x["residual"] for x in out]))
    c = float(np.mean([x["corr"] for x in out]))
    print(f"\nmean residual {r:.3f}   mean correlation {c:.4f}")
    print("  residual << 1 and corr -> 1  : closure sound, build the rollout")
    print("  residual ~ 1                 : expected-gate closure too crude")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
