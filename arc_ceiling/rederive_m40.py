"""Re-derive M40's closure chain against a reference that resolves the endpoint.

M40 (synthetic): next-variance relative RMS 2.564% -> 1.009% (third order)
-> 0.288% (third+fourth).  Measured floor at a 262,144-sample reference is
0.292%, and the largest ratio anything can report there is 8.8x; M40 reports
8.9x.  The endpoint is therefore pinned at the ceiling and its true value is
unknown.

This measures the same chain on real networks with a large reference, and
reports the floor alongside, so each number can be read as resolved or not.
Everything is accumulated in a single streaming pass: the third-cumulant slice
needs E[h_i^2 h_j], which is one extra 256x256 contraction per chunk, so no
second pass over the data is required.

The third-order step is the validation anchor: it is comfortably resolvable even
at 262k, so reproducing ~1% there confirms the implementation before the
fourth-order-sized residual is trusted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import whest.gaussmath as gm  # noqa: E402
from eval_sampling_official import DEFAULT_DATA, _load_rows  # noqa: E402

from bivariate_edgeworth import edgeworth3_mean, edgeworth3_second_moment  # noqa: E402

WIDTH = 256
CHUNK = 32768


def oracle_pass(weights, layer, n_samples, seed):
    """mu, Sigma, c21 of h_l and Var(h_{l+1}), in one streaming pass."""
    s1 = np.zeros(WIDTH)
    s2 = np.zeros((WIDTH, WIDTH))
    s21 = np.zeros((WIDTH, WIDTH))
    v1 = np.zeros(WIDTH)
    v2 = np.zeros(WIDTH)
    rng = np.random.default_rng(seed)
    done = 0
    nxt_w = weights[layer + 1].astype(np.float64)
    while done < n_samples:
        b = min(CHUNK, n_samples - done)
        a = rng.standard_normal((b, WIDTH)).astype(np.float32)
        for li, w in enumerate(weights):
            h = a @ w
            a = np.maximum(h, 0.0)
            if li == layer:
                hd = h.astype(np.float64)
                s1 += hd.sum(0)
                s2 += hd.T @ hd
                s21 += (hd * hd).T @ hd
                nxt = a.astype(np.float64) @ nxt_w
                v1 += nxt.sum(0)
                v2 += (nxt * nxt).sum(0)
                break
        done += b
    mu = s1 / n_samples
    sigma = s2 / n_samples - np.outer(mu, mu)
    m2 = s2 / n_samples
    # c21[i,j] = E[(h_i-mu_i)^2 (h_j-mu_j)]
    c21 = (s21 / n_samples
           - mu[None, :] * np.diag(m2)[:, None]
           - 2.0 * mu[:, None] * m2
           + 2.0 * (mu * mu)[:, None] * mu[None, :])
    vm = v1 / n_samples
    return mu, sigma, c21, v2 / n_samples - vm * vm


def relative_rms(pred, ref):
    good = ref > 0
    return float(np.sqrt(np.mean((pred[good] / ref[good] - 1.0) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--layers", type=int, nargs="+", default=[15, 23])
    ap.add_argument("--n-ref", type=int, default=4_000_000)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "rederive_m40.json")
    args = ap.parse_args()

    rows = _load_rows(DEFAULT_DATA, args.indices)
    print(f"Re-deriving M40's chain, reference N = {args.n_ref:,}")
    print(f"floor at this N ~ {np.sqrt(2/args.n_ref):.4%} "
          f"(262k floor measured at 0.2922%)\n")
    print(f"{'mlp':>4}{'layer':>7}{'Gaussian':>11}{'Edgeworth3':>13}"
          f"{'ratio':>9}{'noise floor':>13}{'resolved?':>11}")
    out = []
    for ri, (name, W, _) in zip(args.indices, rows):
        wl = [w.astype(np.float32) for w in W]
        for layer in args.layers:
            mu, sigma, c21, ref = oracle_pass(wl, layer, args.n_ref, seed=17 + ri)
            ref_b = oracle_pass(wl, layer, args.n_ref, seed=8017 + ri)[3]
            good = (ref > 0) & (ref_b > 0)
            floor = float(np.sqrt(np.mean(
                ((ref - ref_b) / (0.5 * (ref + ref_b)))[good] ** 2)) / np.sqrt(2))
            ref_mean = 0.5 * (ref + ref_b)

            wn = W[layer + 1].astype(np.float64)
            gmean, gcov = gm.relu_cov_from_gauss(mu, sigma, n_nodes=12)
            var_g = np.einsum('ij,jk,ki->i', wn.T, gcov, wn)

            gsecond = gcov + np.outer(gmean, gmean)
            e3second = edgeworth3_second_moment(mu, sigma, c21, gsecond)
            e3mean = edgeworth3_mean(mu, sigma, c21)
            e3cov = e3second - np.outer(e3mean, e3mean)
            var_e3 = np.einsum('ij,jk,ki->i', wn.T, e3cov, wn)

            eg, ee = relative_rms(var_g, ref_mean), relative_rms(var_e3, ref_mean)
            res = "yes" if ee > 3 * floor else ("marginal" if ee > floor else "BURIED")
            print(f"{ri:>4}{layer+1:>7}{eg:11.4%}{ee:13.4%}{eg/ee:9.2f}x"
                  f"{floor:13.4%}{res:>11}", flush=True)
            out.append({"mlp": ri, "layer": layer + 1, "gaussian": eg,
                        "edgeworth3": ee, "ratio": eg / ee, "floor": floor})

    g = float(np.mean([r["gaussian"] for r in out]))
    e = float(np.mean([r["edgeworth3"] for r in out]))
    f = float(np.mean([r["floor"] for r in out]))
    print(f"\nmean: Gaussian {g:.4%}  Edgeworth3 {e:.4%}  ratio {g/e:.2f}x"
          f"  floor {f:.4%}")
    print(f"M40 quoted: Gaussian 2.564%, third order 1.009% (ratio 2.54x)")
    print(f"ceiling at a 262k reference: {2.564/0.2922:.1f}x   "
          f"ceiling at this reference: {g*100/f/100:.0f}x")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
