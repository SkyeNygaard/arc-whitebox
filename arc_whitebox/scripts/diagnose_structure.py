"""Empirically test the structural hypotheses from notes/01 (F3, F4, Q1-Q4).

Measures, as a function of depth l:
  * effective rank of Cov(a_l)  (participation ratio)
  * spread relative to mean:  sqrt(tr Sigma) / ||mu||
  * mean pairwise |correlation| between neurons
  * marginal skewness / excess kurtosis of the pre-activations h_l
  * fraction of Var(a_l) captured by the component linear in x
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from whest.nets import make_mlp  # noqa: E402


def run(width, depth, seed, n_samples, chunk=16384, n_probe=4096):
    mlp = make_mlp(width, depth, seed)
    rng = np.random.default_rng(999)

    # streaming accumulators over h_l (pre-activations) and a_l
    L, n = depth, width
    s1 = np.zeros((L, n))
    s2 = np.zeros((L, n))
    s3 = np.zeros((L, n))
    s4 = np.zeros((L, n))
    a1 = np.zeros((L, n))
    a2 = np.zeros((L, n))
    C = np.zeros((L, n, n))  # E[a a^T]
    XA = np.zeros((L, n, n))  # E[x a^T]  (for the linear component)
    m = 0
    while m < n_samples:
        b = min(chunk, n_samples - m)
        X = rng.standard_normal((b, n)).astype(np.float32)
        H = X
        for li, W in enumerate(mlp.Ws):
            H = H @ W.T
            Hd = H.astype(np.float64)
            s1[li] += Hd.sum(0)
            s2[li] += (Hd**2).sum(0)
            s3[li] += (Hd**3).sum(0)
            s4[li] += (Hd**4).sum(0)
            A = np.maximum(Hd, 0.0)
            a1[li] += A.sum(0)
            a2[li] += (A**2).sum(0)
            C[li] += A.T @ A
            XA[li] += X.astype(np.float64).T @ A
            H = np.maximum(H, 0.0)
        m += b

    s1, s2, s3, s4 = s1 / m, s2 / m, s3 / m, s4 / m
    a1, a2, C, XA = a1 / m, a2 / m, C / m, XA / m

    out = []
    for li in range(L):
        mu = a1[li]
        Sig = C[li] - np.outer(mu, mu)
        d = np.diag(Sig).copy()
        tr = d.sum()
        tr2 = float((Sig * Sig).sum())
        eff_rank = tr**2 / tr2
        # correlation
        sd = np.sqrt(np.maximum(d, 1e-30))
        R = Sig / np.outer(sd, sd)
        off = R[~np.eye(n, dtype=bool)]
        # pre-activation cumulants
        mh, m2, m3, m4 = s1[li], s2[li], s3[li], s4[li]
        var = m2 - mh**2
        cen3 = m3 - 3 * mh * m2 + 2 * mh**3
        cen4 = m4 - 4 * mh * m3 + 6 * mh**2 * m2 - 3 * mh**4
        skew = cen3 / var**1.5
        exkurt = cen4 / var**2 - 3.0
        # linear component of a_l:  Cov(x, a) = XA (since E[x]=0)
        # var explained by best linear predictor in x = ||Cov(x,a_i)||^2 (Cov(x,x)=I)
        lin_var = (XA[li] ** 2).sum(0)
        frac_lin = float(np.mean(lin_var / np.maximum(d, 1e-30)))
        out.append(
            dict(
                layer=li + 1,
                eff_rank=float(eff_rank),
                top_eig_frac=float(np.linalg.eigvalsh(Sig)[-1] / tr),
                mu_norm=float(np.linalg.norm(mu)),
                spread_over_mu=float(np.sqrt(tr) / np.linalg.norm(mu)),
                mean_abs_corr=float(np.abs(off).mean()),
                mean_corr=float(off.mean()),
                h_mean_over_sd=float(np.mean(np.abs(mh) / np.sqrt(var))),
                h_skew_rms=float(np.sqrt(np.mean(skew**2))),
                h_exkurt_mean=float(np.mean(exkurt)),
                h_exkurt_rms=float(np.sqrt(np.mean(exkurt**2))),
                E_h2=float(np.mean(m2)),
                var_a=float(np.mean(d)),
                frac_var_linear_in_x=frac_lin,
            )
        )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=400_000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run(a.width, a.depth, a.seed, a.samples)
    hdr = f"{'l':>3} {'effRank':>9} {'top_eig%':>9} {'sqrtTr/|mu|':>12} {'|corr|':>8} {'corr':>8} {'|mu|/sd_h':>10} {'skew':>8} {'exkurt':>9} {'E[h^2]':>8} {'linFrac':>8}"
    print(hdr)
    for r in res:
        print(
            f"{r['layer']:>3} {r['eff_rank']:>9.2f} {100*r['top_eig_frac']:>8.1f}% "
            f"{r['spread_over_mu']:>12.4f} {r['mean_abs_corr']:>8.3f} {r['mean_corr']:>8.3f} "
            f"{r['h_mean_over_sd']:>10.3f} {r['h_skew_rms']:>8.3f} {r['h_exkurt_mean']:>9.4f} "
            f"{r['E_h2']:>8.3f} {r['frac_var_linear_in_x']:>8.3f}"
        )
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
