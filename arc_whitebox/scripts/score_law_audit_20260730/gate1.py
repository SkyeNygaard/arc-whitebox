"""GATE 1 -- dense oracle bivariate Edgeworth correction.

Pass condition: the next-layer sigma error must fall from ~1.34e-2 to <= 3e-3
using EXACT empirical moments and EXACT dense mixed third cumulants.  If the
dense oracle correction cannot do this, no amount of low-rank implementation
work matters and the branch closes.

Noise discipline (the earlier harness could not resolve a 3e-3 pass):
  * reference is iid GAUSSIAN input, not the fixed-radius design rows, so the
    measured law is the one the estimator actually needs;
  * layer 0 is an EXACT-ZERO check -- p_0 = x W_0 is exactly Gaussian and its
    exact covariance W_0^T W_0 is known in closed form, so any error reported
    there is pure reference noise and bounds the whole measurement;
  * two independent reference halves give a direct noise estimate;
  * convergence is reported against reference sample count.

Truth for the next-layer variance is obtained by contracting the TRUE
Cov(relu(z_l)) with W_{l+1}, which is algebraically exact (z_{l+1} =
relu(z_l) W_{l+1}) and avoids a second sampling error.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, DATA
from closures import relu_pair_gauss, relu_pair_third_order, relu_mean_gauss

LAYERS = [0, 1, 2, 4, 8, 16, 24, 29]
CHUNK = 25_000
SD_FLOOR = 1e-6


def accumulate(weights, n_ref, rng, layers):
    """Propagate iid Gaussian input; accumulate per-layer moment statistics."""
    acc = {l: dict(n=0, s1=np.zeros(WIDTH), s2=np.zeros((WIDTH, WIDTH)),
                   s21=np.zeros((WIDTH, WIDTH)), r1=np.zeros(WIDTH),
                   r2=np.zeros((WIDTH, WIDTH))) for l in layers}
    done = 0
    while done < n_ref:
        m = min(CHUNK, n_ref - done)
        x = rng.standard_normal((m, WIDTH), dtype=np.float32)
        h = x
        for l, w in enumerate(weights):
            z = h @ w
            if l in acc:
                Z = z.astype(np.float64)
                R = np.maximum(Z, 0.0)
                a = acc[l]
                a["n"] += m
                a["s1"] += Z.sum(0)
                a["s2"] += Z.T @ Z
                a["s21"] += (Z * Z).T @ Z
                a["r1"] += R.sum(0)
                a["r2"] += R.T @ R
            h = np.maximum(z, 0.0, dtype=np.float32)
        done += m
    return acc


def moments(a):
    n = a["n"]
    mu = a["s1"] / n
    Sig = a["s2"] / n - np.outer(mu, mu)
    Ez2z = a["s21"] / n
    Ez2 = np.diag(a["s2"] / n)
    Ezz = a["s2"] / n
    # kappa(z_i,z_i,z_j) = E[z_i^2 z_j] - mu_j E[z_i^2] - 2 mu_i E[z_i z_j] + 2 mu_i^2 mu_j
    K = (Ez2z - mu[None, :] * Ez2[:, None] - 2.0 * mu[:, None] * Ezz
         + 2.0 * (mu ** 2)[:, None] * mu[None, :])
    Emean = a["r1"] / n
    Ctrue = a["r2"] / n - np.outer(Emean, Emean)
    return mu, Sig, K, Emean, Ctrue


def sd_err(Cpred, Ctrue, wn, live):
    vt = np.einsum("ij,ik,jk->k", Ctrue, wn, wn)
    vp = np.einsum("ij,ik,jk->k", Cpred, wn, wn)
    ok = live & (vt > 0)
    r = np.sqrt(np.maximum(vp[ok], 0.0) / vt[ok]) - 1.0
    return float(np.sqrt(np.mean(r ** 2)))


def run_net(weights, acc, exact_sigma0):
    rows = []
    for l in LAYERS:
        if l + 1 >= DEPTH:
            continue
        mu, Sig, K, Emean, Ctrue = moments(acc[l])
        if l == 0:
            mu = np.zeros(WIDTH)              # exact
            Sig = exact_sigma0                # exact
            K = np.zeros((WIDTH, WIDTH))      # exact: p_0 is Gaussian
        sd = np.sqrt(np.maximum(np.diag(Sig), 0.0))
        live = sd > SD_FLOOR * sd.max()
        sd_safe = np.where(live, sd, 1.0)
        Cg = relu_pair_gauss(mu, sd_safe, Sig)
        Eg = relu_mean_gauss(mu, sd_safe)
        Cg = Cg - np.outer(Eg, Eg)
        D = relu_pair_third_order(mu, sd_safe, Sig, K)
        Cc = Cg + D
        wn = weights[l + 1]
        eg = sd_err(Cg, Ctrue, wn, live)
        ec = sd_err(Cc, Ctrue, wn, live)
        # pair-moment error, on live pairs, relative to the true covariance scale
        sc = np.sqrt(np.mean(Ctrue[np.ix_(live, live)] ** 2))
        pg = np.sqrt(np.mean((Cg - Ctrue)[np.ix_(live, live)] ** 2)) / sc
        pc = np.sqrt(np.mean((Cc - Ctrue)[np.ix_(live, live)] ** 2)) / sc
        ev = np.linalg.eigvalsh(Cc[np.ix_(live, live)])
        psd = float(ev.min() / max(ev.max(), 1e-30))
        rows.append((l, eg, ec, pg, pc, psd, int(live.sum())))
    return rows


def main():
    n_ref = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    n_nets = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)

    allrows = []
    halves = []
    for net in range(n_nets):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        S0 = (weights[0].astype(np.float64).T @ weights[0].astype(np.float64))
        t0 = time.time()
        accA = accumulate(weights, n_ref // 2, np.random.default_rng(1000 + net), LAYERS)
        accB = accumulate(weights, n_ref // 2, np.random.default_rng(9000 + net), LAYERS)
        acc = {l: {k: (accA[l][k] + accB[l][k]) for k in accA[l]} for l in LAYERS}
        allrows.append(run_net(weights, acc, S0))
        halves.append((run_net(weights, accA, S0), run_net(weights, accB, S0)))
        print(f"  net {net} done in {time.time()-t0:.0f}s", flush=True)

    print(f"\nGATE 1 -- dense oracle bivariate correction "
          f"({n_nets} networks, N_ref={n_ref:,})\n")
    print(f"{'layer':>6} {'sigma Gauss':>12} {'sigma +3rd':>11} {'gain':>7} "
          f"{'pair Gauss':>11} {'pair +3rd':>10} {'minEV/maxEV':>12} {'live':>5}")
    nL = len(allrows[0])
    for k in range(nL):
        l = allrows[0][k][0]
        eg = np.mean([r[k][1] for r in allrows])
        ec = np.mean([r[k][2] for r in allrows])
        pg = np.mean([r[k][3] for r in allrows])
        pc = np.mean([r[k][4] for r in allrows])
        ps = np.mean([r[k][5] for r in allrows])
        lv = np.mean([r[k][6] for r in allrows])
        tag = "  <-- exact-zero check" if l == 0 else ""
        print(f"{l:>6} {eg:>12.3e} {ec:>11.3e} {eg/ec:>6.2f}x "
              f"{pg:>11.3e} {pc:>10.3e} {ps:>12.2e} {lv:>5.0f}{tag}")

    # reference-noise estimate from the independent halves
    dif = []
    for (ha, hb) in halves:
        for k in range(len(ha)):
            dif.append(abs(ha[k][2] - hb[k][2]))
    print(f"\n  half-split spread of the corrected sigma error : {np.mean(dif):.3e}")
    print(f"  layer-0 exact-zero residual (noise floor)      : "
          f"{np.mean([r[0][2] for r in allrows]):.3e}")
    deep = [k for k in range(nL) if allrows[0][k][0] >= 4]
    print(f"  mean corrected sigma error, layers >= 4        : "
          f"{np.mean([[r[k][2] for k in deep] for r in allrows]):.3e}")
    print(f"  mean Gaussian sigma error, layers >= 4         : "
          f"{np.mean([[r[k][1] for k in deep] for r in allrows]):.3e}")
    print("\n  PASS if corrected sigma error <= 3e-3 well above the noise floor")


if __name__ == "__main__":
    main()
