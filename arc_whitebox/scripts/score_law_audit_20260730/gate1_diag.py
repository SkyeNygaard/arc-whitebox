"""Is Gate 1's negative real, or an implementation/conditioning artifact?

Three checks on the same dense oracle correction:

 1. OPTIMAL SCALE.  Fit alpha minimising ||Cg + alpha*D - Ctrue||_F.  If
    alpha* ~ 1 and the residual stays large, the correction is correctly scaled
    but simply insufficient.  If alpha* is far from 1, something is mis-scaled.
 2. CONDITIONING.  The deep layers are rank-collapsed, so many pairs carry
    |rho| -> 1 where the closure has 1/sqrt(1-rho^2) factors.  Report the |rho|
    distribution and re-measure with near-degenerate pairs excluded.
 3. ATTAINABLE CEILING.  Compare against the best possible symmetric rank-1
    rescaling, to see how much of the residual any pairwise correction could
    reach.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, DATA
from closures import relu_pair_gauss, relu_pair_third_order, relu_mean_gauss
from gate1 import accumulate, moments, LAYERS, SD_FLOOR

N = 600_000


def main():
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)

    for net in range(2):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        acc = accumulate(weights, N, np.random.default_rng(500 + net), LAYERS)
        print(f"\n=== network {net} ===")
        print(f"{'layer':>6} {'alpha*':>8} {'|rho|>.99':>10} {'|rho|>.9999':>12} "
              f"{'sigma G':>10} {'sigma +3rd':>11} {'sigma alpha*':>13} {'sigma tame':>11}")
        for l in LAYERS:
            if l + 1 >= DEPTH or l == 0:
                continue
            mu, Sig, K, Emean, Ctrue = moments(acc[l])
            sd = np.sqrt(np.maximum(np.diag(Sig), 0.0))
            live = sd > SD_FLOOR * sd.max()
            sd_s = np.where(live, sd, 1.0)
            rho = np.clip(Sig / np.outer(sd_s, sd_s), -0.999999, 0.999999)
            Cg = relu_pair_gauss(mu, sd_s, Sig)
            Eg = relu_mean_gauss(mu, sd_s)
            Cg = Cg - np.outer(Eg, Eg)
            D = relu_pair_third_order(mu, sd_s, Sig, K)
            R = Ctrue - Cg
            m = np.ix_(live, live)
            alpha = float((D[m] * R[m]).sum() / max((D[m] * D[m]).sum(), 1e-300))
            wn = weights[l + 1]

            def se(C):
                vt = np.einsum("ij,ik,jk->k", Ctrue, wn, wn)
                vp = np.einsum("ij,ik,jk->k", C, wn, wn)
                ok = live & (vt > 0)
                return float(np.sqrt(np.mean(
                    (np.sqrt(np.maximum(vp[ok], 0.0) / vt[ok]) - 1.0) ** 2)))

            # "tame": drop the third-order term on near-degenerate pairs
            tame = Cg + np.where(np.abs(rho) < 0.99, D, 0.0)
            f99 = float(np.mean(np.abs(rho[m]) > 0.99))
            f9999 = float(np.mean(np.abs(rho[m]) > 0.9999))
            print(f"{l:>6} {alpha:>8.3f} {f99:>10.3f} {f9999:>12.5f} "
                  f"{se(Cg):>10.3e} {se(Cg+D):>11.3e} {se(Cg+alpha*D):>13.3e} "
                  f"{se(tame):>11.3e}")


if __name__ == "__main__":
    main()
