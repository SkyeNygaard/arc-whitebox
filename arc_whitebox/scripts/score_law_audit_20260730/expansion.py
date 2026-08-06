"""WHY does the third-order correction fail at depth?

An Edgeworth expansion around a Gaussian is an asymptotic series in the
standardized cumulants.  It is only useful while those are small: the third-order
term carries a factor ~kappa3/6 and the next ~kappa4/24, so once |kappa3| is O(1)
the series stops being a correction and starts being a guess.

Measured per layer on iid Gaussian input: the standardized skewness and excess
kurtosis of the pre-activations, and the effective rank of their covariance.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, DATA

N = 400_000
CHUNK = 25_000
LAYERS = [0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 31]


def main():
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)
    rng = np.random.default_rng(77)
    k3s, k4s, pr = {l: [] for l in LAYERS}, {l: [] for l in LAYERS}, {l: [] for l in LAYERS}

    for net in range(3):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        acc = {l: [np.zeros(WIDTH) for _ in range(4)] + [np.zeros((WIDTH, WIDTH))]
               for l in LAYERS}
        done = 0
        while done < N:
            m = min(CHUNK, N - done)
            h = rng.standard_normal((m, WIDTH), dtype=np.float32)
            for l, w in enumerate(weights):
                z = h @ w
                if l in acc:
                    Z = z.astype(np.float64)
                    acc[l][0] += Z.sum(0)
                    acc[l][1] += (Z ** 2).sum(0)
                    acc[l][2] += (Z ** 3).sum(0)
                    acc[l][3] += (Z ** 4).sum(0)
                    acc[l][4] += Z.T @ Z
                h = np.maximum(z, 0.0, dtype=np.float32)
            done += m
        for l in LAYERS:
            s1, s2, s3, s4, S = [a / N for a in acc[l][:4]] + [acc[l][4] / N]
            mu = s1
            v = s2 - mu ** 2
            sd = np.sqrt(np.maximum(v, 1e-300))
            live = sd > 1e-6 * sd.max()
            m3 = s3 - 3 * mu * s2 + 2 * mu ** 3
            m4 = s4 - 4 * mu * s3 + 6 * mu ** 2 * s2 - 3 * mu ** 4
            k3s[l].append(np.mean(np.abs(m3[live] / sd[live] ** 3)))
            k4s[l].append(np.mean(m4[live] / sd[live] ** 4 - 3.0))
            C = S - np.outer(mu, mu)
            ev = np.maximum(np.linalg.eigvalsh(C), 0)
            pr[l].append(ev.sum() ** 2 / (ev ** 2).sum())

    print("standardized cumulants of the pre-activations (3 networks, "
          f"N={N:,})\n")
    print(f"{'layer':>6} {'mean |skew|':>12} {'excess kurt':>12} "
          f"{'part.ratio':>11} {'Edgeworth term ~|k3|/6':>23}")
    for l in LAYERS:
        a, b, c = np.mean(k3s[l]), np.mean(k4s[l]), np.mean(pr[l])
        print(f"{l:>6} {a:>12.3f} {b:>12.3f} {c:>11.1f} {a/6:>23.3f}")
    print("\nAn Edgeworth correction is only meaningful while |kappa3|/6 << 1.")


if __name__ == "__main__":
    main()
