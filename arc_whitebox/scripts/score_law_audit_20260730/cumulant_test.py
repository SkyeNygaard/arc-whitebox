"""Does the Kerdock design estimate CUMULANTS more efficiently than iid sampling?

notes/03 closed the white-box (Edgeworth moment-propagation) route because
kappa_3 needs ~3% relative accuracy, costed at ~420,000 *iid* samples = 6.5x the
whole budget.  That costing assumed iid sampling.  The Kerdock design delivers
~45x variance reduction for the MEAN.  If a comparable reduction holds for
kappa_3, the route reopens inside budget.

Networks are positively homogeneous, so for the fixed-radius design
    E_gauss[h^k] = (E[chi^k] / R^k) * E_design[h^k],   R = E[chi]
which is exact, and lets the design estimate Gaussian cumulants directly.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from harness import WIDTH, DEPTH, ASSET, DATA, first_layer_design, mean_gaussian_radius

R = mean_gaussian_radius(WIDTH)
N_REF = 300_000
N_CMP = 66_048
LAYERS = [1, 4, 8, 16, 24, 31]


def radial_factor(k, n=WIDTH):
    """E[chi_n^k] / R^k -- exact fixed-radius -> Gaussian moment conversion."""
    log_moment = 0.5 * k * math.log(2.0) + math.lgamma((n + k) / 2.0) - math.lgamma(n / 2.0)
    return math.exp(log_moment - k * math.log(R))


def moments(start_rows, weights, start_index, radial):
    """Propagate and return {layer: (mu, sigma, kappa3)} per neuron."""
    act = start_rows
    out = {}
    for li, w in enumerate(weights, start=start_index):
        pre = act @ w
        if li in LAYERS:
            p64 = pre.astype(np.float64)
            m1, m2, m3 = p64.mean(0), (p64 ** 2).mean(0), (p64 ** 3).mean(0)
            if radial:
                m1, m2, m3 = m1 * radial_factor(1), m2 * radial_factor(2), m3 * radial_factor(3)
            out[li] = (m1, np.sqrt(np.maximum(m2 - m1 ** 2, 0.0)),
                       m3 - 3 * m1 * m2 + 2 * m1 ** 3)
        act = np.maximum(pre, 0.0, dtype=np.float32)
    return out


def main():
    asset = np.load(ASSET)
    chirps = asset["chirps"].astype(np.float32)
    rotation = asset["rotation"].astype(np.float32)
    table = pq.read_table(sorted(DATA.glob("mini-*.parquet"))[0])
    W_all = np.asarray(table.column("weights").to_pylist(), dtype=np.float32)

    rng = np.random.default_rng(7)
    print(f"reference {N_REF} iid | compared: {N_CMP} design rows vs {N_CMP} iid samples")
    print(f"{'net':>4} {'layer':>6} {'design rel err':>15} {'iid rel err':>13} {'var reduction':>15}")
    agg = {li: [] for li in LAYERS}

    for net in range(3):
        weights = [np.ascontiguousarray(W_all[net][i]) for i in range(DEPTH)]
        ref = moments(rng.standard_normal((N_REF, WIDTH), dtype=np.float32),
                      weights, 0, radial=False)
        des = moments(first_layer_design(weights[0], chirps, rotation),
                      weights[1:], 1, radial=True)
        iid = moments(rng.standard_normal((N_CMP, WIDTH), dtype=np.float32),
                      weights, 0, radial=False)

        for li in LAYERS:
            scale = ref[li][1] ** 3 + 1e-30          # standardised kappa_3
            e_des = float(np.sqrt(np.mean(((des[li][2] - ref[li][2]) / scale) ** 2)))
            e_iid = float(np.sqrt(np.mean(((iid[li][2] - ref[li][2]) / scale) ** 2)))
            vr = (e_iid / e_des) ** 2
            agg[li].append(vr)
            print(f"{net:>4} {li:>6} {e_des:>15.5f} {e_iid:>13.5f} {vr:>14.2f}x")

    print("\nmean kappa_3 variance reduction of design over iid, by layer:")
    for li in LAYERS:
        print(f"  layer {li:>2}: {np.mean(agg[li]):>7.2f}x")
    print("\nreference: the design's variance reduction for the MEAN is ~45x.")
    print("kappa_3 needs ~3% (0.03) standardised accuracy to reopen the route.")


if __name__ == "__main__":
    main()
