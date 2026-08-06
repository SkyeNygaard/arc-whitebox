"""Numerically check static-cubature diagnostics against exposed-Mini scores.

The calculations are parameter-free: nothing is fitted to any measurement. They
predict the ratio between two cubature rules from (a) the composed dual-activation
kernel of a depth-32 ReLU net and (b) the rules' exact frame potentials.

This script is a numerical diagnostic, not a replay of the computer-assisted
all-degree theorem described in the public review materials.

The only clean like-for-like comparison available is the official Mini-100 run,
where both rules were scored on all 100 MLPs.  Per-rule comparisons taken from
different MLP subsets are not usable: the per-MLP MSE distribution is
chi-squared-like with one effective degree of freedom (the final-layer
fluctuation is rank-1 dominated), so subset means swing by ~1.5x -- visible here
as the gap between the i.i.d. mean (1.284e-6) and median (8.43e-7) over 10 runs.

Normalise by V_eff = N * MSE, which removes the point-count difference.
"""

from __future__ import annotations

import json
import os

from design_potentials import (D, N_POINTS, antipodal_only_potentials,
                               iid_potentials, kerdock_potentials,
                               predicted_mse_ratio)
from spectrum import spectrum_finite_d

RES = os.path.join(os.path.dirname(__file__), "..", "arc_whitebox", "results")

# Official Mini-100, all 100 MLPs, from results/kerdock_mub5_official_full100.json
# and results/two_nearfull_rqmc.json.
MEASURED = {
    "two-stream Sobol (antipodal)": dict(mse=3.56809476e-7, n=62768),
    "Kerdock/MUB 5-design":         dict(mse=2.28259133e-7, n=66048),
}

BUDGET = 2.72e11


def main():
    shares, const, total, taylor = spectrum_finite_d(32, D)
    g_iid = predicted_mse_ratio(shares, iid_potentials())
    g_anti = predicted_mse_ratio(shares, antipodal_only_potentials())
    g_ker = predicted_mse_ratio(shares, kerdock_potentials())

    print("Ceiling theory -- parameter-free predictions")
    print(f"  Taylor mass captured           {taylor:.6f}")
    print(f"  Gegenbauer coefficient sum     {total:.6f}   (must be 1)")
    print(f"  variance share above degree 12 {1 - shares[:12].sum():.4f}\n")

    print("  predicted gain vs i.i.d. sphere")
    print(f"    i.i.d.                       {g_iid:6.3f}x")
    print(f"    antipodal only               {g_anti:6.3f}x")
    print(f"    Kerdock 5-design             {g_ker:6.3f}x\n")

    print("  --- VALIDATION: like-for-like on official Mini-100 ---")
    rows = {}
    for name, m in MEASURED.items():
        v = m["n"] * m["mse"]
        rows[name] = v
        print(f"    {name:30s} MSE {m['mse']:.4e}  N {m['n']:,}  V_eff {v:.4e}")

    meas = rows["two-stream Sobol (antipodal)"] / rows["Kerdock/MUB 5-design"]
    pred = g_ker / g_anti
    err = abs(meas - pred) / pred
    print(f"\n    measured  Kerdock / Sobol-antipodal  = {meas:.4f}x")
    print(f"    predicted Kerdock / Sobol-antipodal  = {pred:.4f}x")
    print(f"    discrepancy {err:.2%}   {'PASS' if err < 0.10 else 'FAIL'}"
          "   (nothing fitted)\n")

    # what remains
    deg = range(1, len(shares) + 1)
    import numpy as np
    dg = np.arange(1, len(shares) + 1)

    def gain_t(t):
        live = (dg > t) & (dg % 2 == 0)
        return shares.sum() / (2.0 * shares[live].sum())

    print("  --- REMAINING HEADROOM ON THE DESIGN AXIS ---")
    base = gain_t(5)
    for t, pts in ((6, 5_658_112), (8, 366_362_752)):
        print(f"    exact through degree {t}: {gain_t(t)/base:.4f}x more "
              f"({100*(gain_t(t)/base-1):.1f}%), needs {pts:,} points "
              f"= {pts/N_POINTS:.0f}x the current rule")

    ker_mse = MEASURED["Kerdock/MUB 5-design"]["mse"]
    print(f"\n  Kerdock adjusted score today          2.2566e-07")
    print(f"  if a 7-design were free                {2.2566e-7/(gain_t(6)/base):.4e}"
          "   (it is not: 86x budget)")
    print(f"\n  score = V_eff * (FLOPs per direction) / B")
    print(f"    This diagnostic does not establish a global optimum.")
    print(f"    It isolates the arithmetic trade-off for these fixed rules.")
    for save in (0.20, 0.30, 0.40):
        print(f"       {int(save*100)}% arithmetic saving -> score "
              f"{2.2566e-7*(1-save):.4e}")


if __name__ == "__main__":
    main()
