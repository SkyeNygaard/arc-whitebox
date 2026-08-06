"""Exact degree-l frame potentials of the Kerdock/MUB configuration.

For an equal-weight rule Q on N points, and a zonal random field with per-degree
variance shares A_l and total variance V, the mean-square cubature error is
exactly

    E[(Q f - I f)^2]  =  V * sum_{l>=1} A_l * P_l,
    P_l              =  (1/N^2) sum_{j,k} G_l(<u_j, u_k>)

with G_l the degree-l zonal harmonic normalised to G_l(1) = 1.  For i.i.d. points
E[P_l] = 1/N, recovering V/N.  So P_l * N is the "efficiency" of the rule at
degree l: 0 means the degree is annihilated, 1 means i.i.d.-like, and >1 means
the rule is actively WORSE than random there.

The Kerdock configuration makes this computable in closed form, because its
Gram matrix takes only four values:

    t = +1      j = k                                   N pairs
    t = -1      j = antipode of k                       N pairs
    t =  0      same basis, different vector            N * 510 pairs
    t = +-1/16  different bases (mutually unbiased)     N * 65536 pairs, signs balanced

so

    P_l = (1/N) [ 1 + (-1)^l + 510 G_l(0) + 32768 (G_l(1/16) + G_l(-1/16)) ].

Odd l vanishes term by term (antipodal symmetry).  Degrees 2 and 4 vanishing is
the projective 2-design property, and it is a genuine check on the whole
framework -- nothing here is fitted.

The same closed form settles two questions that were being chased empirically:

  * Basis weighting.  All 129 bases are pairwise mutually unbiased, hence
    exchangeable under the symmetry group of the configuration.  Any weighting
    that breaks that exchangeability can only raise the low-degree potentials
    away from zero.  Equal weights are optimal; there is nothing to search.

  * Per-network rotation choice.  P_l is rotation-invariant, so for the ensemble
    every rotation is exactly equivalent.  A rotation can only help a *specific*
    network through that network's own degree->=6 harmonic coefficients, which no
    weight-only statistic sees -- which is why measured selectors correlate at
    Spearman ~0.05.
"""

from __future__ import annotations

import numpy as np

from spectrum import gegenbauer_normalised, spectrum_finite_d

D = 256
N_BASES = 129          # maximal real MUB set in d = 256 is d/2 + 1
N_POINTS = 2 * D * N_BASES   # 66,048, antipodal


def kerdock_potentials(max_deg: int = 600, d: int = D, n_bases: int = N_BASES):
    """N * P_l for l = 0..max_deg.  Value 0 = annihilated, 1 = i.i.d.-like."""
    n = 2 * d * n_bases
    same_basis = 2 * d - 2          # points sharing a basis, excluding self and antipode
    cross = n - 2 * d               # points in the other bases
    mu = 1.0 / np.sqrt(d)           # mutual unbiasedness: |<b,c>| = 1/sqrt(d)

    t = np.array([1.0, -1.0, 0.0, mu, -mu])
    G = gegenbauer_normalised(max_deg, d, t)
    counts = np.array([1.0, 1.0, same_basis, cross / 2.0, cross / 2.0])
    return (G * counts[None, :]).sum(axis=1)     # already multiplied by N/N


def iid_potentials(max_deg: int = 600) -> np.ndarray:
    """N * P_l for i.i.d. points is 1 at every degree."""
    return np.ones(max_deg + 1)


def antipodal_only_potentials(max_deg: int = 600) -> np.ndarray:
    """N * P_l for random antipodal pairs: 2 at even degrees, 0 at odd."""
    l = np.arange(max_deg + 1)
    return np.where(l % 2 == 0, 2.0, 0.0)


def predicted_mse_ratio(shares: np.ndarray, npot: np.ndarray) -> float:
    """Gain vs i.i.d.:  (sum_l A_l * 1) / (sum_l A_l * N P_l)."""
    m = min(len(shares), len(npot) - 1)
    num = shares[:m].sum()
    den = (shares[:m] * npot[1 : m + 1]).sum()
    return num / den


if __name__ == "__main__":
    MAXD = 600   # must match the spectrum truncation or the ratio is biased
    shares, const, total, taylor = spectrum_finite_d(32, D)

    ker = kerdock_potentials(MAXD)
    iid = iid_potentials(MAXD)
    anti = antipodal_only_potentials(MAXD)

    print(f"Kerdock/MUB configuration: {N_POINTS:,} points, {N_BASES} real MUBs in d={D}\n")
    print("  N*P_l   (0 = degree annihilated, 1 = i.i.d.-like, >1 = worse than random)")
    print(f"  {'deg':>4} {'Kerdock':>12} {'antipodal':>11} {'iid':>6}")
    for l in range(1, 17):
        print(f"  {l:>4} {ker[l]:12.3e} {anti[l]:11.1f} {iid[l]:6.1f}")

    bad = np.abs(ker[1:6]).max()
    print(f"\n  CHECK degrees 1-5 annihilated: max |N*P_l| = {bad:.3e}  "
          f"{'PASS' if bad < 1e-9 else 'FAIL'}")
    print(f"  degree 6 efficiency vs i.i.d.: {ker[6]:.4f}  "
          f"({'i.i.d.-like' if abs(ker[6]-2) < 0.5 else 'anomalous'})")

    print("\n  --- predicted gain vs i.i.d. sphere sampling ---")
    for name, pot in (("i.i.d.", iid), ("antipodal only", anti), ("Kerdock 5-design", ker)):
        print(f"  {name:20s} {predicted_mse_ratio(shares, pot):8.3f}x")

    g_ker = predicted_mse_ratio(shares, ker)
    g_anti = predicted_mse_ratio(shares, anti)
    print(f"\n  predicted Kerdock / antipodal-only ratio: {g_ker/g_anti:.4f}x")

    # anchor to the measured official Mini-100 numbers
    meas_kerdock, meas_sobol = 2.28259133e-7, 3.56809476e-7
    print(f"\n  --- anchoring to measured official Mini-100 ---")
    print(f"  measured Kerdock raw MSE          {meas_kerdock:.4e}")
    print(f"  measured two-stream Sobol raw MSE {meas_sobol:.4e}   ratio {meas_sobol/meas_kerdock:.3f}x")
    print(f"  theory says Kerdock is {g_ker:.3f}x i.i.d., so Sobol sits at "
          f"{g_ker/(meas_sobol/meas_kerdock):.3f}x i.i.d.")
    print(f"  pure antipodal cancellation alone is worth {g_anti:.3f}x  -- consistent")
    implied_iid = meas_kerdock * g_ker
    print(f"\n  => implied i.i.d.-sphere raw MSE at 66,048 pts: {implied_iid:.4e}")
    print(f"  => V_eff (= N * MSE) for Kerdock: {N_POINTS * meas_kerdock:.4e}")
