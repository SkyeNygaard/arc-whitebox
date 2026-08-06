"""How rough is the integrand, and how does the difficulty scale with depth?

Two questions the harmonic spectrum answers for free, both of which decide
whole families of methods without implementing any of them.

--------------------------------------------------------------------------
1. Is gradient information usable?
--------------------------------------------------------------------------
The Bayesian-quadrature bound closes off linear estimators built from function
*values*.  The obvious escape is to also use derivatives: a directional
derivative through the network costs the same as a forward pass (propagate a
perturbation, delta_l = D_l W_l delta_{l-1}, which is the same 2n^2 per layer),
so a value+derivative pair costs 2 forward-equivalents and yields 2x256 numbers
-- the same information *count* as two extra points, but a different information
*type*.  Derivative-enhanced cubature (Turan-type rules, gradient-enhanced
kriging) is a real and well-studied technique.

Whether it can possibly help here is decided by one number.  On S^{d-1} the
Laplace-Beltrami eigenvalue of degree l is l(l+d-2), so

    E |grad f|^2 / (d-1)  =  sum_l A_l * l(l+d-2)/(d-1)

is the variance of a directional derivative.  If that sum is much larger than
sum_l A_l -- or worse, divergent -- then derivative observations are dominated by
exactly the high-degree content no design can capture, and each one is a noisier
observation than a plain function value.  A divergent sum means f is not even
mean-square differentiable, and gradient-enhanced quadrature is ill-posed rather
than merely unhelpful.

--------------------------------------------------------------------------
2. How does the difficulty scale with depth?
--------------------------------------------------------------------------
The share of variance a 5-design can remove is a function of depth, computable
in closed form for any depth.  That says how the cubature advantage decays as
networks get deeper -- directly relevant if a later phase goes deeper than 32 --
and it quantifies the sense in which "deeper = harder" for every sampling method
at once.

It also settles a tempting idea: since h_L (the pre-activation) is "one ReLU less
kinked" than a_L = ReLU(h_L), maybe estimating the *moments* of h_L and
reconstructing E[ReLU] via an Edgeworth expansion is easier than estimating
E[ReLU] directly.  The spectra of C_{L-1} and C_L answer that immediately.
"""

from __future__ import annotations

import numpy as np

from spectrum import spectrum_finite_d

D = 256


def derivative_variance_ratio(shares: np.ndarray, d: int = D):
    """Var(directional derivative) / Var(f), and the partial sums.

    Returns (ratio, cumulative) so divergence is visible rather than hidden by
    the truncation of the spectrum.
    """
    l = np.arange(1, len(shares) + 1)
    eig = l * (l + d - 2) / (d - 1)
    terms = shares * eig
    return float(terms.sum() / shares.sum()), np.cumsum(terms) / shares.sum()


def design_removable(shares: np.ndarray, t: int = 5) -> float:
    """Share of variance an antipodal t-design removes (see spectrum.gains)."""
    dg = np.arange(1, len(shares) + 1)
    live = (dg > t) & (dg % 2 == 0)
    return 1.0 - 2.0 * shares[live].sum() / shares.sum()


def tail_exponent(shares: np.ndarray, lo: int = 20, hi: int = 200) -> float:
    """Fit shares ~ l^-alpha over a mid-range window."""
    l = np.arange(1, len(shares) + 1)
    m = (l >= lo) & (l <= hi) & (shares > 0)
    return float(-np.polyfit(np.log(l[m]), np.log(shares[m]), 1)[0])


if __name__ == "__main__":
    print("=" * 74)
    print("1.  IS GRADIENT INFORMATION USABLE?")
    print("=" * 74)
    shares, a0, total, taylor = spectrum_finite_d(32, D)
    ratio, cum = derivative_variance_ratio(shares, D)
    print(f"\n  Var(directional derivative) / Var(f) = {ratio:.3f}")
    print("\n  partial sums (does it converge?):")
    for k in (5, 10, 20, 50, 100, 200, 400, 600):
        if k <= len(cum):
            print(f"    degrees 1..{k:<4d} {cum[k-1]:10.3f}")
    growth = cum[-1] / cum[len(cum) // 2 - 1]
    print(f"\n  ratio of full sum to half-range sum: {growth:.3f}")
    print(f"  spectrum tail exponent alpha ~ {tail_exponent(shares):.2f}"
          "   (shares ~ l^-alpha)")
    print("  CAVEAT: the spectrum is a truncated series (it captures ~99.98% of")
    print("  the mass at depth 32, less when deeper), so this partial sum cannot")
    print("  resolve whether the tail strictly converges.  The ratio below is a")
    print("  lower bound on the true derivative variance, which is enough.")
    print("""
  Reading: a directional derivative is an observation whose variance is this
  multiple of a function value's.  Derivative-enhanced cubature buys higher
  polynomial exactness per point, but each observation carries this much more
  high-degree content -- which is precisely the part no feasible design removes.""")

    print()
    print("=" * 74)
    print("2.  HOW THE DIFFICULTY SCALES WITH DEPTH")
    print("=" * 74)
    print(f"\n  {'depth':>6} {'Var share':>11} {'deg<=5 share':>13} "
          f"{'5-design gain':>14} {'alpha':>7}")
    prev = None
    rows = {}
    for depth in (1, 2, 4, 8, 16, 32, 64):
        sh, a0d, _, _ = spectrum_finite_d(depth, D)
        rem = design_removable(sh, 5)
        gain = 1.0 / (1.0 - rem)
        rows[depth] = (sh, gain)
        print(f"  {depth:>6} {1-a0d:>11.5f} {sh[:5].sum():>13.4f} "
              f"{gain:>13.3f}x {tail_exponent(sh):>7.2f}")
    print("""
  Reading: the cubature advantage decays monotonically with depth.  Every extra
  layer composes one more copy of the arc-cosine kernel's (1-t)^{3/2} branch
  point at t=1, pushing variance up into degrees no design reaches.  If a later
  phase goes deeper, structured designs get *less* valuable, not more, and the
  gap that white-box methods have to close shrinks with them.""")

    print()
    print("  --- is the pre-activation easier to integrate than the activation? ---")
    s31, _, _, _ = spectrum_finite_d(31, D)
    s32 = shares
    g31, g32 = 1 / (1 - design_removable(s31, 5)), 1 / (1 - design_removable(s32, 5))
    print(f"  h_L  (kernel C_31): 5-design gain {g31:.4f}x")
    print(f"  a_L  (kernel C_32): 5-design gain {g32:.4f}x")
    print(f"  advantage of working one ReLU earlier: {g31/g32:.4f}x")
    print("""  So estimating moments of h_L and reconstructing E[ReLU] via Edgeworth
  gains essentially nothing on the location term -- and higher moments h^2, h^3,
  h^4 are *products*, whose spectra are convolutions and therefore heavier still.
  The route is a wash at best.""")
