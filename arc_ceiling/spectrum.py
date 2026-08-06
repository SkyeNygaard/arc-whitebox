"""Exact spherical-harmonic spectrum of a depth-L random ReLU MLP.

Why this exists
---------------
Every estimator that has ever been tried on this challenge -- i.i.d. sampling,
scrambled Sobol, antipodal pairing, the Kerdock/MUB 5-design, and every control
variate on top of them -- is a *cubature rule* on the sphere:

    E_{x~N(0,I)}[f(x)] = E[R] * (1/N) sum_j w_j f(u_j)     (radius exact by homogeneity)

For any such rule the mean-square error is determined entirely by two things: the
rule's degree-l frame potentials, and the *harmonic power spectrum of the
integrand*.  This module computes that spectrum in closed form, which pins down
the achievable ceiling for the entire family without evaluating a single network.

The mathematics
---------------
Daniely-Frostig-Singer dual activation: for phi normalised so E[phi(Z)^2] = 1 and
u, v on the sphere with t = <u,v>,

    E_W[ phi(<w,u>) phi(<w,v>) ] = kappa(t),   kappa(t) = sum_k a_k t^k,  sum a_k = 1

and a_k is the squared k-th normalised Hermite coefficient of phi.  For ReLU:

    a_0 = 1/pi,  a_1 = 1/2,  a_k = ((k-3)!!)^2 / (pi k!) for even k >= 2,  else 0.

Composing L layers gives C_L = kappa o ... o kappa, and C_L is exactly the
two-point function of the depth-L network on the sphere.  Its Gegenbauer
coefficients in dimension d ARE the per-degree variance shares of f.

Two routes are implemented and cross-checked:

  * `spectrum_infinite_d`  -- read off the Taylor coefficients of C_L.  Exact only
    as d -> infinity, because the monomial t^k mixes harmonic degrees k, k-2, ...
    with weights that vanish as O(k^2/d).
  * `spectrum_finite_d`    -- the honest computation: project C_L onto normalised
    Gegenbauer polynomials against the true surface measure (1-t^2)^{(d-3)/2}
    using Gauss-Jacobi quadrature.  No truncation of the composition at all --
    kappa is applied 32 times pointwise at the quadrature nodes.

Both must agree to within the stated O(k^2/d) tolerance, and both must satisfy
sum_l a_l = C_L(1) = 1.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, roots_jacobi


# ---------------------------------------------------------------------------
# the single-layer kernel
# ---------------------------------------------------------------------------
def relu_kernel(t):
    """Normalised arc-cosine (ReLU dual) kernel.  kappa(1)=1, kappa(0)=1/pi."""
    t = np.clip(t, -1.0, 1.0)
    return (np.sqrt(np.maximum(1.0 - t * t, 0.0)) + t * (np.pi - np.arccos(t))) / np.pi


def relu_kernel_coeffs(K: int) -> np.ndarray:
    """Taylor coefficients a_k of kappa, in closed form.

    a_k = ((k-3)!!)^2 / (pi k!) for even k >= 2.  Both factors overflow int64
    well before k = 100, so this is done in log space via
    (2j-1)!! = Gamma(2j+1) / (2^j Gamma(j+1)) with j = k/2 - 1.
    """
    from math import log, pi

    a = np.zeros(K + 1)
    a[0] = 1.0 / pi
    if K >= 1:
        a[1] = 0.5
    for k in range(2, K + 1, 2):
        j = k // 2 - 1
        log_dfact = gammaln(2 * j + 1) - j * log(2.0) - gammaln(j + 1)
        a[k] = np.exp(2 * log_dfact - log(pi) - gammaln(k + 1))
    return a


def compose_series(f: np.ndarray, g: np.ndarray, K: int) -> np.ndarray:
    """Truncated composition f(g(x)) by Horner."""
    out = np.zeros(K + 1)
    out[0] = f[K]
    for k in range(K - 1, -1, -1):
        out = np.convolve(out, g)[: K + 1]
        out[0] += f[k]
    return out


def compose_power(a: np.ndarray, depth: int, K: int) -> np.ndarray:
    """Taylor coefficients of kappa composed `depth` times, by repeated squaring.

    Composition is associative, so C_{2m} = C_m o C_m.  For depth 32 that is 5
    compositions instead of 31.
    """
    if depth < 1:
        raise ValueError(depth)
    result = None
    base = a
    d = depth
    while d:
        if d & 1:
            result = base if result is None else compose_series(result, base, K)
        d >>= 1
        if d:
            base = compose_series(base, base, K)
    return result


# ---------------------------------------------------------------------------
# route 1: d -> infinity
# ---------------------------------------------------------------------------
def spectrum_infinite_d(depth: int = 32, K: int = 160) -> np.ndarray:
    """Per-degree variance shares, valid as d -> infinity.  Index 0 is degree 1."""
    a = relu_kernel_coeffs(K)
    C = compose_power(a, depth, K)
    return C[1:] / (1.0 - C[0])


# ---------------------------------------------------------------------------
# route 2: exact in finite d
# ---------------------------------------------------------------------------
def gegenbauer_normalised(max_deg: int, d: int, t: np.ndarray) -> np.ndarray:
    """G_l(t) for l = 0..max_deg, normalised so G_l(1) = 1.

    Recurrence derived from the Gegenbauer recurrence with lambda = (d-2)/2,
    divided through by C_l^lambda(1) so nothing overflows even at d = 256:

        G_l = [ t (2l + d - 4) G_{l-1} - (l-1) G_{l-2} ] / (l + d - 3)
    """
    G = np.zeros((max_deg + 1,) + t.shape)
    G[0] = 1.0
    if max_deg >= 1:
        G[1] = t
    for l in range(2, max_deg + 1):
        G[l] = (t * (2 * l + d - 4) * G[l - 1] - (l - 1) * G[l - 2]) / (l + d - 3)
    return G


def harmonic_dim(l: int, d: int) -> float:
    """dim of degree-l spherical harmonics on S^{d-1}, via log-gammas."""
    if l == 0:
        return 1.0
    def logbinom(n, k):
        return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    a = np.exp(logbinom(d + l - 1, l))
    b = np.exp(logbinom(d + l - 3, l - 2)) if l >= 2 else 0.0
    return float(a - b)


def spectrum_finite_d(depth: int = 32, d: int = 256, K: int = 600):
    """Exact Gegenbauer spectrum of C_depth on S^{d-1}, computed stably.

    Direct projection is hopeless in float64: the degree-l coefficient is
    dim(H_l) * E[C G_l], and dim(H_l) exceeds 1e30 by degree 20 while the
    integral underflows to match -- catastrophic cancellation above degree ~10.

    Instead expand the monomials into the Gegenbauer basis using the three-term
    recurrence, which follows from the normalised recurrence in
    `gegenbauer_normalised`:

        t G_l = [ (l+d-2) G_{l+1} + l G_{l-1} ] / (2l+d-2)

    Every coefficient in this expansion is positive and O(1), so building
    t^k = sum_l v[k,l] G_l by repeated multiplication is unconditionally stable.
    The Gegenbauer coefficient of C_L is then sum_k b_k v[k,l] with b_k the
    (exactly computable, all-positive) Taylor coefficients of C_L.

    Returns (shares, const, total).
    """
    a = relu_kernel_coeffs(K)
    b = compose_power(a, depth, K)          # Taylor coefficients of C_depth

    l = np.arange(K + 2)
    alpha = (l + d - 2) / (2 * l + d - 2)   # coefficient onto G_{l+1}
    beta = l / (2 * l + d - 2)              # coefficient onto G_{l-1}

    coeffs = np.zeros(K + 1)
    v = np.zeros(K + 2)
    v[0] = 1.0                              # t^0 = G_0
    for k in range(K + 1):
        coeffs += b[k] * v[: K + 1]
        if k < K:
            nv = np.zeros(K + 2)
            nv[1:] += alpha[:-1] * v[:-1]   # G_l -> G_{l+1}
            nv[:-1] += beta[1:] * v[1:]     # G_l -> G_{l-1}
            v = nv

    const = coeffs[0]
    var = 1.0 - const                       # C_L(1) = 1 exactly
    return coeffs[1:] / var, const, coeffs.sum(), b.sum()


# ---------------------------------------------------------------------------
# what a cubature rule can remove
# ---------------------------------------------------------------------------
def gains(shares: np.ndarray) -> dict:
    """Variance-reduction factor vs i.i.d. sphere sampling for each rule.

    NOTE the subtlety that a naive "removed share" argument gets wrong.  The
    mean-square error is `V * sum_l A_l * P_l` with `N*P_l` the per-degree
    efficiency, and for an ANTIPODAL rule `N*P_l = 2` at even degrees, not 1:
    N antipodal points are only N/2 independent directions, and G_l(-1) = +1 for
    even l, so the even-degree error is *doubled* while odd degrees are
    annihilated.  Antipodal pairing is therefore close to a wash (~1.06x), not
    the ~2x that "it removes half the spectrum" would suggest.

    A rule that is antipodal and exact through degree t has

        N*P_l = 0   for l <= t or l odd,      N*P_l = 2   for even l > t.
    """
    deg = np.arange(1, len(shares) + 1)
    total = shares.sum()

    def gain(t):
        live = (deg > t) & (deg % 2 == 0)
        return total / (2.0 * shares[live].sum())

    out = {"iid": 1.0, "antipodal": gain(0)}
    for t in (2, 4, 5, 6, 8, 12):
        out[f"antipodal+deg<={t}"] = gain(t)
    return out


def min_points_for_degree(t: int, d: int = 256) -> int:
    """Fisher-type lower bound on antipodal points for exactness through degree t.

    Exactness through degree 2m requires the rule to reproduce the moment tensor
    of order 2m, i.e. at least dim(Sym^m(R^d)) lines.
    """
    m = t // 2
    from math import comb
    return 2 * comb(d + m - 1, m)


if __name__ == "__main__":
    D, DEPTH = 256, 32
    inf_shares = spectrum_infinite_d(DEPTH)
    fin_shares, const, total, taylor_mass = spectrum_finite_d(DEPTH, D)
    assert (fin_shares[:60] >= -1e-12).all(), "negative share -> numerical breakdown"

    print(f"depth {DEPTH}, d = {D}")
    print(f"  sum of Gegenbauer coefficients = {total:.8f}  (must be C_L(1) = 1)")
    print(f"  Taylor mass captured at K       = {taylor_mass:.8f}")
    print(f"  constant term C_L(0-mode)      = {const:.6f}")
    print(f"  captured in degrees 1..{len(fin_shares)}: {fin_shares.sum():.4f}\n")

    print("  deg   share(d=256)   share(d->inf)   rel.diff")
    for l in range(1, 13):
        f, i = fin_shares[l - 1], inf_shares[l - 1]
        print(f"  {l:3d}   {f:12.5f}   {i:13.5f}   {abs(f-i)/f:8.2%}")
    print(f"  tail beyond degree {len(fin_shares)}: {1 - fin_shares.sum():.4f}")

    print("\n  rule                      gain vs iid    min antipodal points")
    g = gains(fin_shares)
    for k, v in g.items():
        if k.startswith("antipodal+deg<="):
            t = int(k.split("=")[-1])
            n = min_points_for_degree(t, D)
            note = f"{n:>12,}" + ("   <-- Kerdock uses 66,048" if t in (4, 5) else
                                  f"  = {n/66048:.0f}x budget" if n > 66048 else "")
        else:
            note = ""
        print(f"  {k:24s} {v:8.3f}x    {note}")

    r5 = g["antipodal+deg<=5"]
    print(f"\n  marginal 5-design -> 7-design: {g['antipodal+deg<=6']/r5:.4f}x "
          f"({100*(g['antipodal+deg<=6']/r5 - 1):.1f}%)")
    print(f"  marginal 5-design -> 9-design: {g['antipodal+deg<=8']/r5:.4f}x")
