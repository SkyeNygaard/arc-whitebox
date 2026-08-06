"""Exact rectified-Gaussian pair moments, and the dense third-order correction.

GAUSSIAN PAIR MOMENT.  For (X,Y) bivariate normal with means mu, sds sigma and
correlation rho, write a = mu_x/sigma_x, b = mu_y/sigma_y, s = sqrt(1-rho^2):

  E[X+ Y+] = sigma_x sigma_y * Psi(a,b,rho)
  Psi = (ab+rho) Phi2(a,b;rho) + a phi(b) Phi((a-rho b)/s)
                               + b phi(a) Phi((b-rho a)/s) + (1-rho^2) phi2(a,b;rho)

Checks: at rho=0 this factors into [a Phi(a)+phi(a)][b Phi(b)+phi(b)], the product
of the marginal ReLU means; at a=b=0 it reduces to the Cho-Saul arccos kernel
(sqrt(1-r^2) + r(pi/2 + arcsin r))/(2 pi).  Both are asserted below.

THIRD-ORDER CORRECTION.  For a non-Gaussian Z with the same mean and covariance,
the leading Edgeworth term is

  E[g(Z)] = E[g(G)] + (1/6) sum_{abc} kappa_abc E[d3 g / dz_a dz_b dz_c (G)] + ...

With g = X+ Y+ the third derivatives are delta and delta' distributions, giving

  Delta E[X+Y+] = (1/6)[ k_xxx A_x + 3 k_xxy B_xy + 3 k_xyy B_yx + k_yyy A_y ]
  B_xy = f_X(0) Phi(m_x/tau_x)
  A_x  = -f_X(0) [ (a_x/sigma_x) h_x(0) + beta_x Phi(m_x/tau_x) ]
  h_x(0) = m_x Phi(m_x/tau_x) + tau_x phi(m_x/tau_x)
  m_x = mu_y - beta_x mu_x,  beta_x = rho sigma_y/sigma_x,  tau_x = sigma_y s

PHI2 uses the tetrachoric form, which stays well behaved as |rho| -> 1 (it must:
the deep layers are strongly rank-collapsed and carry many near-unit
correlations):

  Phi2(a,b;rho) = Phi(a)Phi(b) + (1/2pi) int_0^{arcsin rho}
                    exp(-(a^2 - 2ab sin th + b^2)/(2 cos^2 th)) d th
"""
import numpy as np
from numpy.polynomial.legendre import leggauss

SQRT2PI = np.sqrt(2.0 * np.pi)
_GL_N = 64
_GL_X, _GL_W = leggauss(_GL_N)


from scipy.special import ndtr


def phi(t):
    return np.exp(-0.5 * np.asarray(t, dtype=np.float64) ** 2) / SQRT2PI


def ncdf(t):
    """Standard normal CDF.  The shipped estimator would substitute the
    pure-flopscope norm_cdf recipe; scipy is used here for research speed."""
    return ndtr(np.asarray(t, dtype=np.float64))


def phi2(a, b, rho):
    s2 = 1.0 - rho * rho
    return np.exp(-(a * a - 2.0 * rho * a * b + b * b) / (2.0 * s2)) / (2.0 * np.pi * np.sqrt(s2))


def Phi2(a, b, rho):
    """Bivariate normal CDF, tetrachoric series, Gauss-Legendre in theta."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rho = np.clip(np.asarray(rho, dtype=np.float64), -0.999999, 0.999999)
    th = np.arcsin(rho)
    # map GL nodes from [-1,1] onto [0, th]
    half = th[..., None] * 0.5
    nodes = half * (_GL_X + 1.0)
    sn = np.sin(nodes)
    cs2 = 1.0 - sn * sn
    ex = np.exp(-(a[..., None] ** 2 - 2.0 * a[..., None] * b[..., None] * sn
                  + b[..., None] ** 2) / (2.0 * cs2))
    integral = (ex * _GL_W).sum(-1) * np.squeeze(half, -1) / (2.0 * np.pi)
    return ncdf(a) * ncdf(b) + integral


def relu_pair_gauss(mu, sd, Sigma):
    """E[relu(z_i) relu(z_j)] for all pairs under the matched Gaussian."""
    a = mu / sd
    rho = np.clip(Sigma / np.outer(sd, sd), -0.999999, 0.999999)
    A = a[:, None] + np.zeros_like(rho)
    B = a[None, :] + np.zeros_like(rho)
    s = np.sqrt(1.0 - rho * rho)
    P2 = Phi2(A, B, rho)
    term = ((A * B + rho) * P2
            + A * phi(B) * ncdf((A - rho * B) / s)
            + B * phi(A) * ncdf((B - rho * A) / s)
            + (1.0 - rho * rho) * phi2(A, B, rho))
    return np.outer(sd, sd) * term


def relu_pair_third_order(mu, sd, Sigma, K):
    """Dense third-order correction to E[relu_i relu_j].

    K[i, j] = kappa(z_i, z_i, z_j) = E[c_i^2 c_j]; its diagonal is kappa_iii.
    """
    a = mu / sd
    rho = np.clip(Sigma / np.outer(sd, sd), -0.999999, 0.999999)
    s = np.sqrt(1.0 - rho * rho)
    sx = sd[:, None] + np.zeros_like(rho)
    sy = sd[None, :] + np.zeros_like(rho)
    mx = mu[:, None] + np.zeros_like(rho)
    my = mu[None, :] + np.zeros_like(rho)
    ax = a[:, None] + np.zeros_like(rho)
    ay = a[None, :] + np.zeros_like(rho)

    def half(sx, sy, mx, my, ax):
        """B_xy and A_x for the ordered pair (x -> y)."""
        beta = rho * sy / sx
        tau = sy * s
        m0 = my - beta * mx
        r = m0 / tau
        Pr = ncdf(r)
        fx0 = phi(ax) / sx
        B = fx0 * Pr
        h0 = m0 * Pr + tau * phi(r)
        A = -fx0 * ((ax / sx) * h0 + beta * Pr)
        return B, A

    Bxy, Ax = half(sx, sy, mx, my, ax)
    Byx, Ay = half(sy, sx, my, mx, ay)
    kxxx = np.diag(K)[:, None] + np.zeros_like(rho)
    kyyy = np.diag(K)[None, :] + np.zeros_like(rho)
    kxxy = K                      # kappa(z_i, z_i, z_j)
    kxyy = K.T                    # kappa(z_i, z_j, z_j)
    return (kxxx * Ax + 3.0 * kxxy * Bxy + 3.0 * kxyy * Byx + kyyy * Ay) / 6.0


def relu_mean_gauss(mu, sd):
    t = mu / sd
    return sd * (t * ncdf(t) + phi(t))


# ----------------------------------------------------------------- self-tests
def _selftest():
    rng = np.random.default_rng(0)
    print("closure self-tests")

    # Phi2 against Monte Carlo
    for (a, b, r) in [(0.0, 0.0, 0.5), (0.7, -1.2, -0.8), (2.0, 1.0, 0.95),
                      (-1.0, 0.3, 0.99), (0.2, 0.2, -0.99)]:
        L = np.array([[1.0, 0.0], [r, np.sqrt(1 - r * r)]])
        z = rng.standard_normal((4_000_000, 2)) @ L.T
        mc = np.mean((z[:, 0] <= a) & (z[:, 1] <= b))
        ex = float(Phi2(np.array(a), np.array(b), np.array(r)))
        assert abs(mc - ex) < 4e-4, (a, b, r, mc, ex)
    print("  Phi2 vs Monte Carlo                        ok")

    # Psi at rho = 0 must factor
    sd = np.array([1.3, 0.7]); mu = np.array([0.4, -0.9])
    Sig = np.diag(sd ** 2)
    got = relu_pair_gauss(mu, sd, Sig)[0, 1]
    m = relu_mean_gauss(mu, sd)
    assert abs(got - m[0] * m[1]) < 1e-12, (got, m[0] * m[1])
    print("  Psi factorises at rho=0                    ok")

    # Psi at zero mean must equal Cho-Saul
    for r in [-0.7, 0.0, 0.3, 0.9]:
        sd0 = np.array([1.0, 1.0]); mu0 = np.zeros(2)
        Sig0 = np.array([[1.0, r], [r, 1.0]])
        got = relu_pair_gauss(mu0, sd0, Sig0)[0, 1]
        cho = (np.sqrt(1 - r * r) + r * (np.pi / 2 + np.arcsin(r))) / (2 * np.pi)
        assert abs(got - cho) < 1e-10, (r, got, cho)
    print("  Psi reduces to Cho-Saul at zero mean       ok")

    # Full Gaussian pair moment against Monte Carlo, general mean/corr
    for (mx, my, sx, sy, r) in [(0.4, -0.9, 1.3, 0.7, 0.6), (-0.2, 0.5, 0.9, 1.1, -0.5),
                                (1.5, 0.2, 0.5, 2.0, 0.9)]:
        L = np.array([[sx, 0.0], [r * sy, sy * np.sqrt(1 - r * r)]])
        z = rng.standard_normal((4_000_000, 2)) @ L.T + np.array([mx, my])
        mc = np.mean(np.maximum(z[:, 0], 0) * np.maximum(z[:, 1], 0))
        Sig = np.array([[sx * sx, r * sx * sy], [r * sx * sy, sy * sy]])
        ex = relu_pair_gauss(np.array([mx, my]), np.array([sx, sy]), Sig)[0, 1]
        assert abs(mc - ex) / abs(mc) < 3e-3, (mx, my, sx, sy, r, mc, ex)
    print("  E[relu relu] vs Monte Carlo, general       ok")

    # third-order term: on a mildly skewed law it must reduce the error
    n = 4_000_000
    g = rng.standard_normal((n, 2))
    x = g[:, 0] + 0.35 * (g[:, 1] ** 2 - 1.0) + 0.5
    y = 0.6 * g[:, 0] + 0.8 * g[:, 1] + 0.25 * (g[:, 0] ** 2 - 1.0) - 0.2
    Z = np.stack([x, y], 1)
    mu = Z.mean(0); C = np.cov(Z, rowvar=False, bias=True); sd = np.sqrt(np.diag(C))
    c = Z - mu
    K = np.einsum("ni,nj->ij", c ** 2, c) / n
    truth = np.mean(np.maximum(x, 0) * np.maximum(y, 0))
    gauss = relu_pair_gauss(mu, sd, C)[0, 1]
    corr = gauss + relu_pair_third_order(mu, sd, C, K)[0, 1]
    print(f"  third-order: |gauss-truth|={abs(gauss-truth):.5f} "
          f"-> |corrected-truth|={abs(corr-truth):.5f}")
    assert abs(corr - truth) < abs(gauss - truth), "third-order term did not help"
    print("  third-order correction reduces error       ok")


if __name__ == "__main__":
    _selftest()
