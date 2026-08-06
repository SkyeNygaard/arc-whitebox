"""Can the third-cumulant slices be recovered from the weights instead of sampled?

The ledger brackets the white-box route precisely:

  M40  bivariate Edgeworth with ORACLE c21/c31/c22 : 2.564% -> 0.288%  (works)
  M42  the same closure with a 6,000-sample pilot  : 1.8-2.4%          (no gain)
  M68  x1/x1a features, finite pilot               : no gain

So the closure is sufficient and the features are the blocker.  Every attempt so
far has estimated them by sampling.  This module tests the structural route.

If the third cumulant of the preactivation has a symmetric CP form

    kappa3(h) ~= sum_r lambda_r u_r (x) u_r (x) u_r

then its (2,1) slice is

    c21[a,b] = cum(h_a, h_a, h_b) = sum_r lambda_r u_ra^2 u_rb,

i.e. the 256x256 MATRIX c21 = sum_r lambda_r (u_r o u_r) u_r^T carries the CP
factors of the 256^3 tensor.  Two consequences:

  * transport through a layer costs r n^2, not n^4 -- the object is cheap to
    propagate exactly once it is in this form;
  * a noisy pilot estimate of c21 can be DENOISED by projecting onto the
    manifold, because i.i.d. sampling noise has no such structure.  M14 already
    measured that rank ~64 captures kappa3 to 3-5%, far tighter than pilot noise.

`compare_estimators` measures whether that projection actually beats the raw
pilot at matched sampling cost.  If it does, M42's blocker is broken; if the
denoised error is no better, the structural route is closed and the whole
white-box line stays gated.

Two denoisers are compared, because they test different claims:
  `svd`        plain rank-r truncation -- tests only that c21 is low rank;
  `structured` alternating fit of sum_r lambda_r (u o u) u^T -- tests the
               stronger CP claim, which is what makes transport exact.
"""

from __future__ import annotations

import numpy as np


def sample_c21(H: np.ndarray) -> np.ndarray:
    """c21[a,b] = cum(h_a, h_a, h_b) = E[(h_a-m_a)^2 (h_b-m_b)] for centred h.

    The third joint cumulant equals the third central moment, so no
    lower-order corrections are needed.
    """
    Z = H - H.mean(0)
    return (Z * Z).T @ Z / Z.shape[0]


def denoise_svd(c21: np.ndarray, rank: int) -> np.ndarray:
    u, s, vt = np.linalg.svd(c21, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vt[:rank]


def denoise_structured(c21: np.ndarray, rank: int, iters: int = 60) -> np.ndarray:
    """Fit c21 ~= sum_r lambda_r (u_r o u_r) u_r^T by alternating least squares.

    Initialised from the SVD right factors, which already span roughly the
    right subspace; the alternation then enforces the (u o u, u) coupling that
    plain truncation ignores.
    """
    n = c21.shape[0]
    _, _, vt = np.linalg.svd(c21, full_matrices=False)
    U = vt[:rank].T.copy()                      # (n, rank) candidate u_r
    lam = np.ones(rank)
    for _ in range(iters):
        # left factors are determined by U; solve for lambda in least squares
        L = (U * U)                              # (n, rank), columns u_r o u_r
        A = np.einsum("ir,jr->ijr", L, U).reshape(n * n, rank)
        lam, *_ = np.linalg.lstsq(A, c21.reshape(-1), rcond=None)
        # refit U given lambda: gradient step on the same objective
        R = c21 - (L * lam) @ U.T
        G = 2.0 * (L * lam).T @ R                # (rank, n) gradient wrt U^T
        step = 0.5 / (np.linalg.norm(L * lam, axis=0) ** 2 + 1e-12)
        U = U + (G * step[:, None]).T
    return ((U * U) * lam) @ U.T


def compare_estimators(H_truth, pilot_sizes, ranks, rng):
    """Relative error of raw pilot vs denoised pilot against the MC truth."""
    truth = sample_c21(H_truth)
    scale = np.sqrt(np.mean(truth ** 2))
    out = []
    for n in pilot_sizes:
        idx = rng.choice(H_truth.shape[0], n, replace=False)
        pilot = sample_c21(H_truth[idx])
        row = {"n": n, "raw": float(np.sqrt(np.mean((pilot - truth) ** 2)) / scale)}
        for r in ranks:
            row[f"svd{r}"] = float(
                np.sqrt(np.mean((denoise_svd(pilot, r) - truth) ** 2)) / scale
            )
            row[f"cp{r}"] = float(
                np.sqrt(np.mean((denoise_structured(pilot, r) - truth) ** 2)) / scale
            )
        out.append(row)
    return out, truth, scale
