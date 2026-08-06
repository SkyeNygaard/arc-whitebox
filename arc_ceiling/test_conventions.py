"""Convention assertions for the shared cumulant branch.

Four implementation mistakes have already contaminated results in this branch,
all of them silent -- each produced a plausible number rather than an error:

  * pre-ReLU moments supplied to an interface expecting post-ReLU state
    (reported as a 4030x control failure);
  * Frobenius norm compared against RMS, a factor of 256 for a 256x256 matrix
    (reported as a 126x cumulant explosion);
  * raw versus connected third moment;
  * c21 orientation, which is NOT symmetric: c21[i,j] = cum(x_i, x_i, x_j).

Every assertion here is checked against direct Monte Carlo on a real network, so
it fails loudly if a convention drifts.  Run before trusting any anchor result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "scripts"))

from eval_sampling_official import DEFAULT_DATA, _load_rows  # noqa: E402

WIDTH = 256
CHUNK = 32768


def mc_state(weights, layer, n_samples, seed, post_relu):
    """Mean, covariance and connected c21 of h_layer (pre) or a_layer (post)."""
    s1 = np.zeros(WIDTH)
    s2 = np.zeros((WIDTH, WIDTH))
    s21 = np.zeros((WIDTH, WIDTH))
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_samples:
        b = min(CHUNK, n_samples - done)
        a = rng.standard_normal((b, WIDTH)).astype(np.float32)
        for li in range(layer + 1):
            h = a @ weights[li]
            a = np.maximum(h, 0.0)
        x = (a if post_relu else h).astype(np.float64)
        s1 += x.sum(0)
        s2 += x.T @ x
        s21 += (x * x).T @ x
        done += b
    mu = s1 / n_samples
    m2 = s2 / n_samples
    sigma = m2 - np.outer(mu, mu)
    c21 = (s21 / n_samples - mu[None, :] * np.diag(m2)[:, None]
           - 2.0 * mu[:, None] * m2 + 2.0 * (mu * mu)[:, None] * mu[None, :])
    return mu, sigma, c21, s21 / n_samples


def raw_m21_from_cumulants(mu, sigma, c21):
    """E[x_i^2 x_j] = c21[i,j] + mu_j Sigma_ii + 2 mu_i Sigma_ij + mu_i^2 mu_j."""
    return (c21 + mu[None, :] * np.diag(sigma)[:, None]
            + 2.0 * mu[:, None] * sigma
            + (mu * mu)[:, None] * mu[None, :])


def main() -> None:
    name, W, _ = _load_rows(DEFAULT_DATA, [0])[0]
    weights = [w.astype(np.float32) for w in W]
    layer = 23
    n = 400_000
    failures = []

    def check(label, ok, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
        if not ok:
            failures.append(label)

    print(f"convention checks on '{name}', layer {layer + 1}, {n:,} samples\n")

    mu_pre, sig_pre, c21_pre, _ = mc_state(weights, layer, n, 1, post_relu=False)
    mu_post, sig_post, c21_post, raw_post = mc_state(weights, layer, n, 1, post_relu=True)

    # 1. pre and post ReLU state are materially different -- mixing them is a bug
    rel = np.linalg.norm(mu_pre - mu_post) / np.linalg.norm(mu_post)
    check("pre vs post ReLU distinguishable", rel > 0.5,
          f"||mu_pre - mu_post|| / ||mu_post|| = {rel:.3f} (post mean is >=0)")
    check("post-ReLU mean is non-negative", float(mu_post.min()) >= 0.0,
          f"min(mu_post) = {mu_post.min():.3e}")
    check("pre-ReLU mean has both signs", float(mu_pre.min()) < 0.0,
          f"min(mu_pre) = {mu_pre.min():.3e}")

    # 2. c21 orientation: c21[i,j] = cum(x_i, x_i, x_j), NOT symmetric
    asym = np.linalg.norm(c21_post - c21_post.T) / np.linalg.norm(c21_post)
    check("c21 is orientation-sensitive", asym > 0.1,
          f"||c21 - c21^T|| / ||c21|| = {asym:.3f}")

    # explicit re-derivation on a few index pairs, no vectorised shortcuts
    rng = np.random.default_rng(0)
    idx = [(int(i), int(j)) for i, j in rng.integers(0, WIDTH, size=(4, 2))]
    direct = []
    rr = np.random.default_rng(1)
    acc = {p: 0.0 for p in idx}
    tot = 0
    while tot < 200_000:
        b = min(CHUNK, 200_000 - tot)
        a = rr.standard_normal((b, WIDTH)).astype(np.float32)
        for li in range(layer + 1):
            h = a @ weights[li]
            a = np.maximum(h, 0.0)
        x = a.astype(np.float64)
        for (i, j) in idx:
            acc[(i, j)] += float((((x[:, i] - mu_post[i]) ** 2)
                                  * (x[:, j] - mu_post[j])).sum())
        tot += b
    for (i, j) in idx:
        direct.append(acc[(i, j)] / tot)
    err = max(abs(d - c21_post[i, j]) / (abs(d) + 1e-30)
              for d, (i, j) in zip(direct, idx))
    check("c21[i,j] = E[(x_i-m)^2 (x_j-m)]", err < 0.05,
          f"max relative deviation over 4 explicit pairs = {err:.4f}")

    # 3. raw vs connected third moment
    rec = raw_m21_from_cumulants(mu_post, sig_post, c21_post)
    e = np.linalg.norm(rec - raw_post) / np.linalg.norm(raw_post)
    check("raw M21 = c21 + mu_j S_ii + 2 mu_i S_ij + mu_i^2 mu_j", e < 1e-10,
          f"relative reconstruction error = {e:.3e}")
    ratio = np.linalg.norm(raw_post) / np.linalg.norm(c21_post)
    check("raw M21 dominated by the mean cube", ratio > 5,
          f"||raw|| / ||connected|| = {ratio:.1f}  "
          f"(mean errors are amplified by this factor)")

    # 4. Frobenius vs RMS -- the factor that produced a phantom 126x
    fro = float(np.linalg.norm(c21_post))
    rms = float(np.sqrt(np.mean(c21_post ** 2)))
    check("||M||_F = sqrt(n_elements) * RMS(M)", abs(fro / (rms * WIDTH) - 1) < 1e-12,
          f"Frobenius {fro:.4e} = {WIDTH} x RMS {rms:.4e}")

    print(f"\n{len(failures)} failure(s)" if failures else "\nall conventions verified")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
