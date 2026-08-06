"""Direct tolerance experiment: how accurate must each anchor component be?

The claim under test is that the exact-anchor control needs the layer-29 MEAN to
roughly 0.01% while propagation delivers 0.65%, which would close the path.  So
far that is an order-of-magnitude argument from the raw-moment identity

    M21[i,j] = c21[i,j] + mu_j Sigma_ii + 2 mu_i Sigma_ij + mu_i^2 mu_j

whose mean-cube term measures 278x the connected part (test_conventions.py).
An argument is not a measurement, so this perturbs mu, Sigma and c21 SEPARATELY
by a controlled relative RMS error and reports the control's final MSE ratio for
each, giving the requirement per component instead of inferring it.

Anchors are written as oracle state with one component perturbed, in the npz
format `eval_crossfit_cumulant_control.py --factorized-k3-dir` consumes, so the
validated harness does the evaluation.  POST-ReLU state, matching that
interface.  Perturbations are structured (symmetric, signal-scaled), not white,
because real closure error is correlated with the signal and isotropic noise
would flatter the method.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from eval_oracle_cumulant_bridge import connected_m21, moment_path  # noqa: E402

WIDTH = 256
CHUNK = 32768


def oracle_post_state(index, layer):
    """POST-ReLU mean, covariance and connected c21 from the stored 1e8-sample moments.

    A locally recomputed 3e5-sample 'oracle' is NOT one: its own sampling noise
    wipes out the entire control gain (measured 0.9976 vs 0.647 for the stored
    state), so the reference must come from the cached high-precision moments.
    `connected_m21` is imported from the harness so the convention cannot drift.
    """
    with np.load(moment_path(index)) as d:
        mean = np.asarray(d["mean"][layer], dtype=np.float64)
        second = np.asarray(d["M11"][layer], dtype=np.float64)
        raw21 = np.asarray(d["M21"][layer], dtype=np.float64)
    sigma = second - np.outer(mean, mean)
    c21 = connected_m21(mean, second, raw21, np.diag(second))
    return mean, sigma, c21


def perturb(array, eps, rng, symmetric):
    """Add structured noise of relative RMS `eps`."""
    if eps == 0.0:
        return array.copy()
    noise = rng.standard_normal(array.shape)
    if symmetric and noise.ndim == 2:
        noise = (noise + noise.T) / np.sqrt(2.0)
    noise *= np.sqrt(np.mean(array ** 2)) / np.sqrt(np.mean(noise ** 2))
    return array + eps * noise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", type=int, nargs="+", required=True)
    ap.add_argument("--weights-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=29)
    ap.add_argument("--components", nargs="+", default=["mu", "sigma", "c21"])
    ap.add_argument("--epsilons", type=float, nargs="+",
                    default=[0.0, 0.0001, 0.001, 0.01, 0.1])
    ap.add_argument("--n-oracle", type=int, default=300_000)
    ap.add_argument("--out-root", type=Path, required=True)
    args = ap.parse_args()

    for index in args.indices:
        mu, sigma, c21 = oracle_post_state(index, args.layer)
        for comp in args.components:
            for eps in args.epsilons:
                rng = np.random.default_rng(9000 + index)
                m, s, c = mu, sigma, c21
                if comp == "mu":
                    m = perturb(mu, eps, rng, False)
                elif comp == "sigma":
                    s = perturb(sigma, eps, rng, True)
                elif comp == "c21":
                    c = perturb(c21, eps, rng, False)
                d = args.out_root / f"{comp}_eps{eps:g}"
                d.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(d / f"mlp_{index:05d}.npz",
                                    mean=m, covariance=s, c21=c)
        print(f"[{index:>5}] wrote {len(args.components) * len(args.epsilons)} "
              f"perturbed anchors", flush=True)


if __name__ == "__main__":
    main()
