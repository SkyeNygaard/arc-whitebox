"""Build the two fixed sphere-frame blocks shipped with the estimator.

This is an offline build tool only.  It reproduces the research harness's
``Sobol -> normal -> exact radius -> six covariance-whitening iterations``
recipe at the largest allocation that passed the real flopscope budget:

* seed 0: 16,384 base directions (+ antipodes at inference = 32,768 rows)
* seed 1: 8,192 base directions (+ antipodes at inference = 16,384 rows)

The submission imports no NumPy/SciPy; it loads these float32 directions using
``flopscope.numpy.load``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.special import gammaln, ndtri
from scipy.stats import qmc


WIDTH = 256
OUTPUT = Path(__file__).with_name("sobol_u32.npz")


def sphere_frame(total_rows: int, seed: int) -> np.ndarray:
    base_rows = total_rows // 2
    uniform = qmc.Sobol(d=WIDTH, scramble=True, seed=seed).random_base2(
        int(math.log2(base_rows))
    )
    x = ndtri(np.clip(uniform, 1e-7, 1.0 - 1e-7)).astype(np.float64)
    expected_radius = float(
        math.sqrt(2.0)
        * math.exp(gammaln((WIDTH + 1) / 2.0) - gammaln(WIDTH / 2.0))
    )
    for _ in range(6):
        x *= expected_radius / np.linalg.norm(x, axis=1, keepdims=True)
        covariance = (x.T @ x) / len(x)
        chol = np.linalg.cholesky(covariance)
        x = np.linalg.solve(chol, x.T).T
    x *= expected_radius / np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float32)


def main() -> None:
    np.savez_compressed(
        OUTPUT,
        directions_a=sphere_frame(1 << 15, 0),
        directions_b=sphere_frame(1 << 14, 1),
    )
    archive = np.load(OUTPUT)
    assert archive["directions_a"].shape == (1 << 14, WIDTH)
    assert archive["directions_b"].shape == (1 << 13, WIDTH)
    assert archive["directions_a"].dtype == np.float32
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
