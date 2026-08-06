"""Build the fixed seed-101 spherical Sobol block shipped with the estimator."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.special import gammaln, ndtri
from scipy.stats import qmc


WIDTH = 256
BASE_ROWS = 1 << 14
OUTPUT = Path(__file__).with_name("sobol_sphere_seed101.npz")


def main() -> None:
    uniform = qmc.Sobol(d=WIDTH, scramble=True, seed=101).random_base2(14)
    x = ndtri(np.clip(uniform, 1e-7, 1.0 - 1e-7)).astype(np.float64)
    expected_radius = float(
        math.sqrt(2.0)
        * math.exp(gammaln((WIDTH + 1) / 2.0) - gammaln(WIDTH / 2.0))
    )
    x *= expected_radius / np.linalg.norm(x, axis=1, keepdims=True)
    directions = x.astype(np.float32)
    np.savez_compressed(OUTPUT, directions=directions)
    archive = np.load(OUTPUT)
    assert archive["directions"].shape == (BASE_ROWS, WIDTH)
    assert archive["directions"].dtype == np.float32
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
