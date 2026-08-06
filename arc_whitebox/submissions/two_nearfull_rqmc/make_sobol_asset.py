"""Build the frozen A/D spherical Sobol streams for the near-full submission."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
from scipy.special import gammaln, ndtri
from scipy.stats import qmc


WIDTH = 256
OUTPUT = Path(__file__).with_name("sobol_sphere_a101_d404.npz")


def make_base(total_rows: int, seed: int) -> np.ndarray:
    """Exactly match eval_multistream_rqmc.make_stream before antipode pairing."""
    base_rows = total_rows // 2
    exponent = int(math.log2(base_rows))
    engine = qmc.Sobol(d=WIDTH, scramble=True, seed=seed)
    if 2**exponent == base_rows:
        uniform = engine.random_base2(exponent)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            uniform = engine.random(base_rows)
    eps = np.finfo(np.float64).eps
    gaussian = ndtri(np.clip(uniform, eps, 1.0 - eps))
    gaussian /= np.linalg.norm(gaussian, axis=1, keepdims=True)
    expected_radius = float(
        math.sqrt(2.0)
        * math.exp(gammaln((WIDTH + 1) / 2.0) - gammaln(WIDTH / 2.0))
    )
    return (gaussian * expected_radius).astype(np.float32)


def main() -> None:
    np.savez_compressed(
        OUTPUT,
        directions_a=make_base(32_768, 101),
        directions_d=make_base(30_000, 404),
    )
    archive = np.load(OUTPUT)
    assert archive["directions_a"].shape == (16_384, WIDTH)
    assert archive["directions_d"].shape == (15_000, WIDTH)
    assert archive["directions_a"].dtype == np.float32
    assert archive["directions_d"].dtype == np.float32
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
