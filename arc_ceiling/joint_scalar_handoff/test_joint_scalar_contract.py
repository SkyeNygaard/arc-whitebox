from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from joint_scalar_contract import (
    ProbeBatch,
    compose_anchor,
    connected_cubic_contractions,
    contract_matrix,
    exact_anchor_matrix,
    ratio_for_retention,
    raw_from_connected,
    retained_oracle_improvement,
    scalarize_moments,
)


def random_psd(rng: np.random.Generator, width: int) -> np.ndarray:
    x = rng.standard_normal((4 * width, width))
    return x.T @ x / len(x)


def test_scalar_anchor_matches_full_matrix() -> None:
    rng = np.random.default_rng(3)
    for width in (5, 17, 32):
        count = min(8, width)
        mean = rng.standard_normal(width)
        second = random_psd(rng, width) + np.outer(mean, mean)
        raw = rng.standard_normal((width, width))
        center = rng.standard_normal(width)
        indices = rng.choice(width, size=count, replace=False)
        directions = rng.standard_normal((width, count))
        directions /= np.linalg.norm(directions, axis=0, keepdims=True)
        probes = ProbeBatch(indices=indices, directions=directions)
        scalars = scalarize_moments(mean, second, raw, probes)
        scalar_anchor = compose_anchor(scalars, probes, center, width=width)
        matrix_anchor = contract_matrix(
            exact_anchor_matrix(mean, second, raw, center, width=width), probes
        )
        np.testing.assert_allclose(scalar_anchor, matrix_anchor, rtol=2e-13, atol=2e-13)


def test_connected_plus_lower_reconstructs_raw() -> None:
    rng = np.random.default_rng(7)
    width = 19
    count = 9
    mean = rng.standard_normal(width)
    second = random_psd(rng, width) + np.outer(mean, mean)
    raw = rng.standard_normal((width, width))
    probes = ProbeBatch(
        indices=rng.choice(width, size=count, replace=False),
        directions=rng.standard_normal((width, count)),
    )
    connected = connected_cubic_contractions(mean, second, raw, probes)
    reconstructed = raw_from_connected(connected, mean, second, probes)
    direct = scalarize_moments(mean, second, raw, probes).cubic_contraction
    np.testing.assert_allclose(reconstructed, direct, rtol=2e-13, atol=2e-13)


def test_gate_math() -> None:
    exact = 0.22399526692739602
    threshold70 = ratio_for_retention(exact, 0.70)
    threshold90 = ratio_for_retention(exact, 0.90)
    np.testing.assert_allclose(threshold70, 0.4567966868491772)
    np.testing.assert_allclose(threshold90, 0.3015957402346564)
    np.testing.assert_allclose(retained_oracle_improvement(threshold70, exact), 0.70)
    np.testing.assert_allclose(retained_oracle_improvement(threshold90, exact), 0.90)


def test_canonical_adjoint_telescope() -> None:
    here = Path(__file__).resolve().parent
    prototype = here / "adjoint_c21_prototype.py"
    spec = importlib.util.spec_from_file_location("adjoint_proto", prototype)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import adjoint prototype")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    rng = np.random.default_rng(11)
    samples, width, layers = 4000, 8, 5
    x = rng.standard_normal((samples, width))
    centered = [module.center(x)]
    maps = []
    for _ in range(layers):
        w = rng.standard_normal((width, width)) / np.sqrt(width / 2)
        y = np.maximum(centered[-1] @ w.T + 0.1 * rng.standard_normal(width), 0.0)
        y = module.center(y)
        maps.append(module.fit_linear_map(centered[-1], y))
        centered.append(y)
    terminal = module.DualControl(
        np.diag(rng.standard_normal(width)), rng.standard_normal(width)
    )
    direct, reconstructed, _, _ = module.decompose_anchor(centered, maps, terminal)
    np.testing.assert_allclose(direct, reconstructed, rtol=2e-12, atol=2e-12)


def main() -> None:
    tests = [
        test_scalar_anchor_matches_full_matrix,
        test_connected_plus_lower_reconstructs_raw,
        test_gate_math,
        test_canonical_adjoint_telescope,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
