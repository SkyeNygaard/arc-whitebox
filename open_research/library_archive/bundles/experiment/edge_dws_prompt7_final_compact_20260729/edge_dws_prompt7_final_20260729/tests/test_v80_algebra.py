import numpy as np

from adapters.extract_v80_contract import optimal_scale


def test_optimal_scale_minimizes_replay():
    rng = np.random.default_rng(2)
    e = rng.normal(size=256)
    c = rng.normal(size=256)
    s = optimal_scale(e, c)
    mse = lambda x: np.mean((e - x * c) ** 2)
    assert mse(s) <= mse(s + 1e-4)
    assert mse(s) <= mse(s - 1e-4)
