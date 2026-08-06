"""Extract K32/K128 basis summaries from an activation already in memory.
No additional network propagation is performed.
"""
from __future__ import annotations

_WIDTH = 256
_ROWS_PER_BASIS = 2 * _WIDTH


def basis_means(fnp, activation, basis_count=32):
    rows = basis_count * _ROWS_PER_BASIS
    shaped = activation[:rows].reshape((basis_count, _ROWS_PER_BASIS, _WIDTH))
    return fnp.mean(shaped.astype(fnp.float64), axis=1)


def selected_mean_and_gram(fnp, activation, coordinates):
    selected = activation[:, coordinates].astype(fnp.float64)
    mean = fnp.mean(selected, axis=0)
    gram = (selected.T @ selected) / activation.shape[0]
    return mean, gram
