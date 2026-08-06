"""Exact penultimate-translation reuse helpers for Path 6.

For a cached final preactivation Z = H @ W and a uniform penultimate activation
translation delta, the corrected final mean is
    mean(ReLU(Z + (delta @ W)))
without a second H @ W replay.
"""
from __future__ import annotations


def corrected_final_mean_from_preactivation(fnp, final_preactivation, delta, final_weight, chunk_rows=512):
    shift = delta @ final_weight
    total = fnp.zeros(final_preactivation.shape[1])
    n_rows = final_preactivation.shape[0]
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        corrected = fnp.maximum(final_preactivation[start:stop] + shift, 0.0)
        total = total + fnp.sum(corrected.astype(fnp.float64), axis=0)
    return total / n_rows


def corrected_final_mean_sparse(fnp, final_preactivation, indices, delta_values, final_weight, chunk_rows=512):
    shift = delta_values @ final_weight[indices, :]
    total = fnp.zeros(final_preactivation.shape[1])
    n_rows = final_preactivation.shape[0]
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        corrected = fnp.maximum(final_preactivation[start:stop] + shift, 0.0)
        total = total + fnp.sum(corrected.astype(fnp.float64), axis=0)
    return total / n_rows
