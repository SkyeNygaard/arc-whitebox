r"""Reference prototype for tensor-free adjoint contraction of connected third moments.

This file is deliberately independent of the ARC repository.  It demonstrates the
identity behind an adjoint-compressed c21/K3 anchor:

    T_L(P_L, q_L)
      = T_0(P_0, q_0) + sum_l <S_l, P_{l+1} \otimes q_{l+1}>,

where

    T_l(P, q) = E[(x_l^T P x_l) (q^T x_l)],
    P_l = A_l^T P_{l+1} A_l,
    q_l = A_l^T q_{l+1},
    K_{l+1} = A_l^{\otimes 3} K_l + S_l.

No n x n x n tensor is formed.  For the elementwise quadratic-Hermite c21
control u^T C21 v, initialize P_L = diag(u), q_L = v.  For a directional
cubic control, initialize P_L = u u^T, q_L = v.

Important: empirical source contractions computed from the same Kerdock cloud
are useful for algebra/unit tests but are not a deployable external anchor; the
layerwise source terms must ultimately come from an analytic, independently
randomized, or otherwise non-degenerate estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class DualControl:
    """Dual representation of a third-moment scalar functional."""

    P: Array  # symmetric quadratic factor, shape (n, n)
    q: Array  # linear factor, shape (n,)

    def validate(self) -> None:
        if self.P.ndim != 2 or self.P.shape[0] != self.P.shape[1]:
            raise ValueError("P must be square")
        if self.q.shape != (self.P.shape[0],):
            raise ValueError("q must have shape (n,)")
        if not np.allclose(self.P, self.P.T, atol=1e-10, rtol=1e-10):
            raise ValueError("P must be symmetric")


def center(samples: Array) -> Array:
    """Center row-major samples."""
    if samples.ndim != 2:
        raise ValueError("samples must have shape (N, n)")
    return samples - samples.mean(axis=0, keepdims=True)


def third_contraction(centered_samples: Array, control: DualControl) -> float:
    """Estimate E[(x^T P x)(q^T x)] without constructing a tensor."""
    control.validate()
    x = centered_samples
    if x.ndim != 2 or x.shape[1] != control.q.size:
        raise ValueError("sample/control dimension mismatch")
    quadratic = np.einsum("ni,ij,nj->n", x, control.P, x, optimize=True)
    linear = x @ control.q
    return float(np.mean(quadratic * linear))


def pullback(A: Array, downstream: DualControl) -> DualControl:
    """Pull a downstream third-moment functional through y = A x.

    Uses column-vector convention y = A x.
    """
    downstream.validate()
    if A.ndim != 2 or A.shape[0] != downstream.q.size:
        raise ValueError("A/downstream dimension mismatch")
    P_prev = A.T @ downstream.P @ A
    P_prev = 0.5 * (P_prev + P_prev.T)  # remove roundoff asymmetry
    q_prev = A.T @ downstream.q
    return DualControl(P_prev, q_prev)


def empirical_source_contraction(
    x_prev_centered: Array,
    x_next_centered: Array,
    A: Array,
    downstream: DualControl,
) -> float:
    """Compute the exact empirical local source contraction.

    This is
      T_{next}(P,q) - T_prev(A^T P A, A^T q).

    It implicitly includes every cross term in the nonlinear residual; there is
    no assumption that the residual is independent or has diagonal K3.
    """
    if x_prev_centered.shape[0] != x_next_centered.shape[0]:
        raise ValueError("paired layers need the same number of samples")
    direct = third_contraction(x_next_centered, downstream)
    inherited = third_contraction(x_prev_centered, pullback(A, downstream))
    return direct - inherited


def decompose_anchor(
    centered_layers: Sequence[Array],
    linear_maps: Sequence[Array],
    terminal: DualControl,
) -> tuple[float, float, list[float], list[DualControl]]:
    """Verify the exact layerwise adjoint/source decomposition.

    centered_layers has length L+1 and linear_maps has length L.  The maps need
    not be accurate linearizations: the source is defined as the exact residual,
    so the telescoping identity still holds empirically.
    """
    if len(centered_layers) != len(linear_maps) + 1:
        raise ValueError("need one more layer of samples than linear maps")

    duals: list[DualControl] = [terminal]
    for A in reversed(linear_maps):
        duals.append(pullback(A, duals[-1]))
    duals.reverse()  # duals[l] acts on layer l

    source_terms: list[float] = []
    for l, A in enumerate(linear_maps):
        source_terms.append(
            empirical_source_contraction(
                centered_layers[l], centered_layers[l + 1], A, duals[l + 1]
            )
        )

    initial = third_contraction(centered_layers[0], duals[0])
    reconstructed = initial + float(np.sum(source_terms))
    direct = third_contraction(centered_layers[-1], terminal)
    return direct, reconstructed, source_terms, duals


def signed_low_rank(P: Array, rank: int) -> Array:
    """Best rank-k symmetric approximation in Frobenius norm.

    This is only a baseline compression rule.  In the ARC experiment, rank must
    be selected by source-weighted anchor error/final MSE, not by Frobenius norm.
    """
    if rank <= 0:
        return np.zeros_like(P)
    vals, vecs = np.linalg.eigh(0.5 * (P + P.T))
    order = np.argsort(np.abs(vals))[::-1][:rank]
    return (vecs[:, order] * vals[order]) @ vecs[:, order].T


def fit_linear_map(x: Array, y: Array, ridge: float = 1e-8) -> Array:
    """L2 map y ~= A x for centered row-major samples, returned in y=A x form."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("x and y must be paired row-major samples")
    gram = x.T @ x + ridge * np.eye(x.shape[1])
    # Row convention Y ~= X B, then column convention A = B^T.
    B = np.linalg.solve(gram, x.T @ y)
    return B.T


def _synthetic_demo(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = 12
    N = 200_000
    L = 4

    x = rng.standard_normal((N, n))
    layers = [center(x)]
    maps: list[Array] = []

    for _ in range(L):
        W = rng.standard_normal((n, n)) / np.sqrt(n / 2)
        y = np.maximum(layers[-1] @ W.T + 0.15 * rng.standard_normal(n), 0.0)
        y = center(y)
        A = fit_linear_map(layers[-1], y)
        maps.append(A)
        layers.append(y)

    u = rng.standard_normal(n)
    v = rng.standard_normal(n)
    terminal = DualControl(np.diag(u), v)

    direct, reconstructed, source_terms, duals = decompose_anchor(layers, maps, terminal)
    abs_err = abs(direct - reconstructed)
    rel_err = abs_err / max(abs(direct), 1e-15)

    print(f"direct anchor:        {direct:+.12e}")
    print(f"adjoint reconstruction:{reconstructed:+.12e}")
    print(f"absolute error:       {abs_err:.3e}")
    print(f"relative error:       {rel_err:.3e}")
    print("source contributions:", np.array(source_terms))
    print("dual P ranks:", [np.linalg.matrix_rank(d.P, tol=1e-10) for d in duals])

    if not np.allclose(direct, reconstructed, atol=2e-11, rtol=2e-11):
        raise AssertionError("adjoint/source identity failed")


if __name__ == "__main__":
    _synthetic_demo()
