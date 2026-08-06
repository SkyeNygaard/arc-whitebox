"""Batched recursive Strassen for the 66,048 x 256 @ 256 x 256 layer products.

Since the ceiling theory pins V_eff within ~16% of its feasible floor, and

    score = V_eff * (FLOPs per direction) / B

the only remaining lever is arithmetic.  flopscope charges einsum analytically as
`M*N*(2K-1)`, so a bilinear algorithm that performs fewer multiplications is
charged less -- this is a genuine reduction in arithmetic, not an accounting
trick.

Two structural facts make this practical here:

1. **Batching keeps the call count at O(L), not O(7^L).**  flopscope charges a
   batched einsum as the sum of its parts, so all 7^L subproblems at a given
   recursion level go through ONE call.  This matters enormously: residual wall
   time is billed at 1e11 FLOP/s, so 7^4 = 2401 separate calls per layer would
   cost more in Python overhead than the multiplications they save.

2. **The weight operand is shared across all 258 blocks.**  66,048 = 258 * 256,
   so the activation matrix is 258 square 256x256 blocks all multiplied by the
   same W.  The entire right-hand Strassen tree over W is built once per layer
   and reused 258 times, and carries no batch dimension at all.

The costs that decide the optimum are the *additions and materialisations*, which
grow as (7/4)^L while the multiplications fall as (7/8)^L.  flopscope charges
reshape/stack/concatenate by element count, so those are counted here too --
that is exactly what this module is for.
"""

from __future__ import annotations

import flopscope.numpy as fnp


def _split4(X):
    """Split the trailing two axes into quadrants.  Slicing is a view."""
    h = X.shape[-1] // 2
    v = X.shape[-2] // 2
    return (X[..., :v, :h], X[..., :v, h:], X[..., v:, :h], X[..., v:, h:])


def _left_combos(A):
    """The 7 Strassen left operands, stacked along a new leading axis."""
    a11, a12, a21, a22 = _split4(A)
    return fnp.stack([a11 + a22, a21 + a22, a11, a22,
                      a11 + a12, a21 - a11, a12 - a22])


def _right_combos(B):
    """The 7 Strassen right operands, stacked along a new leading axis."""
    b11, b12, b21, b22 = _split4(B)
    return fnp.stack([b11 + b22, b11, b12 - b22, b21 - b11,
                      b22, b11 + b12, b21 + b22])


def _merge(M):
    """Recombine the 7 products (leading axis) into the 2x2 output blocks."""
    m1, m2, m3, m4, m5, m6, m7 = (M[i] for i in range(7))
    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6
    top = fnp.concatenate([c11, c12], axis=-1)
    bot = fnp.concatenate([c21, c22], axis=-1)
    return fnp.concatenate([top, bot], axis=-2)


def strassen_matmul(A, W, levels: int):
    """A @ W with `levels` of batched Strassen recursion.

    A : (batch, m, k)   activation blocks
    W : (k, n)          shared weight
    Returns (batch, m, n).

    Left tree carries the batch axis, right tree does not -- the shared-W
    saving.  Both trees are flattened into a single leading axis so each level
    is one einsum.
    """
    if levels == 0:
        return fnp.einsum("bmk,kn->bmn", A, W)

    L, R = A, W
    for _ in range(levels):
        # left: (P, batch, m, k) -> (7P, batch, m/2, k/2)
        Lc = _left_combos(L)                       # (7, P?, batch, m/2, k/2)
        L = Lc.reshape((-1,) + Lc.shape[-3:])
        # right: (P, k, n) -> (7P, k/2, n/2)
        Rc = _right_combos(R)
        R = Rc.reshape((-1,) + Rc.shape[-2:])

    M = fnp.einsum("pbmk,pkn->pbmn", L, R)

    for _ in range(levels):
        M = M.reshape((7, -1) + M.shape[-3:])
        M = _merge(M)
    return M


def dense_matmul(A, W):
    return fnp.einsum("bmk,kn->bmn", A, W)
