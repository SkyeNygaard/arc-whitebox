"""Tracked depth-5 Winograd multiplication for tall 256-column matrices."""

from __future__ import annotations

import flopscope.numpy as fnp


def _encode(left: fnp.ndarray, right: fnp.ndarray) -> tuple[fnp.ndarray, fnp.ndarray]:
    half_rows = left.shape[-2] // 2
    half_inner = left.shape[-1] // 2
    half_output = right.shape[-1] // 2
    a11 = left[..., :half_rows, :half_inner]
    a12 = left[..., :half_rows, half_inner:]
    a21 = left[..., half_rows:, :half_inner]
    a22 = left[..., half_rows:, half_inner:]
    b11 = right[..., :half_inner, :half_output]
    b12 = right[..., :half_inner, half_output:]
    b21 = right[..., half_inner:, :half_output]
    b22 = right[..., half_inner:, half_output:]

    s1 = a21 + a22
    s2 = s1 - a11
    s3 = a11 - a21
    s4 = a12 - s2
    t1 = b12 - b11
    t2 = b22 - t1
    t3 = b22 - b12
    t4 = t2 - b21
    return (
        fnp.stack((a11, a12, s4, a22, s1, s2, s3), axis=-3),
        fnp.stack((b11, b21, b22, t4, t1, t2, t3), axis=-3),
    )


def _decode(products: fnp.ndarray) -> fnp.ndarray:
    p1 = products[..., 0, :, :]
    p2 = products[..., 1, :, :]
    p3 = products[..., 2, :, :]
    p4 = products[..., 3, :, :]
    p5 = products[..., 4, :, :]
    p6 = products[..., 5, :, :]
    p7 = products[..., 6, :, :]
    u1 = p1 + p2
    u2 = p1 + p6
    u3 = u2 + p7
    u4 = u2 + p5
    c11 = u1
    c12 = u4 + p3
    c21 = u3 - p4
    c22 = u3 + p5
    return fnp.block([[c11, c12], [c21, c22]])


def _depth_first(
    left: fnp.ndarray,
    right: fnp.ndarray,
    levels: int,
) -> fnp.ndarray:
    if levels == 0:
        return left @ right

    half_rows = left.shape[-2] // 2
    half_inner = left.shape[-1] // 2
    half_output = right.shape[-1] // 2
    a11 = left[..., :half_rows, :half_inner]
    a12 = left[..., :half_rows, half_inner:]
    a21 = left[..., half_rows:, :half_inner]
    a22 = left[..., half_rows:, half_inner:]
    b11 = right[..., :half_inner, :half_output]
    b12 = right[..., :half_inner, half_output:]
    b21 = right[..., half_inner:, :half_output]
    b22 = right[..., half_inner:, half_output:]

    s1 = a21 + a22
    s2 = s1 - a11
    s3 = a11 - a21
    s4 = a12 - s2
    t1 = b12 - b11
    t2 = b22 - t1
    t3 = b22 - b12
    t4 = t2 - b21
    next_level = levels - 1
    p1 = _depth_first(a11, b11, next_level)
    p2 = _depth_first(a12, b21, next_level)
    p3 = _depth_first(s4, b22, next_level)
    p4 = _depth_first(a22, t4, next_level)
    p5 = _depth_first(s1, t1, next_level)
    p6 = _depth_first(s2, t2, next_level)
    p7 = _depth_first(s3, t3, next_level)
    u1 = p1 + p2
    u2 = p1 + p6
    u3 = u2 + p7
    u4 = u2 + p5
    c11 = u1
    c12 = u4 + p3
    c21 = u3 - p4
    c22 = u3 + p5
    return fnp.block([[c11, c12], [c21, c22]])


def winograd_hybrid_p3_d5(
    left: fnp.ndarray,
    right: fnp.ndarray,
) -> fnp.ndarray:
    """Compute ``left @ right`` with rank ``7**5``.

    The first three levels keep their seven product indices as tensor axes.
    The final two levels recurse depth-first. This avoids billed reshapes,
    limits packed-copy traffic, and uses only 49 leaf matmul calls.
    """
    encoded_left = left
    encoded_right = right
    for _ in range(3):
        encoded_left, encoded_right = _encode(encoded_left, encoded_right)
    products = _depth_first(encoded_left, encoded_right, 2)
    for _ in range(3):
        products = _decode(products)
    return products
