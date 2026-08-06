"""Tracked depth-5 Winograd with a two-level assembly-free suffix."""

from __future__ import annotations

import flopscope.numpy as fnp


def _encode(
    left: fnp.ndarray,
    right: fnp.ndarray,
) -> tuple[fnp.ndarray, fnp.ndarray]:
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
        fnp.stack(
            (a11, a12, s4, a22, s1, s2, s3),
            axis=-3,
        ),
        fnp.stack(
            (b11, b21, b22, t4, t1, t2, t3),
            axis=-3,
        ),
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
    return fnp.block(
        [[u1, u4 + p3], [u3 - p4, u3 + p5]]
    )


def _level_one_quadrants(
    left: fnp.ndarray,
    right: fnp.ndarray,
) -> tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray]:
    """One Winograd level, retaining four output quadrants as a tuple."""
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
    p1 = a11 @ b11
    p2 = a12 @ b21
    p3 = s4 @ b22
    p4 = a22 @ t4
    p5 = s1 @ t1
    p6 = s2 @ t2
    p7 = s3 @ t3
    u1 = p1 + p2
    u2 = p1 + p6
    u3 = u2 + p7
    u4 = u2 + p5
    return u1, u4 + p3, u3 - p4, u3 + p5


def _depth_two_output_tree(
    left: fnp.ndarray,
    right: fnp.ndarray,
) -> tuple[
    tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray],
    tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray],
    tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray],
    tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray],
]:
    """Two levels whose sixteen output leaves are assembled only once."""
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
    products = (
        _level_one_quadrants(a11, b11),
        _level_one_quadrants(a12, b21),
        _level_one_quadrants(s4, b22),
        _level_one_quadrants(a22, t4),
        _level_one_quadrants(s1, t1),
        _level_one_quadrants(s2, t2),
        _level_one_quadrants(s3, t3),
    )
    decoded = []
    for quadrant in range(4):
        p1, p2, p3, p4, p5, p6, p7 = (
            product[quadrant] for product in products
        )
        u1 = p1 + p2
        u2 = p1 + p6
        u3 = u2 + p7
        u4 = u2 + p5
        decoded.append((u1, u4 + p3, u3 - p4, u3 + p5))
    return tuple(
        tuple(decoded[leaf][root] for leaf in range(4))
        for root in range(4)
    )


def winograd_hybrid_p3_d5_partial_tree(
    left: fnp.ndarray,
    right: fnp.ndarray,
) -> fnp.ndarray:
    """Exact rank-7**5 product with one assembly for the deepest two levels."""
    encoded_left = left
    encoded_right = right
    for _ in range(3):
        encoded_left, encoded_right = _encode(
            encoded_left,
            encoded_right,
        )
    tree = _depth_two_output_tree(encoded_left, encoded_right)
    products = fnp.block(
        [
            [tree[0][0], tree[0][1], tree[1][0], tree[1][1]],
            [tree[0][2], tree[0][3], tree[1][2], tree[1][3]],
            [tree[2][0], tree[2][1], tree[3][0], tree[3][1]],
            [tree[2][2], tree[2][3], tree[3][2], tree[3][3]],
        ]
    )
    for _ in range(3):
        products = _decode(products)
    return products
