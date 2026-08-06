"""Tracked depth-5 Winograd with cached right transforms and functional assembly."""
from __future__ import annotations
import flopscope.numpy as fnp

PreparedRight = tuple[
    tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray],
    tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray],
    tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray],
    tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray],
    tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray],
    tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray],
    tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray],
]

def _left_forms(a: fnp.ndarray) -> tuple[fnp.ndarray, ...]:
    half_rows = a.shape[-2] // 2
    half_inner = a.shape[-1] // 2
    a11 = a[..., :half_rows, :half_inner]
    a12 = a[..., :half_rows, half_inner:]
    a21 = a[..., half_rows:, :half_inner]
    a22 = a[..., half_rows:, half_inner:]
    s1 = a21 + a22
    s2 = s1 - a11
    s3 = a11 - a21
    s4 = a12 - s2
    return a11, a12, s4, a22, s1, s2, s3

def _right_forms(b: fnp.ndarray) -> tuple[fnp.ndarray, ...]:
    half_inner = b.shape[-2] // 2
    half_output = b.shape[-1] // 2
    b11 = b[..., :half_inner, :half_output]
    b12 = b[..., :half_inner, half_output:]
    b21 = b[..., half_inner:, :half_output]
    b22 = b[..., half_inner:, half_output:]
    t1 = b12 - b11
    t2 = b22 - t1
    t3 = b22 - b12
    t4 = t2 - b21
    return b11, b21, b22, t4, t1, t2, t3

def _encode_left_once(a: fnp.ndarray) -> fnp.ndarray:
    return fnp.stack(_left_forms(a), axis=-3)

def _encode_right_once(b: fnp.ndarray) -> fnp.ndarray:
    return fnp.stack(_right_forms(b), axis=-3)

def prepare_right_p3_d5(weight: fnp.ndarray) -> PreparedRight:
    """Compute every right-side Winograd form exactly once for one weight."""
    encoded = weight
    for _ in range(3):
        encoded = _encode_right_once(encoded)
    level_four = _right_forms(encoded)
    return tuple(_right_forms(part) for part in level_four)  # type: ignore[return-value]

def _level_one_quadrants_prepared(
    left: fnp.ndarray,
    right_forms: tuple[fnp.ndarray, ...],
) -> tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray]:
    products = tuple(a @ b for a, b in zip(_left_forms(left), right_forms))
    p1, p2, p3, p4, p5, p6, p7 = products
    u1 = p1 + p2
    u2 = p1 + p6
    u3 = u2 + p7
    u4 = u2 + p5
    return u1, u4 + p3, u3 - p4, u3 + p5

def _depth_two_output_tree_prepared(
    left: fnp.ndarray,
    prepared: PreparedRight,
) -> tuple[tuple[fnp.ndarray, ...], ...]:
    products = tuple(
        _level_one_quadrants_prepared(a, b)
        for a, b in zip(_left_forms(left), prepared)
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
    return fnp.block([[u1, u4 + p3], [u3 - p4, u3 + p5]])

def winograd_hybrid_p3_d5_prepared_right(
    left: fnp.ndarray,
    prepared: PreparedRight,
) -> fnp.ndarray:
    """Exact rank-7**5 product with the weight-side forms supplied once."""
    encoded_left = left
    for _ in range(3):
        encoded_left = _encode_left_once(encoded_left)
    tree = _depth_two_output_tree_prepared(encoded_left, prepared)
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
