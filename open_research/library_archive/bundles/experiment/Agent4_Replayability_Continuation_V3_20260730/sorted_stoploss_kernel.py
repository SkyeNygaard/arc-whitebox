#!/usr/bin/env python3
"""Exact stop-loss replay from sorted baseline preactivations.

The implementation is written without item assignment so the same structure can be
ported to immutable flopscope arrays.  The local self-test uses NumPy only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def build_stoploss_table(xp: Any, z: Any, block_cols: int = 8) -> tuple[list[Any], list[Any]]:
    """Return blockwise sorted columns and float64 prefix sums.

    z has shape (N, m).  Keeping blocks separate bounds transient memory and avoids
    requiring one full N x m float64 prefix table.
    """
    n, m = z.shape
    sorted_blocks: list[Any] = []
    prefix_blocks: list[Any] = []
    for j0 in range(0, m, block_cols):
        zb = xp.sort(z[:, j0 : j0 + block_cols], axis=0)
        pb = xp.cumsum(zb, axis=0, dtype=xp.float64)
        sorted_blocks.append(zb)
        prefix_blocks.append(pb)
    return sorted_blocks, prefix_blocks


def query_stoploss_table(
    xp: Any,
    sorted_blocks: list[Any],
    prefix_blocks: list[Any],
    shifts: Any,
    n: int,
) -> Any:
    """Evaluate G(s)=mean(ReLU(z+s)) for shifts with shape (r,m)."""
    r, _ = shifts.shape
    out_blocks: list[Any] = []
    j0 = 0
    for zb, pb in zip(sorted_blocks, prefix_blocks):
        width = zb.shape[1]
        sb = shifts[:, j0 : j0 + width]
        column_outputs: list[Any] = []
        for q in range(width):
            # right side makes z == -s inactive, matching derivative 1[z+s>0].
            k = xp.searchsorted(zb[:, q], -sb[:, q], side="right")
            safe = xp.maximum(k - 1, 0)
            before = xp.where(k > 0, pb[safe, q], 0.0)
            total = pb[-1, q]
            active = n - k
            values = (total - before + active * sb[:, q]) / n
            column_outputs.append(values)
        out_blocks.append(xp.stack(column_outputs, axis=1))
        j0 += width
    return xp.concatenate(out_blocks, axis=1)


def query_active_fraction(
    xp: Any,
    sorted_blocks: list[Any],
    shifts: Any,
    n: int,
) -> Any:
    """Evaluate D(s)=mean(1[z+s>0]) exactly for shifts with shape (r,m)."""
    out_blocks: list[Any] = []
    j0 = 0
    for zb in sorted_blocks:
        width = zb.shape[1]
        sb = shifts[:, j0 : j0 + width]
        columns: list[Any] = []
        for q in range(width):
            k = xp.searchsorted(zb[:, q], -sb[:, q], side="right")
            columns.append((n - k) / n)
        out_blocks.append(xp.stack(columns, axis=1))
        j0 += width
    return xp.concatenate(out_blocks, axis=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rng = np.random.default_rng(20260730)
    n, m, r = 4097, 23, 17
    z = rng.normal(size=(n, m)).astype(np.float32)
    shifts = rng.normal(scale=0.2, size=(r, m)).astype(np.float32)
    sorted_blocks, prefix_blocks = build_stoploss_table(np, z, block_cols=7)
    got = query_stoploss_table(np, sorted_blocks, prefix_blocks, shifts, n)
    direct = np.stack(
        [np.maximum(z + shifts[k][None, :], 0).sum(axis=0, dtype=np.float64) / n for k in range(r)]
    )
    active = query_active_fraction(np, sorted_blocks, shifts, n)
    direct_active = np.stack([(z + shifts[k][None, :] > 0).mean(axis=0) for k in range(r)])
    result = {
        "shape": [n, m],
        "rank": r,
        "block_cols": 7,
        "max_abs_output_error": float(np.max(np.abs(got - direct))),
        "max_abs_active_fraction_error": float(np.max(np.abs(active - direct_active))),
        "all_pass": bool(
            np.max(np.abs(got - direct)) < 2e-8
            and np.max(np.abs(active - direct_active)) < 1e-15
        ),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
