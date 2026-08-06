"""Measure the contracted-K3 adjoint with the challenge's flopscope runtime.

This intentionally executes the same dense/low-rank algebra as
``eval_adjoint_contracted_k3.py`` on deterministic dummy arrays.  It profiles
the contraction core only; local K3 source generation is accounted separately
by the vendor's named counter because that code has not yet been ported from
PyTorch to ``flopscope.numpy``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import flopscope
import flopscope.numpy as fnp
import numpy as np


def pair_dense(matrix, vector, a, c, d):
    ma = fnp.matmul(matrix, a)
    mc = fnp.matmul(matrix, c)
    md = fnp.matmul(matrix, d)
    ac = fnp.sum(a * mc, axis=0)
    ad = fnp.sum(a * md, axis=0)
    cd = fnp.sum(c * md, axis=0)
    ab = fnp.matmul(a.T, vector)
    cb = fnp.matmul(c.T, vector)
    db = fnp.matmul(d.T, vector)
    return fnp.sum(ac * db + ad * cb + cd * ab) / 3.0


def pair_lowrank(basis, eigenvalues, vector, a, c, d):
    pa = fnp.matmul(basis.T, a)
    pc = fnp.matmul(basis.T, c)
    pd = fnp.matmul(basis.T, d)
    ac = fnp.sum(eigenvalues[:, None] * pa * pc, axis=0)
    ad = fnp.sum(eigenvalues[:, None] * pa * pd, axis=0)
    cd = fnp.sum(eigenvalues[:, None] * pc * pd, axis=0)
    ab = fnp.matmul(a.T, vector)
    cb = fnp.matmul(c.T, vector)
    db = fnp.matmul(d.T, vector)
    return fnp.sum(ac * db + ad * cb + cd * ab) / 3.0


def run_one(
    *,
    width: int,
    source_ranks: list[int],
    matrix_rank: int | None,
    handoff: int,
    dtype,
):
    # Setup is outside the measured region.  A real implementation receives
    # these arrays from moment/source generation and from the sampled SVD.
    response = fnp.asarray(
        np.eye(width, dtype=dtype) * np.asarray(0.99, dtype=dtype)
    )
    vector = fnp.asarray(np.ones(width, dtype=dtype))
    matrix = fnp.asarray(np.eye(width, dtype=dtype))
    basis = None
    eigenvalues = None
    source_cache = {
        source_rank: tuple(
            fnp.asarray(np.ones((width, source_rank), dtype=dtype))
            for _ in range(3)
        )
        for source_rank in set(source_ranks)
    }

    if matrix_rank is not None and handoff == len(source_ranks):
        # Count the actual one-off spectral truncation.
        values, vectors = fnp.linalg.eigh(matrix)
        order = fnp.argsort(fnp.abs(values))[::-1][:matrix_rank]
        basis = vectors[:, order]
        eigenvalues = values[order]
        matrix = None

    total = fnp.asarray(np.asarray(0.0, dtype=dtype))
    for layer in range(len(source_ranks) - 1, -1, -1):
        a, c, d = source_cache[source_ranks[layer]]
        if basis is None:
            total = total + pair_dense(matrix, vector, a, c, d)
            if matrix_rank is not None and layer == handoff:
                values, vectors = fnp.linalg.eigh(
                    0.5 * (matrix + matrix.T)
                )
                order = fnp.argsort(fnp.abs(values))[::-1][:matrix_rank]
                basis = vectors[:, order]
                eigenvalues = values[order]
                matrix = None
        else:
            total = total + pair_lowrank(
                basis,
                eigenvalues,
                vector,
                a,
                c,
                d,
            )
        vector = fnp.matmul(response.T, vector)
        if basis is None:
            matrix = fnp.matmul(
                fnp.matmul(response.T, matrix),
                response,
            )
        else:
            basis = fnp.matmul(response.T, basis)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=30)
    parser.add_argument("--first-source-rank", type=int, default=768)
    parser.add_argument("--source-rank", type=int, default=1280)
    parser.add_argument("--controls", type=int, default=2)
    parser.add_argument("--matrix-rank", type=int)
    parser.add_argument("--handoff", type=int, default=24)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    dtype = np.float32 if args.dtype == "float32" else np.float64
    source_ranks = [args.first_source_rank] + [
        args.source_rank
    ] * (args.layers - 1)

    with flopscope.BudgetContext(
        flop_budget=10**15,
        quiet=True,
    ) as context:
        before = int(context.flops_used)
        outputs = [
            run_one(
                width=args.width,
                source_ranks=source_ranks,
                matrix_rank=args.matrix_rank,
                handoff=args.handoff,
                dtype=dtype,
            )
            for _ in range(args.controls)
        ]
        # Force a tiny tracked dependency on every result.
        fnp.stack(outputs)
        after = int(context.flops_used)
    result = {
        "width": args.width,
        "layers": args.layers,
        "source_ranks": source_ranks,
        "controls": args.controls,
        "matrix_rank": args.matrix_rank,
        "handoff": args.handoff,
        "dtype": args.dtype,
        "flopscope_flops": after - before,
        "flopscope_version": getattr(flopscope, "__version__", "unknown"),
        "scope": "adjoint contraction and one-off handoff eigendecomposition",
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
