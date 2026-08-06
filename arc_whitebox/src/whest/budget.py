"""Analytic FLOP accounting that mirrors flopscope 0.9.1's cost model.

We run experiments in plain numpy (flopscope adds ~35x wall-clock overhead) and
account FLOPs separately.  Every rate here was measured directly from
flopscope 0.9.1; `tests/test_budget_matches_flopscope.py` re-verifies them.

Key facts:
  * elementwise ops bill `rate * numel`, where rate is 2 for float64 and
    1 for float32/float16 ("dtype billing", floored at 0.5x).
  * transcendentals (exp/log/arccos/...) bill 32 * dtype_rate.
  * `flops.stats.norm.cdf/pdf/ppf` bill 96 / 54 / 166 per element with **no**
    dtype discount.
  * einsum bills `2 * (output numel) * (contracted dim) * dtype_rate`,
    halved when flopscope detects operand symmetry.
"""

from __future__ import annotations

import numpy as np

# dtype -> billing rate multiplier
_RATE = {np.dtype("float64"): 2.0, np.dtype("float32"): 1.0, np.dtype("float16"): 1.0}

# per-element unit costs *before* the dtype rate multiplier
CHEAP = 1.0  # add, mul, div, sqrt, maximum, abs, sign, comparison, sum
TRANSCENDENTAL = 16.0  # exp, log, arccos, arctan, tanh, sin, power
WHERE = 5.0
RANDN = 32.0  # per sample, dtype-independent in practice

# dtype-independent (billed flat)
NORM_CDF = 96.0
NORM_PDF = 54.0
NORM_PPF = 166.0


def rate(dtype) -> float:
    return _RATE[np.dtype(dtype)]


class Budget:
    """Accumulates FLOPs by category. Raises if the cap is exceeded."""

    def __init__(self, cap: float = np.inf, dtype=np.float32):
        self.cap = cap
        self.dtype = np.dtype(dtype)
        self.by_op: dict[str, float] = {}
        self.total = 0.0

    # -- primitives ---------------------------------------------------------
    def _add(self, op: str, cost: float) -> None:
        self.by_op[op] = self.by_op.get(op, 0.0) + cost
        self.total += cost
        if self.total > self.cap:
            raise RuntimeError(
                f"FLOP budget exceeded: {self.total:.3e} > {self.cap:.3e} (at op {op!r})"
            )

    def elementwise(self, numel: int, unit: float = CHEAP, op: str = "elementwise", dtype=None):
        self._add(op, unit * numel * rate(dtype or self.dtype))

    def transcendental(self, numel: int, op: str = "transcendental", dtype=None):
        self.elementwise(numel, TRANSCENDENTAL, op, dtype)

    def flat(self, numel: int, unit: float, op: str):
        """Cost billed with no dtype discount (the flops.stats family)."""
        self._add(op, unit * numel)

    def randn(self, numel: int):
        self._add("randn", RANDN * numel)

    def matmul(self, m: int, k: int, n: int, symmetric: bool = False, op: str = "matmul", dtype=None):
        """(m,k) @ (k,n) -> (m,n)."""
        cost = 2.0 * m * k * n * rate(dtype or self.dtype)
        if symmetric:
            cost *= 0.5
        self._add(op, cost)

    # -- composites we use a lot -------------------------------------------
    def norm_cdf(self, numel: int):
        self.flat(numel, NORM_CDF, "norm.cdf")

    def norm_pdf(self, numel: int):
        self.flat(numel, NORM_PDF, "norm.pdf")

    def phi_pair_fast(self, numel: int):
        """Our hand-rolled (Phi, phi) pair: one exp + ~12 cheap ops.

        Cheaper than flops.stats (96+54=150 flat) because exp gets the fp32
        discount: 16 + 12 = 28 per element in float32.
        """
        self.transcendental(numel, "phi_fast_exp")
        self.elementwise(12 * numel, CHEAP, "phi_fast_poly")

    # -- reporting ----------------------------------------------------------
    def report(self) -> str:
        lines = [f"total = {self.total:.4e} FLOPs"]
        for k, v in sorted(self.by_op.items(), key=lambda kv: -kv[1]):
            lines.append(f"   {k:24s} {v:12.4e}  ({100*v/max(self.total,1):5.1f}%)")
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Challenge constants (Phase 1)
# ----------------------------------------------------------------------------
WIDTH = 256
DEPTH = 32
BUDGET_PHASE1 = 2.72e11
BUDGET_WARMUP = 6.8e10


def budget_for(width: int, depth: int) -> float:
    """Phase-1 budget scales with depth (warm-up 256x8 -> 6.8e10, x4 depth -> x4)."""
    return BUDGET_PHASE1 * (depth / DEPTH) * (width / WIDTH) ** 2


SCORE_FLOOR = 0.1  # verified against whestbench 0.13.0 scoring.py


def score(mse_final: float, flops_used: float, budget: float) -> float:
    """s_m = mse_final * max(0.1, C_m / B_m), uncapped above.

    The floor is 0.1, NOT 0.5 (whestbench/scoring.py:579,863).  This matters a
    lot: a bias-limited method costing under 10% of budget is scored at 0.1x its
    MSE, so the optimal operating point for white-box methods is C <= 0.1*B, and
    the 2.72e10 FLOPs below that threshold are effectively free.

    Monte Carlo is unaffected: MSE = Vc/(fB) and the factor is f, so its score is
    Vc/B -- flat in compute across the whole range.
    """
    return mse_final * max(SCORE_FLOOR, flops_used / budget)
