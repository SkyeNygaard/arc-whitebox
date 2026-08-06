"""Winograd depth versus the quantity the grader actually charges.

The graded multiplier (0.785) is far above the locally measured one (0.6429).
The difference is residual wall time: tracked FLOPs are 1.709e11, so the grader
is spending ~0.43 s per network on work flopscope does not attribute, versus
~0.048 s here.  At 1e11 FLOP/s that residual is ~16% of the entire budget --
larger than any remaining arithmetic saving.

Residual scales with the NUMBER of tracked calls, not their size: the depth-5
kernel issues 7,592 calls per network (2,953 add, 2,705 subtract, 1,520 matmul).
Each Winograd level multiplies the branch count by 7 while cutting multiplies,
so depth trades tracked FLOPs against call count -- and the ledger has only ever
optimised the first of those.

This sweep measures both for depths 0..5 and projects the graded score using a
per-call residual cost calibrated from the observed 0.43 s at 7,592 calls.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "scripts"))

import flopscope  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402
from eval_sampling_official import DEFAULT_DATA, _load_rows  # noqa: E402

BUDGET = 2.72e11
LAMBDA = 1e11
GRADED_MULTIPLIER = 0.785
GRADED_TRACKED = 1.709e11
BASELINE_CALLS = 7592


def _encode(left, right):
    hr, hi, ho = left.shape[-2] // 2, left.shape[-1] // 2, right.shape[-1] // 2
    a11, a12 = left[..., :hr, :hi], left[..., :hr, hi:]
    a21, a22 = left[..., hr:, :hi], left[..., hr:, hi:]
    b11, b12 = right[..., :hi, :ho], right[..., :hi, ho:]
    b21, b22 = right[..., hi:, :ho], right[..., hi:, ho:]
    s1 = a21 + a22
    s2 = s1 - a11
    s3 = a11 - a21
    s4 = a12 - s2
    t1 = b12 - b11
    t2 = b22 - t1
    t3 = b22 - b12
    t4 = t2 - b21
    return (fnp.stack((a11, a12, s4, a22, s1, s2, s3), axis=-3),
            fnp.stack((b11, b21, b22, t4, t1, t2, t3), axis=-3))


def _decode(p):
    p1, p2, p3 = p[..., 0, :, :], p[..., 1, :, :], p[..., 2, :, :]
    p4, p5 = p[..., 3, :, :], p[..., 4, :, :]
    p6, p7 = p[..., 5, :, :], p[..., 6, :, :]
    u1 = p1 + p2
    u2 = p1 + p6
    u3 = u2 + p7
    u4 = u2 + p5
    return fnp.block([[u1, u4 + p3], [u3 - p4, u3 + p5]])


def winograd(left, right, depth: int):
    """Batched Winograd to `depth` levels; depth 0 is a plain matmul."""
    for _ in range(depth):
        left, right = _encode(left, right)
    products = fnp.matmul(left, right)
    for _ in range(depth):
        products = _decode(products)
    return products


def measure(depth, weights, n_layers=31):
    n = 66048
    rng = np.random.default_rng(0)
    act = np.maximum(rng.standard_normal((n, 256), dtype=np.float32), 0.0)
    t0 = time.perf_counter()
    with flopscope.BudgetContext(flop_budget=10**16, quiet=True) as ctx:
        a = fnp.asarray(act)
        for w in weights[:n_layers]:
            a = fnp.maximum(winograd(a, fnp.asarray(w), depth), 0.0)
    wall = time.perf_counter() - t0
    s = ctx.summary_dict()
    calls = sum(v["calls"] for v in s["operations"].values())
    return {
        "depth": depth,
        "tracked": int(s["flops_used"]),
        "calls": calls,
        "local_residual_s": float(s["residual_wall_time_s"]),
        "wall_s": wall,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--layers", type=int, default=31)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "results" / "winograd_depth_sweep.json")
    args = ap.parse_args()

    _, W, _ = _load_rows(DEFAULT_DATA, [0])[0]
    weights = [w.astype(np.float32) for w in W[1:]]

    # calibrate residual-per-call from the graded observation
    grader_residual_total = (GRADED_MULTIPLIER * BUDGET - GRADED_TRACKED) / LAMBDA
    per_call = grader_residual_total / BASELINE_CALLS
    print(f"Grader residual implied by the graded multiplier: "
          f"{grader_residual_total*1e3:.0f} ms over {BASELINE_CALLS:,} calls"
          f"  =>  {per_call*1e6:.1f} us/call\n")

    rows = []
    for d in args.depths:
        r = measure(d, weights, args.layers)
        r["proj_grader_residual_s"] = r["calls"] * per_call
        r["proj_effective"] = r["tracked"] + LAMBDA * r["proj_grader_residual_s"]
        r["proj_multiplier"] = r["proj_effective"] / BUDGET
        rows.append(r)
        print(f"depth {d}: tracked {r['tracked']/1e9:8.2f}B  calls {r['calls']:>7,}"
              f"  local resid {r['local_residual_s']*1e3:6.1f} ms"
              f"  proj grader resid {r['proj_grader_residual_s']*1e3:6.0f} ms"
              f"  -> multiplier {r['proj_multiplier']:.4f}", flush=True)

    best = min(rows, key=lambda r: r["proj_multiplier"])
    print(f"\nbest projected multiplier: depth {best['depth']} at "
          f"{best['proj_multiplier']:.4f}  (graded baseline 0.785)")
    print(f"projected adjusted score: 2.42e-7 x {best['proj_multiplier']:.4f} = "
          f"{2.42e-7*best['proj_multiplier']:.4e}   (current graded 1.90e-7)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"per_call_s": per_call, "rows": rows}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
