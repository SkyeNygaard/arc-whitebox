"""Joint sweep: Winograd depth x exact dead-column elimination.

Optimises the quantity the grader actually charges.  The graded multiplier
(0.785) sits far above the local one (0.6429) because residual wall time is
charged at 1e11 FLOP/s and the grader is ~11x slower per tracked call than this
machine.  Residual scales with CALL COUNT, and each Winograd level multiplies
branches by 7 while cutting multiplies, so depth trades tracked FLOPs against
calls.  Depth `d` requires the contracted dimension to be a multiple of 2**d,
which also sets the dead-column padding granularity.

Reports tracked FLOPs, call count, projected grader multiplier and final-layer
MSE together, because a kernel that saves FLOPs but changes the answer is not a
saving.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "submissions" / "kerdock_mub5_winograd_tree"))

import flopscope  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402
from eval_sampling_official import DEFAULT_DATA, _load_rows  # noqa: E402

BUDGET = 2.72e11
LAMBDA = 1e11
WIDTH = 256
KERDOCK_BASES = 128
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
# 56.1 us/call, calibrated from the graded 0.785 multiplier at 7,592 calls
GRADER_US_PER_CALL = 56.1e-6


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
    for _ in range(depth):
        left, right = _encode(left, right)
    out = fnp.matmul(left, right)
    for _ in range(depth):
        out = _decode(out)
    return out


def drop_dead(activation, weight, granularity):
    """Exact: zero columns contribute nothing; padding reuses zero columns."""
    alive = fnp.max(activation, axis=0) > 0.0
    k = int(fnp.sum(alive))
    padded = -(-k // granularity) * granularity
    if padded >= activation.shape[1]:
        return activation, weight
    ai = fnp.nonzero(alive)[0]
    if padded > k:
        di = fnp.nonzero(~alive)[0]
        index = fnp.concatenate((ai, di[: padded - k]))
    else:
        index = ai[:padded]
    return activation[:, index], weight[index, :]


def fwht(values):
    span = 1
    while span < WIDTH:
        g = values.reshape((KERDOCK_BASES, WIDTH // (2 * span), 2, span, WIDTH))
        left, right = g[:, :, 0, :, :], g[:, :, 1, :, :]
        values = fnp.stack((left + right, left - right), axis=2).reshape(
            (KERDOCK_BASES, WIDTH, WIDTH))
        span *= 2
    return values


def radius(width):
    return math.sqrt(2.0) * math.exp(
        math.lgamma((width + 1.0) / 2.0) - math.lgamma(width / 2.0))


def first_layer(w0, chirps, rotation):
    ew = rotation @ w0
    r = radius(WIDTH)
    pre = fwht(chirps[:, :, None] * ew[None, :, :]) * (r / math.sqrt(WIDTH))
    kr = fnp.stack((pre, -pre), axis=2).reshape((-1, WIDTH))
    cr = fnp.stack((r * ew, -r * ew), axis=1).reshape((-1, WIDTH))
    return fnp.maximum(fnp.concatenate((kr, cr), axis=0), 0.0)


def run(weights, chirps, rotation, depth, use_dead):
    gran = 2 ** depth
    with flopscope.BudgetContext(flop_budget=10**16, quiet=True) as ctx:
        w = [x.astype(fnp.float32) for x in weights]
        a = first_layer(w[0], fnp.asarray(chirps), fnp.asarray(rotation))
        for weight in w[1:]:
            if use_dead:
                a, weight = drop_dead(a, weight, gran)
            a = fnp.maximum(winograd(a, weight, depth), 0.0)
        final = fnp.mean(a.astype(fnp.float64), axis=0)
    s = ctx.summary_dict()
    return (np.asarray(final), int(s["flops_used"]),
            sum(v["calls"] for v in s["operations"].values()),
            float(s["residual_wall_time_s"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--depths", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--out", type=Path, default=HERE / "results" / "kernel_sweep.json")
    args = ap.parse_args()
    if max(args.indices) >= 50:
        raise ValueError("selection protocol: official IDs 0--49 only")

    asset = np.load(ROOT / "submissions" / "kerdock_mub5_winograd_tree"
                    / "kerdock_mub5_seed3.npz")
    chirps, rotation = asset["chirps"], asset["rotation"]
    rows = _load_rows(DEFAULT_DATA, args.indices)

    print(f"grader residual model: {GRADER_US_PER_CALL*1e6:.1f} us/call "
          f"(calibrated from graded multiplier 0.785 at 7,592 calls)")
    print(f"graded baseline: raw MSE 2.42e-7, multiplier 0.785, "
          f"adjusted 1.90e-7\n")
    print(f"{'variant':<22}{'tracked':>11}{'calls':>8}{'proj mult':>11}"
          f"{'mean MSE':>12}{'proj adj':>12}{'vs graded':>10}")

    out = {}
    for depth in args.depths:
        for dead in (False, True):
            mses, tr, ca, lr = [], [], [], []
            for name, W, tg in rows:
                final, flops, calls, resid = run(W, chirps, rotation, depth, dead)
                mses.append(float(np.mean(np.square(final - tg[-1]))))
                tr.append(flops); ca.append(calls); lr.append(resid)
            tracked = float(np.mean(tr)); calls = float(np.mean(ca))
            mult = (tracked + LAMBDA * calls * GRADER_US_PER_CALL) / BUDGET
            # local MSE ratio transfers; absolute local MSE does not
            mse = float(np.mean(mses))
            label = f"depth{depth}{'+dead' if dead else ''}"
            adj = 2.42e-7 * mult
            out[label] = {"tracked": tracked, "calls": calls, "proj_mult": mult,
                          "mean_mse": mse, "proj_adjusted": adj,
                          "local_residual_ms": float(np.mean(lr)) * 1e3}
            print(f"{label:<22}{tracked/1e9:10.2f}B{calls:8.0f}{mult:11.4f}"
                  f"{mse:12.4e}{adj:12.4e}{adj/1.90e-7:10.4f}", flush=True)

    best = min(out.items(), key=lambda kv: kv[1]["proj_adjusted"])
    print(f"\nbest: {best[0]}  projected adjusted {best[1]['proj_adjusted']:.4e}"
          f"  ({best[1]['proj_adjusted']/1.90e-7:.1%} of graded 1.90e-7)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
