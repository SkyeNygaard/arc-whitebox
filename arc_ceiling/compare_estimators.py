"""Run two submission estimators under flopscope on the same MLPs and compare.

Reports, per estimator: tracked FLOPs, residual wall time, effective compute,
the resulting score multiplier, final-layer MSE, and the adjusted score the
grader would assign (mean of per-MLP scores -- whestbench/scoring.py:981; the
median is NOT scored).  Also checks prediction agreement between the two, which
for an exact arithmetic change should sit at fp32 round-off, not merely close.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "scripts"))

import flopscope  # noqa: E402
from whestbench.domain import MLP  # noqa: E402
from eval_sampling_official import DEFAULT_DATA, _load_rows  # noqa: E402

BUDGET = 2.72e11
LAMBDA = 1e11  # residual seconds -> FLOPs


def load_estimator(path: Path):
    """Import a submission's Estimator with its own directory taking priority.

    Submissions share helper module names (every one of them has a
    `fast_matmul`), so the sibling modules must be evicted from sys.modules or
    the second estimator silently imports the first one's kernel.
    """
    d = str(path.parent)
    while d in sys.path:
        sys.path.remove(d)
    sys.path.insert(0, d)
    for name in [
        m for m, mod in list(sys.modules.items())
        if getattr(mod, "__file__", None)
        and str(Path(mod.__file__).parent) != d
        and Path(mod.__file__).name in {"fast_matmul.py"}
    ]:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(f"est_{path.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    est = mod.Estimator()

    class _Ctx:
        submission_dir = str(path.parent)

    est.setup(_Ctx())
    return est


def run_one(est, weights, targets):
    mlp = MLP(width=256, depth=32, weights=[np.asarray(w) for w in weights])
    t0 = time.perf_counter()
    with flopscope.BudgetContext(flop_budget=10**15, quiet=True) as ctx:
        pred = est.predict(mlp, 10**15)
    wall = time.perf_counter() - t0
    s = ctx.summary_dict()
    pred = np.asarray(pred)
    flops = int(s["flops_used"])
    resid = float(s["residual_wall_time_s"])
    eff = flops + LAMBDA * resid
    mse = float(np.mean(np.square(pred[-1] - targets[-1])))
    return {
        "flops": flops,
        "residual_s": resid,
        "effective": eff,
        "multiplier": max(0.1, eff / BUDGET),
        "mse": mse,
        "score": mse * max(0.1, eff / BUDGET),
        "wall_s": wall,
        "final": pred[-1],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimators", nargs="+", required=True)
    ap.add_argument("--indices", type=int, nargs="+", default=list(range(6)))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if max(args.indices) >= 50:
        raise ValueError("selection protocol: official IDs 0--49 only")

    rows = _load_rows(DEFAULT_DATA, args.indices)
    ests = {Path(p).parent.name: load_estimator(Path(p)) for p in args.estimators}
    results = {k: [] for k in ests}
    finals = {k: [] for k in ests}

    for idx, (name, W, tg) in zip(args.indices, rows, strict=True):
        line = f"[{idx:>3}] {name[:18]:<18}"
        for k, est in ests.items():
            r = run_one(est, W, tg)
            finals[k].append(r.pop("final"))
            results[k].append(r)
            line += (f" | {k[:16]} mse {r['mse']:.3e} mult {r['multiplier']:.4f}"
                     f" sc {r['score']:.3e}")
        print(line, flush=True)

    print(f"\n{'estimator':<28}{'mean MSE':>12}{'mean mult':>11}{'resid ms':>10}"
          f"{'ADJUSTED':>12}{'vs base':>9}")
    base = None
    summary = {}
    for k, rs in results.items():
        mse = float(np.mean([r["mse"] for r in rs]))
        mult = float(np.mean([r["multiplier"] for r in rs]))
        score = float(np.mean([r["score"] for r in rs]))
        resid = float(np.mean([r["residual_s"] for r in rs])) * 1e3
        if base is None:
            base = score
        summary[k] = {
            "mean_mse": mse,
            "mean_multiplier": mult,
            "adjusted": score,
            "residual_ms": resid,
            "mean_flops": float(np.mean([r["flops"] for r in rs])),
        }
        print(f"{k:<28}{mse:12.4e}{mult:11.4f}{resid:10.2f}{score:12.4e}"
              f"{score / base:9.4f}")

    keys = list(ests)
    if len(keys) == 2:
        a = np.array(finals[keys[0]])
        b = np.array(finals[keys[1]])
        rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-30)
        print(f"\nprediction agreement: max rel diff {rel.max():.3e}"
              f"   (exact change => fp32 round-off ~1e-6)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
