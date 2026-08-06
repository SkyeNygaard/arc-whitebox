#!/usr/bin/env python3
"""Synthetic full-shape benchmark for WHestBench signed-mixture final replay.

No competition/protected data are read.  The benchmark uses deterministic Gaussian
preactivations with the production shape N=66,048, m=256 and compares:
  * serialized chunked shifted-ReLU reductions;
  * source-batched chunked shifted-ReLU reductions;
  * exact sorted stop-loss preprocessing and queries.

The numerical contract matches the production implementation's float32 inputs and
float64 chunk accumulation.  Timings are local wall-clock diagnostics only; official
score accounting remains the archived FlopScope model.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import time
from pathlib import Path
from typing import Callable

# Set before importing NumPy when script is launched directly.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


def rss_mib() -> float:
    # Linux ru_maxrss is KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def baseline_mean(z: np.ndarray, chunk_rows: int) -> np.ndarray:
    n, m = z.shape
    total = np.zeros(m, dtype=np.float64)
    for start in range(0, n, chunk_rows):
        x = z[start : start + chunk_rows]
        total += np.maximum(x, np.float32(0.0)).sum(axis=0, dtype=np.float64)
    return total / n


def replay_serial(z: np.ndarray, shifts: np.ndarray, chunk_rows: int) -> np.ndarray:
    n, m = z.shape
    r = shifts.shape[0]
    out = np.zeros((r, m), dtype=np.float64)
    for k in range(r):
        total = np.zeros(m, dtype=np.float64)
        sk = shifts[k]
        for start in range(0, n, chunk_rows):
            x = z[start : start + chunk_rows] + sk
            np.maximum(x, np.float32(0.0), out=x)
            total += x.sum(axis=0, dtype=np.float64)
        out[k] = total / n
    return out


def replay_batched(
    z: np.ndarray,
    shifts: np.ndarray,
    chunk_rows: int,
    source_batch: int,
) -> np.ndarray:
    n, m = z.shape
    r = shifts.shape[0]
    out = np.zeros((r, m), dtype=np.float64)
    for k0 in range(0, r, source_batch):
        k1 = min(k0 + source_batch, r)
        sb = shifts[k0:k1]
        total = np.zeros((k1 - k0, m), dtype=np.float64)
        for start in range(0, n, chunk_rows):
            # Shape: rows x source_batch x outputs.
            x = z[start : start + chunk_rows, None, :] + sb[None, :, :]
            np.maximum(x, np.float32(0.0), out=x)
            total += x.sum(axis=0, dtype=np.float64)
        out[k0:k1] = total / n
    return out


def build_sorted_stoploss(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted columns and float64 prefix sums with leading zero row."""
    zs = np.sort(z, axis=0)
    n, m = zs.shape
    prefix = np.empty((n + 1, m), dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(zs, axis=0, dtype=np.float64, out=prefix[1:])
    return zs, prefix


def query_sorted_stoploss(
    zs: np.ndarray,
    prefix: np.ndarray,
    shifts: np.ndarray,
) -> np.ndarray:
    n, m = zs.shape
    r = shifts.shape[0]
    out = np.empty((r, m), dtype=np.float64)
    totals = prefix[-1]
    for j in range(m):
        idx = np.searchsorted(zs[:, j], -shifts[:, j], side="right")
        count = n - idx
        active_sum = totals[j] - prefix[idx, j]
        out[:, j] = (active_sum + count * shifts[:, j]) / n
    return out


def timed(fn: Callable[[], np.ndarray | tuple[np.ndarray, np.ndarray]], repeats: int) -> tuple[object, list[float], float]:
    times: list[float] = []
    result: object = None
    peak_before = rss_mib()
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return result, times, max(0.0, rss_mib() - peak_before)


def stats(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "min_seconds": float(a.min()),
        "median_seconds": float(np.median(a)),
        "max_seconds": float(a.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260730)
    p.add_argument("--n", type=int, default=66048)
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--ranks", type=int, nargs="+", default=[20, 24, 32])
    p.add_argument("--chunk-rows", type=int, default=2048)
    p.add_argument("--source-batches", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--skip-sorted", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    z = rng.normal(loc=-0.03, scale=1.15, size=(args.n, args.m)).astype(np.float32)
    rmax = max(args.ranks)
    shifts_all = rng.normal(loc=0.0, scale=0.22, size=(rmax, args.m)).astype(np.float32)

    result: dict[str, object] = {
        "protocol": {
            "protected_data_opened": False,
            "shape": [args.n, args.m],
            "ranks": args.ranks,
            "chunk_rows": args.chunk_rows,
            "source_batches": args.source_batches,
            "repeats": args.repeats,
            "dtype_preactivation": str(z.dtype),
            "dtype_accumulation": "float64",
            "seed": args.seed,
            "thread_env": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]},
            "initial_rss_mib": rss_mib(),
        },
        "baseline": {},
        "ranks": {},
    }

    base, base_times, base_peak = timed(lambda: baseline_mean(z, args.chunk_rows), args.repeats)
    assert isinstance(base, np.ndarray)
    result["baseline"] = {**stats(base_times), "peak_rss_delta_mib": base_peak, "checksum": float(base.sum())}

    # Rank-specific scan benchmarks.  Serial reference is run once per rank because
    # source_batch=1 is mathematically identical and expensive.
    for r in args.ranks:
        shifts = shifts_all[:r]
        rank_res: dict[str, object] = {}
        reference: np.ndarray | None = None
        for b in args.source_batches:
            if b > r:
                continue
            out, times, peak = timed(lambda b=b: replay_batched(z, shifts, args.chunk_rows, b), args.repeats)
            assert isinstance(out, np.ndarray)
            if reference is None:
                reference = out.copy()
                err = 0.0
            else:
                err = float(np.max(np.abs(out - reference)))
            rank_res[f"batched_{b}"] = {
                **stats(times),
                "peak_rss_delta_mib": peak,
                "max_abs_vs_batch1": err,
                "checksum": float(out.sum()),
                "throughput_shifted_entries_per_second_median": float(args.n * args.m * r / np.median(times)),
            }
        result["ranks"][str(r)] = rank_res

    if not args.skip_sorted:
        (zs, prefix), prep_times, prep_peak = timed(lambda: build_sorted_stoploss(z), 1)
        assert isinstance(zs, np.ndarray) and isinstance(prefix, np.ndarray)
        sorted_res: dict[str, object] = {
            "preprocess": {
                **stats(prep_times),
                "peak_rss_delta_mib": prep_peak,
                "sorted_bytes": int(zs.nbytes),
                "prefix_bytes": int(prefix.nbytes),
            },
            "queries": {},
        }
        for r in args.ranks:
            shifts = shifts_all[:r]
            out, times, peak = timed(lambda: query_sorted_stoploss(zs, prefix, shifts), args.repeats)
            assert isinstance(out, np.ndarray)
            ref = replay_batched(z, shifts, args.chunk_rows, 1)
            sorted_res["queries"][str(r)] = {
                **stats(times),
                "peak_rss_delta_mib": peak,
                "max_abs_vs_scan": float(np.max(np.abs(out - ref))),
                "rms_vs_scan": float(np.sqrt(np.mean((out - ref) ** 2))),
                "checksum": float(out.sum()),
            }
        result["sorted_stoploss"] = sorted_res

    result["protocol"]["final_peak_rss_mib"] = rss_mib()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
