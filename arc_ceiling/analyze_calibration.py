"""Honest read of the calibration sweep.

The grader averages per-MLP scores (whestbench/scoring.py:981), so the mean
ratio is the only aggregate that maps to the leaderboard -- the median is not
scored.  A mean can be moved by one lucky network, so this also reports a
paired bootstrap over networks, the win rate, and the worst single-network
deterioration.  The ledger is full of variants that looked good on a mean and
reversed on a holdout (C06, C41, C43, M51, M73), so a mean gain whose bootstrap
interval touches 1.0 is not a result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parent / "results" / "layer2_calib_full50.json"
)
data = json.loads(path.read_text())
records = data["records"]
names = sorted(records[0]["final_mse"])
base = np.array([r["final_mse"]["baseline"] for r in records])
n = len(records)
rng = np.random.default_rng(0)
idx = rng.integers(0, n, size=(20000, n))

print(f"{len(records)} networks, IDs {records[0]['index']}--{records[-1]['index']}")
print(f"baseline mean {base.mean():.4e}   median {np.median(base):.4e}\n")
print(f"{'variant':<16}{'mean ratio':>12}{'bootstrap 95%':>22}"
      f"{'wins':>8}{'worst net':>11}")
print("-" * 71)

rows = []
for v in names:
    if v == "baseline":
        continue
    x = np.array([r["final_mse"][v] for r in records])
    ratio = x.mean() / base.mean()
    boot = x[idx].mean(1) / base[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    wins = int((x < base).sum())
    worst = float((x / base).max())
    rows.append((v, ratio, lo, hi, wins, worst))

for v, ratio, lo, hi, wins, worst in sorted(rows, key=lambda r: r[1]):
    flag = "" if hi < 1.0 else "   <- interval includes no gain"
    print(f"{v:<16}{ratio:12.4f}   [{lo:6.4f}, {hi:6.4f}]"
          f"{wins:8d}/{len(records)}{worst:11.2f}x{flag}")

print("\nGRADED-score projection (ratios transfer; absolute local MSE does not)")
graded = 2.2566e-7
best = min(rows, key=lambda r: r[1])
print(f"  best graded Kerdock submission        {graded:.4e}")
print(f"  best variant here ({best[0]}, ratio {best[1]:.4f})")
print(f"    -> projected adjusted              {graded*best[1]:.4e}")
print(f"    -> bootstrap range                 {graded*best[3]:.4e} .. {graded*best[2]:.4e}")
print(f"  combined with A06 arithmetic (0.6453) {graded*best[1]*0.6453:.4e}")
