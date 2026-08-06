"""Evaluate a completed joint-scalar candidate against the frozen rejection gate.

Input schema (JSON):
{
  "protocol": {
    "candidate_name": "...",
    "fresh_block": true,
    "target_scalar_count": 128,
    "added_compute_B": 9.8,
    "residual_wall_seconds": 0.0,
    "all_costs_included": true
  },
  "records": [
    {
      "network_id": 200,
      "baseline_mse": 1.0e-7,
      "exact_anchor_mse": 2.2e-8,
      "candidate_mse": 4.0e-8,
      "correction_cosine": 0.5
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--hard-compute-B", type=float, default=14.0)
    parser.add_argument("--minimum-retention", type=float, default=0.70)
    parser.add_argument("--max-worst-ratio", type=float, default=1.10)
    args = parser.parse_args()

    payload: dict[str, Any] = json.loads(args.input.read_text())
    protocol = payload.get("protocol", {})
    records = payload.get("records", [])
    require(bool(records), "records must be non-empty")
    require(protocol.get("fresh_block") is True, "candidate must use a fresh immutable block")
    require(protocol.get("all_costs_included") is True, "cost must include recurrence, composition, replay, and overhead")
    require(int(protocol.get("target_scalar_count", 0)) in range(32, 129), "target scalar count must be 32-128")

    baseline = np.asarray([float(r["baseline_mse"]) for r in records])
    exact = np.asarray([float(r["exact_anchor_mse"]) for r in records])
    candidate = np.asarray([float(r["candidate_mse"]) for r in records])
    cosine = np.asarray([float(r["correction_cosine"]) for r in records])
    require(np.all(baseline > 0), "baseline MSE must be positive")
    require(np.all(exact >= 0) and np.all(candidate >= 0), "MSEs must be non-negative")
    require(np.all(np.isfinite(cosine)), "correction cosines must be finite")

    exact_ratio = float(np.sum(exact) / np.sum(baseline))
    candidate_ratio = float(np.sum(candidate) / np.sum(baseline))
    retained = float((1.0 - candidate_ratio) / max(1.0 - exact_ratio, 1e-30))
    per_network_ratio = candidate / baseline
    added_compute = float(protocol.get("added_compute_B", np.inf))
    residual_seconds = float(protocol.get("residual_wall_seconds", 0.0))
    # Challenge effective compute convention: 1 second residual = 100B FLOPs.
    effective_added_compute = added_compute + 100.0 * residual_seconds

    checks = {
        "retains_at_least_required_exact_improvement": retained + 1e-12 >= args.minimum_retention,
        "effective_added_compute_below_hard_ceiling": effective_added_compute < args.hard_compute_B,
        "positive_mean_correction_cosine": float(np.mean(cosine)) > 0.0,
        "safe_worst_network": float(np.max(per_network_ratio)) <= args.max_worst_ratio,
        "wins_on_majority": int(np.sum(per_network_ratio < 1.0)) > len(records) / 2,
    }
    result = {
        "candidate_name": protocol.get("candidate_name", "unnamed"),
        "networks": len(records),
        "exact_anchor_ratio": exact_ratio,
        "candidate_ratio": candidate_ratio,
        "retained_exact_improvement": retained,
        "added_compute_B": added_compute,
        "residual_wall_seconds": residual_seconds,
        "effective_added_compute_B": effective_added_compute,
        "mean_correction_cosine": float(np.mean(cosine)),
        "wins": int(np.sum(per_network_ratio < 1.0)),
        "worst_ratio": float(np.max(per_network_ratio)),
        "checks": checks,
        "decision": "PASS" if all(checks.values()) else "FAIL",
    }
    text = json.dumps(result, indent=2) + "\n"
    print(text, end="")
    if args.out is not None:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
