#!/usr/bin/env python3
"""Choose exactly one full-validation hybrid and emit a machine-readable plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("result", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-win", type=float, default=0.625)
    p.add_argument("--max-guard", type=float, default=0.5)
    args = p.parse_args()
    r = json.loads(args.result.read_text())
    ranked = []
    for key, entry in r["hybrid"].items():
        s = entry["summary"]
        if s["fraction_mlps_improved"] >= args.min_win and s.get("fraction_guard_activated", 0.0) <= args.max_guard:
            ranked.append((s["final_mean_mse"], key, entry))
    if not ranked:
        raise SystemExit("no full-validation config passed stability constraints")
    ranked.sort(key=lambda x: x[0])
    _, key, entry = ranked[0]
    c = entry["config"]
    config_text = ",".join(f"{float(c[k]):g}" for k in ("alpha", "beta", "gamma", "corr_cap", "x_clip", "residual_clip"))
    out = {
        "selected_key": key,
        "config": c,
        "config_text": config_text,
        "calibration": r.get("calibration"),
        "validation_summary": entry["summary"],
        "source": str(args.result),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
