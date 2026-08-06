#!/usr/bin/env python3
"""Select validation configurations without touching the held-out test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def config_text(config: dict) -> str:
    return ",".join(f"{float(config[k]):g}" for k in ("alpha", "beta", "gamma", "corr_cap", "x_clip", "residual_clip"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path, nargs="+")
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--min-win", type=float, default=0.5)
    p.add_argument("--max-guard", type=float, default=0.5)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    candidates = []
    for path in args.results:
        r = json.loads(path.read_text())
        calibration = r.get("calibration")
        for key, entry in r["hybrid"].items():
            s = entry["summary"]
            if s["fraction_mlps_improved"] < args.min_win:
                continue
            if s.get("fraction_guard_activated", 0.0) > args.max_guard:
                continue
            candidates.append({
                "source": str(path),
                "calibration": calibration,
                "key": key,
                "config": entry["config"],
                "config_text": config_text(entry["config"]),
                "summary": s,
            })
    candidates.sort(key=lambda x: (x["summary"]["final_mean_mse"], -x["summary"]["fraction_mlps_improved"]))
    selected = candidates[: args.top_k]
    if not selected:
        raise SystemExit("no configurations survived selection constraints")
    result = {"selected": selected, "all_ranked": candidates}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(args.output), "selected": len(selected), "best": selected[0]}, indent=2))


if __name__ == "__main__":
    main()
