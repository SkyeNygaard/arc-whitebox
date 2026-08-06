"""Frozen disjoint check for the random-plane angular proxy."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))

from eval_random_plane_quadrature import evaluate_method, load_npz  # noqa: E402


IDS = list(range(60, 65))
ANGLES = [4, 8, 16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=8192)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "random_plane_proxy_disjoint.json",
    )
    args = parser.parse_args()

    network_records = []
    for mlp_id in IDS:
        weights, target = load_npz(f"/tmp/phase1_mlp{mlp_id}.npz")
        records = [
            evaluate_method(
                weights,
                target,
                method=method,
                total=args.total,
                angles_per_plane=ANGLES[0],
                seeds=args.seeds,
                block_rows=args.total,
            )
            for method in ("iid_antithetic", "sobol_antithetic")
        ]
        records.extend(
            evaluate_method(
                weights,
                target,
                method="plane",
                total=args.total,
                angles_per_plane=angles,
                seeds=args.seeds,
                block_rows=args.total,
            )
            for angles in ANGLES
        )
        sobol_variance = next(
            r["across_seed_prediction_variance"]
            for r in records
            if r["method"] == "sobol_antithetic"
        )
        for record in records:
            record["variance_ratio_vs_sobol_antithetic"] = (
                record["across_seed_prediction_variance"]
                / sobol_variance
            )
        network_records.append({"mlp_id": mlp_id, "records": records})

    summaries = {}
    keys = [
        ("iid_antithetic", None),
        ("sobol_antithetic", None),
        *[("plane", angle) for angle in ANGLES],
    ]
    for method, angle in keys:
        selected = [
            next(
                record
                for record in network["records"]
                if record["method"] == method
                and record["angles_per_plane"] == angle
            )
            for network in network_records
        ]
        label = method if angle is None else f"plane_{angle}"
        summaries[label] = {
            "mean_variance_ratio_vs_sobol": statistics.fmean(
                r["variance_ratio_vs_sobol_antithetic"]
                for r in selected
            ),
            "median_variance_ratio_vs_sobol": statistics.median(
                r["variance_ratio_vs_sobol_antithetic"]
                for r in selected
            ),
            "mean_seconds": statistics.fmean(
                r["mean_seconds"] for r in selected
            ),
        }
    artifact = {
        "ids": IDS,
        "total_forward_directions": args.total,
        "seeds": args.seeds,
        "summaries": summaries,
        "networks": network_records,
        "conclusion_gate": (
            "Proceed to breakpoint-exact propagation only if a multi-angle "
            "plane beats scrambled Sobol-antithetic variance at equal forwards."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
