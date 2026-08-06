"""Reproduce the canonical joint-scalar lane decision from packaged artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from joint_scalar_contract import ratio_for_retention, retained_oracle_improvement


def load(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_ratio(summary: dict[str, Any], key: str) -> float:
    row = summary[key]
    for field in ("aggregate_ratio", "aggregate_mse_ratio", "aggregate", "ratio"):
        if field in row:
            return float(row[field])
    raise KeyError((key, row))


def suffix_cost(flops: dict[str, Any], depth: int) -> float:
    """Interpolate the exact linear FLOP formula used by the packaged table."""

    rows = flops["suffix_table"]
    for row in rows:
        if int(row["suffix_depth"]) == depth:
            return float(row["total_B"])
    x = np.asarray([float(row["suffix_depth"]) for row in rows])
    overhead = np.asarray(
        [
            float(row["total_B"])
            - float(row["source_forward_B"])
            - float(row["source_projection_B"])
            - float(row["sparse_replay_B"])
            for row in rows
        ]
    )
    slope, intercept = np.polyfit(x, overhead, 1)
    a = flops["assumptions"]
    n = int(a["n"])
    h = int(a["source_rows"])
    support = int(a["support"])
    qrank = int(a["qrank"])
    rows_k = int(a["kerdock_rows"])
    source_forward = 2.0 * h * depth * n * n / 1e9
    projection = 2.0 * h * n * (depth + 1) * (support + qrank) / 1e9
    replay = 2.0 * rows_k * support * n / 1e9
    return float(source_forward + projection + replay + slope * depth + intercept)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--arc-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source_paths = {
        "component_ablation": args.sources / "lowerpilot_screen8_merged.json",
        "joint_pilot": args.sources / "joint_adjoint_pilot_screen8.json",
        "lower_holdout": args.sources / "lowerpilot_frozen_holdout16.json",
        "crossfit_lower": args.sources / "crossfit_lower_fast_screen8.json",
        "source_localization": args.sources / "adjoint_source_full8_merged.json",
        "sparse_ceiling": args.sources / "sparse_radial_highref8_merged.json",
        "toy_adjoint": args.sources / "agent2_sparse_adjoint_results.json",
    }
    data = {name: load(path) for name, path in source_paths.items()}

    exact = aggregate_ratio(data["component_ablation"]["summary"], "exact")
    lower = aggregate_ratio(data["component_ablation"]["summary"], "oracle_lower")
    connected = aggregate_ratio(data["component_ablation"]["summary"], "oracle_connected")
    fixed_name = str(data["joint_pilot"]["fixed_candidate"])
    fixed_joint = aggregate_ratio(data["joint_pilot"]["summary"], fixed_name)
    holdout_lower = float(data["lower_holdout"]["summary"]["aggregate_mse_ratio"])
    crossfit = aggregate_ratio(data["crossfit_lower"]["summary"], "a0.02")

    threshold70 = ratio_for_retention(exact, 0.70)
    threshold90 = ratio_for_retention(exact, 0.90)
    localization = data["source_localization"]["summary"]
    suffix = localization["suffix"]

    cost_rows = []
    under_budget_and_localized = []
    for support_key in ("flops_support8", "flops_support16", "flops_support32"):
        flops = data["toy_adjoint"][support_key]
        support = int(flops["assumptions"]["support"])
        for depth in (8, 12, 16, 24, 30):
            cost = suffix_cost(flops, depth)
            signed_fraction = (
                float(suffix[str(depth)]["median_signed"])
                if str(depth) in suffix
                else 1.0
            )
            row = {
                "support": support,
                "depth": depth,
                "added_compute_B": cost,
                "median_signed_connected_fraction": signed_fraction,
                "under_14B": cost < 14.0,
                "at_least_70pct_signed_fraction": signed_fraction >= 0.70,
            }
            cost_rows.append(row)
            if row["under_14B"] and row["at_least_70pct_signed_fraction"]:
                under_budget_and_localized.append(row)

    checkpoint_files = {
        "q32_h24": args.arc_results / "adjoint_k3_flopscope_q32_h24.json",
        "q32_h27": args.arc_results / "adjoint_k3_flopscope_q32_h27.json",
        "q64_h27": args.arc_results / "adjoint_k3_flopscope_q64_h27.json",
        "holdout": args.arc_results / "adjoint_k3_full_holdout8.json",
    }
    checkpoint_data = {name: load(path) for name, path in checkpoint_files.items()}
    holdout_summary = checkpoint_data["holdout"]["summary"]
    checkpoint = {}
    for label in ("q32_h24", "q32_h27", "q64_h27"):
        prof = checkpoint_data[label]
        summary = holdout_summary[label]
        checkpoint[label] = {
            "flopscope_B": float(prof["flopscope_flops"]) / 1e9,
            "profiled_controls": int(prof["controls"]),
            "scope": str(prof["scope"]),
            "relative_error_vs_factorized": float(summary["relative_error_vs_factorized"]),
            "relative_error_vs_oracle": float(summary["relative_error_vs_oracle"]),
            "cosine_vs_factorized": float(summary["cosine_vs_factorized"]),
        }

    payload = {
        "terminal_state": "EXTERNALLY_BLOCKED_WITH_TESTED_SUBFAMILIES_FAILED",
        "scientific_decision": {
            "reject_now": [
                "ordinary short-suffix connected-source regeneration",
                "one-basis lower-moment pilot",
                "same-design fold-crossfit lower anchor",
                "tested Gaussian connected source",
                "tested joint lower-plus-Gaussian-connected pilot",
                "4096-row independent full-depth source stream under the 14B ceiling",
            ],
            "still_open_only": [
                "a full-depth analytic/shared-arithmetic joint-scalar recurrence",
                "an inherited checkpoint that estimates the same joint scalar state, not only factorized connected c21",
            ],
            "reason_not_global_fail": (
                "The packaged artifacts omit official/fresh weights, high-precision reference moments, "
                "frozen sparse arrays, and three helper modules needed for the exact composed-control runner."
            ),
        },
        "hard_gate": {
            "exact_anchor_ratio": exact,
            "minimum_retention": 0.70,
            "candidate_ratio_threshold_70pct": threshold70,
            "preferred_ratio_threshold_90pct": threshold90,
            "hard_added_compute_B": 14.0,
            "preferred_added_compute_B": 10.0,
        },
        "component_ablation": {
            "exact": {
                "ratio": exact,
                "retained_exact_improvement": 1.0,
            },
            "lower_only": {
                "ratio": lower,
                "retained_exact_improvement": retained_oracle_improvement(lower, exact),
            },
            "connected_only": {
                "ratio": connected,
                "retained_exact_improvement": retained_oracle_improvement(connected, exact),
            },
            "tested_fixed_joint": {
                "name": fixed_name,
                "ratio": fixed_joint,
                "retained_exact_improvement": retained_oracle_improvement(fixed_joint, exact),
            },
            "frozen_one_basis_lower_holdout": {
                "ratio": holdout_lower,
                "retained_exact_improvement": retained_oracle_improvement(holdout_lower, exact),
                "wins": int(data["lower_holdout"]["summary"]["wins"]),
                "networks": int(len(data["lower_holdout"]["records"])),
                "worst": float(data["lower_holdout"]["summary"]["worst"]),
            },
            "same_design_crossfit_lower": {
                "ratio": crossfit,
                "retained_exact_improvement": retained_oracle_improvement(crossfit, exact),
            },
        },
        "source_localization": {
            "median_last8_signed_fraction": float(suffix["8"]["median_signed"]),
            "median_last16_signed_fraction": float(suffix["16"]["median_signed"]),
            "median_last24_signed_fraction": float(suffix["24"]["median_signed"]),
            "median_effective_source_layers": float(localization["median_effective_source_layers"]),
            "median_rank4_frobenius_capture": float(localization["median_rank4_frob_capture"]),
            "max_terminal_identity_error": float(localization["max_identity_error"]),
        },
        "source_cost_frontier": {
            "rows": cost_rows,
            "under_14B_and_at_least_70pct_signed_fraction": under_budget_and_localized,
            "intersection_exists": bool(under_budget_and_localized),
            "caveat": (
                "Signed connected-source fraction is not identical to retained final MSE improvement; "
                "it is used here only as a necessary localization diagnostic."
            ),
        },
        "implemented_checkpoint_audit": {
            "profiles": checkpoint,
            "caveat": (
                "These profiles cover two connected controls and a factorized target. "
                "They do not include the full 32-128-scalar joint state, lower-order recurrence, "
                "or the unchanged composed-control replay."
            ),
        },
        "scalar_contract": {
            "probe_count": 32,
            "scalar_slots_before_deduplication": 128,
            "families": [
                "target mean projection z_p = E[v_p^T h]",
                "marginal second moment s_p = E[h_i^2]",
                "row-direction second moment u_p = E[h_i(v_p^T h)]",
                "cubic contraction r_p = E[h_i^2(v_p^T h)]",
            ],
            "composition": (
                "a_p = (r_p - (m^T v_p)s_p - 2m_i u_p + 2m_i^2 z_p)/(D+1)"
            ),
        },
        "missing_assets": [
            "official/fresh weight files used by the local ARC harness",
            "high-precision target moments/corpora",
            "frozen sparse probe arrays and coefficients for a fresh composed-control block",
            "sparse_adjoint_control.py",
            "sparse_crossfit_lower_fast.py",
            "sparse_lower_moment_pilot.py",
        ],
        "provenance_sha256": {
            **{str(path.name): sha256(path) for path in source_paths.values()},
            **{str(path.name): sha256(path) for path in checkpoint_files.values()},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
