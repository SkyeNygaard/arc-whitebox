#!/usr/bin/env python3
"""Adversarial synthetic audit for WHestBench layer-31 anchor claims.

This does NOT reproduce M146. It tests theorem implications and whether a scalar
Euclidean anchor-error threshold can be invariant to downstream direction or
ReLU gate geometry.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def unit(x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n <= eps:
        raise ValueError("cannot normalize near-zero vector")
    return x / n


def random_orthogonal(rng: np.random.Generator, d: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def apply_j(v_basis: np.ndarray, singular: np.ndarray, x: np.ndarray) -> np.ndarray:
    return singular * (v_basis.T @ x)


def fit_m146_quadratic() -> Dict[str, object]:
    eps = np.array([0.0, 5e-4, 1e-3, 2e-3], dtype=float)
    gains = np.array([41.2, 1.32, 0.34, 0.086], dtype=float)
    ratios = 1.0 / gains
    q = ratios[0]
    x = eps[1:] ** 2
    y = ratios[1:] - q
    a = float(np.dot(x, y) / np.dot(x, x))
    predicted_ratios = q + a * eps**2
    predicted_gains = 1.0 / predicted_ratios
    ss_res = float(np.sum((ratios - predicted_ratios) ** 2))
    ss_tot = float(np.sum((ratios - np.mean(ratios)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    break_even = math.sqrt((1.0 - q) / a)
    pointwise_a = [None] + [float((ratios[i] - q) / eps[i] ** 2) for i in range(1, len(eps))]
    return {
        "epsilon": eps.tolist(),
        "reported_gain": gains.tolist(),
        "reported_candidate_base_ratio": ratios.tolist(),
        "exact_anchor_residual_fraction_q": q,
        "fitted_quadratic_coefficient_a": a,
        "pointwise_a": pointwise_a,
        "predicted_gain": predicted_gains.tolist(),
        "predicted_candidate_base_ratio": predicted_ratios.tolist(),
        "r_squared": r2,
        "fitted_break_even_epsilon": break_even,
    }


def linear_direction_audit(
    rng: np.random.Generator,
    n_networks: int,
    d: int,
    eps_design: float,
    exact_gain: float,
    eval_epsilon: float,
) -> Tuple[list[dict], dict]:
    rows: list[dict] = []
    q = 1.0 / exact_gain
    condition_numbers = []

    for net in range(n_networks):
        v = random_orthogonal(rng, d)
        # Randomized condition number from 30 to 300; this deliberately models
        # anisotropic downstream sensitivity without claiming ARC calibration.
        cond = float(10 ** rng.uniform(math.log10(30.0), math.log10(300.0)))
        singular = np.geomspace(1.0, 1.0 / cond, d)
        condition_numbers.append(cond)

        d_dir = unit(rng.standard_normal(d))
        defect = eps_design * d_dir
        s = apply_j(v, singular, defect)
        s2 = float(np.dot(s, s))
        r2 = q / (1.0 - q) * s2
        baseline = s2 + r2

        random_dir = unit(rng.standard_normal(d))
        # Construct stylized estimator residual directions with fixed cosine to
        # the true defect. They are synthetic adversaries, not recovered ARC rows.
        orth = unit(random_dir - float(np.dot(random_dir, d_dir)) * d_dir)
        analytic_dir = unit(0.50 * d_dir + math.sqrt(0.75) * orth)
        companion_dir = unit(-0.20 * d_dir + math.sqrt(0.96) * orth)
        jtjd = v @ ((singular**2) * (v.T @ d_dir))

        directions = {
            "actual_defect": d_dir,
            "opposite_defect": -d_dir,
            "isotropic": random_dir,
            "analytic_residual_synthetic": analytic_dir,
            "companion_residual_synthetic": companion_dir,
            "leading_downstream_singular": v[:, 0],
            "trailing_downstream_singular": v[:, -1],
            "defect_sensitivity_gradient": unit(jtjd),
        }

        for name, direction in directions.items():
            jv = apply_j(v, singular, direction)
            jv2 = float(np.dot(jv, jv))
            threshold = math.sqrt(s2 / jv2)
            xi = eval_epsilon * direction
            n = apply_j(v, singular, xi)
            n2 = float(np.dot(n, n))
            cand = r2 + n2
            gain = baseline / cand
            u = s + n
            su = float(np.dot(s, u))
            u2 = float(np.dot(u, u))
            correction_cos = su / math.sqrt(max(1e-300, baseline * u2))
            downstream_ratio = math.sqrt(n2 / s2)
            rows.append(
                {
                    "network": net,
                    "condition_number": cond,
                    "direction": name,
                    "euclidean_error_epsilon": eval_epsilon,
                    "break_even_epsilon": threshold,
                    "threshold_over_design_error": threshold / eps_design,
                    "downstream_error_ratio_eta_J": downstream_ratio,
                    "candidate_base_ratio": cand / baseline,
                    "gain": gain,
                    "correction_cosine": correction_cos,
                }
            )

    summary: dict[str, dict[str, float]] = {}
    names = sorted({r["direction"] for r in rows})
    for name in names:
        vals = np.array([r["break_even_epsilon"] for r in rows if r["direction"] == name])
        ratios = np.array([r["candidate_base_ratio"] for r in rows if r["direction"] == name])
        cosines = np.array([r["correction_cosine"] for r in rows if r["direction"] == name])
        summary[name] = {
            "threshold_median": float(np.median(vals)),
            "threshold_p10": float(np.quantile(vals, 0.10)),
            "threshold_p90": float(np.quantile(vals, 0.90)),
            "threshold_min": float(np.min(vals)),
            "threshold_max": float(np.max(vals)),
            "candidate_base_ratio_median_at_eval_epsilon": float(np.median(ratios)),
            "correction_cosine_median_at_eval_epsilon": float(np.median(cosines)),
        }
    summary["global"] = {
        "condition_number_median": float(np.median(condition_numbers)),
        "condition_number_min": float(np.min(condition_numbers)),
        "condition_number_max": float(np.max(condition_numbers)),
        "eps_design": eps_design,
        "eval_epsilon": eval_epsilon,
        "exact_anchor_gain": exact_gain,
    }
    return rows, summary


def relu_remainder_audit(
    rng: np.random.Generator,
    n_networks: int,
    d: int,
    particles: int,
    eps_grid: Iterable[float],
) -> Tuple[list[dict], dict]:
    rows: list[dict] = []
    eps_grid = list(eps_grid)

    for net in range(n_networks):
        v = random_orthogonal(rng, d)
        cond = float(10 ** rng.uniform(math.log10(20.0), math.log10(150.0)))
        singular = np.geomspace(1.0, 1.0 / cond, d)

        defect_dir = unit(rng.standard_normal(d))
        isotropic = unit(rng.standard_normal(d))
        # v[:, 0] maps exclusively to the most sensitive output coordinate,
        # which is made kink-rich below.
        kink_focused = v[:, 0]
        directions = {
            "actual_defect": defect_dir,
            "isotropic": isotropic,
            "kink_focused": kink_focused,
        }

        base_h = rng.standard_normal((particles, d))
        # A structured near-kink regime: output coordinate 0 has 30% of particles
        # within ~1e-4 of zero; coordinates 1:16 have 5% near zero.
        kink_h = base_h.copy()
        mask0 = rng.random(particles) < 0.30
        kink_h[mask0, 0] = rng.normal(0.0, 1e-4, int(mask0.sum()))
        mask_block = rng.random((particles, 15)) < 0.05
        replacements = rng.normal(0.0, 2e-4, (particles, 15))
        kink_h[:, 1:16] = np.where(mask_block, replacements, kink_h[:, 1:16])

        for regime, h in (("generic", base_h), ("kink_enriched", kink_h)):
            gates = h > 0
            for direction_name, direction in directions.items():
                wd = apply_j(v, singular, direction)
                for eps in eps_grid:
                    t = eps * wd
                    before = np.maximum(h, 0.0)
                    after = np.maximum(h + t[None, :], 0.0)
                    exact = np.mean(after - before, axis=0)
                    linear = np.mean(gates, axis=0) * t
                    remainder = exact - linear
                    crossings = np.mean((h > 0) != ((h + t[None, :]) > 0))
                    lin_norm = float(np.linalg.norm(linear))
                    rem_norm = float(np.linalg.norm(remainder))
                    exact_norm = float(np.linalg.norm(exact))
                    bound = np.mean(
                        np.abs(t)[None, :] * (np.abs(h) <= np.abs(t)[None, :]),
                        axis=0,
                    )
                    bound_norm = float(np.linalg.norm(bound))
                    rows.append(
                        {
                            "network": net,
                            "regime": regime,
                            "direction": direction_name,
                            "epsilon": eps,
                            "crossing_fraction": float(crossings),
                            "linear_shift_norm": lin_norm,
                            "exact_shift_norm": exact_norm,
                            "remainder_norm": rem_norm,
                            "remainder_over_linear": rem_norm / max(lin_norm, 1e-300),
                            "bound_norm": bound_norm,
                            "bound_ratio_actual": bound_norm / max(rem_norm, 1e-300),
                        }
                    )

    summary: dict[str, dict[str, float]] = {}
    keys = sorted({(r["regime"], r["direction"], r["epsilon"]) for r in rows})
    for regime, direction, eps in keys:
        selected = [r for r in rows if r["regime"] == regime and r["direction"] == direction and r["epsilon"] == eps]
        rem = np.array([r["remainder_over_linear"] for r in selected])
        cross = np.array([r["crossing_fraction"] for r in selected])
        bound = np.array([r["bound_ratio_actual"] for r in selected])
        summary[f"{regime}|{direction}|{eps:.1e}"] = {
            "remainder_over_linear_median": float(np.median(rem)),
            "remainder_over_linear_p90": float(np.quantile(rem, 0.90)),
            "crossing_fraction_median": float(np.median(cross)),
            "bound_ratio_actual_min": float(np.min(bound)),
        }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("no rows")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_internal_consistency(out: Path, fit: dict) -> None:
    eps = np.array(fit["epsilon"])
    gains = np.array(fit["reported_gain"])
    grid = np.linspace(0, 0.0022, 300)
    q = float(fit["exact_anchor_residual_fraction_q"])
    a = float(fit["fitted_quadratic_coefficient_a"])
    plt.figure(figsize=(7.0, 4.4))
    plt.plot(grid, 1.0 / (q + a * grid**2), label="quadratic-noise fit")
    plt.scatter(eps, gains, label="ledger headline points", zorder=3)
    plt.axhline(1.0, linewidth=1.0)
    plt.yscale("log")
    plt.xlabel("reported relative anchor perturbation")
    plt.ylabel("baseline / candidate MSE gain")
    plt.title("M146 headline numbers: internal quadratic consistency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_thresholds(out: Path, rows: list[dict]) -> None:
    order = [
        "leading_downstream_singular",
        "defect_sensitivity_gradient",
        "actual_defect",
        "opposite_defect",
        "isotropic",
        "analytic_residual_synthetic",
        "companion_residual_synthetic",
        "trailing_downstream_singular",
    ]
    data = [[r["break_even_epsilon"] for r in rows if r["direction"] == name] for name in order]
    labels = [
        "top singular",
        "JᵀJ defect",
        "actual defect",
        "opposite defect",
        "isotropic",
        "analytic-like",
        "companion-like",
        "bottom singular",
    ]
    plt.figure(figsize=(9.5, 5.0))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.axhline(5e-4, linewidth=1.0, label="5e-4 headline")
    plt.yscale("log")
    plt.ylabel("Euclidean relative error at linear break-even")
    plt.title("Equal-norm anchor errors have direction-dependent thresholds")
    plt.xticks(rotation=28, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_relu(out: Path, rows: list[dict]) -> None:
    plt.figure(figsize=(8.0, 5.0))
    combinations = [
        ("generic", "actual_defect"),
        ("generic", "kink_focused"),
        ("kink_enriched", "actual_defect"),
        ("kink_enriched", "kink_focused"),
    ]
    for regime, direction in combinations:
        eps_vals = sorted({r["epsilon"] for r in rows if r["regime"] == regime and r["direction"] == direction})
        med = []
        for eps in eps_vals:
            vals = [r["remainder_over_linear"] for r in rows if r["regime"] == regime and r["direction"] == direction and r["epsilon"] == eps]
            med.append(float(np.median(vals)))
        plt.plot(eps_vals, med, marker="o", label=f"{regime}, {direction}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("translation magnitude")
    plt.ylabel("median ||ReLU remainder|| / ||linear shift||")
    plt.title("Gate-crossing remainder depends on kink geometry")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def sha256_manifest(directory: Path) -> None:
    lines = []
    for path in sorted(directory.iterdir()):
        if path.name == "MANIFEST.sha256" or not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    (directory / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/mnt/data/agent6_anchor_review"))
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--networks", type=int, default=60)
    parser.add_argument("--relu-networks", type=int, default=30)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--particles", type=int, default=4096)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    fit = fit_m146_quadratic()
    (args.output / "M146_INTERNAL_CONSISTENCY.json").write_text(json.dumps(fit, indent=2) + "\n")
    with (args.output / "M146_INTERNAL_CONSISTENCY.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epsilon", "reported_gain", "reported_ratio", "predicted_gain", "predicted_ratio"])
        for row in zip(fit["epsilon"], fit["reported_gain"], fit["reported_candidate_base_ratio"], fit["predicted_gain"], fit["predicted_candidate_base_ratio"]):
            writer.writerow(row)

    linear_rows, linear_summary = linear_direction_audit(
        rng=rng,
        n_networks=args.networks,
        d=args.dimension,
        eps_design=5.55e-4,
        exact_gain=41.2,
        eval_epsilon=5e-4,
    )
    write_csv(args.output / "SYNTHETIC_LINEAR_DIRECTION_ROWS.csv", linear_rows)
    (args.output / "SYNTHETIC_LINEAR_DIRECTION_SUMMARY.json").write_text(json.dumps(linear_summary, indent=2) + "\n")

    relu_rows, relu_summary = relu_remainder_audit(
        rng=rng,
        n_networks=args.relu_networks,
        d=args.dimension,
        particles=args.particles,
        eps_grid=[1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3],
    )
    write_csv(args.output / "SYNTHETIC_RELU_REMAINDER_ROWS.csv", relu_rows)
    (args.output / "SYNTHETIC_RELU_REMAINDER_SUMMARY.json").write_text(json.dumps(relu_summary, indent=2) + "\n")

    plot_internal_consistency(args.output / "m146_quadratic_consistency.png", fit)
    plot_thresholds(args.output / "direction_dependent_thresholds.png", linear_rows)
    plot_relu(args.output / "relu_gate_crossing_remainder.png", relu_rows)

    run_meta = {
        "seed": args.seed,
        "networks": args.networks,
        "relu_networks": args.relu_networks,
        "dimension": args.dimension,
        "particles": args.particles,
        "warning": "Synthetic adversarial audit only; not an M146 reproduction.",
    }
    (args.output / "RUN_METADATA.json").write_text(json.dumps(run_meta, indent=2) + "\n")
    sha256_manifest(args.output)

    print(json.dumps({
        "fit_break_even": fit["fitted_break_even_epsilon"],
        "fit_r_squared": fit["r_squared"],
        "actual_defect_threshold": linear_summary["actual_defect"],
        "top_singular_threshold": linear_summary["leading_downstream_singular"],
        "bottom_singular_threshold": linear_summary["trailing_downstream_singular"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
