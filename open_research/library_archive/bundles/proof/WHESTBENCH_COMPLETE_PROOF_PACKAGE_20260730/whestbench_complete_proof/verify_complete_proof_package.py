#!/usr/bin/env python3
"""Deterministic verifier for the WHestBench complete proof package.

This verifier checks the frozen reopened-path artifact hashes, recomputes the
reported terminal metrics from saved row-level arrays, checks the signed-probe
oracle/global summaries, and numerically cross-checks the exact spherical mean
formulas used in the mathematical appendix.

It does not replace the separate T22/T30 interval proof engines. Those are
imported as separately certified dependencies and are identified in the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import betainc, roots_jacobi
from scipy.integrate import quad

D = 256
RADIUS = math.sqrt(2.0) * math.exp(math.lgamma((D + 1) / 2) - math.lgamma(D / 2))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def close(a: float, b: float, atol: float = 5e-13, rtol: float = 5e-11) -> bool:
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def raw_mse(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean((pred - ref) ** 2))


def cross_mse(pred: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((pred - a) * (pred - b)))


def pooled_metrics(root: Path, seeds: list[int], key: str, alpha: float) -> dict[str, Any]:
    br: list[float] = []
    cr: list[float] = []
    bx: list[float] = []
    cx: list[float] = []
    ratios: list[float] = []
    for seed in seeds:
        with np.load(root / "results" / f"network_{seed}.npz") as z:
            base = z["base"].astype(np.float64)
            ref = z["ref"].astype(np.float64)
            ra = z["ref_a"].astype(np.float64)
            rb = z["ref_b"].astype(np.float64)
            pred0 = z[key].astype(np.float64)
            pred = base + alpha * (pred0 - base)
            b0 = raw_mse(base, ref)
            c0 = raw_mse(pred, ref)
            br.append(b0)
            cr.append(c0)
            bx.append(cross_mse(base, ra, rb))
            cx.append(cross_mse(pred, ra, rb))
            ratios.append(c0 / b0)
    return {
        "pooled_raw_ratio": float(sum(cr) / sum(br)),
        "pooled_cross_ratio": float(sum(cx) / sum(bx)),
        "mean_raw_ratio": float(np.mean(ratios)),
        "wins": int(np.sum(np.asarray(ratios) < 1.0)),
        "worst": float(np.max(ratios)),
        "per_seed": {str(s): float(r) for s, r in zip(seeds, ratios)},
    }


def verify_hashes(root: Path) -> dict[str, Any]:
    manifest = root / "SHA256SUMS.txt"
    checked = 0
    failures: list[dict[str, str]] = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        expected, rel = line.split(None, 1)
        rel = rel.lstrip("*").strip()
        path = root / rel
        if not path.exists():
            failures.append({"path": rel, "error": "missing"})
            continue
        actual = sha256(path)
        checked += 1
        if actual != expected:
            failures.append({"path": rel, "expected": expected, "actual": actual})
    return {"checked": checked, "failures": failures, "passed": not failures}


def signed_summary_check(root: Path, family: str) -> dict[str, Any]:
    stored = json.loads((root / "results" / "SIGNED_64_VALIDATION.json").read_text())[family]
    seeds = list(range(3004, 3068))
    br = []
    bc = []
    for seed in seeds:
        with np.load(root / "results" / f"network_{seed}.npz") as z:
            br.append(raw_mse(z["base"], z["ref"]))
            bc.append(cross_mse(z["base"], z["ref_a"], z["ref_b"]))
    cr = np.asarray(stored["fixed"]["candidate_raw"], dtype=np.float64)
    cc = np.asarray(stored["fixed"]["candidate_cross"], dtype=np.float64)
    ratio = float(cr.sum() / np.sum(br))
    cross = float(cc.sum() / np.sum(bc))
    wins = int(sum(r < 1 for r in stored["fixed"]["per_seed_raw_ratio"].values()))
    worst = float(max(stored["fixed"]["per_seed_raw_ratio"].values()))
    return {
        "pooled_raw_ratio": ratio,
        "pooled_cross_ratio": cross,
        "wins": wins,
        "worst": worst,
        "matches_stored": all([
            close(ratio, stored["fixed"]["pooled_raw_ratio"]),
            close(cross, stored["fixed"]["pooled_cross_ratio"]),
            wins == stored["fixed"]["wins_raw"],
            close(worst, stored["fixed"]["worst_raw_ratio"]),
        ]),
    }


def signed_oracle_check(root: Path, family: str) -> dict[str, Any]:
    key = "signed_network" if family == "network" else "signed_random"
    seeds = list(range(3004, 3068))
    br = []
    cr = []
    masses = []
    ratios = []
    for seed in seeds:
        with np.load(root / "results" / f"network_{seed}.npz") as z:
            dmat = z[key].astype(np.float64)
            base = z["base"].astype(np.float64)
            ref = z["ref"].astype(np.float64)
            err = ref - base
            gram = dmat @ dmat.T
            m = dmat.shape[0]
            scale = max(float(np.trace(gram)) / m, 1e-30)
            w = np.linalg.solve(gram + 1e-8 * scale * np.eye(m), dmat @ err)
            pred = base + w @ dmat
            b0 = raw_mse(base, ref)
            c0 = raw_mse(pred, ref)
            br.append(b0)
            cr.append(c0)
            ratios.append(c0 / b0)
            masses.append(float(np.sum(np.abs(w))))
    stored = json.loads((root / "results" / "SIGNED_64_VALIDATION.json").read_text())[family]
    got = {
        "pooled_raw_ratio": float(sum(cr) / sum(br)),
        "mean_ratio": float(np.mean(ratios)),
        "median_mass": float(np.median(masses)),
        "p90_mass": float(np.quantile(masses, 0.9)),
    }
    got["matches_stored"] = all([
        close(got["pooled_raw_ratio"], stored["oracle_pooled_raw_ratio"], atol=1e-11),
        close(got["mean_ratio"], stored["oracle_mean_ratio"], atol=1e-11),
        close(got["median_mass"], stored["oracle_median_mass"], atol=1e-10),
        close(got["p90_mass"], stored["oracle_p90_mass"], atol=1e-10),
    ])
    return got


def spherical_relu_mean_closed(b: float, rho: float = RADIUS, d: int = D) -> float:
    if rho == 0:
        return max(b, 0.0)
    if b >= rho:
        return b
    if b <= -rho:
        return 0.0
    s = -b / rho
    cd = math.exp(math.lgamma(d / 2) - 0.5 * math.log(math.pi) - math.lgamma((d - 1) / 2))
    beta = (d - 1) / 2
    if s == 0:
        tail = 0.5
    else:
        tail = 0.5 - 0.5 * math.copysign(1.0, s) * float(betainc(0.5, beta, s * s))
    return rho * cd / (d - 1) * (1 - s * s) ** beta + b * tail


def exact_mean_cross_checks() -> dict[str, Any]:
    alpha = (D - 3) / 2
    nodes, weights = roots_jacobi(512, alpha, alpha)
    weights = weights / weights.sum()
    # The normalized spherical Poisson kernel must integrate to one.
    poisson_errors = {}
    for r in (0.03, 0.10, 0.20):
        vals = (1 - r * r) / (1 - 2 * r * nodes + r * r) ** (D / 2)
        sym = 0.5 * (vals + (1 - r * r) / (1 + 2 * r * nodes + r * r) ** (D / 2))
        poisson_errors[str(r)] = abs(float(weights @ sym) - 1.0)
    relu_errors = {}
    cd = math.exp(math.lgamma(D / 2) - 0.5 * math.log(math.pi) - math.lgamma((D - 1) / 2))
    for b in (-2.0, -0.5, 0.0, 0.5, 2.0):
        threshold = max(-1.0, min(1.0, -b / RADIUS))
        numeric = quad(
            lambda t: max(RADIUS * t + b, 0.0) * cd * (1 - t * t) ** ((D - 3) / 2),
            threshold, 1.0, epsabs=1e-13, epsrel=1e-13, limit=200
        )[0]
        closed = spherical_relu_mean_closed(b)
        relu_errors[str(b)] = abs(numeric - closed)
    return {
        "radius": RADIUS,
        "poisson_quadrature_abs_errors": poisson_errors,
        "relu_closed_form_abs_errors": relu_errors,
        "passed": max(poisson_errors.values()) < 2e-10 and max(relu_errors.values()) < 2e-12,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "reopened_paths_repro_20260730")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "WHESTBENCH_PROOF_CERTIFICATE_20260730.json")
    args = parser.parse_args()
    root = args.root.resolve()

    terminal = json.loads((root / "results" / "TERMINAL_DESCENDANT_RESULTS.json").read_text())
    extension = json.loads((root / "results" / "PROJECTED_RELU_EXTENSION_RESULTS.json").read_text())
    poisson = pooled_metrics(root, list(range(3020, 3036)), "poisson_mid_r_0.1", terminal["candidates"]["poisson_mid_r_0.1"]["alpha"])
    relu16 = pooled_metrics(root, list(range(3020, 3036)), "nonlinear_k4_m64_b0_0.001", -2.0)
    relu48 = pooled_metrics(root, list(range(3020, 3068)), "nonlinear_k4_m64_b0_0.001", -2.0)

    checks = {
        "artifact_hashes": verify_hashes(root),
        "poisson_terminal": {**poisson, "matches_stored": close(poisson["pooled_raw_ratio"], terminal["candidates"]["poisson_mid_r_0.1"]["metrics"]["pooled_raw_ratio"])},
        "projected_relu_terminal_16": {**relu16, "matches_stored": close(relu16["pooled_raw_ratio"], terminal["candidates"]["nonlinear_k4_m64_b0_0.001"]["metrics"]["pooled_raw_ratio"])},
        "projected_relu_extension_48": {**relu48, "matches_stored": all([
            close(relu48["pooled_raw_ratio"], extension["pooled_raw_ratio"]),
            close(relu48["pooled_cross_ratio"], extension["pooled_cross_ratio"]),
            relu48["wins"] == extension["wins"],
            close(relu48["worst"], extension["worst"]),
        ])},
        "signed_network_fixed": signed_summary_check(root, "network"),
        "signed_random_fixed": signed_summary_check(root, "random"),
        "signed_network_oracle": signed_oracle_check(root, "network"),
        "signed_random_oracle": signed_oracle_check(root, "random"),
        "exact_mean_cross_checks": exact_mean_cross_checks(),
    }
    passed_flags = []
    for name, item in checks.items():
        if name == "artifact_hashes":
            passed_flags.append(bool(item["passed"]))
        elif name == "exact_mean_cross_checks":
            passed_flags.append(bool(item["passed"]))
        else:
            passed_flags.append(bool(item.get("matches_stored", True)))
    certificate = {
        "certificate_version": 1,
        "scope": "Frozen reopened-path binary artifacts and analytic mean cross-checks; excludes separate T22/T30 interval engines.",
        "root": str(root),
        "checks": checks,
        "all_checks_passed": all(passed_flags),
        "mathematical_gate_results": {
            "poisson_terminal_raw_ratio_gt_1": poisson["pooled_raw_ratio"] > 1.0,
            "projected_relu_48_raw_ratio_gt_1": relu48["pooled_raw_ratio"] > 1.0,
            "projected_relu_adjusted_ratio": extension["flop_audit"]["projected_adjusted_ratio"],
            "projected_relu_adjusted_ratio_gt_1": extension["flop_audit"]["projected_adjusted_ratio"] > 1.0,
            "signed_network_fixed_raw_ratio_gt_1": checks["signed_network_fixed"]["pooled_raw_ratio"] > 1.0,
            "signed_network_oracle_raw_ratio_lt_competition_target": checks["signed_network_oracle"]["pooled_raw_ratio"] < 1 / 4.34,
        },
    }
    args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(certificate, indent=2, sort_keys=True))
    if not certificate["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
