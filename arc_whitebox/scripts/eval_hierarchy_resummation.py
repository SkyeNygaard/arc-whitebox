"""Prototype Richardson/Padé resummation across kprop hierarchy levels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "mlp_cumulant_propagation" / "src"
sys.path.insert(0, str(VENDOR))

from mlp_kprop.kprop_harmonic import AUGMENT, SIMPLE, mlp_kprop  # noqa: E402
from mlp_kprop.mlp import MLP  # noqa: E402


def build_mlp(weights: np.ndarray) -> MLP:
    depth, width, _ = weights.shape
    mlp = MLP(
        input_dim=width,
        hidden_dim=width,
        output_dim=width,
        num_layers=depth + 1,
        nonlin="relu",
        init_kind="manual",
        w_var=[2.0] * depth + [1.0],
        b_var=0.0,
        b_mean=0.0,
    )
    with torch.no_grad():
        for layer, weight in enumerate(weights):
            mlp.Ws[layer].weight.copy_(torch.as_tensor(weight.T))
        mlp.Ws[-1].weight.copy_(torch.eye(width))
    return mlp


def predict(mlp, kmax, kind, factor):
    result = mlp_kprop(
        mlp,
        {1: torch.zeros(256), 2: torch.eye(256)},
        k_max=kmax,
        kind=kind,
        factor=factor,
        use_avg_metric=True,
        output_d_max=1,
    )
    return result[1].core.detach().cpu().numpy()


def collect(ids):
    rows = []
    for mlp_id in ids:
        data = np.load(f"/tmp/phase1_mlp{mlp_id}.npz")
        weights = np.asarray(data["weights"], dtype=np.float32)
        target = np.asarray(data["means"], dtype=np.float64)[-1]
        mlp = build_mlp(weights)
        start = time.perf_counter()
        levels = {
            "k1": predict(mlp, 1, SIMPLE, False),
            "k2": predict(mlp, 2, SIMPLE, False),
            "k3": predict(mlp, 3, SIMPLE, True),
            "augment": predict(mlp, 3, AUGMENT, True),
        }
        rows.append(
            {
                "mlp_id": mlp_id,
                "target": target,
                "levels": levels,
                "seconds": time.perf_counter() - start,
            }
        )
        print(json.dumps({"completed": mlp_id, "seconds": rows[-1]["seconds"]}), flush=True)
    return rows


def mean_mse(rows, predictions):
    return float(
        np.mean(
            [
                np.mean((prediction - row["target"]) ** 2)
                for row, prediction in zip(rows, predictions, strict=True)
            ]
        )
    )


def fit_simplex(rows):
    errors = np.stack(
        [
            np.stack(
                [
                    row["levels"][name] - row["target"]
                    for name in ("k1", "k2", "k3", "augment")
                ],
                axis=1,
            )
            for row in rows
        ]
    ).reshape(-1, 4)
    gram = errors.T @ errors / len(errors)
    result = minimize(
        lambda w: float(w @ gram @ w),
        np.full(4, 0.25),
        method="SLSQP",
        bounds=[(0, 1)] * 4,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"ftol": 1e-18, "maxiter": 1000},
    )
    return result.x


def stabilized_shanks(row, floor_fraction, clip_multiple):
    y1, y2, y3 = (
        row["levels"]["k1"],
        row["levels"]["k2"],
        row["levels"]["k3"],
    )
    d1 = y2 - y1
    d2 = y3 - y2
    denominator = d1 - d2
    floor = floor_fraction * (np.median(np.abs(d1)) + 1e-12)
    denominator = np.where(
        np.abs(denominator) < floor,
        np.where(denominator >= 0, floor, -floor),
        denominator,
    )
    correction = d1 * d1 / denominator
    limit = clip_multiple * (np.abs(d1) + 1e-12)
    correction = np.clip(correction, -limit, limit)
    return y1 + correction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", type=int, default=list(range(10)))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "hierarchy_resummation_selection.json",
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float32)

    rows = collect(args.ids)
    baseline = mean_mse(rows, [r["levels"]["augment"] for r in rows])
    direction = np.concatenate(
        [r["levels"]["augment"] - r["levels"]["k3"] for r in rows]
    )
    error = np.concatenate(
        [r["levels"]["augment"] - r["target"] for r in rows]
    )
    alpha = float(np.clip(-(error @ direction) / (direction @ direction), -2, 2))
    richardson = [
        r["levels"]["augment"]
        + alpha * (r["levels"]["augment"] - r["levels"]["k3"])
        for r in rows
    ]
    weights = fit_simplex(rows)
    level_names = ("k1", "k2", "k3", "augment")
    simplex = [
        sum(w * r["levels"][name] for w, name in zip(weights, level_names, strict=True))
        for r in rows
    ]
    shanks_candidates = []
    for floor in (0.1, 0.25, 0.5, 1.0):
        for clip in (1.0, 2.0, 4.0):
            predictions = [stabilized_shanks(r, floor, clip) for r in rows]
            shanks_candidates.append(
                {
                    "floor_fraction": floor,
                    "clip_multiple": clip,
                    "mean_mse": mean_mse(rows, predictions),
                }
            )
    best_shanks = min(shanks_candidates, key=lambda x: x["mean_mse"])
    methods = {
        "augment_baseline": baseline,
        "richardson": mean_mse(rows, richardson),
        "simplex_hierarchy": mean_mse(rows, simplex),
        "best_stabilized_shanks": best_shanks["mean_mse"],
    }
    best_ratio = min(methods.values()) / baseline
    artifact = {
        "ids": args.ids,
        "methods": methods,
        "baseline": baseline,
        "best_ratio": best_ratio,
        "richardson_alpha": alpha,
        "simplex_weights": dict(zip(level_names, weights.tolist(), strict=True)),
        "best_shanks": best_shanks,
        "shanks_grid": shanks_candidates,
        "holdout_gate_passed": best_ratio <= 0.9,
        "per_id_baseline_mse": {
            str(r["mlp_id"]): float(
                np.mean((r["levels"]["augment"] - r["target"]) ** 2)
            )
            for r in rows
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
