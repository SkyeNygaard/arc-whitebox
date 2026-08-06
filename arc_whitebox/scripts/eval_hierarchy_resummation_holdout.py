"""Apply frozen hierarchy-resummation parameters to fresh IDs 70--79."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))

from eval_hierarchy_resummation import collect, mean_mse  # noqa: E402


def main():
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.float32)
    selection = json.loads(
        (ROOT / "results" / "hierarchy_resummation_selection.json").read_text()
    )
    alpha = selection["richardson_alpha"]
    names = ("k1", "k2", "k3", "augment")
    weights = np.asarray(
        [selection["simplex_weights"][name] for name in names]
    )
    rows = collect(list(range(70, 80)))
    baseline_predictions = [r["levels"]["augment"] for r in rows]
    richardson_predictions = [
        r["levels"]["augment"]
        + alpha * (r["levels"]["augment"] - r["levels"]["k3"])
        for r in rows
    ]
    simplex_predictions = [
        sum(
            weight * r["levels"][name]
            for weight, name in zip(weights, names, strict=True)
        )
        for r in rows
    ]
    baseline = mean_mse(rows, baseline_predictions)
    richardson = mean_mse(rows, richardson_predictions)
    simplex = mean_mse(rows, simplex_predictions)
    artifact = {
        "ids": list(range(70, 80)),
        "frozen": {
            "richardson_alpha": alpha,
            "simplex_weights": selection["simplex_weights"],
        },
        "mean_mse": {
            "augment_baseline": baseline,
            "richardson": richardson,
            "simplex_hierarchy": simplex,
        },
        "ratios": {
            "richardson": richardson / baseline,
            "simplex_hierarchy": simplex / baseline,
        },
        "per_id": [
            {
                "mlp_id": row["mlp_id"],
                "augment_mse": float(
                    np.mean(
                        (
                            baseline_prediction - row["target"]
                        )
                        ** 2
                    )
                ),
                "richardson_mse": float(
                    np.mean(
                        (
                            richardson_prediction - row["target"]
                        )
                        ** 2
                    )
                ),
                "simplex_mse": float(
                    np.mean(
                        (
                            simplex_prediction - row["target"]
                        )
                        ** 2
                    )
                ),
            }
            for (
                row,
                baseline_prediction,
                richardson_prediction,
                simplex_prediction,
            ) in zip(
                rows,
                baseline_predictions,
                richardson_predictions,
                simplex_predictions,
                strict=True,
            )
        ],
    }
    output = ROOT / "results" / "hierarchy_resummation_holdout.json"
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
