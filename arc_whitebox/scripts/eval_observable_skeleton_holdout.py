"""Apply frozen observable-skeleton coefficients to official IDs 80--89."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from eval_observable_skeleton import collect_one, mse  # noqa: E402


def main():
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.float32)
    selection = json.loads(
        (ROOT / "results" / "observable_skeleton_selection.json").read_text()
    )
    frozen = {
        key: np.asarray(
            [
                method["coefficients"]["k3"],
                method["coefficients"]["k4"],
            ]
        )
        for key, method in selection["summary"]["methods"].items()
    }
    rows = []
    for mlp_id in range(80, 90):
        row = collect_one(mlp_id)
        rows.append(row)
        print(
            json.dumps(
                {
                    "completed": mlp_id,
                    "seconds": row["seconds"],
                    "baseline_mse": mse(row, row["baseline"]),
                }
            ),
            flush=True,
        )
    baseline_mses = [mse(row, row["baseline"]) for row in rows]
    methods = {}
    per_id = []
    for key, coefficient in frozen.items():
        method_mses = []
        correlations = []
        for row in rows:
            design = np.stack(
                [row["signals"][key]["k3"], row["signals"][key]["k4"]],
                axis=1,
            )
            signal = design @ coefficient
            residual = row["target"] - row["baseline"]
            method_mses.append(mse(row, row["baseline"] + signal))
            correlations.append(
                float(
                    np.corrcoef(signal, residual)[0, 1]
                    if np.std(signal) > 0 and np.std(residual) > 0
                    else 0.0
                )
            )
        methods[key] = {
            "frozen_coefficients": {
                "k3": float(coefficient[0]),
                "k4": float(coefficient[1]),
            },
            "mean_mse": float(np.mean(method_mses)),
            "ratio": float(np.mean(method_mses) / np.mean(baseline_mses)),
            "improved_networks": int(
                sum(
                    method < baseline
                    for method, baseline in zip(
                        method_mses, baseline_mses, strict=True
                    )
                )
            ),
            "mean_within_network_correlation": float(np.mean(correlations)),
        }
        for row, baseline_mse, method_mse in zip(
            rows, baseline_mses, method_mses, strict=True
        ):
            per_id.append(
                {
                    "mlp_id": row["mlp_id"],
                    "method": key,
                    "baseline_mse": baseline_mse,
                    "corrected_mse": method_mse,
                    "ratio": method_mse / baseline_mse,
                }
            )
    artifact = {
        "ids": list(range(80, 90)),
        "baseline_mean_mse": float(np.mean(baseline_mses)),
        "methods": methods,
        "per_id": per_id,
        "selection_source": "observable_skeleton_selection.json",
        "parameters_frozen_before_holdout": True,
    }
    output = ROOT / "results" / "observable_skeleton_holdout.json"
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
