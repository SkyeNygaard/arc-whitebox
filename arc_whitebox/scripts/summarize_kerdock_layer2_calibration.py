"""Aggregate the selection-only layer-2 calibration sweep.

The underlying experiments were split into small files to keep iteration
fast.  This script merges them, computes paired statistics, and performs a
deterministic five-fold ``index % 5`` hyperparameter-selection audit.  It
never reads the official dataset, so no holdout row can be opened here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUTS = (
    ROOT / "results" / "kerdock_layer2_calibration.json",
    ROOT / "results" / "kerdock_layer2_calibration_extended_0_9.json",
    ROOT / "results" / "kerdock_layer2_calibration_lowstrength_10_49.json",
    ROOT / "results" / "kerdock_layer2_calibration_10_49.json",
)
OUT = ROOT / "results" / "kerdock_layer2_calibration_all50.json"
STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75)


def variant_name(strength: float) -> str:
    if strength == 0.0:
        return "baseline"
    return f"mean_variance_{strength:g}"


def load_scores() -> tuple[dict[int, dict[str, float]], dict[int, dict]]:
    scores: dict[int, dict[str, float]] = {}
    metadata: dict[int, dict] = {}
    for path in INPUTS:
        payload = json.loads(path.read_text())
        indices = payload["protocol"]["selection_indices"]
        if min(indices) < 0 or max(indices) >= 50:
            raise AssertionError(f"non-selection row in {path}")
        for record in payload["records"]:
            index = int(record["index"])
            metadata[index] = record["metadata"]
            destination = scores.setdefault(index, {})
            for name, value in record["final_mse"].items():
                prior = destination.get(name)
                if prior is not None and not math.isclose(
                    prior,
                    float(value),
                    rel_tol=1e-10,
                    abs_tol=1e-15,
                ):
                    raise AssertionError(
                        f"inconsistent {index=} {name=}: {prior} vs {value}"
                    )
                destination[name] = float(value)
    expected = {variant_name(strength) for strength in STRENGTHS}
    if set(scores) != set(range(50)):
        raise AssertionError(f"missing selection IDs: {set(range(50))-set(scores)}")
    for index, row in scores.items():
        missing = expected - set(row)
        if missing:
            raise AssertionError(f"ID {index} missing {sorted(missing)}")
    return scores, metadata


def sign_flip_pvalue(
    differences: np.ndarray,
    trials: int = 200_000,
) -> float:
    rng = np.random.default_rng(20260728)
    observed = abs(float(np.mean(differences)))
    exceed = 0
    batch = 10_000
    for _ in range(0, trials, batch):
        count = min(batch, trials - _)
        signs = rng.choice((-1.0, 1.0), size=(count, len(differences)))
        simulated = np.abs(np.mean(signs * differences[None, :], axis=1))
        exceed += int(np.count_nonzero(simulated >= observed))
    return (exceed + 1.0) / (trials + 1.0)


def summarize_strength(
    scores: dict[int, dict[str, float]],
    strength: float,
) -> dict[str, float | int]:
    baseline = np.asarray(
        [scores[index]["baseline"] for index in range(50)]
    )
    values = np.asarray(
        [scores[index][variant_name(strength)] for index in range(50)]
    )
    difference = values - baseline
    standard_error = float(
        np.std(difference, ddof=1) / math.sqrt(len(difference))
    )
    return {
        "mean_final_mse": float(np.mean(values)),
        "median_final_mse": float(np.median(values)),
        "mean_paired_delta": float(np.mean(difference)),
        "relative_mean_change": float(
            np.mean(values) / np.mean(baseline) - 1.0
        ),
        "paired_delta_standard_error": standard_error,
        "paired_delta_t": float(
            np.mean(difference) / standard_error
            if standard_error > 0.0
            else 0.0
        ),
        "networks_improved": int(np.count_nonzero(difference < 0.0)),
        "networks_worsened": int(np.count_nonzero(difference > 0.0)),
        "sign_flip_pvalue_two_sided": sign_flip_pvalue(difference),
    }


def cross_validation(
    scores: dict[int, dict[str, float]],
) -> dict[str, object]:
    folds = []
    heldout_values = []
    heldout_baselines = []
    chosen_strengths = []
    for fold in range(5):
        train = [index for index in range(50) if index % 5 != fold]
        test = [index for index in range(50) if index % 5 == fold]
        train_means = {
            strength: float(
                np.mean(
                    [
                        scores[index][variant_name(strength)]
                        for index in train
                    ]
                )
            )
            for strength in STRENGTHS
        }
        chosen = min(STRENGTHS, key=lambda value: train_means[value])
        chosen_strengths.append(chosen)
        test_values = [
            scores[index][variant_name(chosen)]
            for index in test
        ]
        test_baseline = [scores[index]["baseline"] for index in test]
        heldout_values.extend(test_values)
        heldout_baselines.extend(test_baseline)
        folds.append(
            {
                "fold": fold,
                "train_indices": train,
                "test_indices": test,
                "chosen_strength": chosen,
                "train_mean_by_strength": {
                    str(key): value for key, value in train_means.items()
                },
                "test_mean_final_mse": float(np.mean(test_values)),
                "test_baseline_mean_final_mse": float(
                    np.mean(test_baseline)
                ),
            }
        )
    return {
        "scheme": "five folds by selection_index modulo 5",
        "chosen_strengths": chosen_strengths,
        "folds": folds,
        "cross_validated_mean_final_mse": float(
            np.mean(heldout_values)
        ),
        "cross_validated_baseline_mean_final_mse": float(
            np.mean(heldout_baselines)
        ),
        "relative_change": float(
            np.mean(heldout_values) / np.mean(heldout_baselines) - 1.0
        ),
    }


def main() -> None:
    scores, metadata = load_scores()
    summaries = {
        str(strength): summarize_strength(scores, strength)
        for strength in STRENGTHS
    }
    frozen_strength = min(
        STRENGTHS,
        key=lambda strength: float(
            summaries[str(strength)]["mean_final_mse"]
        ),
    )
    cv = cross_validation(scores)
    canonical_records = []
    for index in range(50):
        canonical_records.append(
            {
                "index": index,
                "metadata": metadata[index],
                "final_mse_by_strength": {
                    str(strength): scores[index][variant_name(strength)]
                    for strength in STRENGTHS
                },
            }
        )
    payload = {
        "protocol": {
            "selection_indices": list(range(50)),
            "holdout_loaded": False,
            "rotation_seed": 3,
            "strength_grid": list(STRENGTHS),
            "strength_selected_once_globally": frozen_strength,
        },
        "transform": {
            "target": (
                "exact fixed-radius layer-2 preactivation mean and variance"
            ),
            "formula": (
                "h'=(h-m_s)*(sqrt(v_target/v_s)**alpha)"
                "+m_s+alpha*(m_target-m_s); then ReLU"
            ),
        },
        "summary_by_strength": summaries,
        "five_fold_selection_audit": cv,
        "recommendation": {
            "frozen_strength": frozen_strength,
            "principled_exact_match_strength": 1.0,
            "selection_mean_final_mse": summaries[str(frozen_strength)][
                "mean_final_mse"
            ],
            "selection_relative_change": summaries[str(frozen_strength)][
                "relative_mean_change"
            ],
            "reason": (
                "0.75 has the lowest all-50 selection mean and neighboring "
                "strengths 0.5 and 1.0 also improve. However, five-fold "
                "strength selection is slightly worse than baseline and the "
                "paired gain is not significant, so this is a holdout "
                "candidate rather than a robustly established replacement."
            ),
        },
        "records": canonical_records,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        {
            "out": str(OUT),
            "recommendation": payload["recommendation"],
            "cv": {
                key: value
                for key, value in cv.items()
                if key != "folds"
            },
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
