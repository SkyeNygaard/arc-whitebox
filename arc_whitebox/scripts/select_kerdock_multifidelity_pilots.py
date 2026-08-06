"""Select small matched Kerdock rotation pilots on official IDs 0--49.

The estimator family is

    full(seed=3) + alpha * (
        (pilot(seed=0, S) - pilot(seed=3, S))
      + (pilot(seed=1, S) - pilot(seed=3, S))
    ) / 2.

Every pilot basis contains its 256 vectors and their antipodes.  The two
alternate rotations therefore add ``1024 * len(S)`` rows.  The seed-3 pilot
is a free subset reduction of the already-computed full design.

Candidates are filtered to the best quartile of both quartic leakage and
sixth-moment mismatch before target MSE is consulted.  This prevents a purely
empirical subset search from destroying the low-order design geometry.
This research script is deliberately hard-limited to IDs 0--49.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "results" / "kerdock_basis_selection_cache_0135.npz"
DEFAULT_MOMENTS = ROOT / "results" / "kerdock_moment_features_013.npz"
DEFAULT_OUT = ROOT / "results" / "kerdock_multifidelity_size_ladder.json"
BASES = 129


def sufficient_statistics(
    features: np.ndarray,
    residual: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    matrix = features[mask].transpose((1, 0, 2)).reshape((BASES, -1))
    target = residual[mask].reshape(-1)
    return matrix @ target, matrix @ matrix.T, float(target @ target), len(target)


def candidate_scores(
    indicator: np.ndarray,
    size: int,
    statistics: tuple[np.ndarray, np.ndarray, float, int],
    alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    correlation, gram, residual_ss, count = statistics
    summed_correlation = indicator @ correlation
    summed_norm = np.sum((indicator @ gram) * indicator, axis=1)
    if alpha is None:
        fitted_alpha = size * summed_correlation / summed_norm
    else:
        fitted_alpha = np.full(len(indicator), alpha)
    squared_error = (
        residual_ss
        - 2.0 * fitted_alpha * summed_correlation / size
        + np.square(fitted_alpha) * summed_norm / (size * size)
    )
    return squared_error / count, fitted_alpha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--moments", type=Path, default=DEFAULT_MOMENTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16, 20])
    parser.add_argument("--candidates", type=int, default=20_000)
    args = parser.parse_args()

    cache = np.load(args.cache)
    indices = cache["indices"]
    seeds = cache["seeds"].tolist()
    if not np.array_equal(indices, np.arange(50)):
        raise ValueError("selection cache must contain exactly IDs 0--49")
    seed_to_index = {int(seed): index for index, seed in enumerate(seeds)}
    predictions = cache["predictions"].astype(np.float64)
    targets = cache["targets"]
    full = predictions.mean(axis=2)
    baseline = full[:, seed_to_index[3]]
    residual = targets - baseline
    features = 0.5 * (
        predictions[:, seed_to_index[0]]
        + predictions[:, seed_to_index[1]]
        - 2.0 * predictions[:, seed_to_index[3]]
    )

    moments = np.load(args.moments)
    moment_seeds = moments["seeds"].tolist()
    moment_index = {int(seed): index for index, seed in enumerate(moment_seeds)}
    probes = moments["quartic"].shape[0]
    quartic = moments["quartic"].reshape((probes, len(moment_seeds), BASES))
    sixth = moments["sixth"].reshape((probes, len(moment_seeds), BASES))
    q_difference = 0.5 * (
        quartic[:, moment_index[0]]
        + quartic[:, moment_index[1]]
        - 2.0 * quartic[:, moment_index[3]]
    )
    s_difference = 0.5 * (
        sixth[:, moment_index[0]]
        + sixth[:, moment_index[1]]
        - 2.0 * sixth[:, moment_index[3]]
    )
    full_s_difference = s_difference.mean(axis=1)

    folds = np.arange(50) % 5
    all_mask = np.ones(50, dtype=bool)
    all_stats = sufficient_statistics(features, residual, all_mask)
    fold_stats = [
        (
            sufficient_statistics(features, residual, folds != fold),
            sufficient_statistics(features, residual, folds == fold),
        )
        for fold in range(5)
    ]
    baseline_mse = float(np.mean(np.square(residual)))
    rng = np.random.default_rng(2026072802)
    records = []
    for size in args.sizes:
        subsets = np.asarray(
            [rng.choice(BASES, size=size, replace=False) for _ in range(args.candidates)],
            dtype=np.int16,
        )
        indicator = np.zeros((args.candidates, BASES), dtype=np.float64)
        indicator[np.arange(args.candidates)[:, None], subsets] = 1.0

        q_pilot = q_difference @ (indicator.T / size)
        s_pilot = s_difference @ (indicator.T / size)
        q_rms = np.sqrt(np.mean(np.square(q_pilot), axis=0))
        s_mismatch = np.sqrt(
            np.mean(
                np.square(s_pilot - full_s_difference[:, None]),
                axis=0,
            )
        )
        admissible = (
            (q_rms <= np.quantile(q_rms, 0.25))
            & (s_mismatch <= np.quantile(s_mismatch, 0.25))
        )

        all_mse, all_alpha = candidate_scores(
            indicator,
            size,
            all_stats,
        )
        # Select by aggregate MSE only after the target-independent harmonic
        # screen.  Five-fold scores below refit only alpha, not the subset.
        admissible_indices = np.flatnonzero(admissible)
        chosen_index = int(
            admissible_indices[np.argmin(all_mse[admissible_indices])]
        )
        chosen_indicator = indicator[chosen_index : chosen_index + 1]
        fold_mses = []
        fold_alphas = []
        for train_stats, test_stats in fold_stats:
            _, train_alpha = candidate_scores(
                chosen_indicator,
                size,
                train_stats,
            )
            test_mse, _ = candidate_scores(
                chosen_indicator,
                size,
                test_stats,
                alpha=float(train_alpha[0]),
            )
            fold_mses.append(float(test_mse[0]))
            fold_alphas.append(float(train_alpha[0]))

        alpha = float(all_alpha[chosen_index])
        simple_alpha_candidates = np.asarray(
            [1 / 32, 3 / 64, 1 / 16, 5 / 64, 3 / 32, 7 / 64, 1 / 8]
        )
        simple_mses = {}
        for simple_alpha in simple_alpha_candidates:
            score, _ = candidate_scores(
                chosen_indicator,
                size,
                all_stats,
                alpha=float(simple_alpha),
            )
            simple_mses[str(float(simple_alpha))] = float(score[0])
        best_simple_alpha = min(simple_mses, key=simple_mses.get)

        records.append(
            {
                "bases_per_alternate_rotation": size,
                "extra_rows": 1024 * size,
                "total_rows": 66048 + 1024 * size,
                "subset": sorted(int(value) for value in subsets[chosen_index]),
                "candidate_index": chosen_index,
                "admissible_candidates": int(np.count_nonzero(admissible)),
                "quartic_rms": float(q_rms[chosen_index]),
                "quartic_percentile": float(
                    np.mean(q_rms <= q_rms[chosen_index])
                ),
                "sixth_mismatch_rms": float(s_mismatch[chosen_index]),
                "sixth_mismatch_percentile": float(
                    np.mean(s_mismatch <= s_mismatch[chosen_index])
                ),
                "fitted_alpha": alpha,
                "selection_mse": float(all_mse[chosen_index]),
                "selection_reduction_fraction": float(
                    1.0 - all_mse[chosen_index] / baseline_mse
                ),
                "fivefold_refit_alpha_mse": float(np.mean(fold_mses)),
                "fivefold_alphas": fold_alphas,
                "simple_alpha_mses": simple_mses,
                "best_simple_alpha": float(best_simple_alpha),
                "best_simple_alpha_mse": simple_mses[best_simple_alpha],
            }
        )
        print(records[-1], flush=True)

    result = {
        "protocol": {
            "split": "official IDs 0--49 only",
            "baseline_rotation_seed": 3,
            "alternate_rotation_seeds": [0, 1],
            "candidate_seed": 2026072802,
            "candidate_count_per_size": args.candidates,
            "harmonic_filter": (
                "intersection of best quartile quartic leakage and best "
                "quartile sixth-moment mismatch"
            ),
            "formula": (
                "F3 + alpha * ((P0_S-P3_S) + (P1_S-P3_S))/2"
            ),
        },
        "baseline_selection_mse": baseline_mse,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
