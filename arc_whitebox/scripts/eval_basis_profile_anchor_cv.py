"""Grouped holdout screen for predicting a quadrature anchor error.

This is an identifiability diagnostic, not a submission component.  It asks
whether the 129 Kerdock basis means contain enough *target-free* information
to predict the error of their uniform average.  Network IDs, rather than
output coordinates, define the train/validation/test split.

The script compares:

* a symmetry-respecting invariant feature map;
* the two Kerdock basis orbits (128 MUB bases plus the coordinate basis);
* an unrestricted 129-coordinate ridge, included as an overfit diagnostic.

The cache was generated on selection IDs 0--49 and therefore this result is
only an analogue for a future late rank-4 cubic-feature cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "results" / "kerdock_basis_selection_cache_0135.npz"
DEFAULT_OUT = ROOT / "results" / "basis_profile_anchor_cv.json"
RIDGES = np.logspace(-8, 6, 29)


def invariant_features(basis: np.ndarray) -> np.ndarray:
    """Return sign-equivariant/exchangeable features for (..., 129) profiles."""
    ordinary = basis[..., :128]
    ordinary_mean = ordinary.mean(axis=-1)
    centered = ordinary - ordinary_mean[..., None]
    scale2 = np.mean(centered**2, axis=-1)
    eps = np.maximum(scale2, 1e-30)
    q10, q25, q75, q90 = np.quantile(
        centered, (0.10, 0.25, 0.75, 0.90), axis=-1
    )
    positive_tail = np.mean(
        np.partition(centered, -16, axis=-1)[..., -16:], axis=-1
    )
    negative_tail = np.mean(
        np.partition(centered, 15, axis=-1)[..., :16], axis=-1
    )
    return np.stack(
        (
            ordinary_mean,
            basis[..., 128] - ordinary_mean,
            np.mean(centered**3, axis=-1) / eps,
            np.mean(centered**5, axis=-1)
            / np.maximum(np.mean(centered**4, axis=-1), 1e-30),
            q90 + q10,
            q75 + q25,
            positive_tail + negative_tail,
            np.sqrt(eps),
            np.mean(centered**4, axis=-1) / np.maximum(eps**2, 1e-30),
        ),
        axis=-1,
    )


def standardized_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    ridge: float,
) -> np.ndarray:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-15] = 1.0
    a = (x_train - mean) / scale
    b = (x_eval - mean) / scale
    # The intercept is not penalized.
    a = np.column_stack((np.ones(len(a)), a))
    b = np.column_stack((np.ones(len(b)), b))
    penalty = np.eye(a.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(a.T @ a + penalty, a.T @ y_train)
    return b @ coef


def mse_by_network(
    correction: np.ndarray,
    target_correction: np.ndarray,
) -> np.ndarray:
    return np.mean((correction - target_correction) ** 2, axis=1)


def fit_family(
    features: np.ndarray,
    target_correction: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> dict[str, object]:
    flat_x = features.reshape((len(features), -1, features.shape[-1]))
    flat_y = target_correction.reshape((len(target_correction), -1))
    xt = flat_x[train].reshape((-1, flat_x.shape[-1]))
    yt = flat_y[train].reshape(-1)
    xv = flat_x[validation].reshape((-1, flat_x.shape[-1]))
    validation_scores = []
    for ridge in RIDGES:
        pred = standardized_ridge(xt, yt, xv, float(ridge))
        validation_scores.append(float(np.mean((pred - flat_y[validation].reshape(-1)) ** 2)))
    chosen = float(RIDGES[int(np.argmin(validation_scores))])
    fit_ids = np.concatenate((train, validation))
    xf = flat_x[fit_ids].reshape((-1, flat_x.shape[-1]))
    yf = flat_y[fit_ids].reshape(-1)
    pred_test = standardized_ridge(
        xf,
        yf,
        flat_x[test].reshape((-1, flat_x.shape[-1])),
        chosen,
    ).reshape((len(test), -1))
    base = mse_by_network(np.zeros_like(flat_y[test]), flat_y[test])
    fitted = mse_by_network(pred_test, flat_y[test])
    return {
        "ridge": chosen,
        "validation_mse": min(validation_scores),
        "test_baseline_mse": float(base.mean()),
        "test_fitted_mse": float(fitted.mean()),
        "test_ratio": float(fitted.mean() / base.mean()),
        "test_wins": int(np.sum(fitted < base)),
        "test_network_ratios": (fitted / base).tolist(),
    }


def five_fold_crossfit(
    features: np.ndarray,
    target_correction: np.ndarray,
) -> dict[str, object]:
    """Nested, network-grouped five-fold predictions for a stability check."""
    folds = [np.arange(start, start + 10) for start in range(0, 50, 10)]
    network_base = np.mean(target_correction**2, axis=1)
    network_fitted = np.empty(50, dtype=np.float64)
    chosen_ridges: list[float] = []
    for outer in range(5):
        test = folds[outer]
        validation = folds[(outer + 1) % 5]
        train = np.concatenate(
            [folds[index] for index in range(5) if index not in (outer, (outer + 1) % 5)]
        )
        flat_x = features.reshape((len(features), -1, features.shape[-1]))
        flat_y = target_correction.reshape((len(target_correction), -1))
        xt = flat_x[train].reshape((-1, flat_x.shape[-1]))
        yt = flat_y[train].reshape(-1)
        xv = flat_x[validation].reshape((-1, flat_x.shape[-1]))
        yv = flat_y[validation].reshape(-1)
        scores = [
            float(
                np.mean(
                    (
                        standardized_ridge(xt, yt, xv, float(ridge))
                        - yv
                    )
                    ** 2
                )
            )
            for ridge in RIDGES
        ]
        ridge = float(RIDGES[int(np.argmin(scores))])
        chosen_ridges.append(ridge)
        fit_ids = np.concatenate((train, validation))
        prediction = standardized_ridge(
            flat_x[fit_ids].reshape((-1, flat_x.shape[-1])),
            flat_y[fit_ids].reshape(-1),
            flat_x[test].reshape((-1, flat_x.shape[-1])),
            ridge,
        ).reshape((len(test), -1))
        network_fitted[test] = np.mean(
            (prediction - flat_y[test]) ** 2,
            axis=1,
        )
    return {
        "baseline_mse": float(network_base.mean()),
        "fitted_mse": float(network_fitted.mean()),
        "ratio": float(network_fitted.mean() / network_base.mean()),
        "wins": int(np.sum(network_fitted < network_base)),
        "chosen_ridges": chosen_ridges,
        "network_ratios": (network_fitted / network_base).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    payload = np.load(args.cache)
    seeds = payload["seeds"].astype(int)
    seed_position = int(np.flatnonzero(seeds == args.seed)[0])
    # (network, output, basis)
    basis = payload["predictions"][:, seed_position].transpose((0, 2, 1)).astype(np.float64)
    target = payload["targets"].astype(np.float64)
    uniform = basis.mean(axis=-1)
    target_correction = target - uniform

    # Frozen contiguous network blocks.  Outputs never cross network folds.
    train = np.arange(0, 30)
    validation = np.arange(30, 40)
    test = np.arange(40, 50)
    centered = basis - uniform[..., None]
    families = {
        "invariant": invariant_features(basis),
        "two_orbit": np.stack((uniform, basis[..., 128] - basis[..., :128].mean(axis=-1)), axis=-1),
        "unrestricted_129": np.concatenate((uniform[..., None], centered), axis=-1),
    }
    results = {
        name: {
            "frozen_30_10_10": fit_family(
                features, target_correction, train, validation, test
            ),
            "nested_five_fold": five_fold_crossfit(features, target_correction),
        }
        for name, features in families.items()
    }
    output = {
        "protocol": {
            "cache": str(args.cache),
            "rotation_seed": args.seed,
            "train_network_positions": train.tolist(),
            "validation_network_positions": validation.tolist(),
            "test_network_positions": test.tolist(),
            "network_grouped": True,
            "selection_cache_only": True,
            "interpretation": "analogue screen, not a fresh scientific claim",
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
