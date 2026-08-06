"""Explore rotated Kerdock-basis mixtures on the selection split only.

This is a research harness, not a submission.  It keeps the 129 antipodal
orthonormal bases separate after a dense forward, which lets us evaluate many
quadrature mixtures without rerunning the networks.

The moment geometry is probed with random one-dimensional contractions:

    q_B(z) = E_{x in B} (x dot z)^4
    s_B(z) = E_{x in B} (x dot z)^6.

For each complete Kerdock MUB set, the average q_B is the exact spherical
fourth moment.  Consequently, the centered quartic feature matrix exposes
whether unions of rotated sets contain nontrivial "MUB trades": changes of
basis weights that preserve the fourth moment.

Only official IDs 0--49 are accepted by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_kerdock_design import (  # noqa: E402
    N_BASES,
    WIDTH,
    make_kerdock_design,
    random_rotation,
)
from eval_sampling_official import DEFAULT_DATA, _load_rows  # noqa: E402


DEFAULT_CACHE = ROOT / "results" / "kerdock_basis_selection_cache.npz"
DEFAULT_OUT = ROOT / "results" / "kerdock_mixture_selection.json"


def forward_basis_means(
    weights: np.ndarray,
    points_by_basis: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """Return one final-layer mean per antipodal basis, shape (129, 256)."""
    effective_first = rotation @ weights[0]
    activation = points_by_basis.reshape((-1, WIDTH)) @ effective_first
    activation = np.maximum(activation, 0.0)
    for weight in weights[1:]:
        activation = np.maximum(activation @ weight, 0.0)
    return activation.reshape((N_BASES, -1, WIDTH)).mean(axis=1)


def build_prediction_cache(
    data: Path,
    seeds: list[int],
    indices: list[int],
    cache: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = _load_rows(data, indices)
    radius_points = make_kerdock_design()
    points_by_basis = radius_points.reshape((N_BASES, -1, WIDTH))
    basis_predictions = np.empty(
        (len(indices), len(seeds), N_BASES, WIDTH), dtype=np.float32
    )
    targets = np.empty((len(indices), WIDTH), dtype=np.float64)
    names: list[str] = []
    for network_index, (name, weights, layer_targets) in enumerate(rows):
        targets[network_index] = layer_targets[-1]
        names.append(name)
        for seed_index, seed in enumerate(seeds):
            started = time.perf_counter()
            basis_predictions[network_index, seed_index] = forward_basis_means(
                weights,
                points_by_basis,
                random_rotation(WIDTH, seed),
            )
            print(
                {
                    "index": indices[network_index],
                    "seed": seed,
                    "seconds": time.perf_counter() - started,
                },
                flush=True,
            )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        indices=np.asarray(indices),
        seeds=np.asarray(seeds),
        predictions=basis_predictions,
        targets=targets,
        names=np.asarray(names),
    )
    return basis_predictions, targets, names


def load_or_build(
    data: Path,
    seeds: list[int],
    indices: list[int],
    cache: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if cache.exists():
        payload = np.load(cache)
        if (
            np.array_equal(payload["indices"], indices)
            and np.array_equal(payload["seeds"], seeds)
        ):
            return (
                payload["predictions"],
                payload["targets"],
                payload["names"].tolist(),
            )
    return build_prediction_cache(data, seeds, indices, cache)


def moment_features(
    seeds: list[int],
    probes: int,
    probe_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return centered quartic and sixth contraction features per basis."""
    rng = np.random.default_rng(probe_seed)
    z = rng.standard_normal((WIDTH, probes))
    z /= np.linalg.norm(z, axis=0, keepdims=True)
    unit_points = make_kerdock_design(radius=1.0)[::2]
    # The stride above is not a line selector because bases are contiguous.
    # Reshape first, then take one vector from every antipodal pair block.
    full = make_kerdock_design(radius=1.0).reshape((N_BASES, 2, WIDTH, WIDTH))
    canonical_bases = full[:, 0]

    quartic_blocks = []
    sixth_blocks = []
    q_target = 3.0 / (WIDTH * (WIDTH + 2.0))
    s_target = 15.0 / (WIDTH * (WIDTH + 2.0) * (WIDTH + 4.0))
    for seed in seeds:
        rotated = canonical_bases.reshape((-1, WIDTH)) @ random_rotation(
            WIDTH, seed
        )
        contractions = (rotated @ z).reshape((N_BASES, WIDTH, probes))
        quartic_blocks.append(np.mean(contractions**4, axis=1).T - q_target)
        sixth_blocks.append(np.mean(contractions**6, axis=1).T - s_target)
    quartic = np.concatenate(quartic_blocks, axis=1)
    sixth = np.concatenate(sixth_blocks, axis=1)
    # Enforce the known exact set-average identities to remove finite-probe
    # roundoff.  The subtraction is mathematically zero for each probe.
    for block_index in range(len(seeds)):
        sl = slice(block_index * N_BASES, (block_index + 1) * N_BASES)
        quartic[:, sl] -= quartic[:, sl].mean(axis=1, keepdims=True)
    return quartic, sixth, {
        "quartic_target": q_target,
        "sixth_target": s_target,
        "unused_unit_points_shape_0": float(len(unit_points)),
    }


def mse(prediction: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.square(prediction - targets)))


def evaluate_weights(
    weights: np.ndarray,
    flat_predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    return mse(np.einsum("nsbo,sb->no", flat_predictions, weights), targets)


def optimize_simplex(
    basis_predictions: np.ndarray,
    targets: np.ndarray,
    quartic: np.ndarray,
    sixth: np.ndarray,
    quartic_penalty: float,
    sixth_penalty: float,
) -> tuple[np.ndarray, dict[str, float]]:
    networks, rotations, bases, outputs = basis_predictions.shape
    flat = basis_predictions.reshape((networks, rotations * bases, outputs))
    columns = rotations * bases
    initial = np.full(columns, 1.0 / columns)
    target_flat = targets.reshape(-1)
    design = flat.transpose((0, 2, 1)).reshape((-1, columns)).astype(np.float64)

    scale = 1.0 / len(target_flat)
    q_scale = 1.0 / max(1, quartic.shape[0])
    s_scale = 1.0 / max(1, sixth.shape[0])

    def objective(candidate: np.ndarray) -> tuple[float, np.ndarray]:
        residual = design @ candidate - target_flat
        q_residual = quartic @ candidate
        s_residual = sixth @ candidate
        value = (
            scale * residual @ residual
            + quartic_penalty * q_scale * (q_residual @ q_residual)
            + sixth_penalty * s_scale * (s_residual @ s_residual)
        )
        gradient = (
            2.0 * scale * (design.T @ residual)
            + 2.0
            * quartic_penalty
            * q_scale
            * (quartic.T @ q_residual)
            + 2.0
            * sixth_penalty
            * s_scale
            * (sixth.T @ s_residual)
        )
        return float(value), gradient

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=True,
        bounds=[(0.0, 1.0)] * columns,
        constraints=[
            {
                "type": "eq",
                "fun": lambda candidate: np.sum(candidate) - 1.0,
                "jac": lambda candidate: np.ones_like(candidate),
            }
        ],
        options={"maxiter": 1000, "ftol": 1e-18, "disp": False},
    )
    candidate = result.x
    return candidate.reshape((rotations, bases)), {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "selection_mse": evaluate_weights(
            candidate.reshape((rotations, bases)),
            basis_predictions,
            targets,
        ),
        "quartic_probe_rms": float(
            np.sqrt(np.mean(np.square(quartic @ candidate)))
        ),
        "sixth_probe_rms": float(
            np.sqrt(np.mean(np.square(sixth @ candidate)))
        ),
        "effective_bases_inverse_simpson": float(
            1.0 / np.sum(np.square(candidate))
        ),
        "nonzero_weights": int(np.count_nonzero(candidate > 1e-10)),
        "max_weight": float(np.max(candidate)),
    }


def cv_weight_fit(
    basis_predictions: np.ndarray,
    targets: np.ndarray,
    quartic: np.ndarray,
    sixth: np.ndarray,
    quartic_penalty: float,
    sixth_penalty: float,
) -> dict[str, float]:
    folds = np.arange(len(targets)) % 5
    train_scores = []
    test_scores = []
    for fold in range(5):
        train = folds != fold
        test = ~train
        weights, summary = optimize_simplex(
            basis_predictions[train],
            targets[train],
            quartic,
            sixth,
            quartic_penalty,
            sixth_penalty,
        )
        train_scores.append(summary["selection_mse"])
        test_scores.append(
            evaluate_weights(weights, basis_predictions[test], targets[test])
        )
    return {
        "mean_train_mse": float(np.mean(train_scores)),
        "mean_test_mse": float(np.mean(test_scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(50)))
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 3, 5, 8, 9, 10, 11]
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--moment-probes", type=int, default=512)
    args = parser.parse_args()
    if not args.indices or min(args.indices) < 0 or max(args.indices) > 49:
        raise ValueError("this selection-only harness accepts IDs 0--49")

    basis_predictions, targets, names = load_or_build(
        args.data,
        args.seeds,
        args.indices,
        args.cache,
    )
    full = basis_predictions.mean(axis=2)
    per_seed = {
        str(seed): mse(full[:, seed_index], targets)
        for seed_index, seed in enumerate(args.seeds)
    }
    equal_prefix = {
        str(count): mse(full[:, :count].mean(axis=1), targets)
        for count in range(1, len(args.seeds) + 1)
    }

    quartic, sixth, moment_meta = moment_features(
        args.seeds,
        args.moment_probes,
        probe_seed=20260728,
    )
    singular = np.linalg.svd(quartic, compute_uv=False)
    rank_threshold = singular[0] * 1e-10
    rank = int(np.count_nonzero(singular > rank_threshold))

    fits = {}
    # Penalties cover free empirical fitting through near-exact fourth moments.
    for quartic_penalty in [0.0, 1e2, 1e4, 1e6, 1e8]:
        key = f"q{quartic_penalty:.0e}_s0"
        weights, summary = optimize_simplex(
            basis_predictions,
            targets,
            quartic,
            sixth,
            quartic_penalty,
            0.0,
        )
        summary["cross_validation"] = cv_weight_fit(
            basis_predictions,
            targets,
            quartic,
            sixth,
            quartic_penalty,
            0.0,
        )
        summary["set_weight_totals"] = np.sum(weights, axis=1).tolist()
        fits[key] = summary

    result = {
        "protocol": {
            "split": "official IDs 0--49 only",
            "indices": args.indices,
            "names": names,
            "seeds": args.seeds,
        },
        "full_design_mse_by_seed": per_seed,
        "equal_full_design_prefix_ensemble_mse": equal_prefix,
        "moment_geometry": {
            **moment_meta,
            "probes": args.moment_probes,
            "quartic_feature_shape": list(quartic.shape),
            "quartic_feature_rank": rank,
            "quartic_feature_nullity": int(quartic.shape[1] - rank),
            "largest_singular_value": float(singular[0]),
            "smallest_singular_values": singular[-12:].tolist(),
        },
        "basis_weight_fits": fits,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
