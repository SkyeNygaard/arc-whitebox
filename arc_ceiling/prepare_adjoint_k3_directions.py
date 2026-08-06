"""Prepare deployable late-layer directions for contracted-K3 experiments.

The expensive analytic K3 calculation does not choose its own observables.
They are the two dominant connected-C21 directions estimated from the Kerdock
activation cloud that the estimator has already evaluated.  This small helper
separates that ordinary NumPy experiment from the vendor PyTorch environment
used by ``eval_adjoint_contracted_k3.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "arc_whitebox"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from eval_crossfit_cumulant_control import (  # noqa: E402
    empirical_c21_state,
)
from eval_exact_anchor_residual import FULL_DATA  # noqa: E402
from eval_kerdock_design import WIDTH, make_kerdock_design, random_rotation  # noqa: E402
from eval_oracle_cumulant_bridge import (  # noqa: E402
    connected_m21,
    moment_path,
    truncated_svd,
)
from eval_sampling_official import _load_rows  # noqa: E402


def forward_with_gates(
    weights: np.ndarray,
    points: np.ndarray,
    rotation: np.ndarray,
    capture_layer: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward the cloud while retaining empirical ReLU response gates."""
    preactivation = points @ (rotation @ weights[0].astype(np.float32))
    gates = [np.mean(preactivation > 0.0, axis=0, dtype=np.float64)]
    activation = np.maximum(preactivation, 0.0)
    captured = activation.copy() if capture_layer == 0 else None
    for layer in range(1, len(weights)):
        preactivation = activation @ weights[layer]
        gates.append(
            np.mean(preactivation > 0.0, axis=0, dtype=np.float64)
        )
        activation = np.maximum(preactivation, 0.0)
        if layer == capture_layer:
            captured = activation.copy()
    if captured is None:
        raise ValueError(capture_layer)
    return captured, activation, np.stack(gates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(160, 168)))
    parser.add_argument("--layer", type=int, default=29)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--rotation-seed", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    points = make_kerdock_design()
    rotation = random_rotation(WIDTH, args.rotation_seed)
    rows = _load_rows(FULL_DATA, args.indices)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        started = time.perf_counter()
        activation, final, sample_gates = forward_with_gates(
            weights,
            points,
            rotation,
            args.layer,
        )
        sample_left, sample_right, sample_raw_m21 = empirical_c21_state(
            activation,
            args.rank,
        )
        sample_mean = np.mean(activation, axis=0, dtype=np.float64)
        with np.load(moment_path(index)) as moments:
            mean = np.asarray(moments["mean"][args.layer], dtype=np.float64)
            second = np.asarray(moments["M11"][args.layer], dtype=np.float64)
            raw_m21 = np.asarray(moments["M21"][args.layer], dtype=np.float64)
            marginal_second = np.asarray(
                moments["m2"][args.layer],
                dtype=np.float64,
            )
            oracle_c21 = connected_m21(
                mean,
                second,
                raw_m21,
                marginal_second,
            )
        oracle_left, oracle_right = truncated_svd(oracle_c21, args.rank)
        baseline = np.mean(final, axis=0, dtype=np.float64)
        output = args.out_dir / f"mlp_{index:05d}.npz"
        np.savez_compressed(
            output,
            global_index=index,
            name=name,
            layer=args.layer,
            rank=args.rank,
            rotation_seed=args.rotation_seed,
            sample_left=sample_left,
            sample_right=sample_right,
            sample_raw_m21=sample_raw_m21,
            sample_mean=sample_mean,
            sample_gates=sample_gates,
            oracle_left=oracle_left,
            oracle_right=oracle_right,
            oracle_c21=oracle_c21,
            oracle_mean=mean,
            oracle_second=second,
            baseline=baseline,
            target=np.asarray(targets[-1], dtype=np.float64),
        )
        print(
            f"[{index:>4}] wrote {output} "
            f"({time.perf_counter() - started:.2f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
