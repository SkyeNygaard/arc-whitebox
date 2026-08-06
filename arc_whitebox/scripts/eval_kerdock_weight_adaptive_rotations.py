"""Evaluate deterministic rotations derived from each network's weights.

Fixed random orientations ignore the integrand.  This selection-only study
aligns the Kerdock/MUB rule with input-space metrics extracted from the first
weight and a diagonal approximation to downstream path sensitivity.

The adaptive maps are orthogonal, so every resulting rule remains a spherical
5-design.  No target value is used to construct a rotation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from eval_kerdock_adaptive_orientation import backward_squared_salience
from eval_kerdock_design import (
    WIDTH,
    forward_final,
    make_kerdock_design,
    random_rotation,
)
from eval_sampling_official import DEFAULT_DATA, _load_rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "kerdock_weight_adaptive_rotations.json"


def canonicalize_columns(matrix: np.ndarray) -> np.ndarray:
    """Fix eigenvector/QR signs by their largest-magnitude entry."""
    result = matrix.copy()
    pivots = np.argmax(np.abs(result), axis=0)
    signs = np.sign(result[pivots, np.arange(result.shape[1])])
    signs[signs == 0.0] = 1.0
    return result * signs[None, :]


def eigen_rotation(metric: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (metric.astype(np.float64) + metric.astype(np.float64).T)
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = canonicalize_columns(eigenvectors[:, order])
    return eigenvectors.T.astype(np.float32)


def adaptive_rotations(weights: np.ndarray) -> dict[str, np.ndarray]:
    first = weights[0].astype(np.float64)
    fixed = random_rotation(WIDTH, 3).astype(np.float64)

    input_metric = first @ first.T
    input_eigen = eigen_rotation(input_metric).astype(np.float64)

    salience = backward_squared_salience(weights)
    downstream_metric = (first * salience[None, :]) @ first.T
    downstream_eigen = eigen_rotation(downstream_metric).astype(np.float64)

    q, r = np.linalg.qr(first)
    q = canonicalize_columns(q)
    qr_rotation = q.T

    rotations = {
        "fixed_seed3": fixed,
        "input_eigen": input_eigen,
        "fixed_after_input_eigen": fixed @ input_eigen,
        "input_eigen_after_fixed": input_eigen @ fixed,
        "salience_eigen": downstream_eigen,
        "fixed_after_salience_eigen": fixed @ downstream_eigen,
        "salience_eigen_after_fixed": downstream_eigen @ fixed,
        "qr": qr_rotation,
        "fixed_after_qr": fixed @ qr_rotation,
    }
    return {
        name: rotation.astype(np.float32)
        for name, rotation in rotations.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=list(range(10)),
    )
    parser.add_argument("--modes", nargs="+")
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.indices or min(args.indices) < 0 or max(args.indices) >= 50:
        raise ValueError("adaptive rotations are restricted to IDs 0--49")

    points = make_kerdock_design()
    rows = _load_rows(args.data, args.indices)
    records: list[dict[str, object]] = []
    for index, (name, weights, targets) in zip(args.indices, rows, strict=True):
        rotations = adaptive_rotations(weights)
        if args.modes:
            missing = set(args.modes) - set(rotations)
            if missing:
                raise ValueError(f"unknown modes: {sorted(missing)}")
            rotations = {
                mode: rotation
                for mode, rotation in rotations.items()
                if mode in args.modes
            }
        for mode, rotation in rotations.items():
            prediction, seconds = forward_final(
                weights,
                points,
                args.chunk,
                rotation,
            )
            record = {
                "index": index,
                "name": name,
                "mode": mode,
                "seconds": seconds,
                "final_mse": float(
                    np.mean(np.square(prediction - targets[-1]))
                ),
                "prediction": prediction.tolist(),
            }
            records.append(record)
            print(
                {
                    "index": index,
                    "mode": mode,
                    "seconds": seconds,
                    "final_mse": record["final_mse"],
                },
                flush=True,
            )

    modes = sorted({str(record["mode"]) for record in records})
    summary = {
        mode: {
            "mean_final_mse": float(
                np.mean(
                    [
                        float(record["final_mse"])
                        for record in records
                        if record["mode"] == mode
                    ]
                )
            ),
            "median_final_mse": float(
                np.median(
                    [
                        float(record["final_mse"])
                        for record in records
                        if record["mode"] == mode
                    ]
                )
            ),
        }
        for mode in modes
    }
    payload = {
        "protocol": {
            "selection_indices": args.indices,
            "holdout_loaded": False,
            "target_free_rotation_construction": True,
        },
        "summary": summary,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print({"out": str(args.out), "summary": summary}, flush=True)


if __name__ == "__main__":
    main()
