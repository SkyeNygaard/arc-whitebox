from __future__ import annotations

"""Reference contract for exporting the frozen V80 correction as a 1-D label.

This file intentionally does not recreate missing ARC helper modules. Run it in
the original V80 environment after wiring `compute_one` to the frozen blockwise
H3 implementation. It documents the exact label algebra required by Prompt 7.
"""

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np


def optimal_scale(baseline_error: np.ndarray, correction_direction: np.ndarray) -> float:
    # Candidate = baseline - scale * direction.
    den = float(np.dot(correction_direction, correction_direction))
    return float(np.dot(correction_direction, baseline_error) / max(den, 1e-30))


def export_v80(
    records: list[dict],
    output: Path,
) -> None:
    arrays = {
        "weights": np.stack([r["weights"] for r in records]).astype(np.float32),
        "base_network_id": np.asarray([r["base_network_id"] for r in records]),
        "rotation_id": np.asarray([r["rotation_id"] for r in records], dtype=np.int16),
        "baseline_error": np.stack([r["baseline"] - r["target"] for r in records]).astype(np.float32),
        # error(scale)=baseline_error - direction*scale
        "replay_jacobian": np.stack([-r["correction_direction"][:, None] for r in records]).astype(np.float32),
        "anchor_coeffs": np.full((len(records), 1), 0.25, dtype=np.float32),
        "target_coeffs": np.asarray([[optimal_scale(r["baseline"] - r["target"], r["correction_direction"])] for r in records], dtype=np.float32),
    }
    anchor_err = arrays["baseline_error"] + np.einsum("nod,nd->no", arrays["replay_jacobian"], arrays["anchor_coeffs"])
    arrays["target_confidence"] = (np.mean(anchor_err ** 2, axis=1) < np.mean(arrays["baseline_error"] ** 2, axis=1)).astype(np.float32)
    if arrays["target_coeffs"].shape[1] != 1:
        raise AssertionError("V80 must remain a scalar scale/sign target")
    np.savez_compressed(output, **arrays)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--records-json", type=Path, required=True, help="Upstream frozen records containing array file paths")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    spec = json.loads(args.records_json.read_text())
    records = []
    for row in spec["records"]:
        records.append({
            "base_network_id": row["base_network_id"],
            "rotation_id": row["rotation_id"],
            "weights": np.load(row["weights_npy"]),
            "baseline": np.load(row["baseline_npy"]),
            "target": np.load(row["target_npy"]),
            "correction_direction": np.load(row["correction_direction_npy"]),
        })
    export_v80(records, args.output)


if __name__ == "__main__":
    main()
