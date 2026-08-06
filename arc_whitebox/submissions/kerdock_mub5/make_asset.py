"""Generate the small Kerdock chirp and frozen rotation asset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from eval_kerdock_design import kerdock_chirp, random_rotation  # noqa: E402


def main() -> None:
    chirps = np.stack([kerdock_chirp(u) for u in range(128)])
    rotation = random_rotation(256, seed=3)
    path = HERE / "kerdock_mub5_seed3.npz"
    np.savez_compressed(path, chirps=chirps, rotation=rotation)
    print(path)
    print(
        {
            "chirps_shape": chirps.shape,
            "chirps_values": np.unique(chirps).tolist(),
            "rotation_shape": rotation.shape,
            "orthogonality_max_error": float(
                np.max(np.abs(rotation.T @ rotation - np.eye(256)))
            ),
        }
    )


if __name__ == "__main__":
    main()

