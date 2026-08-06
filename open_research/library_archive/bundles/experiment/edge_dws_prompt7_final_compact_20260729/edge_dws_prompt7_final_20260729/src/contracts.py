from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ALLOWED_KEYS = {
    "weights",
    "base_network_id",
    "rotation_id",
    "node_observables",
    "layer_observables",
    "baseline_error",
    "replay_jacobian",
    "anchor_coeffs",
    "target_coeffs",
    "target_confidence",
}
REQUIRED_KEYS = {
    "weights",
    "base_network_id",
    "rotation_id",
    "baseline_error",
    "replay_jacobian",
    "anchor_coeffs",
    "target_coeffs",
}


@dataclass
class Bundle:
    arrays: dict[str, np.ndarray]
    manifest: dict[str, Any]
    splits: dict[str, np.ndarray]

    @property
    def n(self) -> int:
        return int(self.arrays["weights"].shape[0])

    @property
    def label_dim(self) -> int:
        return int(self.arrays["target_coeffs"].shape[1])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _as_str_ids(x: np.ndarray) -> np.ndarray:
    return np.asarray([str(v) for v in x], dtype=object)


def load_bundle(data_path: Path, manifest_path: Path, split_path: Path) -> Bundle:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "frozen":
        raise RuntimeError("label manifest is not frozen")
    if manifest.get("data_sha256") != sha256_file(data_path):
        raise RuntimeError("label bundle SHA-256 mismatch")
    if manifest.get("surrogate", {}).get("verified") is not True:
        raise RuntimeError("exact replay surrogate is not verified")

    with np.load(data_path, allow_pickle=False) as z:
        keys = set(z.files)
        unknown = keys - ALLOWED_KEYS
        missing = REQUIRED_KEYS - keys
        if unknown:
            raise RuntimeError(f"unknown bundle keys: {sorted(unknown)}")
        if missing:
            raise RuntimeError(f"missing bundle keys: {sorted(missing)}")
        arrays = {k: np.asarray(z[k]) for k in z.files}

    w = arrays["weights"]
    if w.ndim != 4 or w.shape[1] != 32 or w.shape[2:] != (256, 256):
        raise RuntimeError(f"expected width-256 depth-32 weights, got {w.shape}")
    n = len(w)
    for k, v in arrays.items():
        if len(v) != n:
            raise RuntimeError(f"first dimension mismatch for {k}")
    d = arrays["target_coeffs"].shape[1]
    out_dim = arrays["baseline_error"].shape[1]
    if arrays["anchor_coeffs"].shape != (n, d):
        raise RuntimeError("anchor_coeffs shape mismatch")
    if arrays["replay_jacobian"].shape != (n, out_dim, d):
        raise RuntimeError("replay_jacobian shape mismatch")
    if not np.isfinite(w).all():
        raise RuntimeError("non-finite weights")
    for k in ("baseline_error", "replay_jacobian", "anchor_coeffs", "target_coeffs"):
        if not np.isfinite(arrays[k]).all():
            raise RuntimeError(f"non-finite {k}")

    split_doc = json.loads(split_path.read_text())
    if split_doc.get("status") != "frozen":
        raise RuntimeError("split registry is not frozen")
    split_hash = canonical_hash(split_doc.get("splits", {}))
    if split_doc.get("splits_sha256") != split_hash:
        raise RuntimeError("split registry hash mismatch")
    ids = _as_str_ids(arrays["base_network_id"])
    assignments: dict[str, str] = {}
    splits: dict[str, np.ndarray] = {}
    for name in ("train", "calibration", "validation", "test"):
        base_ids = [str(v) for v in split_doc["splits"].get(name, [])]
        for base_id in base_ids:
            if base_id in assignments:
                raise RuntimeError(f"base network {base_id} appears in multiple splits")
            assignments[base_id] = name
        splits[name] = np.flatnonzero(np.isin(ids, base_ids))
    missing_ids = sorted(set(ids) - set(assignments))
    if missing_ids:
        raise RuntimeError(f"unassigned base networks: {missing_ids[:5]}")
    for base_id in set(ids):
        idx = np.flatnonzero(ids == base_id)
        names = {assignments[str(ids[i])] for i in idx}
        if len(names) != 1:
            raise RuntimeError(f"rotations of {base_id} cross splits")
    disallowed = {str(v) for v in split_doc.get("disallowed_base_network_ids", [])}
    overlap = sorted(set(ids) & disallowed)
    if overlap:
        raise RuntimeError(f"disallowed/exposed base IDs present: {overlap[:10]}")
    if any(len(splits[k]) == 0 for k in splits):
        raise RuntimeError("every split must contain examples")
    return Bundle(arrays=arrays, manifest=manifest, splits=splits)


def feature_inputs(arrays: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray | None]:
    # Explicitly excludes all labels and replay arrays.
    return {
        "weights": arrays["weights"][idx],
        "node_observables": arrays.get("node_observables", None)[idx] if "node_observables" in arrays else None,
        "layer_observables": arrays.get("layer_observables", None)[idx] if "layer_observables" in arrays else None,
    }
