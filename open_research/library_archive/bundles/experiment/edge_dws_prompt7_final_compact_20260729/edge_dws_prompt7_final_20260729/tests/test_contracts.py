import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.contracts import canonical_hash, feature_inputs, load_bundle


def make_bundle(tmp_path: Path):
    n, d, o = 8, 2, 5
    data = tmp_path / "labels.npz"
    np.savez_compressed(
        data,
        weights=np.zeros((n, 32, 256, 256), np.float16),
        base_network_id=np.arange(300, 308),
        rotation_id=np.zeros(n, np.int16),
        baseline_error=np.ones((n, o), np.float32),
        replay_jacobian=np.ones((n, o, d), np.float32),
        anchor_coeffs=np.zeros((n, d), np.float32),
        target_coeffs=np.zeros((n, d), np.float32),
    )
    sha = hashlib.sha256(data.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"status": "frozen", "data_sha256": sha, "surrogate": {"verified": True}}))
    splits_obj = {"train": [300, 301], "calibration": [302, 303], "validation": [304, 305], "test": [306, 307]}
    split = tmp_path / "splits.json"
    split.write_text(json.dumps({"status": "frozen", "splits": splits_obj, "splits_sha256": canonical_hash(splits_obj), "disallowed_base_network_ids": list(range(200))}))
    return data, manifest, split


def test_valid_grouped_bundle(tmp_path):
    data, manifest, split = make_bundle(tmp_path)
    b = load_bundle(data, manifest, split)
    assert b.label_dim == 2


def test_hash_mismatch_fails(tmp_path):
    data, manifest, split = make_bundle(tmp_path)
    m = json.loads(manifest.read_text()); m["data_sha256"] = "0" * 64; manifest.write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="SHA-256"):
        load_bundle(data, manifest, split)


def test_split_overlap_fails(tmp_path):
    data, manifest, split = make_bundle(tmp_path)
    doc = json.loads(split.read_text())
    doc["splits"]["test"].append(300)
    doc["splits_sha256"] = canonical_hash(doc["splits"])
    split.write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match="multiple splits"):
        load_bundle(data, manifest, split)


def test_exposed_id_fails(tmp_path):
    data, manifest, split = make_bundle(tmp_path)
    with np.load(data, allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    arrays["base_network_id"][0] = 42
    np.savez_compressed(data, **arrays)
    m = json.loads(manifest.read_text())
    m["data_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(m))
    doc = json.loads(split.read_text())
    doc["splits"]["train"][0] = 42
    doc["splits_sha256"] = canonical_hash(doc["splits"])
    split.write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match="disallowed/exposed"):
        load_bundle(data, manifest, split)


def test_target_shuffle_cannot_change_model_inputs(tmp_path):
    data, manifest, split = make_bundle(tmp_path)
    b = load_bundle(data, manifest, split)
    idx = b.splits["train"]
    before = feature_inputs(b.arrays, idx)
    shuffled = dict(b.arrays)
    shuffled["target_coeffs"] = shuffled["target_coeffs"][::-1].copy()
    shuffled["baseline_error"] = shuffled["baseline_error"][::-1].copy()
    shuffled["replay_jacobian"] = shuffled["replay_jacobian"][::-1].copy()
    after = feature_inputs(shuffled, idx)
    assert set(before) == {"weights", "node_observables", "layer_observables"}
    np.testing.assert_array_equal(before["weights"], after["weights"])
    assert before["node_observables"] is None and after["node_observables"] is None
    assert before["layer_observables"] is None and after["layer_observables"] is None
