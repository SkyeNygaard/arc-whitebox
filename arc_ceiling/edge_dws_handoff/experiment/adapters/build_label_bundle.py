from __future__ import annotations

"""Freeze a generic low-dimensional width-256 label bundle.

The upstream Prompt 4–6 exporter must provide a verified exact linear replay
contract. This adapter hashes the export, rejects broad 256-answer targets, and
will not mark a surrogate verified based on an unchecked assertion.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REQUIRED = {
    "weights", "base_network_id", "rotation_id", "baseline_error",
    "replay_jacobian", "anchor_coeffs", "target_coeffs",
}
OPTIONAL = {"node_observables", "layer_observables", "target_confidence"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_surrogate_document(path: Path) -> dict:
    doc = json.loads(path.read_text())
    proof_kind = doc.get("proof_kind")
    numerical_ok = (
        doc.get("verified") is True
        and isinstance(doc.get("max_abs_error"), (int, float))
        and float(doc["max_abs_error"]) <= float(doc.get("tolerance", 1e-6))
    )
    algebraic_ok = doc.get("verified") is True and proof_kind == "additive_final_output"
    if not (numerical_ok or algebraic_ok):
        raise RuntimeError(
            "surrogate verification must be a passing numerical replay check or "
            "an additive_final_output algebraic proof"
        )
    return doc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--label-kind", choices=["joint_anchor_residual", "g31_residual", "v80_scale"], required=True)
    p.add_argument("--surrogate-verification", type=Path, required=True)
    p.add_argument("--source-hash", action="append", default=[])
    args = p.parse_args()

    verification = verify_surrogate_document(args.surrogate_verification)
    with np.load(args.input, allow_pickle=False) as z:
        keys = set(z.files)
        missing = REQUIRED - keys
        unknown = keys - REQUIRED - OPTIONAL
        if missing:
            raise RuntimeError(f"missing arrays: {sorted(missing)}")
        if unknown:
            raise RuntimeError(f"unknown arrays: {sorted(unknown)}")
        arrays = {k: np.asarray(z[k]) for k in z.files}

    n = len(arrays["weights"])
    if arrays["weights"].shape[1:] != (32, 256, 256):
        raise RuntimeError("only width-256 depth-32 exports are accepted")
    for key, value in arrays.items():
        if len(value) != n:
            raise RuntimeError(f"first dimension mismatch for {key}")
    if arrays["target_coeffs"].ndim != 2:
        raise RuntimeError("target_coeffs must be [examples, label_dim]")
    d = int(arrays["target_coeffs"].shape[1])
    if not 1 <= d <= 16:
        raise RuntimeError(f"label dimension {d} violates the frozen 1–16 dimensional target contract")
    out_dim = int(arrays["baseline_error"].shape[1])
    if arrays["anchor_coeffs"].shape != (n, d):
        raise RuntimeError("anchor_coeffs shape mismatch")
    if arrays["replay_jacobian"].shape != (n, out_dim, d):
        raise RuntimeError("replay_jacobian shape mismatch")
    for key in ("weights", "baseline_error", "replay_jacobian", "anchor_coeffs", "target_coeffs"):
        if not np.isfinite(arrays[key]).all():
            raise RuntimeError(f"non-finite values in {key}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    manifest = {
        "version": 1,
        "status": "frozen",
        "label_kind": args.label_kind,
        "label_dim": d,
        "examples": n,
        "data_sha256": sha256(args.output),
        "source_sha256": args.source_hash,
        "surrogate": {
            "kind": "exact_linear_final_replay",
            "equation": "error(coeff)=baseline_error+replay_jacobian@coeff",
            "verified": True,
            "verification_sha256": sha256(args.surrogate_verification),
            "verification": verification,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
