"""Release integrity check for the public WHestBench surface.

The release policy is deliberately wide: every script, note, figure, audit,
ledger row, and small result record is published. This check therefore does
*not* enforce an allowlist of publishable paths. It enforces the four things
that actually matter once the surface is open:

1. the load-bearing documents exist;
2. the pinned source and archive hashes still authenticate;
3. nothing that must stay out has leaked in — third-party contact details,
   bulk numerical arrays, oversized blobs;
4. published Python still parses.

Run from the repository root::

    python scripts/check_competition_release.py
"""

from __future__ import annotations

import hashlib
import py_compile

import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_SIZE = 20 * 1024 * 1024

REQUIRED_FILES = (
    "LICENSE",
    "CITATION.cff",
    "README.md",
    "whestbench/README.md",
    "whestbench/COMPETITION.md",
    "whestbench/RELEASE_STATUS.md",
    "whestbench/claims.csv",
    "whestbench/phase1_320802.json",
    "whestbench/ledger/README.md",
    "whestbench/papers/Phase1_Algorithmic_Contribution_320802.pdf",
    "arc_whitebox/README.md",
    "arc_whitebox/submissions/production_baseline_320802/README.md",
    "arc_ceiling/README.md",
    "arc_ceiling/requirements-public.txt",
    "theory/README.md",
    "theory/paper/main.tex",
    "theory/proof_archive/SHA256SUMS",
    "open_research/README.md",
    "open_research/RECONCILIATION_20260804.md",
)

SOURCE_HASHES = {
    "arc_whitebox/submissions/kerdock_mub5/estimator.py": "ab7fdbaff2dc6943cccfc54e4761ad952f43dbdbdf6e31e042d553b8f3ae0749",
    "arc_whitebox/submissions/kerdock_mub5/kerdock_mub5_seed3.npz": "58eac1b69707b204d00f6d50cf4e1996b1fcd566154ec93a7ecb5668c1acbfad",
    "arc_whitebox/submissions/kerdock_mub5_winograd_tree/estimator.py": "7c3fcc2ac542bda41ab568e62428ec75b7edec6f146d61a036c0710d9ee49694",
    "arc_whitebox/submissions/kerdock_mub5_winograd_tree/fast_matmul.py": "fb1b93cb625b66ce5f26220ea3b6b685dbb9887d50f8756cafa9426577d45085",
    "arc_whitebox/submissions/kerdock_mub5_winograd_tree/kerdock_mub5_seed3.npz": "58eac1b69707b204d00f6d50cf4e1996b1fcd566154ec93a7ecb5668c1acbfad",
    "arc_whitebox/submissions/production_baseline_320802/estimator.py": "f1e32ce44fe43b53eba3f70f9cf6383da588ec1bbb3d82c047edbc916a98d8df",
    "arc_whitebox/submissions/production_baseline_320802/fast_matmul.py": "fb1b93cb625b66ce5f26220ea3b6b685dbb9887d50f8756cafa9426577d45085",
    "arc_whitebox/submissions/production_baseline_320802/kerdock_mub5_seed3.npz": "58eac1b69707b204d00f6d50cf4e1996b1fcd566154ec93a7ecb5668c1acbfad",
}

ARCHIVES = {
    "arc_whitebox/submissions/kerdock_mub5/submission.tar.gz": (
        "e60c0a686188f9fe030c1a3769b29859d539902d9a43be40e4b6f9883dd663ae",
        {"estimator.py", "kerdock_mub5_seed3.npz", "manifest.json"},
    ),
    "arc_whitebox/submissions/kerdock_mub5_winograd_tree/submission.tar.gz": (
        "a7f5e1e58639192e33e0886e776b4c8392399a7879e372bed557811516ec93e7",
        {"estimator.py", "fast_matmul.py", "kerdock_mub5_seed3.npz", "manifest.json"},
    ),
    "arc_whitebox/submissions/production_baseline_320802/submission.tar.gz": (
        "77be0e8865b2aeee6c6c16314cac4d38496efefed6b2b758f75bc3033bb6b7bc",
        {
            "README.md",
            "estimator.py",
            "fast_matmul.py",
            "kerdock_mub5_seed3.npz",
            "manifest.json",
            "manifest_actual.json",
        },
    ),
}

COMPILE_TARGETS = (
    "arc_whitebox/submissions/kerdock_mub5/estimator.py",
    "arc_whitebox/submissions/kerdock_mub5/make_asset.py",
    "arc_whitebox/submissions/kerdock_mub5_winograd_tree/estimator.py",
    "arc_whitebox/submissions/kerdock_mub5_winograd_tree/fast_matmul.py",
    "arc_whitebox/submissions/production_baseline_320802/estimator.py",
    "arc_whitebox/submissions/production_baseline_320802/fast_matmul.py",
    "arc_ceiling/spectrum.py",
    "arc_ceiling/design_potentials.py",
    "arc_ceiling/validate_ceiling.py",
    "scripts/export_ledger.py",
    "theory/proof_archive/scripts/run_verification_portable.py",
)

# Third-party contact details that were removed on import and must not return.
# Stored split so this file does not itself republish them.
FORBIDDEN_CONTACT_PATTERN = r"[A-Za-z0-9._%+-]+@" + "northeastern" + r"\.edu"
FORBIDDEN_PATH_FRAGMENT = "Northeastern_Outreach"

# Bulk arrays stay out, with one deliberate exception: the frozen Kerdock
# design asset, which is the single numerical input needed to rerun a package.
ARRAY_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".safetensors", ".joblib"}
ALLOWED_ARRAY_NAME = "kerdock_mub5_seed3.npz"

TEXT_SUFFIXES = {".md", ".csv", ".txt", ".json", ".py", ".tex", ".yml", ".yaml", ".cff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_for_contact_details(tracked: list[Path]) -> list[str]:
    """One grep pass over published text files.

    Doing this per file in Python costs minutes on a surface this size, so
    hand the whole set to grep once and read back the filenames it flags.
    """
    candidates = [
        str(path) for path in tracked
        if path.suffix.lower() in TEXT_SUFFIXES and (ROOT / path).is_file()
    ]
    if not candidates:
        return []

    flagged: set[str] = set()
    batch_size = 400
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        completed = subprocess.run(
            ["grep", "-lIiE", FORBIDDEN_CONTACT_PATTERN, "--", *batch],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        # grep exits 1 when nothing matched, which is the expected case.
        if completed.returncode not in (0, 1):
            return [f"contact-detail scan failed: {completed.stderr.strip()}"]
        flagged.update(line for line in completed.stdout.splitlines() if line)

    return [
        f"third-party contact details in the public surface: {name}"
        for name in sorted(flagged)
    ]


def release_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    return [Path(item) for item in output.decode().split("\0") if item]


def main() -> int:
    errors: list[str] = []

    for relative_path in REQUIRED_FILES:
        if not (ROOT / relative_path).is_file():
            errors.append(f"missing required release file: {relative_path}")

    claims = (ROOT / "whestbench/claims.csv").read_text(encoding="utf-8")
    if "reported" not in claims or "not independently reproduced" not in claims:
        errors.append("claims.csv must retain the reported-result evidence boundary")

    tracked = release_files()

    for relative_path in tracked:
        path = ROOT / relative_path
        if not path.is_file():
            continue

        if path.stat().st_size > MAX_FILE_SIZE:
            size_mb = path.stat().st_size / (1024 * 1024)
            errors.append(f"file exceeds {MAX_FILE_SIZE // 1024 // 1024} MiB: {relative_path} ({size_mb:.1f} MiB)")

        if path.suffix.lower() in ARRAY_SUFFIXES and path.name != ALLOWED_ARRAY_NAME:
            errors.append(f"bulk array in the public surface: {relative_path}")

        if FORBIDDEN_PATH_FRAGMENT in relative_path.as_posix():
            errors.append(f"withheld outreach material is in the public surface: {relative_path}")

    errors.extend(scan_for_contact_details(tracked))

    for relative_path, expected_hash in SOURCE_HASHES.items():
        path = ROOT / relative_path
        if not path.is_file():
            errors.append(f"pinned source is missing: {relative_path}")
        elif sha256(path) != expected_hash:
            errors.append(f"source hash mismatch: {relative_path}")

    for relative_path, (expected_hash, expected_members) in ARCHIVES.items():
        archive = ROOT / relative_path
        if not archive.is_file():
            errors.append(f"pinned archive is missing: {relative_path}")
            continue
        if sha256(archive) != expected_hash:
            errors.append(f"archive hash mismatch: {relative_path}")
        with tarfile.open(archive, "r:gz") as bundle:
            members = {member.name.rstrip("/") for member in bundle.getmembers() if member.isfile()}
        if members != expected_members:
            errors.append(f"unexpected archive contents: {relative_path}")

    for relative_path in COMPILE_TARGETS:
        try:
            py_compile.compile(str(ROOT / relative_path), doraise=True)
        except py_compile.PyCompileError as error:
            errors.append(str(error))

    if errors:
        print("Competition release check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Competition release check passed ({len(tracked)} files in the public surface).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
