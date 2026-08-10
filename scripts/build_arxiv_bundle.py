"""Package the theory paper and its proof archive as an arXiv upload.

Writes dist/arxiv_submission_static_cubature.tar.gz containing main.tex,
main.bbl, references.bib, and the proof archive under anc/ (arXiv's reserved
directory for ancillary files, which it publishes but does not compile).

The proof archive's own SHA256SUMS are proof inputs, so the copy is verified
after staging: if packaging ever perturbs a hashed file, this fails loudly
rather than shipping a certificate that no longer replays.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "theory/paper"
ARCHIVE = ROOT / "theory/proof_archive"
OUT = ROOT / "dist/arxiv_submission_static_cubature.tar.gz"

PAPER_FILES = ["main.tex", "main.bbl", "references.bib"]
JUNK = {"__pycache__", ".DS_Store", ".ipynb_checkpoints"}


def stage(tmp: Path) -> None:
    for name in PAPER_FILES:
        src = PAPER / name
        if not src.exists():
            sys.exit(f"missing paper file: {src}")
        shutil.copy2(src, tmp / name)
    shutil.copytree(
        ARCHIVE, tmp / "anc", ignore=shutil.ignore_patterns(*JUNK)
    )


def verify(anc: Path) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_package.py"],
        cwd=anc,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(
            "staged ancillary archive failed its own hash check:\n"
            + result.stdout
            + result.stderr
        )
    print(result.stdout.strip())


def main() -> None:
    tmp = ROOT / "dist/_arxiv_stage"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        stage(tmp)
        verify(tmp / "anc")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(OUT, "w:gz") as tar:
            for path in sorted(tmp.rglob("*")):
                # recursive=False: rglob already yields every entry, and
                # tar.add() would otherwise re-add each directory's contents.
                tar.add(path, arcname=str(path.relative_to(tmp)), recursive=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    anc_count = sum(1 for p in ARCHIVE.rglob("*") if p.is_file())
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {anc_count} ancillary files)")


if __name__ == "__main__":
    main()
