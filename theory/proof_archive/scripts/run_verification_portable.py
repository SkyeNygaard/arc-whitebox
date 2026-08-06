#!/usr/bin/env python3
"""Portable replacement for ``run_verification.py``.

``run_verification.py`` is shipped byte-identical to the arXiv ancillary
archive because its SHA-256 is listed in ``SHA256SUMS`` and is therefore a
proof input. It invokes the checks with the literal command ``python``, which
does not exist on systems that only provide ``python3`` (most current macOS
and Debian installs), so it fails there before running any mathematics.

This wrapper runs exactly the same five checks in the same order, using
``sys.executable`` instead. It is additive: it changes no hashed file, and
``check_package.py`` still passes with it present.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKS = [
    ("all-degree exact reduced costs", "proof/nonnegative", "prove_t16_all_degree.py"),
    ("nonnegative recovered-record consistency", "proof/nonnegative/v5_2", "verify_bundle.py"),
    ("Kerdock risk high-precision sanity check", "proof/kerdock", "sanity_check_kerdock_risk.py"),
    (
        "original frozen signed rational witness",
        "proof/signed",
        "verify_signed_near_optimality_certificate_blocktrace_order320.py",
    ),
    (
        "positive-index and sign-count strengthening",
        "proof/signed",
        "verify_inertia_strengthened_frozen_witness.py",
    ),
]


def main() -> int:
    results = []
    for name, relative_dir, script in CHECKS:
        started = time.time()
        completed = subprocess.run(
            [sys.executable, script],
            cwd=ROOT / relative_dir,
            text=True,
            capture_output=True,
            timeout=900,
        )
        results.append(
            {
                "name": name,
                "returncode": completed.returncode,
                "seconds": round(time.time() - started, 3),
            }
        )
        if completed.returncode:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            print("FAILED: " + name, file=sys.stderr)
            return 1

    print(json.dumps({"passed": True, "checks": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
