#!/usr/bin/env python3
"""Run one self-contained v23 verification step.

Running theorem-critical high-degree replays in separate fresh processes is
recommended because each constructs a large exact Gegenbauer basis.
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
TASKS={
 "inertia_exact_replay":"code/verify_inertia_certificates_exact.py",
 "v21_degree280":"source_v21/verify_signed_near_optimality_certificate_blocktrace_order320.py",
 "shared_profile_and_sturm":"code/verify_shared_profile_and_sturm_gap.py",
 "gaussian_suffix_nonexpansivity":"code/verify_gaussian_relu_nonexpansivity.py",
 "gaussian_crossing_formula":"code/verify_gaussian_crossing_formula.py",
 "finite_width_monotonicity_no_go":"code/verify_finite_width_monotonicity_counterexample.py",
}
ap=argparse.ArgumentParser()
ap.add_argument("step", nargs="?", choices=sorted(TASKS))
ap.add_argument("--list", action="store_true")
a=ap.parse_args()
if a.list or a.step is None:
 print("\n".join(sorted(TASKS)))
 raise SystemExit(0)
raise SystemExit(subprocess.call([sys.executable,str(ROOT/TASKS[a.step])],cwd=ROOT))
