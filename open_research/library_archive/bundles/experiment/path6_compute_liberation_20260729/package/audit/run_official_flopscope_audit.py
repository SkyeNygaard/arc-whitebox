"""Run inside the real WHestBench subprocess environment.

Usage is repository-specific; point BASELINE_DIR and CANDIDATE_DIR at unpacked
submissions and invoke the existing official Mini/full-suite runner on both.
The script intentionally refuses to manufacture a local zero-FLOP result when
flopscope/whestbench are unavailable.
"""
from __future__ import annotations
import importlib.util
import json
import sys

required = ['flopscope', 'whestbench']
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit('Missing official dependencies: ' + ', '.join(missing) +
                     '. Run this audit only in the final subprocess package.')
print(json.dumps({'official_dependencies_present': True,
                  'next': 'Run the repository official paired baseline/candidate grader and retain its immutable FlopScope arrays.'}, indent=2))
