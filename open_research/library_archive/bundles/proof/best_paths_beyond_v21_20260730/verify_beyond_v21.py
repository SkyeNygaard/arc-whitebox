#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
scripts=[
 'verify_joint_sharpness_strictness.py',
 'verify_gaussian_crossing_formula.py',
 'verify_finite_width_monotonicity_counterexample.py',
 'verify_a51_interface_frontier.py',
]
rows=[]
for s in scripts:
 p=subprocess.run([sys.executable,str(ROOT/s)],capture_output=True,text=True,check=True)
 rows.append({'script':s,'status':'PASS','stdout':p.stdout.strip()})
out={'status':'PASS','scripts':rows}
(ROOT/'combined_verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':'PASS','scripts':[r['script'] for r in rows]},indent=2))
