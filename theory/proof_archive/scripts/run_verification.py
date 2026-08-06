#!/usr/bin/env python3
from pathlib import Path
import json, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1]
checks=[
 ('all-degree exact reduced costs',ROOT/'proof/nonnegative',['python','prove_t16_all_degree.py']),
 ('nonnegative recovered-record consistency',ROOT/'proof/nonnegative/v5_2',['python','verify_bundle.py']),
 ('Kerdock risk high-precision sanity check',ROOT/'proof/kerdock',['python','sanity_check_kerdock_risk.py']),
 ('original frozen signed rational witness',ROOT/'proof/signed',['python','verify_signed_near_optimality_certificate_blocktrace_order320.py']),
 ('positive-index and sign-count strengthening',ROOT/'proof/signed',['python','verify_inertia_strengthened_frozen_witness.py']),
]
results=[]
for name,cwd,cmd in checks:
    t=time.time(); cp=subprocess.run(cmd,cwd=cwd,text=True,capture_output=True,timeout=900)
    results.append({'name':name,'returncode':cp.returncode,'seconds':round(time.time()-t,3)})
    if cp.returncode:
        print(cp.stdout); print(cp.stderr,file=sys.stderr); raise SystemExit('FAILED: '+name)
print(json.dumps({'passed':True,'checks':results},indent=2))
