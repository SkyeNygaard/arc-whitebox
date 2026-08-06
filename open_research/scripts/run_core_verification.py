#!/usr/bin/env python3
from pathlib import Path
import json, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[1]
checks = [
    ('T16 all-degree exact recurrence', ROOT/'evidence/primary_theory', ['python','prove_t16_all_degree.py']),
    ('v5.2 recovered-bundle consistency', ROOT, ['python','scripts/verify_v5_2_bundle.py']),
    ('independent Kerdock-risk numerical sanity check', ROOT, ['python','scripts/sanity_check_kerdock_risk.py']),
    ('original signed comparison witness', ROOT/'evidence/primary_theory/signed_replay', ['python','verify_signed_near_optimality_certificate_blocktrace_order320.py']),
    ('audited frozen-witness inertia/sign-count', ROOT/'evidence/primary_theory/signed_replay', ['python','verify_inertia_strengthened_frozen_witness.py']),
]
results=[]
for name,cwd,cmd in checks:
    t=time.time()
    cp=subprocess.run(cmd,cwd=cwd,text=True,capture_output=True,timeout=600)
    results.append({'name':name,'returncode':cp.returncode,'seconds':round(time.time()-t,3)})
    if cp.returncode:
        print(cp.stdout)
        print(cp.stderr,file=sys.stderr)
        raise SystemExit(f'FAILED: {name}')
print(json.dumps({'passed':True,'checks':results},indent=2))
