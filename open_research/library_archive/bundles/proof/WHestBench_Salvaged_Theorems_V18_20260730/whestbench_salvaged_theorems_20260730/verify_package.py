#!/usr/bin/env python3
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent

def run(name):
    p=subprocess.run([sys.executable,str(HERE/name)],cwd=HERE,text=True,capture_output=True)
    return {'script':name,'returncode':p.returncode,'stdout_tail':p.stdout[-1000:],'stderr_tail':p.stderr[-1000:]}

def main():
    results=[run('verify_salvaged_theorems.py'),run('certify_k32_mub_line_spectrum.py'),run('certify_t16_endpoints.py')]
    control=[]
    for p in HERE.rglob('*'):
        if p.is_file() and p.suffix in {'.md','.py','.json','.txt'}:
            data=p.read_bytes()
            bad=[i for i,b in enumerate(data) if b<32 and b not in (9,10,13)]
            if bad:
                control.append({'file':str(p.relative_to(HERE)),'positions':bad[:20]})
    stale=[]
    for p in HERE.glob('*.md'):
        s=p.read_text()
        if 'composed 31 times' in s or 'range(31)' in s:
            stale.append(p.name)
    status='PASS' if all(r['returncode']==0 for r in results) and not control and not stale else 'FAIL'
    out={'status':status,'subchecks':results,'control_character_issues':control,'stale_depth_wording':stale}
    (HERE/'PACKAGE_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
    raise SystemExit(0 if status=='PASS' else 1)

if __name__=='__main__':
    main()
