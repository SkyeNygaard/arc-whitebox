#!/usr/bin/env python3
from __future__ import annotations
import hashlib,importlib.util,json,os,platform,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def main():
    expected_data=ROOT.parent/'arc_code'/'arc_whitebox'/'data'/'official_phase1_mini'/'data'
    deps={m:bool(importlib.util.find_spec(m)) for m in ['numpy','scipy','pyarrow','flopscope','whestbench','psutil','pytest']}
    assets={}
    for rel in ['assets/kerdock_mub5_seed3.npz','vendor/original_estimator.py','vendor/original_fast_matmul.py']:
        p=ROOT/rel;assets[rel]={'exists':p.exists(),'size_bytes':p.stat().st_size if p.exists() else None,'sha256':sha(p) if p.exists() else None}
    payload={'terminal_state':'externally_blocked','python':sys.version,'platform':platform.platform(),'dependencies':deps,'official_data_expected_path':str(expected_data),'official_data_exists':expected_data.exists(),'official_parquet_count':len(list(expected_data.glob('*.parquet'))) if expected_data.exists() else 0,'assets':assets,'blockers':[]}
    if not payload['official_data_exists']:payload['blockers'].append('official Mini-100 parquet data absent')
    if not deps['flopscope']:payload['blockers'].append('flopscope runtime absent')
    if not deps['whestbench']:payload['blockers'].append('whestbench runtime absent')
    if not deps['pyarrow']:payload['blockers'].append('pyarrow parquet loader absent')
    out=ROOT/'results'/'environment_diagnostic.json';out.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
