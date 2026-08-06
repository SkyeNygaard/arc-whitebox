#!/usr/bin/env python3
from pathlib import Path
import csv, hashlib, sys

root = Path(__file__).resolve().parents[1]
required = [
    'README.md', 'RELEASE_STATUS.md', 'LEDGER_GUIDE.md', 'REPRODUCIBILITY.md',
    'OPEN_PROBLEMS.md', 'BASELINE_PACKAGE_MISSING.md',
    'papers/Paper_A_Kerdock_Near_Optimality.md',
    'papers/Paper_B_Oracle_Headroom_Open_Ledger.md',
    'ledger/whestbench_canonical_research_ledger_20260731_reconciled_v31_final_local_writeup.xlsx',
    'ledger/WHestBench_Current_State_v31_20260731.md',
    'ledger/WHestBench_Public_Claim_Manifest.csv',
]
missing = [p for p in required if not (root/p).exists()]
if missing:
    print('Missing required files:')
    for p in missing: print(' -', p)
    sys.exit(1)

manifest = root/'release_manifest.csv'
if manifest.exists():
    with manifest.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    bad=[]
    for row in rows:
        p=root/row['path']
        if not p.exists():
            bad.append((row['path'],'missing'))
            continue
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        if h != row['sha256']:
            bad.append((row['path'],'hash mismatch'))
    if bad:
        print('Manifest failures:')
        for p,why in bad: print(' -',p,why)
        sys.exit(2)
print('Release check passed.')
