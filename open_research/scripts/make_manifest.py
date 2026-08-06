#!/usr/bin/env python3
from pathlib import Path
import csv, hashlib

root = Path(__file__).resolve().parents[1]
out = root/'release_manifest.csv'
rows=[]
for p in sorted(root.rglob('*')):
    if not p.is_file() or p == out or '.git' in p.parts:
        continue
    rows.append({
        'path': str(p.relative_to(root)),
        'bytes': p.stat().st_size,
        'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
    })
with out.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['path','bytes','sha256'])
    w.writeheader(); w.writerows(rows)
print(f'wrote {len(rows)} entries to {out}')
