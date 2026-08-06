#!/usr/bin/env python3
from pathlib import Path
import csv, hashlib, sys
ROOT = Path(__file__).resolve().parents[1]
manifest = ROOT / 'library_archive' / 'manifests' / 'archive_members.csv'
errors = []
with manifest.open(newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        p = ROOT / row['extracted_path']
        if not p.is_file():
            errors.append(f"missing: {p}")
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != row['file_sha256']:
            errors.append(f"hash mismatch: {p}")
if errors:
    print('\n'.join(errors[:100]), file=sys.stderr)
    raise SystemExit(1)
print('Library import verification passed.')
