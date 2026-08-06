#!/usr/bin/env python3
from pathlib import Path
import hashlib, sys
ROOT=Path(__file__).resolve().parents[1]
errors=[]
for line in (ROOT/'SHA256SUMS').read_text().splitlines():
    if not line.strip(): continue
    expected,rel=line.split('  ',1); p=ROOT/rel
    if not p.exists(): errors.append('missing '+rel); continue
    got=hashlib.sha256(p.read_bytes()).hexdigest()
    if got!=expected: errors.append('hash mismatch '+rel)
if errors:
    print('PACKAGE CHECK FAILED')
    for e in errors: print(' -',e)
    sys.exit(1)
print('Package hashes verified.')
