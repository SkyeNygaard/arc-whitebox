from pathlib import Path
import hashlib, sys
root=Path(__file__).resolve().parent
manifest=root/'SHA256SUMS.txt'
errors=[]; checked=0
for line in manifest.read_text().splitlines():
    if not line.strip(): continue
    digest, rel=line.split('  ',1)
    p=root/rel
    if not p.is_file():
        errors.append(f'MISSING {rel}'); continue
    got=hashlib.sha256(p.read_bytes()).hexdigest(); checked+=1
    if got!=digest: errors.append(f'MISMATCH {rel}: {got}')
print(f'checked={checked} errors={len(errors)}')
for e in errors: print(e)
sys.exit(1 if errors else 0)
