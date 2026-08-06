#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,csv,tarfile,sys
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
errors=[]
# Required node artifacts
req=['NODE_SPEC.md','MANIFEST.json','ROWS.csv','RESULTS.json','COST_MODEL.json','DECISION.md','CHILDREN.md']
for node in ['T0.1_basis_curve','T0.2_compute_calibration','T0.3_A42_A43_grade']:
 for n in req:
  if not (ROOT/node/n).exists():errors.append(f'missing {node}/{n}')
# Parse JSON/CSV
for p in ROOT.rglob('*.json'):
 try:json.load(open(p))
 except Exception as e:errors.append(f'bad json {p}: {e}')
for p in ROOT.rglob('*.csv'):
 try:list(csv.DictReader(open(p)))
 except Exception as e:errors.append(f'bad csv {p}: {e}')
# Package tar safety and manifest hashes
for p in (ROOT/'packages').glob('*.tar.gz'):
 try:
  with tarfile.open(p,'r:gz') as tf:
   names=tf.getnames()
   if any(n.startswith('/') or '..' in Path(n).parts for n in names):errors.append(f'unsafe path {p}')
   by={Path(n).name:n for n in names}
   m=json.loads(tf.extractfile(by['manifest.json']).read())
   for x in m['files']:
    got=hashlib.sha256(tf.extractfile(by[x['name']]).read()).hexdigest()
    if got!=x['sha256']:errors.append(f'manifest mismatch {p}:{x["name"]}')
 except Exception as e:errors.append(f'bad package {p}: {e}')
# Freeze hashes (exclude this verifier if added after original freeze; caller refreshes freeze file)
freeze=ROOT/'FREEZE_HASHES.sha256'
if freeze.exists():
 for line in freeze.read_text().splitlines():
  if not line.strip():continue
  h,rel=line.split('  ',1);q=ROOT/rel
  if not q.exists() or sha(q)!=h:errors.append(f'freeze mismatch {rel}')
print(json.dumps({'ok':not errors,'errors':errors,'package_count':len(list((ROOT/'packages').glob('*.tar.gz')))},indent=2))
sys.exit(1 if errors else 0)
