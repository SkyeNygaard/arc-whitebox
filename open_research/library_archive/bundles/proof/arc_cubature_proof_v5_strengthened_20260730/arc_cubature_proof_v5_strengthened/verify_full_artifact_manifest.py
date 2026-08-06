#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
LIST=ROOT/'FULL_ARTIFACT_FILES.json'
MANIFEST=ROOT/'FULL_ARTIFACT_MANIFEST.sha256'
IGNORE={'FULL_ARTIFACT_MANIFEST.sha256'}
def files_on_disk():
 out=[]
 for p in ROOT.rglob('*'):
  if not p.is_file(): continue
  rel=p.relative_to(ROOT).as_posix()
  if '__pycache__/' in rel or rel.endswith('.pyc') or rel in IGNORE: continue
  out.append(rel)
 return sorted(out)
def main():
 expected=json.loads(LIST.read_text())['files']; assert expected==sorted(expected) and len(expected)==len(set(expected))
 actual=files_on_disk()
 if actual!=expected:
  raise SystemExit(json.dumps({'missing':sorted(set(expected)-set(actual)),'unexpected':sorted(set(actual)-set(expected))},indent=2))
 rows={}
 for line in MANIFEST.read_text().splitlines():
  digest,rel=line.split('  ',1); rows[rel]=digest
 if sorted(rows)!=expected: raise SystemExit('manifest/list mismatch')
 bad=[]
 for rel in expected:
  got=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
  if got!=rows[rel]: bad.append({'file':rel,'expected':rows[rel],'actual':got})
 if bad: raise SystemExit(json.dumps({'hash_mismatches':bad},indent=2))
 print(json.dumps({'passed':True,'tracked_files':len(expected),'manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),'all_23_curvature_chunks_tracked':sum('/formal_gpp_chunk_' in ('/'+x) for x in expected)==23},indent=2))
if __name__=='__main__':main()
