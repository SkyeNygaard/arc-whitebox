#!/usr/bin/env python3
from __future__ import annotations
import argparse,ast,json,tarfile,tempfile
from pathlib import Path
FORBIDDEN={'numpy','scipy','torch','jax','tensorflow'}
p=argparse.ArgumentParser();p.add_argument('path',type=Path)
# Offline tools (weight packers, cost estimators) legitimately import numpy and are
# never bundled into the graded payload. Point --exclude at them when auditing a
# working directory rather than a built submission.
p.add_argument('--exclude',action='append',default=[],metavar='GLOB')
a=p.parse_args();root=a.path
if root.suffix in {'.gz','.tgz'}:
 td=tempfile.TemporaryDirectory();tarfile.open(root).extractall(td.name);root=Path(td.name)
files=[x for x in root.rglob('*') if x.is_file()
       and not any(x.match(g) for g in a.exclude)];issues=[]
for f in files:
 if f.suffix=='.py':
  try:tree=ast.parse(f.read_text())
  except Exception as e:issues.append(f'{f}: parse error {e}');continue
  for n in ast.walk(tree):
   names=[]
   if isinstance(n,ast.Import):names=[x.name.split('.')[0] for x in n.names]
   elif isinstance(n,ast.ImportFrom) and n.module:names=[n.module.split('.')[0]]
   for name in names:
    if name in FORBIDDEN:issues.append(f'{f}: forbidden runtime import {name}')
report={'files':len(files),'bytes':sum(f.stat().st_size for f in files),'python_files':sum(f.suffix=='.py' for f in files),'issues':issues,
        'within_50_files':len(files)<=50,'within_50_mib':sum(f.stat().st_size for f in files)<=50*1024*1024}
print(json.dumps(report,indent=2));raise SystemExit(bool(issues or not report['within_50_files'] or not report['within_50_mib']))
