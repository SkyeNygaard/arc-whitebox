#!/usr/bin/env python3
"""Download selected per-MLP moment files without duplicating the HF cache."""
from __future__ import annotations
import argparse,os,shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
REPO='keenanpepper/arc-whestbench-higher-moments-2026'

def parse_indices(spec):
 out=set()
 for piece in spec.split(','):
  piece=piece.strip()
  if not piece:continue
  if '-' in piece:
   lo,hi=map(int,piece.split('-',1));out.update(range(lo,hi+1))
  else:out.add(int(piece))
 if any(i<0 or i>999 for i in out):raise ValueError('indices must be 0..999')
 return sorted(out)

def link_or_copy(source,destination):
 if destination.exists():return 'exists'
 try:os.link(source,destination);return 'hardlink'
 except OSError:
  try:destination.symlink_to(source);return 'symlink'
  except OSError:shutil.copy2(source,destination);return 'copy'

def main():
 p=argparse.ArgumentParser();p.add_argument('indices');p.add_argument('--output',type=Path,required=True);p.add_argument('--revision',default='main');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 for pos,i in enumerate(parse_indices(a.indices),1):
  cached=Path(hf_hub_download(REPO,f'full/mlp_{i:05d}.npz',repo_type='dataset',revision=a.revision));dest=a.output/cached.name;mode=link_or_copy(cached,dest);print({'position':pos,'index':i,'path':str(dest),'mode':mode},flush=True)
if __name__=='__main__':main()
