#!/usr/bin/env python3
"""Benchmark memory-bounded exact sorted stop-loss replay on synthetic full shape."""
from __future__ import annotations
import argparse, gc, json, os, resource, time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1'); os.environ.setdefault('NUMEXPR_NUM_THREADS','1')
import numpy as np

def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0

def scan(z,s,chunk=2048):
 n,m=z.shape; r=s.shape[0]; out=np.zeros((r,m),np.float64)
 for k in range(r):
  total=np.zeros(m,np.float64)
  for i in range(0,n,chunk):
   x=z[i:i+chunk]+s[k]; np.maximum(x,np.float32(0),out=x); total += x.sum(0,dtype=np.float64)
  out[k]=total/n
 return out

def sorted_block(z,s,block):
 n,m=z.shape; outs=[]
 for j0 in range(0,m,block):
  j1=min(j0+block,m)
  zs=np.sort(z[:,j0:j1],axis=0)
  pref=np.cumsum(zs,axis=0,dtype=np.float64)
  cols=[]
  for q in range(j1-j0):
   sj=s[:,j0+q]
   idx=np.searchsorted(zs[:,q],-sj,side='right')
   before=np.where(idx>0,pref[np.maximum(idx-1,0),q],0.0)
   active=pref[-1,q]-before+(n-idx)*sj
   cols.append(active/n)
  outs.append(np.stack(cols,axis=1))
 return np.concatenate(outs,axis=1)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--n',type=int,default=66048);p.add_argument('--m',type=int,default=256);p.add_argument('--rank',type=int,default=32);p.add_argument('--blocks',type=int,nargs='+',default=[4,8,16,32,64,128,256]);p.add_argument('--repeats',type=int,default=2);a=p.parse_args()
 rng=np.random.default_rng(20260731);z=rng.normal(-.03,1.15,(a.n,a.m)).astype(np.float32);s=rng.normal(0,.22,(a.rank,a.m)).astype(np.float32)
 t=time.perf_counter(); ref=scan(z,s); scan_t=time.perf_counter()-t
 res={'protected_data_opened':False,'shape':[a.n,a.m],'rank':a.rank,'scan_seconds':scan_t,'initial_rss_mib':rss(),'blocks':{}}
 for b in a.blocks:
  ts=[]; err=[]; before=rss()
  for _ in range(a.repeats):
   gc.collect();t=time.perf_counter();o=sorted_block(z,s,b);ts.append(time.perf_counter()-t);err.append(float(np.max(np.abs(o-ref))))
  res['blocks'][str(b)]={'median_seconds':float(np.median(ts)),'min_seconds':min(ts),'max_seconds':max(ts),'max_abs_vs_scan':max(err),'peak_rss_delta_mib':max(0.0,rss()-before),'checksum':float(o.sum())}
 res['final_peak_rss_mib']=rss();a.output.write_text(json.dumps(res,indent=2,sort_keys=True)+'\n');print(json.dumps(res,indent=2,sort_keys=True))
if __name__=='__main__':main()
