#!/usr/bin/env python3
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,os,json,sys

def one(t):
 r,k=t
 import sys
 if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
 from one_pair_complete_fast import energy
 z=energy(r,k)
 p=f'/mnt/data/onepair_tensor{r}_degree_{k}.json'
 with open(p,'w') as f:json.dump({'rank':r,'degree':k,'lo':str(z.lo),'hi':str(z.hi)},f,indent=2)
 return p
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--rmin',type=int,default=8);p.add_argument('--rmax',type=int,default=22);p.add_argument('--kmax',type=int,default=14);p.add_argument('--workers',type=int,default=10);a=p.parse_args()
 ts=[(r,k) for r in range(a.rmin,a.rmax+1) for k in range(a.kmax+1) if not os.path.exists(f'/mnt/data/onepair_tensor{r}_degree_{k}.json')]
 print('tasks',len(ts),flush=True)
 with ProcessPoolExecutor(max_workers=a.workers) as ex:
  fs={ex.submit(one,t):t for t in ts}
  for i,f in enumerate(as_completed(fs),1):
   try:f.result()
   except Exception as e:print('ERR',fs[f],repr(e),flush=True)
   if i%10==0:print('done',i,flush=True)
