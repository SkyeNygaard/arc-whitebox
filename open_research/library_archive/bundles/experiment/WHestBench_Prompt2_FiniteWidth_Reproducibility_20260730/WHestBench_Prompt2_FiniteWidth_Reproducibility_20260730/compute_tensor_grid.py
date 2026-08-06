#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,time,os
from concurrent.futures import ProcessPoolExecutor,as_completed

def one(task):
    import sys
    if hasattr(sys, 'set_int_max_str_digits'): sys.set_int_max_str_digits(0)
    r,k,s=task
    import prompt2_tensor_partition_core as pc
    import prompt2_full_hermite_core as c
    t=time.time(); z=pc.TensorEnergy(r).energy(k,max_support=s)
    rec={'rank':r,'degree':k,'max_support':s,'lo':str(z.lo),'hi':str(z.hi),'seconds':time.time()-t}
    path=f'/mnt/data/tensor{r}_degree_{k}_s{s}.json'
    with open(path,'w') as f: json.dump(rec,f,indent=2)
    return path,rec['seconds']

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);p.add_argument('--ranks',default='4,5,6,7,8');p.add_argument('--degrees',default='9,10,11,12,13,14,15,16');p.add_argument('--support',type=int,default=2)
 a=p.parse_args();tasks=[]
 for r in map(int,a.ranks.split(',')):
  for k in map(int,a.degrees.split(',')):
   path=f'/mnt/data/tensor{r}_degree_{k}_s{a.support}.json'
   if not os.path.exists(path):tasks.append((r,k,a.support))
 print('tasks',len(tasks),flush=True)
 with ProcessPoolExecutor(max_workers=a.workers) as ex:
  fs={ex.submit(one,t):t for t in tasks}
  for f in as_completed(fs):
   t=fs[f]
   try: print(t,f.result(),flush=True)
   except Exception as e: print('ERR',t,repr(e),flush=True)
