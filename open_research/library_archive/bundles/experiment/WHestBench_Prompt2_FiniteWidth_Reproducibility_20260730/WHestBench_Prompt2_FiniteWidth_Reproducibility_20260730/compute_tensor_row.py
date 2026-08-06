#!/usr/bin/env python3
import sys,os,json,time,argparse
if hasattr(sys,'set_int_max_str_digits'): sys.set_int_max_str_digits(0)
import prompt2_tensor_partition_core as pc
p=argparse.ArgumentParser();p.add_argument('rank',type=int);p.add_argument('--degrees',required=True);p.add_argument('--support',type=int,default=None);a=p.parse_args()
te=pc.TensorEnergy(a.rank)
for k in map(int,a.degrees.split(',')):
 path=f'/mnt/data/tensor{a.rank}_degree_{k}' + (f'_s{a.support}' if a.support is not None else '') + '.json'
 if os.path.exists(path): print('skip',path,flush=True);continue
 t=time.time();z=te.energy(k,max_support=a.support);dt=time.time()-t
 rec={'rank':a.rank,'degree':k,'max_support':a.support,'lo':str(z.lo),'hi':str(z.hi),'seconds':dt}
 with open(path,'w') as f:json.dump(rec,f,indent=2)
 print('done',path,dt,flush=True)
