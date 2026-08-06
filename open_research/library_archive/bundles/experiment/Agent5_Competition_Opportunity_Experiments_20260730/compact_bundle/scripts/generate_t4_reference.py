#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
T4=Path('/mnt/data/work/T4_legal_layer31_anchor_hedge_20260729_review/T4_legal_layer31_anchor_hedge_20260729/code');sys.path.insert(0,str(T4));import frozen_reference_impl as fr
def main():
 p=argparse.ArgumentParser();p.add_argument('--network',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=2);a=p.parse_args();torch.set_num_threads(a.threads);a.out.mkdir(parents=True,exist_ok=True);(a.out/'results/references').mkdir(parents=True,exist_ok=True)
 path=a.out/'results/references'/f'network_{a.network:04d}_reference.npz'
 if path.exists():print(json.dumps({'network':a.network,'reference':'reused'}));return
 ws,_,_=fr.make_weights(a.network);refs=[];seeds=[]
 for j in range(6):
  seed=93_000_000+100*a.network+j;seeds.append(seed);refs.append(fr.stream_reference(ws,65536,seed,4096));print(json.dumps({'network':a.network,'stream':j}),flush=True)
 A={k:np.mean([r[k] for r in refs[:3]],axis=0) for k in ('y','mu','M')};B={k:np.mean([r[k] for r in refs[3:]],axis=0) for k in ('y','mu','M')};truth={k:.5*(A[k]+B[k]) for k in A}
 np.savez_compressed(path,refA_y=A['y'],refB_y=B['y'],truth_y=truth['y'],truth_mu=truth['mu'],truth_M=truth['M'],seeds=np.asarray(seeds));print(json.dumps({'network':a.network,'saved':str(path)}),flush=True)
if __name__=='__main__':main()
