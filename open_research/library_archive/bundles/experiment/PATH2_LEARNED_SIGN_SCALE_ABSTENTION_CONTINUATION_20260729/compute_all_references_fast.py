from __future__ import annotations
import argparse, math, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
SRC=Path('/mnt/data/whest_path2/additional/INDEPENDENT_LAYER31_RESIDUAL_FINAL_BUNDLE')
sys.path.insert(0,str(SRC))
import run_layer31_residual as r
from fast_reference_audit import stream_target_fast, stream_anchor_fast

MOD=2_147_483_647

def qseed(network_seed:int, stream:str)->int:
    if stream=='anchor1': return (1_300_000_000+2*network_seed)%MOD
    if stream=='anchor2': return (1_300_000_001+2*network_seed)%MOD
    if stream=='target1': return (1_700_000_000+2*network_seed)%MOD
    if stream=='target2': return (1_700_000_001+2*network_seed)%MOD
    raise ValueError(stream)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--network-seed',type=int,required=True)
    ap.add_argument('--n',type=int,default=262144)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--threads',type=int,default=5)
    ap.add_argument('--chunk',type=int,default=32768)
    a=ap.parse_args()
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.out.exists():
        print({'status':'exists','out':str(a.out)})
        return
    ws=r.make_weights(a.network_seed)
    t=time.time()
    t1=stream_target_fast(ws,a.n,qseed(a.network_seed,'target1'),a.chunk); print({'stage':'t1','elapsed':time.time()-t},flush=True)
    t2=stream_target_fast(ws,a.n,qseed(a.network_seed,'target2'),a.chunk); print({'stage':'t2','elapsed':time.time()-t},flush=True)
    a1=stream_anchor_fast(ws,a.n,qseed(a.network_seed,'anchor1'),a.chunk,'f32'); print({'stage':'a1','elapsed':time.time()-t},flush=True)
    a2=stream_anchor_fast(ws,a.n,qseed(a.network_seed,'anchor2'),a.chunk,'f32'); print({'stage':'a2','elapsed':time.time()-t},flush=True)
    data={
      'target_final1':t1['final'],'target_final2':t2['final'],
      'target_pen1':t1['penultimate'],'target_pen2':t2['penultimate'],
      'anchor_mu1':a1['mu_h'],'anchor_mu2':a2['mu_h'],
      'anchor_M1':a1['M_h'],'anchor_M2':a2['M_h'],
      'anchor_raw1':a1['raw_h'],'anchor_raw2':a2['raw_h'],
      'qmc_n':np.array(a.n,dtype=np.int64),
      'target_seed1':np.array(qseed(a.network_seed,'target1'),dtype=np.int64),
      'target_seed2':np.array(qseed(a.network_seed,'target2'),dtype=np.int64),
      'anchor_seed1':np.array(qseed(a.network_seed,'anchor1'),dtype=np.int64),
      'anchor_seed2':np.array(qseed(a.network_seed,'anchor2'),dtype=np.int64),
      'fast_reference_version':np.array('target-f32-anchor-f32matmul-f64reduce-v1'),
      'chunk':np.array(a.chunk,dtype=np.int64),
    }
    tmp=a.out.with_suffix(a.out.suffix+'.tmp.npz')
    np.savez(tmp,**data)
    tmp.replace(a.out)
    print({'status':'done','seed':a.network_seed,'out':str(a.out),'runtime_seconds':time.time()-t})
if __name__=='__main__': main()
