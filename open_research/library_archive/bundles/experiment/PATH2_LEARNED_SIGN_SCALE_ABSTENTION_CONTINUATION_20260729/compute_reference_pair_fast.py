from __future__ import annotations
import argparse,sys,time
from pathlib import Path
import numpy as np, torch
SRC=Path('/mnt/data/whest_path2/additional/INDEPENDENT_LAYER31_RESIDUAL_FINAL_BUNDLE');sys.path.insert(0,str(SRC))
import run_layer31_residual as r
from fast_reference_audit import stream_target_fast,stream_anchor_fast
MOD=2_147_483_647
def qs(s,k):
 b=1_700_000_000 if k=='target' else 1_300_000_000
 return ((b+2*s)%MOD,(b+2*s+1)%MOD)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['target','anchor'],required=True);ap.add_argument('--seeds',type=int,nargs='+',required=True);ap.add_argument('--n',type=int,default=262144);ap.add_argument('--outdir',type=Path,required=True);ap.add_argument('--threads',type=int,default=5);ap.add_argument('--chunk',type=int,default=32768);a=ap.parse_args()
 torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);a.outdir.mkdir(parents=True,exist_ok=True)
 for s in a.seeds:
  p=a.outdir/f'{a.kind}_{s}_pair_n{a.n}.npz'
  if p.exists(): print({'status':'exists','kind':a.kind,'seed':s},flush=True);continue
  t=time.time();ws=r.make_weights(s);q1,q2=qs(s,a.kind)
  if a.kind=='target':
   z1=stream_target_fast(ws,a.n,q1,a.chunk);z2=stream_target_fast(ws,a.n,q2,a.chunk)
   data={'target_final1':z1['final'],'target_final2':z2['final'],'target_pen1':z1['penultimate'],'target_pen2':z2['penultimate']}
  else:
   z1=stream_anchor_fast(ws,a.n,q1,a.chunk,'f32');z2=stream_anchor_fast(ws,a.n,q2,a.chunk,'f32')
   data={'anchor_mu1':z1['mu_h'],'anchor_mu2':z2['mu_h'],'anchor_M1':z1['M_h'],'anchor_M2':z2['M_h'],'anchor_raw1':z1['raw_h'],'anchor_raw2':z2['raw_h']}
  data|={'network_seed':np.array(s),'qmc_n':np.array(a.n),'seed1':np.array(q1),'seed2':np.array(q2),'chunk':np.array(a.chunk),'fast_version':np.array('f32matmul-f64reduce-v1')}
  np.savez(p,**data);print({'status':'done','kind':a.kind,'seed':s,'runtime_seconds':time.time()-t,'out':str(p)},flush=True)
if __name__=='__main__':main()
