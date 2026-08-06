from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np,torch
SRC=Path('/mnt/data/whest_path2/additional/INDEPENDENT_LAYER31_RESIDUAL_FINAL_BUNDLE');sys.path.insert(0,str(SRC))
import run_layer31_residual as r

def stream_target_fast(ws,total_n,seed,chunk=16384):
 eng=torch.quasirandom.SobolEngine(r.D,scramble=True,seed=seed); final=torch.zeros(r.D,dtype=torch.float64); pen=torch.zeros(r.D,dtype=torch.float64); done=0
 with torch.no_grad():
  while done<total_n:
   n=min(chunk,total_n-done);u=eng.draw(n,dtype=torch.float32).clamp_(1e-7,1-1e-7);x=math.sqrt(2.)*torch.erfinv(2*u-1);p=None
   for li,w in enumerate(ws):
    x=torch.relu(x@w)
    if li==r.PENULTIMATE_LAYER:p=x
   final+=x.sum(0,dtype=torch.float64);pen+=p.sum(0,dtype=torch.float64);done+=n
 return {'final':final.numpy()/total_n,'penultimate':pen.numpy()/total_n}

def stream_anchor_fast(ws,total_n,seed,chunk=16384,accum='f32'):
 eng=torch.quasirandom.SobolEngine(r.D,scramble=True,seed=seed); hs=torch.zeros(r.D,dtype=torch.float64);M=torch.zeros((r.D,r.D),dtype=torch.float64);raw=torch.zeros_like(M);done=0
 with torch.no_grad():
  while done<total_n:
   n=min(chunk,total_n-done);u=eng.draw(n,dtype=torch.float32).clamp_(1e-7,1-1e-7);x=math.sqrt(2.)*torch.erfinv(2*u-1)
   for li,w in enumerate(ws):
    x=torch.relu(x@w)
    if li==r.TARGET_FEATURE_LAYER:break
   H=x;hs+=H.sum(0,dtype=torch.float64)
   if accum=='f32':
    M+=(H.T@H).double();raw+=((H*H).T@H).double()
   else:
    Hd=H.double();M+=Hd.T@Hd;raw+=(Hd*Hd).T@Hd
   done+=n
 return {'mu_h':hs.numpy()/total_n,'M_h':M.numpy()/total_n,'raw_h':raw.numpy()/total_n}

def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=493891104);ap.add_argument('--qmc-seed',type=int,default=1234567);ap.add_argument('--n',type=int,default=4096);ap.add_argument('--threads',type=int,default=5);a=ap.parse_args();torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);ws=r.make_weights(a.seed)
 t=time.time();slow=r.stream_reference(ws,a.n,a.qmc_seed,chunk=4096);ts=time.time()-t
 for ch in [4096,8192,16384,32768]:
  t=time.time();tar=stream_target_fast(ws,a.n,a.qmc_seed,ch);tt=time.time()-t
  t=time.time();anc=stream_anchor_fast(ws,a.n,a.qmc_seed,ch,'f32');ta=time.time()-t
  print(json.dumps({'n':a.n,'chunk':ch,'slow_total_s':ts,'fast_target_s':tt,'fast_anchor_s':ta,'fast_sum_s':tt+ta,
   'errors':{'final':rel(tar['final'],slow['final']),'pen':rel(tar['penultimate'],slow['penultimate']),'mu':rel(anc['mu_h'],slow['mu_h']),'M':rel(anc['M_h'],slow['M_h']),'raw':rel(anc['raw_h'],slow['raw_h'])}},indent=2))
if __name__=='__main__':main()
