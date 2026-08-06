from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
from scipy.stats import norm
ROOT=Path('/mnt/data/whest_reopened');A9=ROOT/'agent9_10_oracle_bundle/agent9_10_oracle_bundle';WHITE=ROOT/'arc_code/arc_whitebox'
sys.path[:0]=[str(A9),str(WHITE/'src')]
import arc_experiments as ae
D=256;OUT=Path('/mnt/data/competition_relevance_20260730/edgeworth_final_mean');OUT.mkdir(parents=True,exist_ok=True)

def mc_final(w,n,seed,chunk=4096):
 rng=np.random.default_rng(seed);acc=np.zeros(D);done=0
 while done<n:
  b=min(chunk,n-done);a=rng.standard_normal((b,D),dtype=np.float32)
  for W in w:a=np.maximum(a@W,0)
  acc+=a.sum(0,dtype=np.float64);done+=b
 return acc/n

def kerdock_final_pre(w,rot):
 a=ae.first_activation(w[0],rot)
 for li,W in enumerate(w[1:],start=1):
  h=a@W
  if li==len(w)-1:return h.astype(np.float64)
  a=np.maximum(h,0)

def predictors(H):
 mu=H.mean(0);z=H-mu;v=np.mean(z*z,0);sd=np.sqrt(np.maximum(v,1e-30));t=mu/sd
 c3=np.mean(z**3,0);c4=np.mean(z**4,0)-3*v*v
 g=mu*norm.cdf(t)+sd*norm.pdf(t)
 e3=g-(c3/6)*t*norm.pdf(t)/(sd*sd)
 e4=e3+(c4/24)*((t*t-1)*norm.pdf(t))/(sd**3)
 base=np.maximum(H,0).mean(0)
 return {'base':base,'gaussian':g,'edgeworth3':e3,'edgeworth4':e4}, {'mean_skew_abs':float(np.mean(np.abs(c3/sd**3))),'mean_excess_abs':float(np.mean(np.abs(c4/sd**4)))}
def mse(a,b):return float(np.mean((a-b)**2))
def run(seed,nref,rot):
 t0=time.time();w=ae.make_weights(seed);r1=mc_final(w,nref,3_000_000+seed);r2=mc_final(w,nref,3_100_000+seed);truth=.5*(r1+r2);noise=mse(r1,r2)/4;H=kerdock_final_pre(w,rot);P,diag=predictors(H);bm=mse(P['base'],truth);rows={}
 for name,p in P.items(): rows[name]={'mse':mse(p,truth),'ratio':mse(p,truth)/bm,'delta_norm':float(np.linalg.norm(p-P['base']))}
 for name in ['gaussian','edgeworth3','edgeworth4']:
  d=P[name]-P['base']
  for a in [0.05,0.1,0.2,0.3,0.5,0.75,1.0]:
   q=P['base']+a*d;rows[f'{name}_blend{a:g}']={'mse':mse(q,truth),'ratio':mse(q,truth)/bm,'delta_norm':float(np.linalg.norm(a*d))}
 out={'seed':seed,'rot':rot,'nref_each':nref,'base_mse':bm,'reference_noise':noise,'diagnostics':diag,'rows':rows,'seconds':time.time()-t0}
 (OUT/f'seed{seed}_rot{rot}.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,nargs='+',required=True);ap.add_argument('--nref',type=int,default=65536);ap.add_argument('--rot',type=int,default=3);a=ap.parse_args()
 for s in a.seeds:run(s,a.nref,a.rot)
