from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.linalg import solve
OGAP=Path('/mnt/data/priority_v26_inputs/ogap/whest_experiments_oracle_gap_20260730')
OUT=Path('/mnt/data/priority_v26_work/adaptive_direct_late_checkpoint_socp.json')
D=256;N=66048;L=32;TARGET=1/4.34;ENERGY=.9939595959595959;NP=1024;TS=[27,29,30,31];ROTS=[31001,31013,31033]
sys.path.insert(0,str(OGAP/'src'));import run_fresh_suite as core

def mch():return math.sqrt(2)*math.exp(gammaln(128.5)-gammaln(128))
def cv(a,b):a=a-a.mean(0);b=b-b.mean(0);return a.T@b/(len(a)-1)
def psd_pinv(a,rc=1e-8):
 e,v=np.linalg.eigh((a+a.T)/2);keep=e>max(e[-1]*rc,0);return (v[:,keep]/e[keep])@v[:,keep].T,int(keep.sum()),float(e[-1]/e[keep].min())
def states(W,seed):
 rng=np.random.default_rng(seed);n=2*NP;u=rng.standard_normal((n,D)).astype(np.float32);u/=np.linalg.norm(u,axis=1,keepdims=True);h=np.concatenate([(mch()*u).astype(np.float32),(-mch()*u).astype(np.float32)]);o={};wanted=set([1,32]+TS)
 for i,w in enumerate(W,1):
  h=np.maximum(h@w.T,0).astype(np.float32,copy=False)
  if i in wanted:
   p=.5*(h[:n].astype(float)+h[n:].astype(float));o[i]=p[NP:] # untouched/test half only
 return o
def trace_block(C,U,A,B,Dm):
 return max(float(np.trace(U.T@A@U)+np.trace(C.T@Dm@C)-2*np.trace(U.T@B@C)),0.0)
def solve_one(st,U,t):
 h1,ht,hL=st[1],st[t],st[32]
 D1=cv(h1,h1);B1=cv(ht,h1);A1=cv(ht,ht);D2=A1;B2=cv(hL,ht);A2=cv(hL,hL)
 P,rank1,cond1=psd_pinv(D1);R=(A1-B1@P@B1.T);R=(R+R.T)/2
 er,vr=np.linalg.eigh(R);er=np.maximum(er,0.0);R=(vr*er)@vr.T
 g1=2*t/(N*L);g2=2/N;rhs=B2.T@U
 scale=max(float(np.trace(D2+R))/D,1e-20);ridge=scale*1e-10
 I=np.eye(D)
 def at(logrho):
  rho=math.exp(float(logrho));H=D2+rho*R+ridge*I
  C=solve(H,rhs,assume_a='pos',check_finite=False)
  q1=max(float(np.trace(C.T@R@C)),0.0);q2=trace_block(C,U,A2,B2,D2)
  obj=math.sqrt(g1*q1)+math.sqrt(g2*q2)
  return obj,C,q1,q2
 grid=np.linspace(-14,14,29);vals=[]
 for x in grid:
  try:vals.append(at(x)[0])
  except Exception:vals.append(float('inf'))
 j=int(np.argmin(vals));lo=grid[max(0,j-1)];hi=grid[min(len(grid)-1,j+1)]
 res=minimize_scalar(lambda x:at(x)[0],bounds=(lo,hi),method='bounded',options={'xatol':1e-6,'maxiter':80})
 candidates=[at(res.x),at(grid[j])]
 # Explicit boundaries C=0 and terminal regression.
 q20=max(float(np.trace(U.T@A2@U)),0.0);candidates.append((math.sqrt(g2*q20),np.zeros_like(rhs),0.0,q20))
 P2,rank2,cond2=psd_pinv(D2);C2=P2@rhs;q12=max(float(np.trace(C2.T@R@C2)),0.0);q22=trace_block(C2,U,A2,B2,D2);candidates.append((math.sqrt(g1*q12)+math.sqrt(g2*q22),C2,q12,q22))
 obj,C,q1,q2=min(candidates,key=lambda x:x[0])
 return {'objective_unscaled':obj,'q1':q1,'q2':q2,'rho_log_opt':float(res.x),'optimizer_success':bool(res.success),'h1_rank':rank1,'h1_condition':cond1,'ht_rank':rank2,'ht_condition':cond2,'ridge':ridge}

def source(seed,rot):
 z=np.load(f'/mnt/data/priority_v26_work/sources/source_{seed}_{rot}.npz');s=z['s'];q=s**2;k=int(np.searchsorted(np.cumsum(q)/q.sum(),ENERGY)+1);return z['U40'][:,:k],k,z['base']

ap=argparse.ArgumentParser();ap.add_argument('seed',type=int);a=ap.parse_args();seed=a.seed
t0=time.time();W=core.make_weights(seed);st=states(W,seed+990000);new=[]
for rot in ROTS:
 U,k,base=source(seed,rot);p=OGAP/'results'/'confirmation'/f'seed_{seed}'/f'seed{seed}_rot{rot}.npz'
 with np.load(p) as z:truth=z['truth'].astype(float);stored=z['baseline'].astype(float)
 assert np.max(abs(stored-base))<2e-12;e=base-truth;den=float(e@e);b=U.T@e;rstar=float((den-b@b)/den);smax=math.sqrt(TARGET)-math.sqrt(rstar)
 for t in TS:
  sol=solve_one(st,U,t);S=sol['objective_unscaled']/math.sqrt(den);score=(math.sqrt(rstar)+S)**2
  new.append({'case_id':f'seed{seed}_rot{rot}','seed':seed,'rotation':rot,'rank':k,'partition':[1,t,32],'base_error_sq':den,'oracle_ratio':rstar,'Smax':smax,'S_oracle':S,'score_oracle':score,'pass':score<TARGET,**sol})
 print(seed,rot,'scores',[round(r['score_oracle'],6) for r in new[-4:]],flush=True)
if OUT.exists():o=json.load(open(OUT));rows=[r for r in o.get('rows',[]) if r['seed']!=seed]
else:rows=[]
rows+=new;o={'status':'RUNNING','source':'Agent8 adaptive direct-output PCA','sampling':'oracle covariance on untouched half of 1024+1024 fixed-radius antithetic pairs','rows':rows,'target':TARGET};OUT.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n');print('seconds',time.time()-t0)
