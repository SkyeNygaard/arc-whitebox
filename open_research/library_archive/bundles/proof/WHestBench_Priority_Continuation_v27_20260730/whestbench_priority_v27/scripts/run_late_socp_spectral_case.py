from __future__ import annotations
import argparse,json,math,sys,time,warnings
from pathlib import Path
import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
warnings.filterwarnings('ignore')
OGAP=Path('/mnt/data/priority_v26_inputs/ogap/whest_experiments_oracle_gap_20260730');OUT=Path('/mnt/data/priority_v26_work/adaptive_direct_late_checkpoint_socp.json')
D=256;N=66048;L=32;TARGET=1/4.34;ENERGY=.9939595959595959;NP=1024;TS=[29,30,31]
sys.path.insert(0,str(OGAP/'src'));import run_fresh_suite as core
def mch():return math.sqrt(2)*math.exp(gammaln(128.5)-gammaln(128))
def cv(a,b):a=a-a.mean(0);b=b-b.mean(0);return a.T@b/(len(a)-1)
def pinv(a,rc=1e-9):
 e,v=np.linalg.eigh((a+a.T)/2);keep=e>max(e[-1]*rc,1e-18);return (v[:,keep]/e[keep])@v[:,keep].T,int(keep.sum()),float(e[-1]/e[keep].min())
def spectral_solve(H,rhs,rc=1e-10):
 e,v=np.linalg.eigh((H+H.T)/2);keep=e>max(e[-1]*rc,1e-18)
 if not np.any(keep):return np.zeros_like(rhs)
 return v[:,keep]@((v[:,keep].T@rhs)/e[keep,None])
def states(W,seed):
 rng=np.random.default_rng(seed);n=2*NP;u=rng.standard_normal((n,D)).astype(np.float32);u/=np.linalg.norm(u,axis=1,keepdims=True);h=np.concatenate([(mch()*u).astype(np.float32),(-mch()*u).astype(np.float32)]);o={};wanted=set([1,32]+TS)
 for i,w in enumerate(W,1):
  h=np.maximum(h@w.T,0).astype(np.float32,copy=False)
  if i in wanted:
   p=.5*(h[:n].astype(float)+h[n:].astype(float));o[i]=p[NP:]
 return o
def block(C,U,A,B,Dm):return max(float(np.trace(U.T@A@U)+np.trace(C.T@Dm@C)-2*np.trace(U.T@B@C)),0)
def solve(st,U,t):
 h1,ht,hL=st[1],st[t],st[32];D1=cv(h1,h1);B1=cv(ht,h1);A1=cv(ht,ht);D2=A1;B2=cv(hL,ht);A2=cv(hL,hL);P,r1,c1=pinv(D1);R=(A1-B1@P@B1.T);R=(R+R.T)/2;er,vr=np.linalg.eigh(R);er=np.maximum(er,0);R=(vr*er)@vr.T;rhs=B2.T@U;g1=2*t/(N*L);g2=2/N
 def ev(x):
  rho=math.exp(float(x));C=spectral_solve(D2+rho*R,rhs);q1=max(float(np.trace(C.T@R@C)),0);q2=block(C,U,A2,B2,D2);return math.sqrt(g1*q1)+math.sqrt(g2*q2),q1,q2,C
 grid=np.linspace(-10,10,17);vals=[ev(x)[0] for x in grid];j=int(np.argmin(vals));lo=grid[max(j-1,0)];hi=grid[min(j+1,len(grid)-1)];res=minimize_scalar(lambda x:ev(x)[0],bounds=(lo,hi),method='bounded',options={'maxiter':40,'xatol':1e-5});cand=[ev(res.x),ev(grid[j])]
 # Boundaries
 q20=max(float(np.trace(U.T@A2@U)),0);cand.append((math.sqrt(g2*q20),0,q20,np.zeros_like(rhs)));P2,r2,c2=pinv(D2);C2=P2@rhs;q12=max(float(np.trace(C2.T@R@C2)),0);q22=block(C2,U,A2,B2,D2);cand.append((math.sqrt(g1*q12)+math.sqrt(g2*q22),q12,q22,C2));obj,q1,q2,C=min(cand,key=lambda z:z[0]);return obj,q1,q2,r1,c1,r2,c2,float(res.x),bool(res.success)
def source(seed,rot):
 z=np.load(f'/mnt/data/priority_v26_work/sources/source_{seed}_{rot}.npz');q=z['s']**2;k=int(np.searchsorted(np.cumsum(q)/q.sum(),ENERGY)+1);return z['U40'][:,:k],k,z['base']
ap=argparse.ArgumentParser();ap.add_argument('seed',type=int);ap.add_argument('rot',type=int);a=ap.parse_args();seed,rot=a.seed,a.rot;W=core.make_weights(seed);st=states(W,seed+990000);U,k,base=source(seed,rot);p=OGAP/'results'/'confirmation'/f'seed_{seed}'/f'seed{seed}_rot{rot}.npz'
with np.load(p) as z:truth=z['truth'].astype(float);stored=z['baseline'].astype(float)
assert np.max(abs(stored-base))<2e-12;e=base-truth;den=float(e@e);b=U.T@e;rstar=float((den-b@b)/den);new=[]
for t in TS:
 obj,q1,q2,r1,c1,r2,c2,x,succ=solve(st,U,t);S=obj/math.sqrt(den);sc=(math.sqrt(rstar)+S)**2;new.append({'case_id':f'seed{seed}_rot{rot}','seed':seed,'rotation':rot,'rank':k,'partition':[1,t,32],'base_error_sq':den,'oracle_ratio':rstar,'Smax':math.sqrt(TARGET)-math.sqrt(rstar),'S_oracle':S,'score_oracle':sc,'pass':sc<TARGET,'objective_unscaled':obj,'q1':q1,'q2':q2,'h1_rank':r1,'h1_condition':c1,'ht_rank':r2,'ht_condition':c2,'logrho_opt':x,'optimizer_success':succ,'solver':'spectral_pseudoinverse_favorable'})
if OUT.exists():o=json.load(open(OUT));rows=[r for r in o.get('rows',[]) if not(r['seed']==seed and r['rotation']==rot and r['partition'][1] in TS)]
else:rows=[]
rows+=new;o={'status':'RUNNING','source':'Agent8 adaptive direct-output PCA','sampling':'oracle covariance on untouched half of 1024+1024 fixed-radius antithetic pairs','rows':rows,'target':TARGET};OUT.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n');print(seed,rot,[round(r['score_oracle'],6) for r in new])
