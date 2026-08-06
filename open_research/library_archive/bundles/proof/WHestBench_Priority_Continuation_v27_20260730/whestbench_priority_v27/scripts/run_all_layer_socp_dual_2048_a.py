from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
from scipy.special import gammaln
from scipy.linalg import cho_factor,cho_solve
OGAP=Path('/mnt/data/priority_v26_inputs/ogap/whest_experiments_oracle_gap_20260730');OUT=Path('/mnt/data/priority_v26_work/all_layer_socp_dual_2048_a.json')
D=256;N=66048;L=32;TARGET=1/4.34;ENERGY=.9939595959595959;NP=2048;DEPTHS=list(range(1,33))
sys.path.insert(0,str(OGAP/'src'));import run_fresh_suite as core

def mch():return math.sqrt(2)*math.exp(gammaln(128.5)-gammaln(128))
def psd_pinv(a,rc=1e-8):
 e,v=np.linalg.eigh((a+a.T)/2);keep=e>max(e[-1]*rc,1e-18);return (v[:,keep]/e[keep])@v[:,keep].T

def sample_states(W,seed):
 rng=np.random.default_rng(seed);u=rng.standard_normal((NP,D)).astype(np.float32);u/=np.linalg.norm(u,axis=1,keepdims=True);h=np.concatenate([(mch()*u).astype(np.float32),(-mch()*u).astype(np.float32)]);out={}
 for i,w in enumerate(W,1):
  h=np.maximum(h@w.T,0).astype(np.float32,copy=False);p=.5*(h[:NP].astype(float)+h[NP:].astype(float));p-=p.mean(0);out[i]=p/math.sqrt(NP-1)
 return out

def source(seed,rot):
 z=np.load(f'/mnt/data/priority_v26_work/sources/source_{seed}_{rot}.npz');q=z['s']**2;k=int(np.searchsorted(np.cumsum(q)/q.sum(),ENERGY)+1);return z['U40'][:,:k],k,z['base']

def block_data(X):
 bl=[]
 for prev,cur in zip(DEPTHS[:-1],DEPTHS[1:]):
  xp,xc=X[prev],X[cur];bl.append((xc.T@xc,xc.T@xp,xp.T@xp))
 return bl

def qvals(Cs,U,X):
 full=Cs+[U];return np.array([float(np.sum((X[cur]@full[j+1]-X[prev]@full[j])**2)) for j,(prev,cur) in enumerate(zip(DEPTHS[:-1],DEPTHS[1:]))])
def objective(Cs,U,X,a):return float(np.sum(a*np.sqrt(np.maximum(qvals(Cs,U,X),0))))

def factor_spd(M,base_ridge):
 M=(M+M.T)/2;ridge=base_ridge
 for _ in range(8):
  try:return cho_factor(M+ridge*np.eye(D),lower=True,check_finite=False),ridge
  except Exception:ridge*=10
 raise np.linalg.LinAlgError('factor failed')

def block_solve(diag,upper,lower,rhs,base_ridge):
 m=len(diag);fac=[None]*m;dt=[None]*m;rt=[None]*m;used=[]
 dt[0]=(diag[0]+diag[0].T)/2;rt[0]=rhs[0];fac[0],rg=factor_spd(dt[0],base_ridge);used.append(rg)
 for i in range(1,m):
  cat=np.concatenate([upper[i-1],rt[i-1]],axis=1);sol=cho_solve(fac[i-1],cat,check_finite=False);xu=sol[:,:D];xr=sol[:,D:]
  dt[i]=(diag[i]-lower[i-1]@xu);dt[i]=(dt[i]+dt[i].T)/2;rt[i]=rhs[i]-lower[i-1]@xr;fac[i],rg=factor_spd(dt[i],base_ridge);used.append(rg)
 x=[None]*m;x[-1]=cho_solve(fac[-1],rt[-1],check_finite=False)
 for i in range(m-2,-1,-1):x[i]=cho_solve(fac[i],rt[i]-upper[i]@x[i+1],check_finite=False)
 return x,max(used)

def greedy(U,bl):
 m=len(bl);out=[None]*m;cur=U
 for j in range(m-1,-1,-1):
  A,B,Dm=bl[j];out[j]=psd_pinv(Dm)@B.T@cur;cur=out[j]
 return out

def primal(U,X,bl,a):
 m=len(bl);starts=[greedy(U,bl),[np.zeros((D,U.shape[1])) for _ in range(m)]];best=None
 for si,Cs in enumerate(starts[:1]):
  hist=[];maxridge=0
  for it in range(4):
   q=qvals(Cs,U,X);eps=max(float(q.sum())*1e-12,1e-24);w=a/np.sqrt(q+eps)
   diag=[np.zeros((D,D)) for _ in range(m)];up=[];lo=[];rhs=[np.zeros((D,U.shape[1])) for _ in range(m)]
   for j,(A,B,Dm) in enumerate(bl):
    diag[j]+=w[j]*Dm
    if j+1<m:
     diag[j+1]+=w[j]*A;up.append(-w[j]*B.T);lo.append(-w[j]*B)
    else:rhs[j]+=w[j]*B.T@U
   scale=max(sum(float(np.trace(z)) for z in diag)/(m*D),1e-20);Cn,rg=block_solve(diag,up,lo,rhs,scale*1e-10);maxridge=max(maxridge,rg)
   damp=.8;Cn=[damp*n+(1-damp)*o for n,o in zip(Cn,Cs)];val=objective(Cn,U,X,a);hist.append(val)
   if it>1 and abs(hist[-1]-hist[-2])<2e-8*max(hist[-2],1e-30):Cs=Cn;break
   Cs=Cn
  rec={'Cs':Cs,'objective':objective(Cs,U,X,a),'iterations':len(hist),'history_tail':hist[-6:],'max_ridge':maxridge,'start':si}
  if best is None or rec['objective']<best['objective']:best=rec
 return best

def dual_certificate(Cs,U,X,a):
 m=len(Cs);full=Cs+[U];res=[];y=[]
 for j,(prev,cur) in enumerate(zip(DEPTHS[:-1],DEPTHS[1:])):
  r=X[cur]@full[j+1]-X[prev]@full[j];nr=float(np.linalg.norm(r));res.append(r);y.append((a[j]/max(nr,1e-30))*r)
 # K y = A^* y for each control variable.
 ky=[]
 for i,d in enumerate(DEPTHS[:-1]):
  val=-X[d].T@y[i]
  if i>0:val+=X[d].T@y[i-1]
  ky.append(val)
 # Project onto ker(A^*) by y0=y-A lambda, solve A^* A lambda=K y.
 bl=block_data(X);diag=[np.zeros((D,D)) for _ in range(m)];up=[];lo=[]
 for j,(A,B,Dm) in enumerate(bl):
  diag[j]+=Dm
  if j+1<m:diag[j+1]+=A;up.append(-B.T);lo.append(-B)
 scale=max(sum(float(np.trace(z)) for z in diag)/(m*D),1e-20);lam,rg=block_solve(diag,up,lo,ky,scale*1e-10)
 yp=[]
 for j,(prev,cur) in enumerate(zip(DEPTHS[:-1],DEPTHS[1:])):
  corr=-X[prev]@lam[j]
  if j+1<m:corr+=X[cur]@lam[j+1]
  yp.append(y[j]-corr)
 # global scaling ensures all SOC constraints exactly.
 norms=np.array([np.linalg.norm(z) for z in yp]);alpha=min(1.0,float(np.min(a/np.maximum(norms,1e-30))));yp=[alpha*z for z in yp]
 stat=[]
 for i,d in enumerate(DEPTHS[:-1]):
  val=-X[d].T@yp[i]
  if i>0:val+=X[d].T@yp[i-1]
  stat.append(np.linalg.norm(val))
 b=X[32]@U;dual=float(np.sum(yp[-1]*b));
 if dual<0:dual=-dual
 return {'dual_objective':dual,'dual_scale':alpha,'max_stationarity_norm':float(max(stat)),'max_ball_ratio':float(max(np.linalg.norm(z)/aa for z,aa in zip(yp,a))),'projection_ridge':rg,'block_norms':[float(x) for x in norms]}

ap=argparse.ArgumentParser();ap.add_argument('seed',type=int);ap.add_argument('rot',type=int);a0=ap.parse_args();seed,rot=a0.seed,a0.rot;t0=time.time();W=core.make_weights(seed);X=sample_states(W,seed+4410000);U,k,base=source(seed,rot);cp=OGAP/'results'/'confirmation'/f'seed_{seed}'/f'seed{seed}_rot{rot}.npz'
with np.load(cp) as z:truth=z['truth'].astype(float);stored=z['baseline'].astype(float)
assert np.max(abs(stored-base))<2e-12;e=base-truth;den=float(e@e);b=U.T@e;rstar=float((den-b@b)/den);Smax=math.sqrt(TARGET)-math.sqrt(rstar);bl=block_data(X);a=np.sqrt(np.array([2*cur/(N*L) for cur in DEPTHS[1:]],float));pr=primal(U,X,bl,a);du=dual_certificate(pr['Cs'],U,X,a);Spr=pr['objective']/math.sqrt(den);Sdu=du['dual_objective']/math.sqrt(den);score_pr=(math.sqrt(rstar)+Spr)**2;score_du=(math.sqrt(rstar)+Sdu)**2
row={'case_id':f'seed{seed}_rot{rot}','seed':seed,'rotation':rot,'rank':k,'partition':DEPTHS,'n_pairs':NP,'base_error_sq':den,'oracle_ratio':rstar,'Smax':Smax,'primal_S':Spr,'dual_S':Sdu,'primal_score':score_pr,'dual_score_lower_bound':score_du,'target':TARGET,'primal_pass':score_pr<TARGET,'dual_closes':score_du>=TARGET,'primal_dual_relative_gap':(pr['objective']-du['dual_objective'])/max(pr['objective'],1e-30),'primal':{kk:vv for kk,vv in pr.items() if kk!='Cs'},'dual':du,'seconds':time.time()-t0}
if OUT.exists():o=json.load(open(OUT));rows=[r for r in o.get('rows',[]) if r['case_id']!=row['case_id']]
else:rows=[]
rows.append(row);OUT.write_text(json.dumps({'status':'RUNNING','scope':'All 32 checkpoints, empirical oracle covariance, direct-output adaptive source','rows':rows},indent=2,sort_keys=True)+'\n');print(json.dumps(row,indent=2,sort_keys=True))
