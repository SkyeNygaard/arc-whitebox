from __future__ import annotations
import argparse,json,math,pickle,sys,time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0,'/mnt/data/whest_prior_followup')
from experiment_suite import make_network,gaussian_relu_covariance

@dataclass
class Case:
 X:np.ndarray;y:np.ndarray;scale:np.ndarray;base:np.ndarray;true:np.ndarray;W:np.ndarray;next_var:np.ndarray;iu:np.ndarray;ju:np.ndarray;net:int;layer:int

def cov(x):x=x.astype(np.float64);x-=x.mean(0);return x.T@x/max(1,len(x)-1)
def forward_selected(weights,n,seed,layers):
 rng=np.random.default_rng(seed);x=rng.standard_normal((n,len(weights[0])),dtype=np.float32);o={}
 for li,w in enumerate(weights):
  z=x@w
  if li in layers:o[li]=z.copy()
  x=np.maximum(z,0)
 return o

def jf(zraw,lf,seed,nproj=3):
 x=zraw.astype(np.float64);mu=x.mean(0);s=np.maximum(x.std(0,ddof=1),1e-10);z=(x-mu)/s;n,d=z.shape;iu,ju=np.triu_indices(d,1);rho=np.clip(z.T@z/max(1,n-1),-1,1);g3=np.mean(z**3,0);g4=np.mean(z**4,0)-3;q=z*z-1;k21=q.T@z/n
 # Fiber mean: mean_p kappa(i,j,p) = E[z_i z_j mean_p z_p]
 sm=z.mean(1);y1=z.T@(z*sm[:,None])/n
 rng=np.random.default_rng(seed);acc=np.zeros((d,d));facc=np.zeros(d)
 for _ in range(nproj):
  r=rng.choice([-1.,1.],d)/math.sqrt(d);u=z@r;G=z.T@(z*u[:,None])/n;acc+=G*G
  r2=rng.choice([-1.,1.],d)/math.sqrt(d);u2=z@r2;h=np.mean(z*(u*u2)[:,None],0);facc+=h*h
 y2=np.sqrt(np.maximum(acc/nproj,0));tn=np.mean(z*q.mean(1)[:,None],0);fn=np.sqrt(np.maximum(facc/nproj,0));a=mu/s
 X=np.column_stack([np.full(len(iu),lf),a[iu],a[ju],rho[iu,ju],g3[iu],g3[ju],g4[iu],g4[ju],.5*(k21[iu,ju]+k21[ju,iu]),.5*(k21[iu,ju]-k21[ju,iu]),y1[iu,ju],y2[iu,ju],tn[iu],tn[ju],fn[iu],fn[ju]]).astype(np.float32)
 return X,mu,s,rho,iu,ju

def make_case(weights,z,li,depth,net,seed):
 X,mu,s,rho,iu,ju=jf(z,(li+1)/depth,seed);relu=np.maximum(z.astype(np.float64),0);true=cov(relu);_,base=gaussian_relu_covariance(mu,s,rho);np.fill_diagonal(base,np.diag(true));scale=np.outer(s,s);y=((true-base)/np.maximum(scale,1e-12))[iu,ju].astype(np.float32);W=weights[li+1].astype(np.float64);nv=np.diag(W.T@true@W);return Case(X,y,scale[iu,ju].astype(np.float32),base,true,W,nv,iu,ju,net,li)
def gen(width,depth,nets,nsamp,layers,seed):
 cs=[]
 for ni in range(nets):
  w=make_network(width,depth,seed+ni*109);zs=forward_selected(w,nsamp,seed+100000+ni*113,layers)
  for li in layers:cs.append(make_case(w,zs[li],li,depth,ni,seed+li+ni*1000))
  print(json.dumps({'generated':ni+1,'width':width}),flush=True)
 return cs
def train_rows(cases,per,seed):
 rng=np.random.default_rng(seed);X=[];y=[]
 for c in cases:
  idx=rng.choice(len(c.y),min(per,len(c.y)),replace=False);X.append(c.X[idx]);y.append(c.y[idx])
 return np.concatenate(X),np.concatenate(y)
def ev(m,cases):
 rows=[]
 for c in cases:
  p=m.predict(c.X);C=c.base.copy();d=p*c.scale;C[c.iu,c.ju]+=d;C[c.ju,c.iu]+=d;v=lambda A:np.diag(c.W.T@A@c.W);rel=lambda q:float(np.sqrt(np.mean(((q-c.next_var)/np.maximum(c.next_var,1e-12))**2)));pm=np.mean((p-c.y)**2);zm=np.mean(c.y**2);rows.append({'net':c.net,'layer':c.layer,'gaussian':rel(v(c.base)),'model':rel(v(C)),'r2':float(1-pm/max(zm,1e-30))})
 a=lambda k:np.array([r[k] for r in rows]);return {'rows':rows,'summary':{'gaussian':float(a('gaussian').mean()),'model':float(a('model').mean()),'gain':float(a('gaussian').mean()/a('model').mean()),'fraction':float(np.mean(a('model')<a('gaussian'))),'r2':float(a('r2').mean())}}
def main():
 p=argparse.ArgumentParser();p.add_argument('--train-width',type=int,default=64);p.add_argument('--train-nets',type=int,default=30);p.add_argument('--test-width',type=int,default=256);p.add_argument('--test-nets',type=int,default=5);p.add_argument('--samples-train',type=int,default=8000);p.add_argument('--samples-test',type=int,default=18000);p.add_argument('--per-case',type=int,default=1600);p.add_argument('--seed',type=int,default=20260802);p.add_argument('--out',type=Path,default=Path('/mnt/data/whest_ml/joint_synth.json'));a=p.parse_args();layers=[8,16,24,30];tr=gen(a.train_width,32,a.train_nets,a.samples_train,layers,a.seed);X,y=train_rows(tr,a.per_case,a.seed);del tr;print(json.dumps({'rows':len(y)}),flush=True);m=HistGradientBoostingRegressor(max_iter=400,max_leaf_nodes=63,learning_rate=.05,l2_regularization=4,min_samples_leaf=50,random_state=a.seed);m.fit(X,y);pickle.dump(m,open(a.out.with_suffix('.pkl'),'wb'));te=gen(a.test_width,32,a.test_nets,a.samples_test,layers,a.seed+900000);r=ev(m,te);a.out.write_text(json.dumps({'config':vars(a),'result':r},indent=2,default=str));print(json.dumps(r['summary']))
if __name__=='__main__':main()
