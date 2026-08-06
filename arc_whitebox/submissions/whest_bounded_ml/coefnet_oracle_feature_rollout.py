from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,torch
from torch import nn
sys.path.insert(0,'/mnt/data/whest_prior_followup');sys.path.insert(0,'/mnt/data/whest_ml')
from experiment_suite import make_network,gaussian_relu_covariance
S=20260811;N=22000;NETS=4;D=256;DEPTH=32;OUT=Path('/mnt/data/whest_ml/coefnet_oracle_feature_rollout.json')
class M(nn.Module):
 def __init__(self,w=96):super().__init__();self.body=nn.Sequential(nn.Linear(5,w),nn.SiLU(),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,2))
 def forward(self,b,d,x,xA):c=self.body(b);return c[:,0]*x+d*c[:,1]*xA
ck=torch.load('/mnt/data/whest_ml/symmetric_x1_model_d256.pt',map_location='cpu',weights_only=False);m=M();m.load_state_dict(ck['state_dict']);m.eval();mn=ck['mean'];sd=ck['std'];iu,ju=np.triu_indices(D,1)
def cov(x):
 x=x.astype(np.float64);mu=x.mean(0);q=x-mu;return mu,q.T@q/(len(x)-1)
def predict(layer,mu,sig,Capprox,k21):
 s=np.sqrt(np.maximum(sig,1e-12));rho=np.clip(Capprox/np.maximum(np.outer(s,s),1e-12),-1,1);a=mu/s;d=a[iu]-a[ju];x1=.5*(k21[iu,ju]+k21[ju,iu]);x1a=.5*(k21[iu,ju]-k21[ju,iu]);b=np.column_stack([np.full(len(iu),(layer+1)/DEPTH),a[iu]+a[ju],a[iu]*a[ju],np.abs(d),rho[iu,ju]]).astype('f4');b=(b-mn)/sd
 with torch.no_grad():p=m(torch.from_numpy(b),torch.from_numpy(d.astype('f4')),torch.from_numpy(x1.astype('f4')),torch.from_numpy(x1a.astype('f4'))).numpy()
 return p,rho
rows=[]
for ni in range(NETS):
 W=make_network(D,DEPTH,S+ni*117);rng=np.random.default_rng(S+100000+ni*131);h=rng.standard_normal((N,D),dtype=np.float32);truth=[]
 for li,w in enumerate(W):
  z=h@w;mu,C=cov(z);s=np.sqrt(np.maximum(np.diag(C),1e-12));zs=(z.astype(np.float64)-mu)/s;q=zs*zs-1;k21=q.T@zs/len(zs);hp=np.maximum(z,0);mh,Ch=cov(hp);truth.append((mu,C,k21,mh,Ch));h=hp
 # exact first pre state
 Cg=truth[0][1].copy();Cm=Cg.copy();netrows=[]
 for li in range(DEPTH):
  mu,Ctrue,k21,mh,Chtrue=truth[li];vartrue=np.diag(Ctrue)
  one={'layer':li,'gaussian_pre_var':float(np.sqrt(np.mean(((np.diag(Cg)-vartrue)/np.maximum(vartrue,1e-12))**2))),'model_pre_var':float(np.sqrt(np.mean(((np.diag(Cm)-vartrue)/np.maximum(vartrue,1e-12))**2)))}
  netrows.append(one)
  if li==DEPTH-1:break
  # Stabilized diagnostic: oracle current marginal mean/variance, recursively propagated off-diagonal rho.
  def post(Ccur,use_model):
   Cuse=Ccur.copy();np.fill_diagonal(Cuse,vartrue);s=np.sqrt(vartrue);rho=np.clip(Cuse/np.maximum(np.outer(s,s),1e-12),-1,1)
   _,base=gaussian_relu_covariance(mu,s,rho);np.fill_diagonal(base,np.diag(Chtrue))
   if use_model:
    p,_=predict(li,mu,vartrue,Cuse,k21);scale=s[iu]*s[ju];base[iu,ju]+=p*scale;base[ju,iu]+=p*scale
   return (base+base.T)/2
  Pg=post(Cg,False);Pm=post(Cm,True);w=W[li+1].astype(np.float64);Cg=w.T@Pg@w;Cm=w.T@Pm@w
 final=netrows[-1];rows.append({'net':ni,'layers':netrows,'final_gaussian':final['gaussian_pre_var'],'final_model':final['model_pre_var'],'gain':final['gaussian_pre_var']/max(final['model_pre_var'],1e-30)});print(json.dumps({k:rows[-1][k] for k in ['net','final_gaussian','final_model','gain']}),flush=True)
OUT.write_text(json.dumps({'config':{'samples':N,'nets':NETS},'rows':rows,'summary':{'gaussian':float(np.mean([r['final_gaussian'] for r in rows])),'model':float(np.mean([r['final_model'] for r in rows])),'gain':float(np.mean([r['final_gaussian'] for r in rows])/np.mean([r['final_model'] for r in rows])),'fraction':float(np.mean([r['final_model']<r['final_gaussian'] for r in rows]))}},indent=2));print(json.dumps(json.loads(OUT.read_text())['summary']),flush=True)
