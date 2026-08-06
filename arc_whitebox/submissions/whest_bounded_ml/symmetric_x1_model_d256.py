from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
sys.path.insert(0,'/mnt/data/whest_ml')
from joint_feature_closure_synth import gen, train_rows, ev
SEED=20260806; OUT=Path('/mnt/data/whest_ml/symmetric_x1_model_d256')
torch.manual_seed(SEED);np.random.seed(SEED)
def sym_base(X):
 l,ai,aj,rho=X[:,0],X[:,1],X[:,2],X[:,3];d=ai-aj
 return np.column_stack([l,ai+aj,ai*aj,np.abs(d),rho]).astype(np.float32),d.astype(np.float32),X[:,8].astype(np.float32),X[:,9].astype(np.float32)
class CoefNet(nn.Module):
 def __init__(self,w=96):
  super().__init__();self.body=nn.Sequential(nn.Linear(5,w),nn.SiLU(),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,2))
 def forward(self,b,d,x1,x1a):
  c=self.body(b);return c[:,0]*x1+(d*c[:,1])*x1a
layers=[8,16,24,30]
print('train cases',flush=True);tr=gen(256,32,18,10000,layers,SEED);X,y=train_rows(tr,1200,SEED);del tr
B,d,x1,x1a=sym_base(X);rng=np.random.default_rng(SEED);p=rng.permutation(len(y));nv=len(y)//8;va=p[:nv];tt=p[nv:]
mean=B[tt].mean(0);std=B[tt].std(0)+1e-6;B=(B-mean)/std
D=TensorDataset(*map(torch.from_numpy,[B[tt],d[tt],x1[tt],x1a[tt],y[tt]]));loader=DataLoader(D,batch_size=8192,shuffle=True)
m=CoefNet();opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=3e-5);bestloss=1e9;best=None;stale=0
for ep in range(100):
 m.train()
 for b,dd,xs,xa,yy in loader:
  pr=m(b,dd,xs,xa);loss=((pr-yy)**2).mean();opt.zero_grad();loss.backward();opt.step()
 m.eval()
 with torch.no_grad():vl=float(((m(torch.from_numpy(B[va]),torch.from_numpy(d[va]),torch.from_numpy(x1[va]),torch.from_numpy(x1a[va]))-torch.from_numpy(y[va]))**2).mean())
 if vl<bestloss:bestloss=vl;best={k:v.detach().clone() for k,v in m.state_dict().items()};stale=0
 else:stale+=1
 if ep%10==0:print(json.dumps({'epoch':ep,'val':vl}),flush=True)
 if stale>=15:break
m.load_state_dict(best);torch.save({'state_dict':m.state_dict(),'mean':mean,'std':std},OUT.with_suffix('.pt'))
print('test cases',flush=True);te=gen(256,32,10,18000,layers,SEED+900000)
class W:
 def predict(self,Z):
  b,dd,xs,xa=sym_base(Z);b=(b-mean)/std
  with torch.no_grad():return m(torch.from_numpy(b),torch.from_numpy(dd),torch.from_numpy(xs),torch.from_numpy(xa)).numpy()
r=ev(W(),te);OUT.with_suffix('.json').write_text(json.dumps({'best_val':bestloss,'parameters':sum(q.numel() for q in m.parameters()),'result':r},indent=2));print(json.dumps(r['summary']),flush=True)
