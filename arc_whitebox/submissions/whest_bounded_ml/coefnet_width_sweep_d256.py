from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
sys.path.insert(0,'/mnt/data/whest_ml')
from joint_feature_closure_synth import gen,train_rows,ev
S=20260809; layers=[8,16,24,30]; out=Path('/mnt/data/whest_ml/coefnet_width_sweep_d256.json')
torch.manual_seed(S);np.random.seed(S)
def feat(X):
 l,ai,aj,r=X[:,0],X[:,1],X[:,2],X[:,3];d=ai-aj
 return np.column_stack([l,ai+aj,ai*aj,np.abs(d),r]).astype(np.float32),d.astype(np.float32),X[:,8].astype(np.float32),X[:,9].astype(np.float32)
class M(nn.Module):
 def __init__(self,w):super().__init__();self.b=nn.Sequential(nn.Linear(5,w),nn.SiLU(),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,2))
 def forward(self,b,d,x,xA):c=self.b(b);return c[:,0]*x+d*c[:,1]*xA
print('generate train',flush=True);tr=gen(256,32,14,9000,layers,S);X,y=train_rows(tr,1100,S);del tr
B,d,x,xA=feat(X);rng=np.random.default_rng(S);p=rng.permutation(len(y));nv=len(y)//8;va=p[:nv];tt=p[nv:];mean=B[tt].mean(0);std=B[tt].std(0)+1e-6;B=(B-mean)/std
D=TensorDataset(*map(torch.from_numpy,[B[tt],d[tt],x[tt],xA[tt],y[tt]]));loader=DataLoader(D,batch_size=8192,shuffle=True)
print('generate test',flush=True);te=gen(256,32,8,16000,layers,S+900000)
res={}
for w in [8,16,24,32,48,64,96]:
 torch.manual_seed(S+w);m=M(w);opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=3e-5);bestloss=1e9;best=None;stale=0
 for ep in range(110):
  m.train()
  for q in loader:
   pr=m(*q[:4]);loss=((pr-q[4])**2).mean();opt.zero_grad();loss.backward();opt.step()
  m.eval()
  with torch.no_grad():vl=float(((m(torch.from_numpy(B[va]),torch.from_numpy(d[va]),torch.from_numpy(x[va]),torch.from_numpy(xA[va]))-torch.from_numpy(y[va]))**2).mean())
  if vl<bestloss:bestloss=vl;best={k:v.detach().clone() for k,v in m.state_dict().items()};stale=0
  else:stale+=1
  if stale>=15:break
 m.load_state_dict(best)
 class W:
  def predict(self,Z):
   b,dd,xx,aa=feat(Z);b=(b-mean)/std
   with torch.no_grad():return m(torch.from_numpy(b),torch.from_numpy(dd),torch.from_numpy(xx),torch.from_numpy(aa)).numpy()
 r=ev(W(),te)['summary'];r['params']=sum(z.numel() for z in m.parameters());r['val']=bestloss;r['epochs']=ep+1;res[str(w)]=r
 torch.save({'state_dict':m.state_dict(),'mean':mean,'std':std,'width':w},Path(f'/mnt/data/whest_ml/coefnet_d256_w{w}.pt'))
 print(w,json.dumps(r),flush=True)
out.write_text(json.dumps(res,indent=2));print('done',flush=True)
