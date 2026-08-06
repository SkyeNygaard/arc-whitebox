from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/mnt/data/arc_research/agent34/agent3_agent4_repro');sys.path.insert(0,str(ROOT))
from sampling import full_real_kerdock_bases,haar_rotation,chi_mean
import agent34_screen_fast as a
D=256

def make_kerdock(seed=3):
 q=haar_rotation(D,seed);r=chi_mean(D)
 return torch.cat([torch.cat([b@q,-(b@q)],0)*r for b in full_real_kerdock_bases(D)],0).contiguous()

def forward_target_final(x,ws,target=29):
 ht=None
 with torch.no_grad():
  for l,w in enumerate(ws):
   x=torch.relu(x@w)
   if l==target:ht=x.clone()
 return ht,x

def radial_features(H,m,L,R,rho):
 # L,R: D x p, implements sum_ij L_i R_j psi_ij
 h2=H*H; hm=H*m[None,:]
 t1=(h2@L)*(H@R)/(rho*rho)
 t2=(h2@L)*(m@R)[None,:]*(D/(D+1))/(rho*rho)
 t3=(hm@L)*(H@R)*(2*D/(D+1))/(rho*rho)
 t4=(m*m@L)[None,:]*(H@R)*(2/(D+1))
 return t1-t2-t3+t4

def sample_anchor_matrix(H,m,rho):
 raw=(H*H).T@H/len(H);M=H.T@H/len(H);m2=np.diag(M)
 return raw/(rho*rho)-D/(D+1)*m2[:,None]*m[None,:]/(rho*rho)-2*D/(D+1)*m[:,None]*M/(rho*rho)+2/(D+1)*(m*m)[:,None]*m[None,:]

def exact_anchor_matrix(mu,M,raw,m):
 return (raw-np.diag(M)[:,None]*m[None,:]-2*m[:,None]*M+2*(m*m)[:,None]*mu[None,:])/(D+1)

def stream_truth(ws,target,total_n,seed,chunk=8192):
 eng=torch.quasirandom.SobolEngine(D,scramble=True,seed=seed);ys=np.zeros(D);hs=np.zeros(D);M=np.zeros((D,D));raw=np.zeros((D,D));done=0
 with torch.no_grad():
  while done<total_n:
   n=min(chunk,total_n-done);u=eng.draw(n,dtype=torch.float32).clamp_(1e-7,1-1e-7);x=math.sqrt(2)*torch.erfinv(2*u-1)
   h,y=forward_target_final(x,ws,target);H=h.double().cpu().numpy();Y=y.double().cpu().numpy()
   ys+=Y.sum(0);hs+=H.sum(0);M+=H.T@H;raw+=(H*H).T@H;done+=n
 return ys/total_n,hs/total_n,M/total_n,raw/total_n

def crossfit(X,Y,anchor,folds=6,ridge=.1):
 bid=np.repeat(np.arange(129),512);groups=np.array_split(np.arange(129),folds);est=np.zeros(D);tot=0
 for g in groups:
  te=np.isin(bid,g);tr=~te;xt=X[tr];yt=Y[tr];xc=xt-xt.mean(0);yc=yt-yt.mean(0)
  gram=xc.T@xc;scale=max(np.trace(gram)/len(anchor),1e-12);beta=np.linalg.solve(gram+ridge*scale*np.eye(len(anchor)),xc.T@yc)
  e=Y[te].mean(0)-(X[te].mean(0)-anchor)@beta;est+=e*te.sum();tot+=te.sum()
 return est/tot

def run_one(net,truth_n=131072,p=128,ridge=.1,target=29):
 t=time.time();ws=a.make_weights(51000+net);xk=make_kerdock();hk,yk=forward_target_final(xk,ws,target)
 H=hk.double().cpu().numpy();Y=yk.double().cpu().numpy();m=H.mean(0);rho=chi_mean(D);base=Y.mean(0)
 y1,mu1,M1,R1=stream_truth(ws,target,truth_n,710000+2*net);y2,mu2,M2,R2=stream_truth(ws,target,truth_n,710001+2*net)
 ty=.5*(y1+y2);bm=np.mean((base-ty)**2);yn=.5*np.mean((y1-y2)**2)
 mu=.5*(mu1+mu2);M=.5*(M1+M2);raw=.5*(R1+R2);E=exact_anchor_matrix(mu,M,raw,m);Q=sample_anchor_matrix(H,m,rho);defect=E-Q
 variants={}
 # diagonal probes
 score=np.abs(np.diag(Q));ix=np.argsort(score)[::-1][:p];L=np.eye(D)[:,ix];R=L.copy();variants['diag']=(L,R,ix)
 # sample-row probes
 score=np.linalg.norm(Q,axis=1);ix=np.argsort(score)[::-1][:p];L=np.eye(D)[:,ix];rr=Q[ix].copy();rr/=np.maximum(np.linalg.norm(rr,axis=1,keepdims=True),1e-30);R=rr.T;variants['sample_rows']=(L,R,ix)
 # oracle defect rows ceiling only
 score=np.linalg.norm(defect,axis=1);ix=np.argsort(score)[::-1][:p];L=np.eye(D)[:,ix];rr=defect[ix].copy();rr/=np.maximum(np.linalg.norm(rr,axis=1,keepdims=True),1e-30);R=rr.T;variants['oracle_defect_rows']=(L,R,ix)
 out={}
 for name,(L,R,ix) in variants.items():
  X=radial_features(H,m,L,R,rho);anchor=np.einsum('ir,ij,jr->r',L,E,R);pred=crossfit(X,Y,anchor,6,ridge);mse=np.mean((pred-ty)**2)
  A1=exact_anchor_matrix(mu1,M1,R1,m);A2=exact_anchor_matrix(mu2,M2,R2,m);a1=np.einsum('ir,ij,jr->r',L,A1,R);a2=np.einsum('ir,ij,jr->r',L,A2,R)
  out[name]={'mse':float(mse),'mse_ratio':float(mse/bm),'anchor_noise_ratio':float(.5*np.linalg.norm(a1-a2)/max(np.linalg.norm(anchor-X.mean(0)),1e-30)),'indices':ix.tolist()}
 return {'network':net,'baseline_mse':float(bm),'truth_noise_mse':float(yn),'truth_n_per_split':truth_n,'variants':out,'runtime_seconds':time.time()-t}

def main():
 p=argparse.ArgumentParser();p.add_argument('--nets',nargs='+',type=int,default=[0]);p.add_argument('--truth-n',type=int,default=131072);p.add_argument('--probes',type=int,default=128);p.add_argument('--ridge',type=float,default=.1);p.add_argument('--out',type=Path,default=Path('/mnt/data/arc_research/sparse_radial.json'));args=p.parse_args();torch.set_num_threads(min(16,torch.get_num_threads()))
 rs=[]
 for n in args.nets:
  r=run_one(n,args.truth_n,args.probes,args.ridge);rs.append(r);print(n,{k:round(v['mse_ratio'],4) for k,v in r['variants'].items()},'noise/base',round(r['truth_noise_mse']/r['baseline_mse'],3),flush=True)
 sm={k:{'aggregate_ratio':float(sum(r['variants'][k]['mse'] for r in rs)/sum(r['baseline_mse'] for r in rs)),'median':float(np.median([r['variants'][k]['mse_ratio'] for r in rs])),'wins':int(sum(r['variants'][k]['mse_ratio']<1 for r in rs)),'worst':float(max(r['variants'][k]['mse_ratio'] for r in rs))} for k in rs[0]['variants']}
 o={'config':vars(args)|{'out':str(args.out)},'records':rs,'summary':sm};args.out.write_text(json.dumps(o,indent=2));print(json.dumps(sm,indent=2))
if __name__=='__main__':main()
