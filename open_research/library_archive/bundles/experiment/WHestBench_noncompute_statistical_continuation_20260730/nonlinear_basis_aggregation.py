from __future__ import annotations
import argparse,json,math,sys,time,gc
from pathlib import Path
import numpy as np, torch
SRC=Path('/mnt/data/work/T4_legal_layer31_anchor_hedge_20260729_review/T4_legal_layer31_anchor_hedge_20260729/code')
sys.path.insert(0,str(SRC))
import frozen_reference_impl as fr
D=fr.D

def design(seed):
 radius=fr.chi_mean(D);H=fr.walsh_hadamard()/math.sqrt(D);R=fr.haar_rotation(seed);blocks=[]
 for u in range(128):
  b=(H*fr.kerdock_chirp(u)[None,:])@R;blocks += [(radius*b).astype(np.float32),(-radius*b).astype(np.float32)]
 c=(radius*R).astype(np.float32);blocks += [c,-c]
 return np.concatenate(blocks)

def basis_means(ws,X):
 x=torch.from_numpy(X)
 with torch.no_grad():
  for w in ws:x=torch.relu(x@w)
 Y=x.double().numpy();B=Y.reshape(129,2,256,D).mean((1,2));return B

def geom_median(X,it=50):
 y=np.median(X,axis=0)
 for _ in range(it):
  d=np.linalg.norm(X-y,axis=1);w=1/np.maximum(d,1e-12);yn=(w[:,None]*X).sum(0)/w.sum()
  if np.linalg.norm(yn-y)<1e-10*max(np.linalg.norm(y),1):break
  y=yn
 return y

def huber_coord(X,c):
 med=np.median(X,0);s=1.4826*np.median(np.abs(X-med),0)+1e-12;y=med.copy()
 for _ in range(20):
  r=(X-y)/s;w=np.minimum(1,c/np.maximum(np.abs(r),1e-12));yn=(w*X).sum(0)/np.maximum(w.sum(0),1e-12)
  if np.linalg.norm(yn-y)<1e-10*max(np.linalg.norm(y),1):break
  y=yn
 return y

def methods(B):
 base=B.mean(0);med=np.median(B,0);out={}
 out['median']=med
 for q in [.05,.1,.2]:
  k=int(round(q*len(B)));S=np.sort(B,axis=0);out[f'trim{q:g}']=S[k:len(B)-k].mean(0)
 for c in [1.,1.5,2.,3.]:out[f'huber{c:g}']=huber_coord(B,c)
 out['geomed']=geom_median(B)
 dist=np.linalg.norm(B-med,axis=1)
 for k in [32,64,96]:out[f'central{k}']=B[np.argsort(dist)[:k]].mean(0)
 eps=1e-12
 for p in [.25,.5,.75,1.25,1.5,2.,3.]:out[f'power{p:g}']=np.maximum(np.mean(np.maximum(B,eps)**p,axis=0),eps)**(1/p)
 # Log mean/geometric mean.
 out['geomean_coord']=np.exp(np.mean(np.log(np.maximum(B,eps)),axis=0))
 return base,out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,default=7400);ap.add_argument('--networks',type=int,default=12);ap.add_argument('--rots',type=int,nargs='+',default=[3,11,97]);ap.add_argument('--out',type=Path,default=Path('/mnt/data/competition_relevance_20260730/nonlinear_basis_aggregation/data.npz'));a=ap.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True);torch.set_num_threads(min(8,torch.get_num_threads()));Xs={r:design(r) for r in a.rots};records=[]
 for n in range(a.start,a.start+a.networks):
  t=time.time();ws,_,_=fr.make_weights(n);allB={r:basis_means(ws,Xs[r]) for r in a.rots};bases={r:allB[r].mean(0) for r in a.rots}
  for r in a.rots:
   ref=np.mean([bases[s] for s in a.rots if s!=r],axis=0);base,sources=methods(allB[r]);records.append({'network':n,'rot':r,'base':base,'ref':ref,'sources':sources})
  print(json.dumps({'network':n,'seconds':time.time()-t}),flush=True);gc.collect()
 names=sorted(records[0]['sources']);np.savez_compressed(a.out,network=np.array([x['network'] for x in records]),rot=np.array([x['rot'] for x in records]),base=np.stack([x['base'] for x in records]),ref=np.stack([x['ref'] for x in records]),names=np.array(names),sources=np.stack([[x['sources'][k] for k in names] for x in records]))
if __name__=='__main__':main()
