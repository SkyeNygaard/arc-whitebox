#!/usr/bin/env python3
from __future__ import annotations
import argparse,gc,json,math,sys,time
from pathlib import Path
import numpy as np
T4=Path('/mnt/data/work/T4_legal_layer31_anchor_hedge_20260729_review/T4_legal_layer31_anchor_hedge_20260729/code');sys.path.insert(0,str(T4));import frozen_reference_impl as fr
from run_stein_screen import effective_j,qmc_final,forward_kerdock,gegen_raw,highpass,sphere_stein,normalize_cols,mse_obs,mse_unb
D=256;RHO=fr.chi_mean(D);NB=129;RPB=512;OUT=Path('/mnt/data/work/new_opportunities/aligned');OUT.mkdir(parents=True,exist_ok=True)
KS=[2,4,8,16];RIDGES=[1e-8,1e-4,1e-2,.1,1.,10.,100.]
def make_groups(xk,U):
 t=(xk@U[:,:16])/RHO;out={x:[] for x in ['harm68','harm6810','sigmoid','tanh','sigtanh']}
 for q in range(16):
  z=t[:,q];out['harm68'].append(normalize_cols(np.stack([gegen_raw(z,6),gegen_raw(z,8)],1)));out['harm6810'].append(normalize_cols(np.stack([gegen_raw(z,6),gegen_raw(z,8),gegen_raw(z,10)],1)))
  sg=normalize_cols(np.stack([highpass(z,sphere_stein(z,'sigmoid',8.,b)) for b in (-.1,0,.1)],1));th=normalize_cols(np.stack([highpass(z,sphere_stein(z,'tanh',6.,b)) for b in (-.1,0,.1)],1));out['sigmoid'].append(sg);out['tanh'].append(th);out['sigtanh'].append(normalize_cols(np.c_[sg,th]))
 return out
def raw(G,z,m):
 X=G[m];y=z[m];return (len(X),X.sum(0),float(y.sum()),X.T@X,X.T@y)
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def solve(st,ridge):
 n,sg,sy,gg,gy=st;mg=sg/n;my=sy/n;gc=gg-n*np.outer(mg,mg);yc=gy-n*mg*my;scale=max(np.trace(gc)/len(mg),1e-30);return np.linalg.solve(gc+ridge*scale*np.eye(len(mg)),yc),mg,my
def metric(base,p,truth,a,b):
 e=base-truth;c=p-base;inn=float(e@c/D);nn=float(c@c/D);ee=float(e@e/D);return {'observed_mse':mse_obs(p,truth),'unbiased_mse':mse_unb(p,a,b),'inner':inn,'norm_sq':nn,'cosine':float(-inn/max(math.sqrt(max(ee*nn,0)),1e-30)),'oracle_alpha':float(-inn/max(nn,1e-30))}
def run(net,nref,xk,bid,stage):
 t0=time.time();ws,wh,_=fr.make_weights(net);a=qmc_final(ws,nref,2000000+2*net);b=qmc_final(ws,nref,2000001+2*net);truth=(a+b)/2;Y,gates=forward_kerdock(xk,ws);Y=Y.astype(float);base=Y.mean(0);U,s,Vh=np.linalg.svd(effective_j(ws,gates),full_matrices=False);V=Vh.T;Z=Y@V[:,:16];gs=make_groups(xk,U)
 # Stats indexed fam/q: full and fold test/train.
 ST={}
 allm=np.ones(len(Y),bool)
 for fam,lst in gs.items():
  ST[fam]=[]
  for q,G in enumerate(lst):
   fu=raw(G,Z[:,q],allm);te=[];tr=[]
   for f in range(4):
    m=(bid%4)==f;t=raw(G,Z[:,q],m);te.append(t);tr.append(sub(fu,t))
   ST[fam].append((fu,tr,te))
 r={'stage':stage,'network_id':net,'nref_per_half':nref,'weight_sha256':wh,'base_observed_mse':mse_obs(base,truth),'base_unbiased_mse':mse_unb(base,a,b),'reference_noise_mse':float(np.mean((a-b)**2)/4),'configs':{}}
 for fam in gs:
  for ridge in RIDGES:
   full_corr=[];fold_corr=[[] for _ in range(4)]
   for q in range(16):
    fu,tr,te=ST[fam][q];coef,mg,my=solve(fu,ridge);full_corr.append(-(mg@coef)*V[:,q])
    for f in range(4):
     coef,_,_=solve(tr[f],ridge);n,sg,sy,_,_=te[f];fold_corr[f].append(-((sg/n)@coef)*V[:,q])
   for k in KS:
    pf=base+np.sum(full_corr[:k],axis=0);vals=[]
    for f in range(4):
     m=(bid%4)==f;vals.append(Y[m].mean(0)+np.sum(fold_corr[f][:k],axis=0))
    pc=np.mean(np.stack(vals),axis=0)
    key=f'{fam}_k{k}|ridge={ridge:g}';r['configs'][key]={'full':metric(base,pf,truth,a,b),'cf4':metric(base,pc,truth,a,b)}
 r['seconds']=time.time()-t0;(OUT/f'{stage}_network_{net}.json').write_text(json.dumps(r,sort_keys=True));best=min((v['full']['unbiased_mse'],k) for k,v in r['configs'].items());print(json.dumps({'network':net,'seconds':r['seconds'],'base_unb':r['base_unbiased_mse'],'noise_frac':r['reference_noise_mse']/max(r['base_observed_mse'],1e-30),'best_full':best}),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('nets',nargs='+',type=int);ap.add_argument('--nref',type=int,default=262144);ap.add_argument('--stage',default='screen');a=ap.parse_args();xk,_=fr.make_kerdock();bid=np.repeat(np.arange(NB),RPB)
 for n in a.nets:run(n,a.nref,xk,bid,a.stage);gc.collect()
if __name__=='__main__':main()
