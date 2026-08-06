#!/usr/bin/env python3
from __future__ import annotations
import argparse,gc,json,math,os,sys,time
from pathlib import Path
import numpy as np, torch
T4=Path('/mnt/data/work/T4_legal_layer31_anchor_hedge_20260729_review/T4_legal_layer31_anchor_hedge_20260729/code')
sys.path.insert(0,str(T4)); import frozen_reference_impl as fr
from run_stein_screen import sphere_stein,gegen_raw,highpass,normalize_cols,effective_j,qmc_final,forward_kerdock,mse_obs,mse_unb
D=256;RHO=fr.chi_mean(D);NB=129;RPB=512
OUT=Path('/mnt/data/work/new_opportunities/grid');OUT.mkdir(parents=True,exist_ok=True)
RIDGES=[1e-8,1e-5,1e-3,1e-1,1.,10.,100.];RANKS=[0,2,4,8,16];KS=[2,4,8]

def build_master(xk,U):
 t=(xk@U[:,:8])/RHO; blocks=[]; mp={}; pos=0
 for n in (6,8,10):
  G=normalize_cols(np.stack([gegen_raw(t[:,q],n) for q in range(8)],1)); blocks.append(G);mp[f'h{n}']=np.arange(pos,pos+8);pos+=8
 for fam,k in [('sigmoid',8.),('tanh',6.)]:
  cols=[]
  for q in range(8):
   for b in (-.1,0,.1): cols.append(highpass(t[:,q],sphere_stein(t[:,q],fam,k,b)))
  G=normalize_cols(np.stack(cols,1));blocks.append(G);mp[fam]=np.arange(pos,pos+24).reshape(8,3);pos+=24
 M=np.concatenate(blocks,1)
 sets={}
 for k in KS:
  h68=np.r_[mp['h6'][:k],mp['h8'][:k]];h6810=np.r_[h68,mp['h10'][:k]]
  sg=mp['sigmoid'][:k].ravel();th=mp['tanh'][:k].ravel()
  sets[f'harm_k{k}_d6_8']=h68;sets[f'harm_k{k}_d6_8_10']=h6810
  sets[f'sigmoid_k{k}_hp']=sg;sets[f'tanh_k{k}_hp']=th;sets[f'sigtanh_k{k}_hp']=np.r_[sg,th]
 return M,sets

def raw_stats(G,Y,mask):
 X=G[mask];Z=Y[mask];n=len(X)
 return {'n':n,'sumg':X.sum(0),'sumy':Z.sum(0),'gg':X.T@X,'gy':X.T@Z}
def subtract(a,b):return {k:(a[k]-b[k]) for k in a}
def centered(st,idx):
 n=st['n'];mg=st['sumg'][idx]/n;my=st['sumy']/n
 gg=st['gg'][np.ix_(idx,idx)]-n*np.outer(mg,mg)
 gy=st['gy'][idx]-n*np.outer(mg,my)
 return gg,gy,mg,my

def solve(gg,gy,ridge,rank):
 scale=max(float(np.trace(gg))/max(1,len(gg)),1e-30);B=np.linalg.solve(gg+ridge*scale*np.eye(len(gg)),gy)
 if rank and rank<min(B.shape):
  u,s,vh=np.linalg.svd(B,full_matrices=False);B=(u[:,:rank]*s[:rank])@vh[:rank]
 return B

def metric(base,pred,truth,a,b):
 e=base-truth;c=pred-base;inn=float(e@c/D);nn=float(c@c/D);ee=float(e@e/D)
 return {'observed_mse':mse_obs(pred,truth),'unbiased_mse':mse_unb(pred,a,b),'inner':inn,'norm_sq':nn,'cosine':float(-inn/max(math.sqrt(max(ee*nn,0)),1e-30)),'oracle_alpha':float(-inn/max(nn,1e-30))}

def run(net,nref,xk,bid):
 t0=time.time();ws,whash,_=fr.make_weights(net);a=qmc_final(ws,nref,1000000+2*net);b=qmc_final(ws,nref,1000001+2*net);truth=(a+b)/2
 Y,gates=forward_kerdock(xk,ws);Y=Y.astype(np.float64,copy=False);base=Y.mean(0)
 U=np.linalg.svd(effective_j(ws,gates),full_matrices=False)[0];G,sets=build_master(xk,U)
 allmask=np.ones(len(G),bool);fullst=raw_stats(G,Y,allmask);testst=[];trainst=[]
 for f in range(4):
  te=(bid%4)==f;ts=raw_stats(G,Y,te);testst.append(ts);trainst.append(subtract(fullst,ts))
 row={'network_id':net,'nref_per_half':nref,'weight_sha256':whash,'base_observed_mse':mse_obs(base,truth),'base_unbiased_mse':mse_unb(base,a,b),'reference_noise_mse':float(np.mean((a-b)**2)/4),'configs':{}}
 for fname,idx in sets.items():
  ggf,gyf,mgf,myf=centered(fullst,idx)
  folddata=[]
  for tr,te in zip(trainst,testst):
   gg,gy,mg,my=centered(tr,idx);mgt=te['sumg'][idx]/te['n'];myt=te['sumy']/te['n'];folddata.append((gg,gy,mgt,myt,te['n']))
  for ridge in RIDGES:
   Bf=solve(ggf,gyf,ridge,0);svdf=np.linalg.svd(Bf,full_matrices=False)
   Bs=[]
   for gg,gy,mgt,myt,n in folddata:
    B=solve(gg,gy,ridge,0);Bs.append((B,np.linalg.svd(B,full_matrices=False),mgt,myt,n))
   for rank in RANKS:
    if rank and rank>min(len(idx),16):continue
    if rank:
     u,s,vh=svdf;Br=(u[:,:rank]*s[:rank])@vh[:rank]
    else:Br=Bf
    full=myf-mgf@Br
    vals=[];ns=[]
    for B,sv,mgt,myt,n in Bs:
     if rank:
      u,s,vh=sv;B=(u[:,:rank]*s[:rank])@vh[:rank]
     vals.append(myt-mgt@B);ns.append(n)
    cf=np.average(np.stack(vals),axis=0,weights=ns)
    key=f'{fname}|ridge={ridge:g}|rank={rank or "full"}'
    row['configs'][key]={'features':len(idx),'full':metric(base,full,truth,a,b),'cf4':metric(base,cf,truth,a,b)}
 row['seconds']=time.time()-t0;(OUT/f'grid_network_{net}.json').write_text(json.dumps(row,sort_keys=True))
 best=min((v['cf4']['observed_mse'],k) for k,v in row['configs'].items());print(json.dumps({'network':net,'seconds':row['seconds'],'base':row['base_observed_mse'],'best_cf4':best}),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('networks',nargs='+',type=int);ap.add_argument('--nref',type=int,default=32768);args=ap.parse_args();xk,_=fr.make_kerdock();bid=np.repeat(np.arange(NB),RPB)
 for n in args.networks:run(n,args.nref,xk,bid);gc.collect()
if __name__=='__main__':main()
