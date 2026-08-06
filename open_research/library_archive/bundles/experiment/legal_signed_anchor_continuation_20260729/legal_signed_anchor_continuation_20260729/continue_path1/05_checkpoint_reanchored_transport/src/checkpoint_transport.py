#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
from typing import Any
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import frozen_reference_impl as fr
from base import relu_bivariate,relu_univariate,observed_covariance,observed_marginals,anchor_from_defect,lower_anchor_selected,bootstrap_ratio
D=fr.D;TARGET=fr.TARGET
DEFAULT_CHECKPOINTS=(0,4,8,12,16,20,24,28)

def propagate(n,xk,cps):
 ws,_,_=fr.make_weights(n);x=torch.from_numpy(xk.copy());gm=np.zeros(D);gc=np.eye(D);states={};target={};Ht=Yt=None
 with torch.no_grad():
  for l,wt in enumerate(ws):
   w=wt.double().numpy();pre=x@wt;post=torch.relu(pre)
   gpm=gm@w;gpc=w.T@gc@w;gm,gc=relu_bivariate(gpm,gpc)
   # Transport existing checkpoint defects without injecting fresh Gaussian source.
   if l>0:
    kpm,kpv=observed_marginals(pre);base_m,base_v,_=relu_univariate(kpm,kpv);emp=(pre>0).double().mean(0).cpu().numpy()
    for cp,(dm,dc) in list(states.items()):
     pdm=dm@w;pdc=w.T@dc@w;tm,tv,tg=relu_univariate(kpm+pdm,np.maximum(kpv+np.diag(pdc),1e-18));ndm=tm-base_m;ndv=tv-base_v;gate=np.sqrt(np.maximum(tg*emp,0));ndc=pdc*np.outer(gate,gate);np.fill_diagonal(ndc,ndv);states[cp]=(ndm,.5*(ndc+ndc.T))
   if l in cps:
    km,kc=observed_covariance(post);states[l]=(gm-km,gc-kc)
   x=post
   if l==TARGET:
    Ht=x.clone();target={str(cp):(dm.copy(),dc.copy()) for cp,(dm,dc) in states.items()}
   if l==len(ws)-1:Yt=x.clone()
 assert Ht is not None and Yt is not None
 return ws,Ht,Yt,target

def run(n,xk,cps,truth_n,chunk):
 t=time.perf_counter();ws,wh,seed=fr.make_weights(n);ws2,Ht,Yt,target=propagate(n,xk,cps);assert all(torch.equal(a,b) for a,b in zip(ws,ws2));H=Ht.double().numpy();Y=Yt.double().numpy();m=H.mean(0);rho=fr.chi_mean(D);Q=fr.sample_anchor_matrix(H,m,rho);idx,dirs=fr.sample_row_probes(Q);X=fr.radial_features_sample_rows(H,m,idx,dirs,rho);fit=fr.fit_crossfit(X,Y);sa=fr.contract_rows(Q,idx,dirs);base,_=fr.estimate_from_fit(fit,sa)
 r1=fr.stream_reference(ws,truth_n,81_000_000+2*n,chunk);r2=fr.stream_reference(ws,truth_n,81_000_001+2*n,chunk);ty=.5*(r1['y']+r2['y']);mu=.5*(r1['mu']+r2['mu']);M=.5*(r1['M']+r2['M']);oracle=lower_anchor_selected(m,mu,np.diag(M),np.sum(M[idx]*dirs,axis=1),idx,dirs);alph=np.array([-1,-.5,-.25,0,.025,.05,.1,.2,.35,.5,.75,1,1.25])
 mse=lambda p:float(np.mean((p-ty)**2));ub=lambda p:float(np.mean((p-r1['y'])*(p-r2['y'])));bm=mse(base);methods={}
 for cp in cps:
  dm,dc=target[str(cp)];a=anchor_from_defect(H,dm,dc,idx,dirs);pred=[fr.estimate_from_fit(fit,sa+s*a)[0] for s in alph];methods[str(cp)]={'cosine':float(a@oracle/max(np.linalg.norm(a)*np.linalg.norm(oracle),1e-30)),'relative_error':float(np.linalg.norm(a-oracle)/max(np.linalg.norm(oracle),1e-30)),'anchor_norm':float(np.linalg.norm(a)),'mse_grid':[mse(p) for p in pred],'unbiased_grid':[ub(p) for p in pred]}
 op,_=fr.estimate_from_fit(fit,sa+oracle)
 return {'network_id':n,'weight_seed':seed,'weight_sha256':wh,'checkpoints':list(cps),'truth_n':truth_n,'baseline_mse':bm,'oracle_mse':mse(op),'alpha_grid':alph.tolist(),'methods':methods,'runtime_seconds':time.perf_counter()-t}

def summarize(rs,tune_n):
 base=np.array([r['baseline_mse'] for r in rs]);grid=np.array(rs[0]['alpha_grid']);ti=np.arange(tune_n);vi=np.arange(tune_n,len(rs));rows=[]
 for cp in rs[0]['methods']:
  cm=np.array([r['methods'][cp]['mse_grid'] for r in rs]);
  for j,a in enumerate(grid):
   rat=cm[ti,j]/base[ti];rows.append((float(cm[ti,j].sum()/base[ti].sum()),float(rat.max()),-int((rat<1).sum()),int(cp),j))
 safe=[x for x in rows if x[1]<=1.15];sel=min(safe or rows);_,_,_,cp,j=sel;cm=np.array([r['methods'][str(cp)]['mse_grid'] for r in rs]);
 def block(ix,seed):
  c=cm[ix,j];rat=c/base[ix];return {'n':len(ix),'candidate_over_base':float(c.sum()/base[ix].sum()),'ci95':bootstrap_ratio(base[ix],c,seed),'wins':int((rat<1).sum()),'worst':float(rat.max()),'per_network':rat.tolist(),'mean_cosine':float(np.mean([rs[i]['methods'][str(cp)]['cosine'] for i in ix]))}
 return {'selected_checkpoint':cp,'selected_alpha':float(grid[j]),'tune_ids':[rs[i]['network_id'] for i in ti],'validation_ids':[rs[i]['network_id'] for i in vi],'tuning':block(ti,20260810),'validation':block(vi,20260811) if len(vi) else {},'top_safe':[{'ratio':x[0],'worst':x[1],'wins':-x[2],'checkpoint':x[3],'alpha':float(grid[x[4]])} for x in sorted(safe)[:20]],'top_any':[{'ratio':x[0],'worst':x[1],'wins':-x[2],'checkpoint':x[3],'alpha':float(grid[x[4]])} for x in sorted(rows)[:20]]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--network',type=int);p.add_argument('--checkpoints',type=int,nargs='*',default=list(DEFAULT_CHECKPOINTS));p.add_argument('--truth-n',type=int,default=8192);p.add_argument('--chunk',type=int,default=4096);p.add_argument('--threads',type=int,default=8);p.add_argument('--out',type=Path,required=True);p.add_argument('--records-dir',type=Path);p.add_argument('--tune-n',type=int,default=6);a=p.parse_args();torch.set_num_threads(a.threads)
 if a.records_dir:
  rs=[json.loads(q.read_text()) for q in sorted(a.records_dir.glob('network_*.json'))];z=summarize(rs,a.tune_n);a.out.write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2));return
 xk,_=fr.make_kerdock();z=run(a.network,xk,tuple(a.checkpoints),a.truth_n,a.chunk);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(z));print(json.dumps({'network':a.network,'runtime':z['runtime_seconds'],'best':{cp:round(min(v['mse_grid'])/z['baseline_mse'],3) for cp,v in z['methods'].items()}},indent=2))
if __name__=='__main__':main()
