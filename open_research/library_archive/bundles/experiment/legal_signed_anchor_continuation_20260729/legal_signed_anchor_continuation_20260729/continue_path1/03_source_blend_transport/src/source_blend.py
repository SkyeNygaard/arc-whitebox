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
D=fr.D;TARGET=fr.TARGET;RADIAL_SCALE=D/(fr.chi_mean(D)**2)
BETAS=(0.0,0.125,0.25,0.5,0.75,1.0)


def propagate(network_id:int,xk_np:np.ndarray):
 ws,_,_=fr.make_weights(network_id);x=torch.from_numpy(xk_np.copy())
 states={b:(np.zeros(D),np.zeros((D,D))) for b in BETAS}; target={};trace={str(b):[] for b in BETAS};target_h=final=None
 with torch.no_grad():
  for layer,wt in enumerate(ws):
   w=wt.double().numpy();pre=x@wt;post=torch.relu(pre)
   if layer==0:
    tm,tc=relu_bivariate(np.zeros(D),w.T@w);km,kc=observed_covariance(post);init=(tm-km,tc-kc)
    for b in BETAS: states[b]=(init[0].copy(),init[1].copy())
   else:
    kpm,kpv=observed_marginals(pre);kom,kov=observed_marginals(post)
    base_m,base_v,base_gate=relu_univariate(kpm,kpv)
    emp_gate=(pre>0).double().mean(0).cpu().numpy()
    for b in BETAS:
     dm,dc=states[b];pdm=dm@w;pdc=w.T@dc@w
     tm,tv,tg=relu_univariate(kpm+pdm,np.maximum(kpv+np.diag(pdc),1e-18))
     transport_m=tm-base_m; source_m=base_m-kom
     transport_v=tv-base_v; source_v=base_v-kov
     ndm=transport_m+b*source_m
     gate=np.sqrt(np.maximum(tg*emp_gate,0.0))
     ndc=pdc*np.outer(gate,gate)
     np.fill_diagonal(ndc,transport_v+b*source_v)
     ndc=.5*(ndc+ndc.T);states[b]=(ndm,ndc)
     trace[str(b)].append({'layer':layer,'mean_norm':float(np.linalg.norm(ndm)),'cov_fro':float(np.linalg.norm(ndc)),'source_mean_norm':float(np.linalg.norm(source_m)),'transport_mean_norm':float(np.linalg.norm(transport_m))})
   x=post
   if layer==TARGET:
    target_h=x.clone();target={str(b):(states[b][0].copy(),states[b][1].copy()) for b in BETAS}
   if layer==len(ws)-1: final=x.clone()
 assert target_h is not None and final is not None
 return ws,target_h,final,target,trace


def run_one(n:int,xk:np.ndarray,truth_n:int,chunk:int)->dict[str,Any]:
 t0=time.perf_counter();ws,wh,seed=fr.make_weights(n);ws2,Ht,Yt,target,trace=propagate(n,xk);assert all(torch.equal(a,b) for a,b in zip(ws,ws2))
 H=Ht.double().numpy();Y=Yt.double().numpy();m=H.mean(0);rho=fr.chi_mean(D);Q=fr.sample_anchor_matrix(H,m,rho);idx,dirs=fr.sample_row_probes(Q);X=fr.radial_features_sample_rows(H,m,idx,dirs,rho);fit=fr.fit_crossfit(X,Y);sa=fr.contract_rows(Q,idx,dirs)
 r1=fr.stream_reference(ws,truth_n,61_000_000+2*n,chunk);r2=fr.stream_reference(ws,truth_n,61_000_001+2*n,chunk);ty=.5*(r1['y']+r2['y']);mu=.5*(r1['mu']+r2['mu']);M=.5*(r1['M']+r2['M']);oracle=lower_anchor_selected(m,mu,np.diag(M),np.sum(M[idx]*dirs,axis=1),idx,dirs)
 base,_=fr.estimate_from_fit(fit,sa);alphas=np.array([-1,-.5,-.25,0,.025,.05,.1,.15,.2,.35,.5,.75,1.0])
 mse=lambda p:float(np.mean((p-ty)**2));ub=lambda p:float(np.mean((p-r1['y'])*(p-r2['y'])));bm=mse(base)
 methods={}
 for b in BETAS:
  a=anchor_from_defect(H,target[str(b)][0],target[str(b)][1],idx,dirs);preds=[fr.estimate_from_fit(fit,sa+s*a)[0] for s in alphas]
  methods[str(b)]={'anchor':a.tolist(),'cosine':float(a@oracle/max(np.linalg.norm(a)*np.linalg.norm(oracle),1e-30)),'relative_error':float(np.linalg.norm(a-oracle)/max(np.linalg.norm(oracle),1e-30)),'norm':float(np.linalg.norm(a)),'mse_grid':[mse(p) for p in preds],'unbiased_grid':[ub(p) for p in preds]}
 op,_=fr.estimate_from_fit(fit,sa+oracle)
 return {'network_id':n,'weight_seed':seed,'weight_sha256':wh,'truth_n':truth_n,'baseline_mse':bm,'baseline_unbiased':ub(base),'oracle_mse':mse(op),'alpha_grid':alphas.tolist(),'methods':methods,'trace':trace,'runtime_seconds':time.perf_counter()-t0}


def summarize(rs,tune_n):
 base=np.array([r['baseline_mse'] for r in rs]);grid=np.array(rs[0]['alpha_grid']);ti=np.arange(tune_n);vi=np.arange(tune_n,len(rs));candidates=[]
 for b in rs[0]['methods']:
  cm=np.array([r['methods'][b]['mse_grid'] for r in rs]);j=int(np.argmin(cm[ti].sum(0)/base[ti].sum())); candidates.append((float(cm[ti,j].sum()/base[ti].sum()),b,j))
 _,bb,j=min(candidates);alpha=float(grid[j]);cm=np.array([r['methods'][bb]['mse_grid'] for r in rs])
 def block(ix,seed):
  c=cm[ix,j];return {'n':len(ix),'candidate_over_base':float(c.sum()/base[ix].sum()),'ci95':bootstrap_ratio(base[ix],c,seed),'wins':int((c<base[ix]).sum()),'worst':float(np.max(c/base[ix])),'per_network':(c/base[ix]).tolist(),'mean_cosine':float(np.mean([rs[i]['methods'][bb]['cosine'] for i in ix]))}
 return {'betas':list(BETAS),'selected_beta':float(bb),'selected_alpha':alpha,'tune_ids':[rs[i]['network_id'] for i in ti],'validation_ids':[rs[i]['network_id'] for i in vi],'tuning':block(ti,20260801),'validation':block(vi,20260802) if len(vi) else {},'tuning_grid':[{'beta':float(b),'best_tuning_ratio':v,'alpha':float(grid[jj])} for v,b,jj in sorted(candidates)]}


def main():
 p=argparse.ArgumentParser();p.add_argument('--network',type=int);p.add_argument('--truth-n',type=int,default=8192);p.add_argument('--chunk',type=int,default=4096);p.add_argument('--threads',type=int,default=8);p.add_argument('--out',type=Path,required=True);p.add_argument('--records-dir',type=Path);p.add_argument('--tune-n',type=int,default=6);a=p.parse_args();torch.set_num_threads(a.threads)
 if a.records_dir:
  rs=[json.loads(q.read_text()) for q in sorted(a.records_dir.glob('network_*.json'))];z=summarize(rs,a.tune_n);a.out.write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2));return
 xk,_=fr.make_kerdock();z=run_one(a.network,xk,a.truth_n,a.chunk);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(z,indent=2));print(json.dumps({'network':a.network,'runtime':z['runtime_seconds'],'methods':{b:{'cos':round(v['cosine'],3),'best':round(min(v['mse_grid'])/z['baseline_mse'],3)} for b,v in z['methods'].items()}},indent=2))
if __name__=='__main__':main()
