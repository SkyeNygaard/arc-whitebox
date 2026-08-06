#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import frozen_reference_impl as fr
from base import relu_bivariate,relu_univariate,observed_covariance,observed_marginals,lower_anchor_selected,bootstrap_ratio
D=fr.D;TARGET=fr.TARGET
WINDOWS=(0,4,8,12,16,24,29) # 0 means first exact only; 29 means all later sources
BETAS=(0.125,0.25,0.5,0.75,1.0)

def forward_capture(n,xk):
 ws,_,_=fr.make_weights(n);x=torch.from_numpy(xk.copy());gates=[];sm=[];sv=[];first=None;Ht=Yt=None
 with torch.no_grad():
  for l,wt in enumerate(ws):
   w=wt.double().numpy();pre=x@wt;post=torch.relu(pre);kpm,kpv=observed_marginals(pre);kom,kov=observed_marginals(post);gm,gv,_=relu_univariate(kpm,kpv);gates.append((pre>0).double().mean(0).cpu().numpy());sm.append(gm-kom);sv.append(gv-kov)
   if l==0:
    tm,tc=relu_bivariate(np.zeros(D),w.T@w);km,kc=observed_covariance(post);first=(tm-km,tc-kc)
   x=post
   if l==TARGET: Ht=x.clone()
   if l==len(ws)-1:Yt=x.clone()
 assert first is not None and Ht is not None and Yt is not None
 return ws,Ht,Yt,gates,sm,sv,first

def backward(ws,gates,idx,dirs):
 p=len(idx);U=[None]*(TARGET+1);V=[None]*(TARGET+1);U[TARGET]=np.eye(D)[:,idx];V[TARGET]=dirs.T
 for l in range(TARGET-1,-1,-1):
  A=ws[l+1].double().numpy()*gates[l+1][None,:];U[l]=A@U[l+1];V[l]=A@V[l+1]
 return U,V

def probe_anchor(H,idx,dirs,dm_i,dm_v,dc_ii,dc_iv):
 m=H.mean(0);mi=m[idx];mv=dirs@m;mean_i=mi+dm_i;mean_v=mv+dm_v;sd=(H*H).mean(0)*(D/(fr.chi_mean(D)**2));sr=(H[:,idx]*(H@dirs.T)).mean(0)*(D/(fr.chi_mean(D)**2));diag=sd[idx]+dc_ii+mean_i*mean_i-mi*mi;row=sr+dc_iv+mean_i*mean_v-mi*mv
 d=dm_i
 return (diag*dm_v+2*d*row+2*(mi*mi-mean_i*mean_i)*mean_v)/(D+1.0)

def projected_sources(U,V,sm,sv,first):
 dm0,dc0=first;dm_i=dm0@U[0];dm_v=dm0@V[0];dc_ii=np.einsum('ip,ij,jp->p',U[0],dc0,U[0]);dc_iv=np.einsum('ip,ij,jp->p',U[0],dc0,V[0]);layers=[]
 for l in range(1,TARGET+1):
  u=U[l];v=V[l];layers.append((sm[l]@u,sm[l]@v,np.sum((u*u)*sv[l][:,None],axis=0),np.sum((u*v)*sv[l][:,None],axis=0)))
 return (dm_i,dm_v,dc_ii,dc_iv),layers

def run(n,xk,truth_n,chunk):
 t=time.perf_counter();ws,wh,seed=fr.make_weights(n);ws2,Ht,Yt,gates,sm,sv,first=forward_capture(n,xk);assert all(torch.equal(a,b) for a,b in zip(ws,ws2));H=Ht.double().numpy();Y=Yt.double().numpy();m=H.mean(0);rho=fr.chi_mean(D);Q=fr.sample_anchor_matrix(H,m,rho);idx,dirs=fr.sample_row_probes(Q);X=fr.radial_features_sample_rows(H,m,idx,dirs,rho);fit=fr.fit_crossfit(X,Y);sa=fr.contract_rows(Q,idx,dirs);base,_=fr.estimate_from_fit(fit,sa);U,V=backward(ws,gates,idx,dirs);init,layers=projected_sources(U,V,sm,sv,first)
 r1=fr.stream_reference(ws,truth_n,111_000_000+2*n,chunk);r2=fr.stream_reference(ws,truth_n,111_000_001+2*n,chunk);ty=.5*(r1['y']+r2['y']);mu=.5*(r1['mu']+r2['mu']);M=.5*(r1['M']+r2['M']);oracle=lower_anchor_selected(m,mu,np.diag(M),np.sum(M[idx]*dirs,axis=1),idx,dirs);alph=np.array([-1,-.5,-.25,0,.025,.05,.1,.2,.35,.5,.75,1.0]);mse=lambda p:float(np.mean((p-ty)**2));ub=lambda p:float(np.mean((p-r1['y'])*(p-r2['y'])));methods={}
 for wlen in WINDOWS:
  selected=[] if wlen==0 else layers[-wlen:]
  sums=[np.sum([z[k] for z in selected],axis=0) if selected else np.zeros_like(init[k]) for k in range(4)]
  for beta in BETAS:
   vals=[init[k]+beta*sums[k] for k in range(4)];a=probe_anchor(H,idx,dirs,*vals);pred=[fr.estimate_from_fit(fit,sa+s*a)[0] for s in alph];key=f'w{wlen}_b{beta}'
   methods[key]={'window':wlen,'beta':beta,'cosine':float(a@oracle/max(np.linalg.norm(a)*np.linalg.norm(oracle),1e-30)),'relative_error':float(np.linalg.norm(a-oracle)/max(np.linalg.norm(oracle),1e-30)),'anchor':a.tolist(),'delta_output':(fr.estimate_from_fit(fit,sa+a)[0]-base).tolist(),'mse_grid':[mse(p) for p in pred],'unbiased_grid':[ub(p) for p in pred]}
 return {'network_id':n,'weight_seed':seed,'weight_sha256':wh,'truth_n':truth_n,'baseline_mse':mse(base),'alpha_grid':alph.tolist(),'methods':methods,'source_norms':{'mean':[float(np.linalg.norm(x)) for x in sm],'var':[float(np.linalg.norm(x)) for x in sv]},'runtime_seconds':time.perf_counter()-t}

def summarize(rs,tune_n):
 base=np.array([r['baseline_mse'] for r in rs]);grid=np.array(rs[0]['alpha_grid']);ti=np.arange(tune_n);vi=np.arange(tune_n,len(rs));rows=[]
 for key in rs[0]['methods']:
  cm=np.array([r['methods'][key]['mse_grid'] for r in rs]);
  for j,a in enumerate(grid):
   rat=cm[ti,j]/base[ti];rows.append((float(cm[ti,j].sum()/base[ti].sum()),float(rat.max()),-int((rat<1).sum()),key,j))
 safe=[x for x in rows if x[1]<=1.15];sel=min(safe or rows);_,_,_,key,j=sel;cm=np.array([r['methods'][key]['mse_grid'] for r in rs])
 def block(ix,seed):
  c=cm[ix,j];rat=c/base[ix];return {'n':len(ix),'candidate_over_base':float(c.sum()/base[ix].sum()),'ci95':bootstrap_ratio(base[ix],c,seed),'wins':int((rat<1).sum()),'worst':float(rat.max()),'per_network':rat.tolist(),'mean_cosine':float(np.mean([rs[i]['methods'][key]['cosine'] for i in ix]))}
 return {'selected_method':key,'selected_window':rs[0]['methods'][key]['window'],'selected_beta':rs[0]['methods'][key]['beta'],'selected_alpha':float(grid[j]),'tune_ids':[rs[i]['network_id'] for i in ti],'validation_ids':[rs[i]['network_id'] for i in vi],'tuning':block(ti,20260830),'validation':block(vi,20260831) if len(vi) else {},'top_safe':[{'ratio':x[0],'worst':x[1],'wins':-x[2],'method':x[3],'alpha':float(grid[x[4]])} for x in sorted(safe)[:30]],'top_any':[{'ratio':x[0],'worst':x[1],'wins':-x[2],'method':x[3],'alpha':float(grid[x[4]])} for x in sorted(rows)[:30]]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--network',type=int);p.add_argument('--truth-n',type=int,default=8192);p.add_argument('--chunk',type=int,default=4096);p.add_argument('--threads',type=int,default=8);p.add_argument('--out',type=Path,required=True);p.add_argument('--records-dir',type=Path);p.add_argument('--tune-n',type=int,default=6);a=p.parse_args();torch.set_num_threads(a.threads)
 if a.records_dir:
  rs=[json.loads(q.read_text()) for q in sorted(a.records_dir.glob('network_*.json'))];z=summarize(rs,a.tune_n);a.out.write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2));return
 xk,_=fr.make_kerdock();z=run(a.network,xk,a.truth_n,a.chunk);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(z));print(json.dumps({'network':a.network,'runtime':z['runtime_seconds'],'best':sorted([(min(v['mse_grid'])/z['baseline_mse'],k) for k,v in z['methods'].items()])[:8]},indent=2))
if __name__=='__main__':main()
