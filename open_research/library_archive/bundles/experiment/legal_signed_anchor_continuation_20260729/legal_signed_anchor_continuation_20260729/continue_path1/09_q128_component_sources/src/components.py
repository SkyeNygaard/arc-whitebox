#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import frozen_reference_impl as fr
from base import relu_bivariate,relu_univariate,observed_covariance,observed_marginals,lower_anchor_selected,bootstrap_ratio
D=fr.D;TARGET=fr.TARGET;WINDOWS=(4,8,16,24,29);LAM=(0,.025,.05,.1,.2,.35,.5,.75,1.0)

def capture(n,xk):
 ws,_,_=fr.make_weights(n);x=torch.from_numpy(xk.copy());g=[];sm=[];sv=[];first=None;Ht=Yt=None
 with torch.no_grad():
  for l,wt in enumerate(ws):
   w=wt.double().numpy();pre=x@wt;post=torch.relu(pre);kpm,kpv=observed_marginals(pre);kom,kov=observed_marginals(post);gm,gv,_=relu_univariate(kpm,kpv);g.append((pre>0).double().mean(0).cpu().numpy());sm.append(gm-kom);sv.append(gv-kov)
   if l==0:
    tm,tc=relu_bivariate(np.zeros(D),w.T@w);km,kc=observed_covariance(post);first=(tm-km,tc-kc)
   x=post
   if l==TARGET:Ht=x.clone()
   if l==len(ws)-1:Yt=x.clone()
 return ws,Ht,Yt,g,sm,sv,first

def back(ws,g,idx,dirs):
 U=np.eye(D)[:,idx];V=dirs.T;Us=[None]*(TARGET+1);Vs=[None]*(TARGET+1);Us[TARGET]=U;Vs[TARGET]=V
 for l in range(TARGET-1,-1,-1):
  A=ws[l+1].double().numpy()*g[l+1][None,:];U=A@U;V=A@V;Us[l]=U;Vs[l]=V
 return Us,Vs

def anchor(H,idx,dirs,v):
 dmi,dmv,dcii,dciv=v;m=H.mean(0);mi=m[idx];mv=dirs@m;mei=mi+dmi;mev=mv+dmv;sc=D/(fr.chi_mean(D)**2);sd=(H*H).mean(0)*sc;sr=(H[:,idx]*(H@dirs.T)).mean(0)*sc;diag=sd[idx]+dcii+mei*mei-mi*mi;row=sr+dciv+mei*mev-mi*mv;return (diag*dmv+2*dmi*row+2*(mi*mi-mei*mei)*mev)/(D+1)
def add(a,b):return tuple(x+y for x,y in zip(a,b))
def scale(a,s):return tuple(s*x for x in a)

def run(n,xk,truth_n,chunk):
 t=time.perf_counter();ws,wh,seed=fr.make_weights(n);ws2,Ht,Yt,g,sm,sv,first=capture(n,xk);H=Ht.double().numpy();Y=Yt.double().numpy();m=H.mean(0);rho=fr.chi_mean(D);Q=fr.sample_anchor_matrix(H,m,rho);idx,dirs=fr.sample_row_probes(Q);X=fr.radial_features_sample_rows(H,m,idx,dirs,rho);fit=fr.fit_crossfit(X,Y);sa=fr.contract_rows(Q,idx,dirs);base,_=fr.estimate_from_fit(fit,sa);U,V=back(ws,g,idx,dirs);dm0,dc0=first;init=(dm0@U[0],dm0@V[0],np.einsum('ip,ij,jp->p',U[0],dc0,U[0]),np.einsum('ip,ij,jp->p',U[0],dc0,V[0]));a0=anchor(H,idx,dirs,init)
 layers=[]
 for l in range(1,TARGET+1):
  u,v=U[l],V[l];layers.append(((sm[l]@u,sm[l]@v,np.zeros(len(idx)),np.zeros(len(idx))),(np.zeros(len(idx)),np.zeros(len(idx)),np.sum(u*u*sv[l][:,None],0),np.sum(u*v*sv[l][:,None],0))))
 r1=fr.stream_reference(ws,truth_n,121_000_000+2*n,chunk);r2=fr.stream_reference(ws,truth_n,121_000_001+2*n,chunk);ty=.5*(r1['y']+r2['y']);mu=.5*(r1['mu']+r2['mu']);M=.5*(r1['M']+r2['M']);oracle=lower_anchor_selected(m,mu,np.diag(M),np.sum(M[idx]*dirs,1),idx,dirs);mse=lambda p:float(np.mean((p-ty)**2));methods={}
 p0,_=fr.estimate_from_fit(fit,sa+a0);d0=p0-base
 for wlen in WINDOWS:
  sel=layers[-wlen:];mean_src=tuple(np.sum([z[0][k] for z in sel],0) for k in range(4));var_src=tuple(np.sum([z[1][k] for z in sel],0) for k in range(4))
  for name,src in [('mean',mean_src),('var',var_src),('both',add(mean_src,var_src))]:
   afull=anchor(H,idx,dirs,add(init,src));inc=afull-a0;grid=[]
   for lam in LAM:
    p,_=fr.estimate_from_fit(fit,sa+a0+lam*inc);grid.append(mse(p))
   dfull=fr.estimate_from_fit(fit,sa+afull)[0]-base
   methods[f'{name}_w{wlen}']={'component':name,'window':wlen,'mse_grid':grid,'source_output_cosine_with_first':float(d0@(dfull-d0)/max(np.linalg.norm(d0)*np.linalg.norm(dfull-d0),1e-30)),'source_output_norm_ratio':float(np.linalg.norm(dfull-d0)/max(np.linalg.norm(d0),1e-30)),'anchor_cosine':float(afull@oracle/max(np.linalg.norm(afull)*np.linalg.norm(oracle),1e-30))}
 return {'network_id':n,'weight_seed':seed,'weight_sha256':wh,'truth_n':truth_n,'baseline_mse':mse(base),'lambda_grid':list(LAM),'methods':methods,'runtime_seconds':time.perf_counter()-t}

def summarize(rs,tune_n):
 base=np.array([r['baseline_mse'] for r in rs]);grid=np.array(rs[0]['lambda_grid']);ti=np.arange(tune_n);vi=np.arange(tune_n,len(rs));rows=[]
 for k in rs[0]['methods']:
  cm=np.array([r['methods'][k]['mse_grid'] for r in rs]);
  for j,l in enumerate(grid):
   rat=cm[ti,j]/base[ti];rows.append((float(cm[ti,j].sum()/base[ti].sum()),float(rat.max()),-int((rat<1).sum()),k,j))
 safe=[x for x in rows if x[1]<=1.15];sel=min(safe or rows);_,_,_,k,j=sel;cm=np.array([r['methods'][k]['mse_grid'] for r in rs])
 def block(ix,seed):
  c=cm[ix,j];rat=c/base[ix];return {'n':len(ix),'candidate_over_base':float(c.sum()/base[ix].sum()),'ci95':bootstrap_ratio(base[ix],c,seed),'wins':int((rat<1).sum()),'worst':float(rat.max()),'per_network':rat.tolist()}
 return {'selected_method':k,'selected_lambda':float(grid[j]),'tune_ids':[rs[i]['network_id'] for i in ti],'validation_ids':[rs[i]['network_id'] for i in vi],'tuning':block(ti,20260901),'validation':block(vi,20260902) if len(vi) else {},'top_safe':[{'ratio':x[0],'worst':x[1],'wins':-x[2],'method':x[3],'lambda':float(grid[x[4]])} for x in sorted(safe)[:30]],'top_any':[{'ratio':x[0],'worst':x[1],'wins':-x[2],'method':x[3],'lambda':float(grid[x[4]])} for x in sorted(rows)[:30]]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--network',type=int);p.add_argument('--truth-n',type=int,default=8192);p.add_argument('--chunk',type=int,default=4096);p.add_argument('--threads',type=int,default=8);p.add_argument('--out',type=Path,required=True);p.add_argument('--records-dir',type=Path);p.add_argument('--tune-n',type=int,default=6);a=p.parse_args();torch.set_num_threads(a.threads)
 if a.records_dir:
  rs=[json.loads(q.read_text()) for q in sorted(a.records_dir.glob('network_*.json'))];z=summarize(rs,a.tune_n);a.out.write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2));return
 xk,_=fr.make_kerdock();z=run(a.network,xk,a.truth_n,a.chunk);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(z));print(json.dumps({'network':a.network,'runtime':z['runtime_seconds'],'best':sorted([(min(v['mse_grid'])/z['baseline_mse'],k) for k,v in z['methods'].items()])[:10]},indent=2))
if __name__=='__main__':main()
