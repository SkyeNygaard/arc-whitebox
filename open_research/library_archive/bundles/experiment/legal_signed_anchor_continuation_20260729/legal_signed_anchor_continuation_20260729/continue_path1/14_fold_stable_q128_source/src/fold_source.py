#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np, torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import frozen_reference_impl as fr
import q128_sources as q
from base import relu_bivariate,relu_univariate,observed_covariance,observed_marginals,lower_anchor_selected,bootstrap_ratio
D=fr.D;TARGET=fr.TARGET
LAMBDAS=(0.05,0.1,0.2,0.35,0.5)
COS_THRESH=(-1.0,0.0,0.25,0.5,0.75)
DISP_THRESH=(0.5,1.0,1.5,2.0,3.0,10.0)

def capture_fold_sources(n,xk):
 ws,wh,seed=fr.make_weights(n);x=torch.from_numpy(xk.copy());gates=[];sm=[];sv=[];fsm=[];fsv=[];first=None;Ht=Yt=None
 bid=np.repeat(np.arange(fr.N_BASES),fr.ROWS_PER_BASIS);groups=fr.fold_groups();masks=[torch.from_numpy(np.isin(bid,g)) for g in groups]
 with torch.no_grad():
  for l,wt in enumerate(ws):
   w=wt.double().numpy();pre=x@wt;post=torch.relu(pre);kpm,kpv=observed_marginals(pre);kom,kov=observed_marginals(post);gm,gv,_=relu_univariate(kpm,kpv);gates.append((pre>0).double().mean(0).cpu().numpy());sm.append(gm-kom);sv.append(gv-kov)
   lm=[];lv=[]
   for mask in masks:
    fp=pre[mask];fo=post[mask];pm,pv=observed_marginals(fp);om,ov=observed_marginals(fo);ggm,ggv,_=relu_univariate(pm,pv);lm.append(ggm-om);lv.append(ggv-ov)
   fsm.append(lm);fsv.append(lv)
   if l==0:
    tm,tc=relu_bivariate(np.zeros(D),w.T@w);km,kc=observed_covariance(post);first=(tm-km,tc-kc)
   x=post
   if l==TARGET:Ht=x.clone()
   if l==len(ws)-1:Yt=x.clone()
 return ws,wh,seed,Ht,Yt,gates,sm,sv,np.asarray(fsm),np.asarray(fsv),first

def cosine(a,b):return float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))

def run(n,xk,truth_n,chunk):
 t=time.perf_counter();ws,wh,seed,Ht,Yt,gates,sm,sv,fsm,fsv,first=capture_fold_sources(n,xk);H=Ht.double().numpy();Y=Yt.double().numpy();m=H.mean(0);rho=fr.chi_mean(D);Q=fr.sample_anchor_matrix(H,m,rho);idx,dirs=fr.sample_row_probes(Q);X=fr.radial_features_sample_rows(H,m,idx,dirs,rho);fit=fr.fit_crossfit(X,Y);sa=fr.contract_rows(Q,idx,dirs);base,_=fr.estimate_from_fit(fit,sa);U,V=q.backward(ws,gates,idx,dirs);init,layers=q.projected_sources(U,V,sm,sv,first)
 sums=[np.sum([z[k] for z in layers],axis=0) for k in range(4)];a0=q.probe_anchor(H,idx,dirs,*init);af=q.probe_anchor(H,idx,dirs,*[init[k]+sums[k] for k in range(4)]);p0,_=fr.estimate_from_fit(fit,sa+a0);pf,_=fr.estimate_from_fit(fit,sa+af);ds=pf-p0
 fold_ds=[]
 for f in range(len(fr.fold_groups())):
  flayers=[]
  for l in range(1,TARGET+1):
   u=U[l];v=V[l];flayers.append((fsm[l,f]@u,fsm[l,f]@v,np.sum((u*u)*fsv[l,f][:,None],axis=0),np.sum((u*v)*fsv[l,f][:,None],axis=0)))
  fsums=[np.sum([z[k] for z in flayers],axis=0) for k in range(4)];fa=q.probe_anchor(H,idx,dirs,*[init[k]+fsums[k] for k in range(4)]);fp,_=fr.estimate_from_fit(fit,sa+fa);fold_ds.append(fp-p0)
 fold_ds=np.asarray(fold_ds);cos=np.array([cosine(z,ds) for z in fold_ds]);disp=float(np.sqrt(np.mean(np.sum((fold_ds-ds[None,:])**2,axis=1)))/max(np.linalg.norm(ds),1e-30));pair=[]
 for i in range(len(fold_ds)):
  for j in range(i):pair.append(cosine(fold_ds[i],fold_ds[j]))
 r1=fr.stream_reference(ws,truth_n,171_000_000+2*n,chunk);r2=fr.stream_reference(ws,truth_n,171_000_001+2*n,chunk);truth=.5*(r1['y']+r2['y']);mse=lambda p:float(np.mean((p-truth)**2));grid={}
 for ct in COS_THRESH:
  for dt in DISP_THRESH:
   stable=(float(cos.min())>=ct and disp<=dt)
   for lam in LAMBDAS:grid[f'c{ct}_d{dt}_l{lam}']=mse(p0+lam*ds if stable else p0)
 return {'network_id':n,'weight_seed':seed,'weight_sha256':wh,'truth_n':truth_n,'baseline_mse':mse(base),'first_mse':mse(p0),'full_source_mse':mse(pf),'stability':{'fold_cosines':cos.tolist(),'min_fold_cosine':float(cos.min()),'median_fold_cosine':float(np.median(cos)),'min_pair_cosine':float(min(pair)),'median_pair_cosine':float(np.median(pair)),'relative_dispersion':disp},'grid':grid,'runtime_seconds':time.perf_counter()-t}

def metric(base,c,seed):
 rat=c/base;return {'n':len(base),'candidate_over_base':float(c.sum()/base.sum()),'ci95':bootstrap_ratio(base,c,seed),'wins':int((rat<1).sum()),'worst':float(rat.max()),'per_network':rat.tolist()}

def summarize(rs,tune_n):
 rs=sorted(rs,key=lambda r:r['network_id']);base=np.array([r['baseline_mse'] for r in rs]);first=np.array([r['first_mse'] for r in rs]);ti=np.arange(tune_n);vi=np.arange(tune_n,len(rs));rows=[]
 for key in rs[0]['grid']:
  c=np.array([r['grid'][key] for r in rs]);rat=c[ti]/base[ti];rows.append((float(c[ti].sum()/base[ti].sum()),float(rat.max()),-int((rat<1).sum()),key))
 safe=[x for x in rows if x[1]<=1.15];sel=min(safe or rows);key=sel[3];c=np.array([r['grid'][key] for r in rs])
 return {'selected_rule':key,'tune_ids':[rs[i]['network_id'] for i in ti],'validation_ids':[rs[i]['network_id'] for i in vi],'first_tuning':metric(base[ti],first[ti],20261500),'first_validation':metric(base[vi],first[vi],20261501) if len(vi) else {},'tuning':metric(base[ti],c[ti],20261502),'validation':metric(base[vi],c[vi],20261503) if len(vi) else {},'top_safe':[{'ratio':x[0],'worst':x[1],'wins':-x[2],'rule':x[3]} for x in sorted(safe)[:30]],'top_any':[{'ratio':x[0],'worst':x[1],'wins':-x[2],'rule':x[3]} for x in sorted(rows)[:30]]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--network',type=int);p.add_argument('--truth-n',type=int,default=8192);p.add_argument('--chunk',type=int,default=4096);p.add_argument('--threads',type=int,default=4);p.add_argument('--out',type=Path,required=True);p.add_argument('--records-dir',type=Path);p.add_argument('--tune-n',type=int,default=6);a=p.parse_args();torch.set_num_threads(a.threads)
 if a.records_dir:
  rs=[json.loads(x.read_text()) for x in sorted(a.records_dir.glob('network_*.json'))];z=summarize(rs,a.tune_n);a.out.write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2));return
 xk,_=fr.make_kerdock();z=run(a.network,xk,a.truth_n,a.chunk);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(z));print(json.dumps({'network':a.network,'runtime':z['runtime_seconds'],'first':z['first_mse']/z['baseline_mse'],'full':z['full_source_mse']/z['baseline_mse'],'stability':z['stability'],'best':min(z['grid'].values())/z['baseline_mse']},indent=2))
if __name__=='__main__':main()
