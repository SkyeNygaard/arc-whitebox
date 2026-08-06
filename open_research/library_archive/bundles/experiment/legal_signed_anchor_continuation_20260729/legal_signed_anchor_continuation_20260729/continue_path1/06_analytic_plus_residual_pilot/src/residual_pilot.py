#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import frozen_reference_impl as fr
from source_blend import propagate
from base import anchor_from_defect,lower_anchor_selected,bootstrap_ratio
D=fr.D;TARGET=fr.TARGET
SIZES=(512,1024,2048)
GAMMAS=(-.1,0,.005,.01,.02,.05,.1,.2,.4,1.0)

def antithetic_sobol_target(ws,total,seed):
 assert total%2==0
 eng=torch.quasirandom.SobolEngine(D,scramble=True,seed=seed);u=eng.draw(total//2,dtype=torch.float32).clamp_(1e-7,1-1e-7);z=math.sqrt(2)*torch.erfinv(2*u-1);x=torch.cat([z,-z],0)
 with torch.no_grad():
  for l,w in enumerate(ws):
   x=torch.relu(x@w)
   if l==TARGET:return x.double().numpy()
 raise AssertionError

def pilot_anchor(H,center,idx,dirs):
 mu=H.mean(0);M=H.T@H/len(H);return lower_anchor_selected(center,mu,np.diag(M),np.sum(M[idx]*dirs,axis=1),idx,dirs)

def run(n,xk,truth_n,chunk):
 t=time.perf_counter();ws,wh,seed=fr.make_weights(n);ws2,Ht,Yt,target,_=propagate(n,xk);assert all(torch.equal(a,b) for a,b in zip(ws,ws2));H=Ht.double().numpy();Y=Yt.double().numpy();m=H.mean(0);rho=fr.chi_mean(D);Q=fr.sample_anchor_matrix(H,m,rho);idx,dirs=fr.sample_row_probes(Q);X=fr.radial_features_sample_rows(H,m,idx,dirs,rho);fit=fr.fit_crossfit(X,Y);sa=fr.contract_rows(Q,idx,dirs);analytic=anchor_from_defect(H,target['0.0'][0],target['0.0'][1],idx,dirs)
 max_group=max(SIZES)//2;P1=antithetic_sobol_target(ws,max_group,91_000_000+n*2);P2=antithetic_sobol_target(ws,max_group,91_000_001+n*2)
 pilots={}
 for size in SIZES:
  g=size//2;k=g//2
  # Each group is [z_0..z_{M-1}, -z_0..-z_{M-1}]. Preserve pairs.
  h1=np.concatenate([P1[:k],P1[max_group//2:max_group//2+k]],0);h2=np.concatenate([P2[:k],P2[max_group//2:max_group//2+k]],0)
  a1=pilot_anchor(h1,m,idx,dirs);a2=pilot_anchor(h2,m,idx,dirs);avg=.5*(a1+a2);pilots[str(size)]={'avg':avg,'disagreement':float(np.linalg.norm(a1-a2)/max(np.linalg.norm(avg),1e-30))}
 r1=fr.stream_reference(ws,truth_n,101_000_000+2*n,chunk);r2=fr.stream_reference(ws,truth_n,101_000_001+2*n,chunk);ty=.5*(r1['y']+r2['y']);mu=.5*(r1['mu']+r2['mu']);M=.5*(r1['M']+r2['M']);oracle=lower_anchor_selected(m,mu,np.diag(M),np.sum(M[idx]*dirs,axis=1),idx,dirs);base,_=fr.estimate_from_fit(fit,sa);mse=lambda p:float(np.mean((p-ty)**2));ub=lambda p:float(np.mean((p-r1['y'])*(p-r2['y'])));methods={}
 for size,d in pilots.items():
  avg=d['avg'];grid=[];ug=[]
  for g in GAMMAS:
   a=analytic+g*(avg-analytic);p,_=fr.estimate_from_fit(fit,sa+a);grid.append(mse(p));ug.append(ub(p))
  methods[size]={'mse_grid':grid,'unbiased_grid':ug,'pilot_disagreement':d['disagreement'],'pilot_cosine':float(avg@oracle/max(np.linalg.norm(avg)*np.linalg.norm(oracle),1e-30)),'analytic_cosine':float(analytic@oracle/max(np.linalg.norm(analytic)*np.linalg.norm(oracle),1e-30)),'residual_cosine':float((avg-analytic)@(oracle-analytic)/max(np.linalg.norm(avg-analytic)*np.linalg.norm(oracle-analytic),1e-30))}
 return {'network_id':n,'weight_seed':seed,'weight_sha256':wh,'truth_n':truth_n,'baseline_mse':mse(base),'gamma_grid':list(GAMMAS),'methods':methods,'runtime_seconds':time.perf_counter()-t}

def summarize(rs,tune_n):
 base=np.array([r['baseline_mse'] for r in rs]);grid=np.array(rs[0]['gamma_grid']);ti=np.arange(tune_n);vi=np.arange(tune_n,len(rs));rows=[]
 for size in rs[0]['methods']:
  cm=np.array([r['methods'][size]['mse_grid'] for r in rs]);
  for j,g in enumerate(grid):
   rat=cm[ti,j]/base[ti];rows.append((float(cm[ti,j].sum()/base[ti].sum()),float(rat.max()),-int((rat<1).sum()),int(size),j))
 safe=[x for x in rows if x[1]<=1.15];sel=min(safe or rows);_,_,_,size,j=sel;cm=np.array([r['methods'][str(size)]['mse_grid'] for r in rs])
 # Bounded disagreement gate chosen on tuning; fallback is analytic gamma=0 for same size.
 zero=int(np.where(grid==0)[0][0]);dis=np.array([r['methods'][str(size)]['pilot_disagreement'] for r in rs]);ths=np.unique(np.r_[np.quantile(dis[ti],[.25,.5,.75]),np.inf]);gates=[]
 for th in ths:
  c=np.where(dis[ti]<=th,cm[ti,j],cm[ti,zero]);rat=c/base[ti];gates.append((float(c.sum()/base[ti].sum()),float(rat.max()),float(th)))
 _,_,th=min(gates)
 def block(ix,seed):
  raw=cm[ix,j];gated=np.where(dis[ix]<=th,raw,cm[ix,zero]);return {'n':len(ix),'raw_over_base':float(raw.sum()/base[ix].sum()),'raw_wins':int((raw<base[ix]).sum()),'raw_worst':float(np.max(raw/base[ix])),'gated_over_base':float(gated.sum()/base[ix].sum()),'gated_wins':int((gated<base[ix]).sum()),'gated_worst':float(np.max(gated/base[ix])),'gated_ci95':bootstrap_ratio(base[ix],gated,seed),'applied':int((dis[ix]<=th).sum()),'per_network_gated':(gated/base[ix]).tolist(),'mean_residual_cosine':float(np.mean([rs[i]['methods'][str(size)]['residual_cosine'] for i in ix]))}
 return {'selected_size':size,'selected_gamma':float(grid[j]),'disagreement_threshold':th,'tune_ids':[rs[i]['network_id'] for i in ti],'validation_ids':[rs[i]['network_id'] for i in vi],'tuning':block(ti,20260820),'validation':block(vi,20260821) if len(vi) else {},'top_safe':[{'ratio':x[0],'worst':x[1],'wins':-x[2],'size':x[3],'gamma':float(grid[x[4]])} for x in sorted(safe)[:20]],'top_any':[{'ratio':x[0],'worst':x[1],'wins':-x[2],'size':x[3],'gamma':float(grid[x[4]])} for x in sorted(rows)[:20]]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--network',type=int);p.add_argument('--truth-n',type=int,default=8192);p.add_argument('--chunk',type=int,default=4096);p.add_argument('--threads',type=int,default=8);p.add_argument('--out',type=Path,required=True);p.add_argument('--records-dir',type=Path);p.add_argument('--tune-n',type=int,default=6);a=p.parse_args();torch.set_num_threads(a.threads)
 if a.records_dir:
  rs=[json.loads(q.read_text()) for q in sorted(a.records_dir.glob('network_*.json'))];z=summarize(rs,a.tune_n);a.out.write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2));return
 xk,_=fr.make_kerdock();z=run(a.network,xk,a.truth_n,a.chunk);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(z));print(json.dumps({'network':a.network,'runtime':z['runtime_seconds'],'best':{s:round(min(v['mse_grid'])/z['baseline_mse'],3) for s,v in z['methods'].items()}},indent=2))
if __name__=='__main__':main()
