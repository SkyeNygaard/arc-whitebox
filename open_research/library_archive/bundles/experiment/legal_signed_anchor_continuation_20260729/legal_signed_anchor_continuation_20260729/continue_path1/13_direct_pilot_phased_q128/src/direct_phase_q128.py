#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np, torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import frozen_reference_impl as fr
import q128_sources as q
from direct_output_cv import capture_baseline_and_affine,pilot_group
from base import lower_anchor_selected,bootstrap_ratio
D=fr.D
TOTAL_SIZES=(1024,2048,4096)
LAMBDAS=(0.05,0.1,0.2,0.35,0.5)
THRESHOLDS=(-0.25,0.0,0.05,0.1,0.2,0.35)

def run(n,xk,truth_n,chunk):
 t=time.perf_counter();ws,wh,seed=fr.make_weights(n);ws2,Ht,Yt,gates,sm,sv,first=q.forward_capture(n,xk);assert all(torch.equal(a,b) for a,b in zip(ws,ws2))
 H=Ht.double().numpy();Y=Yt.double().numpy();baseK=Y.mean(0);m=H.mean(0);rho=fr.chi_mean(D);Q=fr.sample_anchor_matrix(H,m,rho);idx,dirs=fr.sample_row_probes(Q);X=fr.radial_features_sample_rows(H,m,idx,dirs,rho);fit=fr.fit_crossfit(X,Y);sa=fr.contract_rows(Q,idx,dirs);base,_=fr.estimate_from_fit(fit,sa)
 U,V=q.backward(ws,gates,idx,dirs);init,layers=q.projected_sources(U,V,sm,sv,first);sums=[np.sum([z[k] for z in layers],axis=0) for k in range(4)]
 a0=q.probe_anchor(H,idx,dirs,*init);af=q.probe_anchor(H,idx,dirs,*[init[k]+sums[k] for k in range(4)])
 p0,_=fr.estimate_from_fit(fit,sa+a0);pf,_=fr.estimate_from_fit(fit,sa+af);d0=p0-base;ds=pf-p0
 # Affine CV built from the same Kerdock propagation. Recomputing the baseline is an audit convenience;
 # production can reuse gates, means, and baseK.
 baseK2,A,c,lstats=capture_baseline_and_affine(ws,xk);assert np.max(np.abs(baseK2-baseK))<2e-7
 r1=fr.stream_reference(ws,truth_n,161_000_000+2*n,chunk);r2=fr.stream_reference(ws,truth_n,161_000_001+2*n,chunk);truth=.5*(r1['y']+r2['y']);mse=lambda p:float(np.mean((p-truth)**2));methods={};den=max(float(ds@ds),1e-30)
 for total in TOTAL_SIZES:
  half=total//2;dp1,_=pilot_group(half,162_000_000+20*n+total,ws,A,c);dp2,_=pilot_group(half,162_000_001+20*n+total,ws,A,c)
  # Pilot estimates truth-baseK. Convert to residual around radial base + exact first anchor.
  rr1=dp1+(baseK-base)-d0;rr2=dp2+(baseK-base)-d0
  proj1=float(rr1@ds/den);proj2=float(rr2@ds/den);pmin=min(proj1,proj2);pmean=.5*(proj1+proj2)
  grid={}
  for th in THRESHOLDS:
   for lam in LAMBDAS:
    apply=pmin>=th;pred=p0+lam*ds if apply else p0;grid[f't{th}_l{lam}']=mse(pred)
  methods[str(total)]={'projection1':proj1,'projection2':proj2,'projection_min':pmin,'projection_mean':pmean,
                       'pilot_mutual_cosine':float(rr1@rr2/max(np.linalg.norm(rr1)*np.linalg.norm(rr2),1e-30)),
                       'source_norm':float(np.linalg.norm(ds)),'first_mse':mse(p0),'full_source_mse':mse(pf),'grid':grid}
 return {'network_id':n,'weight_seed':seed,'weight_sha256':wh,'truth_n':truth_n,'baseline_mse':mse(base),'first_mse':mse(p0),'methods':methods,'max_affine_identity_error':max(x['mean_identity_error'] for x in lstats),'runtime_seconds':time.perf_counter()-t}

def metric(base,cand,seed):
 rat=cand/base;return {'n':len(base),'candidate_over_base':float(cand.sum()/base.sum()),'ci95':bootstrap_ratio(base,cand,seed),'wins':int((rat<1).sum()),'worst':float(rat.max()),'per_network':rat.tolist()}

def summarize(rs,tune_n):
 rs=sorted(rs,key=lambda r:r['network_id']);base=np.array([r['baseline_mse'] for r in rs]);first=np.array([r['first_mse'] for r in rs]);ti=np.arange(tune_n);vi=np.arange(tune_n,len(rs));rows=[]
 for size in rs[0]['methods']:
  keys=rs[0]['methods'][size]['grid'].keys()
  for key in keys:
   c=np.array([r['methods'][size]['grid'][key] for r in rs]);rat=c[ti]/base[ti];rows.append((float(c[ti].sum()/base[ti].sum()),float(rat.max()),-int((rat<1).sum()),int(size),key))
 safe=[x for x in rows if x[1]<=1.15];sel=min(safe or rows);_,_,_,size,key=sel;c=np.array([r['methods'][str(size)]['grid'][key] for r in rs])
 return {'selected_total_pilot':size,'selected_rule':key,'tune_ids':[rs[i]['network_id'] for i in ti],'validation_ids':[rs[i]['network_id'] for i in vi],
         'first_tuning':metric(base[ti],first[ti],20261300),'first_validation':metric(base[vi],first[vi],20261301) if len(vi) else {},
         'tuning':metric(base[ti],c[ti],20261302),'validation':metric(base[vi],c[vi],20261303) if len(vi) else {},
         'top_safe':[{'ratio':x[0],'worst':x[1],'wins':-x[2],'size':x[3],'rule':x[4]} for x in sorted(safe)[:30]],
         'top_any':[{'ratio':x[0],'worst':x[1],'wins':-x[2],'size':x[3],'rule':x[4]} for x in sorted(rows)[:30]]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--network',type=int);p.add_argument('--truth-n',type=int,default=8192);p.add_argument('--chunk',type=int,default=4096);p.add_argument('--threads',type=int,default=4);p.add_argument('--out',type=Path,required=True);p.add_argument('--records-dir',type=Path);p.add_argument('--tune-n',type=int,default=6);a=p.parse_args();torch.set_num_threads(a.threads)
 if a.records_dir:
  rs=[json.loads(x.read_text()) for x in sorted(a.records_dir.glob('network_*.json'))];z=summarize(rs,a.tune_n);a.out.write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2));return
 xk,_=fr.make_kerdock();z=run(a.network,xk,a.truth_n,a.chunk);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(z));print(json.dumps({'network':a.network,'runtime':z['runtime_seconds'],'first':z['first_mse']/z['baseline_mse'],'projections':{s:[round(v['projection1'],3),round(v['projection2'],3)] for s,v in z['methods'].items()},'best':{s:round(min(v['grid'].values())/z['baseline_mse'],3) for s,v in z['methods'].items()}},indent=2))
if __name__=='__main__':main()
