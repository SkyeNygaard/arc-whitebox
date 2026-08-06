#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','5');os.environ.setdefault('OMP_NUM_THREADS','5');os.environ.setdefault('MKL_NUM_THREADS','5')
import numpy as np
from scipy.stats import spearmanr
import exact_kerdock_coreset_diagnostic as base
import set_level_basis_feasibility as sl

def parse(s):
 if ':' in s:
  a,b=s.split(':');return list(range(int(a),int(b)))
 return [int(x) for x in s.split(',') if x]

def run(seed,asset,dataset,dims,ridges,C=64):
 t=time.time()
 with np.load(asset,allow_pickle=False) as a: chirps=a['chirps'].astype(np.float32);rot=a['rotation'].astype(np.float32)
 with np.load(dataset/f'setlevel_seed{seed}_q128_c{C}.npz',allow_pickle=False) as d: labels=d['labels'].astype(np.float64)
 quota=base.quotas(4096); lib=np.concatenate([sl.fixed_random_library(C,quota,202607295),sl.affine_stratified_library(C,quota,202607295+99173)])
 W=base.gen_weights(seed);A=base.propagate_to_anchor(W,chirps,rot,28)
 for w in W[28:31]: A=base.relu(A@w)
 coords=sl.pilot_output_coordinates(A,W[31],max(dims)); raw=base.pair_average(base.relu(A@W[31][:,coords]))
 z=base.standardize(raw).astype(np.float64); Q=z.shape[1]
 methods={f'q{q}_r{r:g}':np.empty(len(lib)) for q in dims for r in ridges}
 methods.update({f'q{q}_global':np.empty(len(lib)) for q in dims})
 feasibility={k:0 for k in methods if '_r' in k}
 for ci,sel in enumerate(lib):
  S=z[sel].copy();u=base.base_weights(sel,quota);target=S.T@u; basis=sel//base.PAIRS_PER_BASIS
  for b in range(base.ALL_BASES):
   ii=np.flatnonzero(basis==b);S[ii]-=S[ii].mean(0,keepdims=True)
  gram=S.T@S
  for q in dims:
   slc=slice(Q-q,Q); tq=target[slc]; gq=gram[slc,slc];Sq=S[:,slc];
   methods[f'q{q}_global'][ci]=float(tq@tq/q)
   eye=np.eye(q)
   for r in ridges:
    coef=np.linalg.solve(gq+(r*len(sel))*eye,tq);ww=u-Sq@coef;rel=ww/u;ess=1/np.sum(ww*ww)/len(ww)
    e=z[sel,slc].T@ww
    score=float(e@e/q)
    if rel.min()>=.05 and rel.max()<=4 and ess>=.8: feasibility[f'q{q}_r{r:g}']+=1
    else:
     pen=max(0,.05-rel.min())**2+max(0,rel.max()-4)**2+10*max(0,.8-ess)**2;score+=pen
    methods[f'q{q}_r{r:g}'][ci]=score
 out={'seed':seed,'runtime_s':time.time()-t,'library_best':float(labels.min()),'library_passes':int(np.sum(labels<=1.1e-8)),'methods':{}}
 for n,s in methods.items():
  i=int(np.argmin(s));out['methods'][n]={'candidate':i,'label':float(labels[i]),'pass11':bool(labels[i]<=1.1e-8),'pass22':bool(labels[i]<=2.2e-8),'rho':float(spearmanr(s,np.log10(labels+1e-20)).statistic),'score':float(s[i]),'feasible_count':feasibility.get(n)}
 return out

def summary(rs):
 names=rs[0]['methods'];o={}
 for n in names:
  v=np.array([r['methods'][n]['label'] for r in rs]);rho=np.array([r['methods'][n]['rho'] for r in rs])
  o[n]={'mean':float(v.mean()),'median':float(np.median(v)),'worst':float(v.max()),'pass11':int(np.sum(v<=1.1e-8)),'pass22':int(np.sum(v<=2.2e-8)),'rho':float(rho.mean()),'values':v.tolist(),'candidates':[r['methods'][n]['candidate'] for r in rs]}
 return o

def main():
 p=argparse.ArgumentParser();p.add_argument('--asset',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--seeds',required=True);p.add_argument('--dims',default='16,32,64,128');p.add_argument('--ridges',default='.0001,.01,1,100');p.add_argument('--output',type=Path,required=True);a=p.parse_args();dims=[int(x) for x in a.dims.split(',')];ridges=[float(x) for x in a.ridges.split(',')]
 rs=[]
 for seed in parse(a.seeds):
  r=run(seed,a.asset,a.dataset,dims,ridges);rs.append(r);a.output.write_text(json.dumps({'records':rs,'summary':summary(rs),'config':{'dims':dims,'ridges':ridges}},indent=2));best=min(r['methods'].items(),key=lambda kv:kv[1]['label']);print(seed,best[0],f"{best[1]['label']:.3e}",f"{r['runtime_s']:.1f}s",flush=True)
 print(json.dumps(summary(rs),indent=2))
if __name__=='__main__':main()
