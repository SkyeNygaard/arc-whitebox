#!/usr/bin/env python3
"""Frozen eight-support library selector using a direct output-coordinate sketch."""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','5');os.environ.setdefault('OMP_NUM_THREADS','5');os.environ.setdefault('MKL_NUM_THREADS','5')
import numpy as np
import exact_kerdock_coreset_diagnostic as base
import set_level_basis_feasibility as sl
TOP8=np.array([81,111,88,35,91,78,51,38],dtype=np.int32)

def parse(s):
 if ':' in s:
  a,b=s.split(':');return list(range(int(a),int(b)))
 return [int(x) for x in s.split(',') if x]

def score_supports(z,lib,quota,dims,ridges):
 Q=z.shape[1];out={f'q{q}_r{r:g}':np.empty(len(lib)) for q in dims for r in ridges};out.update({f'q{q}_global':np.empty(len(lib)) for q in dims})
 for ci,sel in enumerate(lib):
  S=z[sel].astype(np.float64,copy=True);u=base.base_weights(sel,quota);t=S.T@u;bids=sel//base.PAIRS_PER_BASIS
  for b in range(base.ALL_BASES):
   ii=np.flatnonzero(bids==b);S[ii]-=S[ii].mean(0,keepdims=True)
  G=S.T@S
  for q in dims:
   slc=slice(Q-q,Q);tq=t[slc];Sq=S[:,slc];Gq=G[slc,slc];out[f'q{q}_global'][ci]=float(tq@tq/q);I=np.eye(q)
   for r in ridges:
    c=np.linalg.solve(Gq+r*len(sel)*I,tq);w=u-Sq@c;rel=w/u;ess=1/np.sum(w*w)/len(w);e=z[sel,slc].T@w;sc=float(e@e/q)
    sc+=max(0,.05-rel.min())**2+max(0,rel.max()-4)**2+10*max(0,.8-ess)**2
    out[f'q{q}_r{r:g}'][ci]=sc
 return out

def run(seed,asset,dataset_dir,dims,ridges,compute_labels=False):
 t0=time.time()
 with np.load(asset,allow_pickle=False) as a:ch=a['chirps'].astype(np.float32);rot=a['rotation'].astype(np.float32)
 quota=base.quotas(4096);alllib=np.concatenate([sl.fixed_random_library(64,quota,202607295),sl.affine_stratified_library(64,quota,202607295+99173)]);lib=alllib[TOP8]
 W=base.gen_weights(seed);A=base.propagate_to_anchor(W,ch,rot,28)
 for w in W[28:31]:A=base.relu(A@w)
 coords=sl.pilot_output_coordinates(A,W[31],max(dims));sk=base.pair_average(base.relu(A@W[31][:,coords]));z=base.standardize(sk)
 scores=score_supports(z,lib,quota,dims,ridges)
 if dataset_dir is not None and (dataset_dir/f'setlevel_seed{seed}_q128_c64.npz').exists():
  with np.load(dataset_dir/f'setlevel_seed{seed}_q128_c64.npz',allow_pickle=False) as d: labels=d['labels'][TOP8].astype(np.float64)
 elif compute_labels:
  Y=base.pair_average(base.relu(A@W[31]));mu=Y.mean(0);OF=base.standardize(Y);labels=[]
  for sel in lib:
   ww,_=base.calibrated_weights(OF[sel],sel,quota);labels.append(base.added_mse(Y,sel,ww,mu))
  labels=np.asarray(labels)
 else: raise RuntimeError('labels unavailable')
 methods={}
 for n,s in scores.items():
  j=int(np.argmin(s));methods[n]={'top8_slot':j,'candidate':int(TOP8[j]),'label':float(labels[j]),'score':float(s[j])}
 return {'seed':seed,'labels':{str(int(TOP8[i])):float(labels[i]) for i in range(8)},'oracle_best':float(labels.min()),'oracle_best_candidate':int(TOP8[int(np.argmin(labels))]),'methods':methods,'runtime_s':time.time()-t0}

def summary(rs):
 o={};names=rs[0]['methods']
 for n in names:
  v=np.array([r['methods'][n]['label'] for r in rs]);o[n]={'mean':float(v.mean()),'median':float(np.median(v)),'worst':float(v.max()),'pass11':int(np.sum(v<=1.1e-8)),'pass22':int(np.sum(v<=2.2e-8)),'values':v.tolist(),'candidates':[r['methods'][n]['candidate'] for r in rs]}
 ov=np.array([r['oracle_best'] for r in rs]);o['oracle_best_top8']={'mean':float(ov.mean()),'worst':float(ov.max()),'pass11':int(np.sum(ov<=1.1e-8)),'pass22':int(np.sum(ov<=2.2e-8)),'values':ov.tolist()}
 return o

def main():
 p=argparse.ArgumentParser();p.add_argument('--asset',type=Path,required=True);p.add_argument('--dataset-dir',type=Path);p.add_argument('--seeds',required=True);p.add_argument('--dims',default='16,32,64,128');p.add_argument('--ridges',default='.0001,.01,1,100');p.add_argument('--compute-labels',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();dims=[int(x) for x in a.dims.split(',')];ridges=[float(x) for x in a.ridges.split(',')]
 rs=[]
 for seed in parse(a.seeds):
  r=run(seed,a.asset,a.dataset_dir,dims,ridges,a.compute_labels);rs.append(r);a.output.write_text(json.dumps({'config':{'top8':TOP8.tolist(),'dims':dims,'ridges':ridges},'records':rs,'summary':summary(rs)},indent=2));print(seed,r['oracle_best_candidate'],f"{r['oracle_best']:.2e}",f"{r['runtime_s']:.1f}s",flush=True)
 print(json.dumps(summary(rs),indent=2))
if __name__=='__main__':main()
