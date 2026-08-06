#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import frozen_reference_impl as fr
from source_blend import BETAS,propagate
from base import anchor_from_defect,lower_anchor_selected
D=fr.D

def run(n,xk,truth_n,chunk):
 t=time.perf_counter();ws,wh,seed=fr.make_weights(n);ws2,Ht,Yt,target,_=propagate(n,xk);assert all(torch.equal(a,b) for a,b in zip(ws,ws2))
 H=Ht.double().numpy();Y=Yt.double().numpy();m=H.mean(0);rho=fr.chi_mean(D);Q=fr.sample_anchor_matrix(H,m,rho);idx,dirs=fr.sample_row_probes(Q);X=fr.radial_features_sample_rows(H,m,idx,dirs,rho);fit=fr.fit_crossfit(X,Y);sa=fr.contract_rows(Q,idx,dirs);base,_=fr.estimate_from_fit(fit,sa)
 r1=fr.stream_reference(ws,truth_n,71_000_000+2*n,chunk);r2=fr.stream_reference(ws,truth_n,71_000_001+2*n,chunk);truth=.5*(r1['y']+r2['y'])
 methods={}
 for b in BETAS:
  a=anchor_from_defect(H,target[str(b)][0],target[str(b)][1],idx,dirs);p,_=fr.estimate_from_fit(fit,sa+a);methods[str(b)]={'anchor_norm':float(np.linalg.norm(a)),'delta_output':(p-base).tolist()}
 return {'network_id':n,'weight_seed':seed,'weight_sha256':wh,'truth_n':truth_n,'baseline_pred':base.tolist(),'truth_half1':r1['y'].tolist(),'truth_half2':r2['y'].tolist(),'methods':methods,'runtime_seconds':time.perf_counter()-t}

def main():
 p=argparse.ArgumentParser();p.add_argument('--network',type=int,required=True);p.add_argument('--truth-n',type=int,default=8192);p.add_argument('--chunk',type=int,default=4096);p.add_argument('--threads',type=int,default=8);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(a.threads);xk,_=fr.make_kerdock();z=run(a.network,xk,a.truth_n,a.chunk);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(z));print(json.dumps({'network':a.network,'runtime':z['runtime_seconds']}))
if __name__=='__main__':main()
