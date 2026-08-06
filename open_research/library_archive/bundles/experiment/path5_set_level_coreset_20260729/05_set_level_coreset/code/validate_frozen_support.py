#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,time,os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','5');os.environ.setdefault('OMP_NUM_THREADS','5');os.environ.setdefault('MKL_NUM_THREADS','5')
import numpy as np
import exact_kerdock_coreset_diagnostic as base
import set_level_basis_feasibility as sl

def parse(s):
 if ':' in s:
  a,b=s.split(':');return range(int(a),int(b))
 return [int(x) for x in s.split(',') if x]

def eval_sel(Y,OF,mu,sel,q):
 w,info=base.calibrated_weights(OF[sel],sel,q)
 return base.added_mse(Y,sel,w,mu),info
ap=argparse.ArgumentParser();ap.add_argument('--asset',type=Path,required=True);ap.add_argument('--seeds',required=True);ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args()
with np.load(a.asset,allow_pickle=False) as z:c=z['chirps'].astype(np.float32);r=z['rotation'].astype(np.float32)
q=base.quotas(4096);rf=sl.fixed_random_library(64,q,202607295);af=sl.affine_stratified_library(64,q,202607295+99173);lib=np.concatenate([rf,af])
frozen={'dev_selected_candidate81':81,'candidate35':35,'candidate47':47,'candidate98':98,'fixed_random0':0}
records=[]
for seed in parse(a.seeds):
 t=time.time();W=base.gen_weights(seed);H=base.propagate_to_anchor(W,c,r,28);A=H
 for w in W[28:]:A=base.relu(A@w)
 Y=base.pair_average(A);mu=Y.mean(0);OF=base.standardize(Y)
 vals={}
 for name,i in frozen.items():
  mse,info=eval_sel(Y,OF,mu,lib[i],q);vals[name]={'mse':mse,'info':info,'pass_1.1e-8':mse<=1.1e-8,'pass_2.2e-8':mse<=2.2e-8}
 # Diagnostic only: oracle best within the frozen top-8 prior library from dev.
 top8=[81,111,88,35,91,78,51,38]
 best=(1e99,None,None)
 for i in top8:
  mse,info=eval_sel(Y,OF,mu,lib[i],q)
  if mse<best[0]:best=(mse,i,info)
 vals['oracle_best_of_dev_top8']={'mse':best[0],'candidate':best[1],'info':best[2],'diagnostic_only':True,'pass_1.1e-8':best[0]<=1.1e-8,'pass_2.2e-8':best[0]<=2.2e-8}
 rec={'seed':seed,'values':vals,'runtime_s':time.time()-t};records.append(rec)
 a.output.write_text(json.dumps({'records':records},indent=2))
 print(seed,vals['dev_selected_candidate81']['mse'],vals['fixed_random0']['mse'],vals['oracle_best_of_dev_top8']['mse'],rec['runtime_s'],flush=True)
# summary
names=records[0]['values'];summary={}
for n in names:
 v=np.array([x['values'][n]['mse'] for x in records]);summary[n]={'mean':float(v.mean()),'median':float(np.median(v)),'worst':float(v.max()),'passes_1.1e-8':int(sum(v<=1.1e-8)),'passes_2.2e-8':int(sum(v<=2.2e-8)),'values':v.tolist()}
a.output.write_text(json.dumps({'records':records,'summary':summary},indent=2));print(json.dumps(summary,indent=2))
