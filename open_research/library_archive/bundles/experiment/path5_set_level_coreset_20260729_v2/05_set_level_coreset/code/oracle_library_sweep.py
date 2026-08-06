#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr
import exact_kerdock_coreset_diagnostic as base
import set_level_basis_feasibility as sl

ap=argparse.ArgumentParser()
ap.add_argument('--asset',type=Path,required=True)
ap.add_argument('--seed',type=int,required=True)
ap.add_argument('--qcoords',type=int,default=128)
ap.add_argument('--library-count',type=int,default=64)
ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args()
with np.load(a.asset,allow_pickle=False) as asset:
 c=asset['chirps'].astype(np.float32); r=asset['rotation'].astype(np.float32)
q=base.quotas(4096)
t=time.time(); W=base.gen_weights(a.seed); H=base.propagate_to_anchor(W,c,r,28)
A=H
for w in W[28:31]: A=base.relu(A@w)
coords=sl.pilot_output_coordinates(A,W[31],a.qcoords)
sketch=base.pair_average(base.relu(A@W[31][:,coords]))
z,wi=sl.whiten(sketch)
Y=base.pair_average(base.relu(A@W[31])); mu=Y.mean(0)
rf=sl.fixed_random_library(a.library_count,q,202607295)
af=sl.affine_stratified_library(a.library_count,q,202607295+99173)
lib=np.concatenate([rf,af])
rs=sl.score_library(z,rf,q); ass=sl.score_library(z,af,q)
scores={k:np.concatenate([rs[k],ass[k]]) for k in rs}
OF=base.standardize(Y)
rows=[]
for i,s in enumerate(lib):
    ow,info=base.calibrated_weights(OF[s],s,q)
    mse=base.added_mse(Y,s,ow,mu)
    row={'candidate':i,'family':'random' if i<a.library_count else 'affine','mse':mse,'info':info}
    for k,v in scores.items(): row[k]=float(v[i])
    rows.append(row)
    if (i+1)%16==0: print(i+1, 'best', min(x['mse'] for x in rows), 'elapsed',time.time()-t,flush=True)
summary={'seed':a.seed,'qcoords':a.qcoords,'library_count_each':a.library_count,'whiten':wi,
         'best_mse':min(x['mse'] for x in rows),'best_candidate':min(rows,key=lambda x:x['mse'])['candidate'],
         'passes_1.1e-8':sum(x['mse']<=1.1e-8 for x in rows),'passes_2.2e-8':sum(x['mse']<=2.2e-8 for x in rows),
         'median_mse':float(np.median([x['mse'] for x in rows])),'worst_mse':max(x['mse'] for x in rows),
         'runtime_s':time.time()-t,'correlations':{}}
logm=np.log10([x['mse'] for x in rows])
for k in scores:
 sv=np.log10(np.asarray(scores[k])+1e-30)
 summary['correlations'][k]={'spearman':float(spearmanr(sv,logm).statistic),'pearson_log':float(pearsonr(sv,logm).statistic),
                             'winner_candidate':int(np.argmin(scores[k])),'winner_mse':float(rows[int(np.argmin(scores[k]))]['mse'])}
a.output.write_text(json.dumps({'summary':summary,'rows':rows},indent=2))
print(json.dumps(summary,indent=2))
