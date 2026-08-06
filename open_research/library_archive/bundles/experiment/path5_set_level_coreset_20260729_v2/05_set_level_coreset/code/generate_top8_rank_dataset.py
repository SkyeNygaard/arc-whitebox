#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','3');os.environ.setdefault('OMP_NUM_THREADS','3');os.environ.setdefault('MKL_NUM_THREADS','3')
import numpy as np
import exact_kerdock_coreset_diagnostic as base
import set_level_basis_feasibility as sl
from top8_sketch_selector import TOP8,score_supports

def build(seed,asset,outdir,dims,ridges,labels_dir=None):
 t=time.time()
 with np.load(asset,allow_pickle=False) as a:ch=a['chirps'].astype(np.float32);rot=a['rotation'].astype(np.float32)
 quota=base.quotas(4096);alllib=np.concatenate([sl.fixed_random_library(64,quota,202607295),sl.affine_stratified_library(64,quota,202607295+99173)]);lib=alllib[TOP8]
 W=base.gen_weights(seed);A=base.propagate_to_anchor(W,ch,rot,28)
 for w in W[28:31]:A=base.relu(A@w)
 coords=sl.pilot_output_coordinates(A,W[31],max(dims));sk=base.pair_average(base.relu(A@W[31][:,coords]));z=base.standardize(sk)
 scores=score_supports(z,lib,quota,dims,ridges);names=sorted(scores)
 raw=np.stack([scores[n] for n in names],axis=1);log=np.log10(raw+1e-30)
 ranks=np.empty_like(log);zs=np.empty_like(log)
 for j in range(log.shape[1]):
  order=np.argsort(log[:,j],kind='stable');rr=np.empty(8);rr[order]=np.arange(8)/7;ranks[:,j]=rr
  zs[:,j]=(log[:,j]-log[:,j].mean())/(log[:,j].std()+1e-8)
 agg=np.stack([ranks.mean(1),ranks.std(1),ranks.min(1),ranks.max(1)],axis=1)
 onehot=np.eye(8,dtype=np.float64)
 X=np.concatenate([log,ranks,zs,agg,onehot],axis=1).astype(np.float32)
 if labels_dir is not None and (labels_dir/f'setlevel_seed{seed}_q128_c64.npz').exists():
  with np.load(labels_dir/f'setlevel_seed{seed}_q128_c64.npz',allow_pickle=False) as oldd:
   labels=oldd['labels'][TOP8].astype(np.float64); infos=[{'min_relative':float(oldd['min_relative'][i]),'max_relative':float(oldd['max_relative'][i]),'ess_fraction':float(oldd['ess'][i])} for i in TOP8]
 else:
  Y=base.pair_average(base.relu(A@W[31]));mu=Y.mean(0);OF=base.standardize(Y);labels=[];infos=[]
  for sel in lib:
   ww,info=base.calibrated_weights(OF[sel],sel,quota);labels.append(base.added_mse(Y,sel,ww,mu));infos.append(info)
  labels=np.asarray(labels,dtype=np.float64)
 outdir.mkdir(parents=True,exist_ok=True);p=outdir/f'top8_rank_seed{seed}.npz'
 np.savez_compressed(p,X=X,labels=labels,candidates=TOP8,score_names=np.array(names),raw_scores=raw,coords=coords,
                     min_relative=np.array([x['min_relative'] for x in infos]),max_relative=np.array([x['max_relative'] for x in infos]),ess=np.array([x['ess_fraction'] for x in infos]))
 meta={'seed':seed,'runtime_s':time.time()-t,'best':float(labels.min()),'worst':float(labels.max()),'passes':int(np.sum(labels<=1.1e-8)),'path':str(p),'feature_dim':int(X.shape[1])}
 (outdir/f'top8_rank_seed{seed}_meta.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--asset',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True);p.add_argument('--dims',default='16,32,64,128');p.add_argument('--ridges',default='.0001,.01,1,100');p.add_argument('--labels-dir',type=Path);a=p.parse_args();build(a.seed,a.asset,a.outdir,[int(x) for x in a.dims.split(',')],[float(x) for x in a.ridges.split(',')],a.labels_dir)
if __name__=='__main__':main()
