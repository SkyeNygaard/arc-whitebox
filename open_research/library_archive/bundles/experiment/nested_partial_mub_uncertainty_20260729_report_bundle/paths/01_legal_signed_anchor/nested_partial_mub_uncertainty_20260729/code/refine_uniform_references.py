#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np, torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import frozen_reference_impl as fr
from nested_partial_mub_experiment import lower_anchor, cosine, mse, PREFIXES, D

def weighted_beta(z):
    fs=z['fit_fold_sizes'].astype(float);return np.tensordot(fs/fs.sum(),z['fit_betas'],axes=(0,0))

def main():
    p=argparse.ArgumentParser();p.add_argument('--network',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--n',type=int,default=65536);p.add_argument('--chunk',type=int,default=4096);p.add_argument('--threads',type=int,default=2);p.add_argument('--reuse-reference',action='store_true');a=p.parse_args();torch.set_num_threads(a.threads)
    recdir=a.out/'results'/'records';vecdir=a.out/'results'/'vectors';refdir=a.out/'results'/'refined_references';refdir.mkdir(parents=True,exist_ok=True)
    refpath=refdir/f'network_{a.network:04d}_reference.npz'
    if a.reuse_reference and refpath.exists():
        q=np.load(refpath);A={'y':q['refA_y']};B={'y':q['refB_y']};truth={'y':q['truth_y'],'mu':q['truth_mu'],'M':q['truth_M']};seeds=q['seeds'].astype(int).tolist()
    else:
        ws,_,_=fr.make_weights(a.network); refs=[]; seeds=[]
        for j in range(6):
            seed=92_000_000+100*a.network+j;seeds.append(seed);refs.append(fr.stream_reference(ws,a.n,seed,a.chunk));print(json.dumps({'network':a.network,'stream':j,'seed':seed}),flush=True)
        A={k:np.mean([r[k] for r in refs[:3]],axis=0) for k in ('y','mu','M')};B={k:np.mean([r[k] for r in refs[3:]],axis=0) for k in ('y','mu','M')};truth={k:.5*(A[k]+B[k]) for k in A}
        np.savez_compressed(refpath,refA_y=A['y'],refB_y=B['y'],truth_y=truth['y'],truth_mu=truth['mu'],truth_M=truth['M'],seeds=np.asarray(seeds),samples_per_stream=np.asarray(a.n))
    for rp in sorted(recdir.glob(f'network_{a.network:04d}_rotation_*.json')):
        rec=json.loads(rp.read_text());vp=vecdir/rec['vectors_file'];old=np.load(vp);d={k:old[k] for k in old.files};Bmap=weighted_beta(old)
        oracle_anchor=lower_anchor(old['sample_center'],truth['mu'],truth['M'],old['probe_indices'],old['probe_directions']);oracle_corr=oracle_anchor@Bmap
        d['truth_y']=truth['y'];d['ref1_y']=A['y'];d['ref2_y']=B['y'];d['oracle_anchor']=oracle_anchor;d['oracle_correction']=oracle_corr
        np.savez_compressed(vp,**d)
        base=d['basefull'];y0=d['reduced_base'];corrs=d['corrections'];c17=corrs[16];ideal=truth['y']-y0;den=mse(base,truth['y'])
        rec['full_baseline_mse']=den;rec['full_baseline_unbiased_mse']=fr.unbiased_mse(base,A['y'],B['y']);rec['reduced_base_mse']=mse(y0,truth['y']);rec['reduced_ratio']=rec['reduced_base_mse']/max(den,1e-300);rec['c17_mse']=mse(y0+c17,truth['y']);rec['c17_ratio']=rec['c17_mse']/max(den,1e-300)
        rec['prefix_metrics']={}
        for k in PREFIXES:
            c=corrs[k-1];mm=mse(y0+c,truth['y']);rec['prefix_metrics'][str(k)]={'mse':mm,'ratio_to_full_baseline':mm/max(den,1e-300),'correction_norm':float(np.linalg.norm(c)),'correction_cosine':cosine(c,ideal)}
        err=y0-truth['y'];rec['correction_metrics']={'error_correction_inner_product':float(err@c17/D),'ideal_correction_inner_product':float(ideal@c17/D),'correction_norm':float(np.linalg.norm(c17)),'correction_cosine':cosine(c17,ideal),'oracle_lower_correction_cosine':cosine(c17,oracle_corr)}
        pc=d['paired_correction'];rec['paired'].update({'cosine_with_ideal':cosine(pc,ideal)})
        rec['oracle_lower_ratio']=mse(y0+oracle_corr,truth['y'])/max(den,1e-300);rec['reference_noise_mse']=float(.25*mse(A['y'],B['y']));rec['reference_noise_fraction']=rec['reference_noise_mse']/max(den,1e-300);rec['truth_n_per_half']=3*a.n;rec['reference_seeds']=seeds;rec['reference_refinement']='uniform_6x65536';rec['vectors_sha256']=fr.sha256_file(vp)
        rp.write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps({'network':a.network,'done':True,'effective_per_half':3*a.n}),flush=True)
if __name__=='__main__':main()
