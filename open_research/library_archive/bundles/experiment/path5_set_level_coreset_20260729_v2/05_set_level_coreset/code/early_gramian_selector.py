#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Set before numpy import in child processes.
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np

import exact_kerdock_coreset_diagnostic as base
import set_level_basis_feasibility as sl
from top8_sketch_selector import TOP8

LIB_SEED=202607295
GATE=1.1e-8

def parse_seeds(s: str):
    if ':' in s:
        a,b=s.split(':'); return list(range(int(a),int(b)))
    return [int(x) for x in s.split(',') if x]

def downstream_gramian(weights: list[np.ndarray], start_weight_index: int=2) -> np.ndarray:
    # Gramian on activation A_start, with scalar mean-gate factors omitted since
    # they do not affect eigenvectors or relative eigenvalues after trace normalization.
    g=np.eye(base.WIDTH,dtype=np.float64)
    for k in range(len(weights)-1,start_weight_index-1,-1):
        w=weights[k].astype(np.float64)
        g=w@g@w.T
        g=(g+g.T)*0.5
        tr=float(np.trace(g))
        if not np.isfinite(tr) or tr<=1e-300:
            raise RuntimeError(f'bad Gramian trace at layer {k}: {tr}')
        g/=tr/base.WIDTH
    return g

def standardize(x: np.ndarray) -> np.ndarray:
    x=x.astype(np.float64)
    m=x.mean(0); s=x.std(0)
    s=np.maximum(s,1e-10)
    return ((x-m)/s).astype(np.float64)

def score_one_bank(z: np.ndarray, lib: np.ndarray, quota: np.ndarray, dims, ridges, prefix: str):
    out={}
    Q=z.shape[1]
    valid=[q for q in dims if q<=Q]
    for q in valid:
        out[f'{prefix}_q{q}_global']=np.empty(len(lib),dtype=np.float64)
        for r in ridges: out[f'{prefix}_q{q}_r{r:g}']=np.empty(len(lib),dtype=np.float64)
    for ci,sel in enumerate(lib):
        S=z[sel].copy(); u=base.base_weights(sel,quota); t=S.T@u
        bids=sel//base.PAIRS_PER_BASIS
        for b in range(base.ALL_BASES):
            ii=np.flatnonzero(bids==b); S[ii]-=S[ii].mean(0,keepdims=True)
        G=S.T@S
        for q in valid:
            # Columns are ordered low-to-high importance, so the last q are retained.
            slc=slice(Q-q,Q); tq=t[slc]; Sq=S[:,slc]; Gq=G[slc,slc]
            out[f'{prefix}_q{q}_global'][ci]=float(tq@tq/q)
            I=np.eye(q)
            for r in ridges:
                c=np.linalg.solve(Gq+r*len(sel)*I,tq)
                ww=u-Sq@c; rel=ww/u; ess=1/np.sum(ww*ww)/len(ww)
                e=z[sel,slc].T@ww; sc=float(e@e/q)
                sc+=max(0,.05-rel.min())**2+max(0,rel.max()-4)**2+10*max(0,.8-ess)**2
                out[f'{prefix}_q{q}_r{r:g}'][ci]=sc
    return out

def labels_for(seed: int, label_dirs: list[Path]) -> np.ndarray:
    for d in label_dirs:
        p=d/f'top8_rank_seed{seed}.npz'
        if p.exists():
            with np.load(p,allow_pickle=False) as z: return z['labels'].astype(np.float64)
        p=d/f'setlevel_seed{seed}_q128_c64.npz'
        if p.exists():
            with np.load(p,allow_pickle=False) as z: return z['labels'][TOP8].astype(np.float64)
    raise FileNotFoundError(f'labels for {seed}')

def run_one(seed:int, asset:Path, label_dirs:list[Path], dims, ridges, outdir:Path):
    t=time.time()
    with np.load(asset,allow_pickle=False) as a:
        ch=a['chirps'].astype(np.float32); rot=a['rotation'].astype(np.float32)
    quota=base.quotas(4096)
    alllib=np.concatenate([sl.fixed_random_library(64,quota,LIB_SEED),sl.affine_stratified_library(64,quota,LIB_SEED+99173)])
    lib=alllib[TOP8]
    W=base.gen_weights(seed)
    h1=base.first_activation(W[0],ch,rot)
    z2=(h1@W[1]).astype(np.float32)
    a2=base.relu(z2).astype(np.float32)
    resid=(a2-0.5*z2).astype(np.float32)
    gram=downstream_gramian(W,2)
    eig,U=np.linalg.eigh(gram)
    eig=np.maximum(eig,0)
    scale=np.sqrt(eig/np.maximum(eig.max(),1e-30))
    # np.linalg.eigh returns ascending eigenvalues; preserve that nesting.
    pa=base.pair_average(a2@U).astype(np.float64)*scale[None,:]
    pr=base.pair_average(resid@U).astype(np.float64)*scale[None,:]
    # Coordinate control, ordered by Gramian diagonal importance.
    order=np.argsort(np.diag(gram))
    pc=base.pair_average(a2[:,order]).astype(np.float64)*np.sqrt(np.maximum(np.diag(gram)[order],0))[None,:]
    za=standardize(pa); zr=standardize(pr); zc=standardize(pc)
    # Joint bank keeps matched low-to-high eigenmode ordering by interleaving A/R.
    joint=np.empty((za.shape[0],2*za.shape[1]),dtype=np.float64)
    joint[:,0::2]=za; joint[:,1::2]=zr
    scores={}
    scores.update(score_one_bank(za,lib,quota,dims,ridges,'gram_a'))
    scores.update(score_one_bank(zr,lib,quota,dims,ridges,'gram_resid'))
    scores.update(score_one_bank(zc,lib,quota,dims,ridges,'diag_a'))
    scores.update(score_one_bank(joint,lib,quota,[2*q for q in dims],ridges,'gram_joint'))
    labels=labels_for(seed,label_dirs)
    names=sorted(scores)
    raw=np.stack([scores[n] for n in names],axis=1)
    # Candidate-level feature matrix: log scores, within-network ranks/z-scores,
    # candidate one-hot, and network-wide Gramian spectrum summaries.
    log=np.log10(raw+1e-30); ranks=np.empty_like(log); zs=np.empty_like(log)
    for j in range(log.shape[1]):
        o=np.argsort(log[:,j],kind='stable'); rr=np.empty(8); rr[o]=np.arange(8)/7; ranks[:,j]=rr
        zs[:,j]=(log[:,j]-log[:,j].mean())/(log[:,j].std()+1e-8)
    spectrum=np.array([eig[-1],eig[-4:].sum(),eig[-8:].sum(),eig[-16:].sum(),eig[-32:].sum(),
                       np.sum(eig*eig)/(eig.sum()**2+1e-30)],dtype=np.float64)
    spectrum=np.tile(spectrum[None,:],(8,1))
    X=np.concatenate([log,ranks,zs,np.eye(8),spectrum],axis=1).astype(np.float32)
    methods={}
    for n,s in scores.items():
        j=int(np.argmin(s)); methods[n]={'slot':j,'candidate':int(TOP8[j]),'label':float(labels[j]),'score':float(s[j])}
    out={'seed':seed,'runtime_s':time.time()-t,'oracle_best':float(labels.min()),'oracle_passes':int(np.sum(labels<=GATE)),
         'methods':methods,'score_names':names,'labels':labels.tolist(),'candidates':TOP8.tolist(),
         'gramian_top_eigs':eig[-16:].tolist()}
    outdir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(outdir/f'early_gramian_seed{seed}.npz',X=X,labels=labels,candidates=TOP8,raw_scores=raw,score_names=np.array(names),eig=eig)
    (outdir/f'early_gramian_seed{seed}.json').write_text(json.dumps(out,indent=2))
    return out

def summarize(rs):
    names=sorted(rs[0]['methods']); out={}
    for n in names:
        v=np.array([r['methods'][n]['label'] for r in rs])
        out[n]={'pass11':int(np.sum(v<=GATE)),'pass22':int(np.sum(v<=2.2e-8)),'mean':float(v.mean()),'median':float(np.median(v)),'worst':float(v.max()),
                'values':v.tolist(),'candidates':[r['methods'][n]['candidate'] for r in rs]}
    o=np.array([r['oracle_best'] for r in rs])
    out['oracle_best_top8']={'pass11':int(np.sum(o<=GATE)),'pass22':int(np.sum(o<=2.2e-8)),'mean':float(o.mean()),'worst':float(o.max())}
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--asset',type=Path,required=True);p.add_argument('--seeds',required=True)
    p.add_argument('--label-dir',type=Path,action='append',required=True);p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--dims',default='8,16,32,64,128');p.add_argument('--ridges',default='.0001,.01,1,100');p.add_argument('--workers',type=int,default=4);p.add_argument('--summary',type=Path,required=True)
    a=p.parse_args(); seeds=parse_seeds(a.seeds); dims=[int(x) for x in a.dims.split(',')];ridges=[float(x) for x in a.ridges.split(',')]
    rs=[]
    if a.workers==1:
        for s in seeds:
            r=run_one(s,a.asset,a.label_dir,dims,ridges,a.outdir);rs.append(r);print(s,f"{r['oracle_best']:.2e}",f"{r['runtime_s']:.1f}s",flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            fut={ex.submit(run_one,s,a.asset,a.label_dir,dims,ridges,a.outdir):s for s in seeds}
            for f in as_completed(fut):
                r=f.result();rs.append(r);print(r['seed'],f"{r['oracle_best']:.2e}",f"{r['runtime_s']:.1f}s",flush=True)
    rs.sort(key=lambda x:x['seed']); payload={'config':{'seeds':seeds,'dims':dims,'ridges':ridges},'records':rs,'summary':summarize(rs)}
    a.summary.write_text(json.dumps(payload,indent=2)); print(json.dumps(payload['summary'],indent=2))
if __name__=='__main__':main()
