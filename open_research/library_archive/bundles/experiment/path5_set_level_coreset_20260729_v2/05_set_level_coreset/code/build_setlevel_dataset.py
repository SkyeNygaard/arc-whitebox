#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','5'); os.environ.setdefault('OMP_NUM_THREADS','5'); os.environ.setdefault('MKL_NUM_THREADS','5')
import numpy as np
import exact_kerdock_coreset_diagnostic as base
import set_level_basis_feasibility as sl


def descriptors(sketch_pairs: np.ndarray, library: np.ndarray, quota: np.ndarray):
    x=sketch_pairs.astype(np.float64)
    mu0=x.mean(0); sd=x.std(0)+1e-8
    x=(x-mu0)/sd
    full=x.reshape(base.ALL_BASES,base.PAIRS_PER_BASIS,x.shape[1])
    fm=full.mean(1); fv=full.var(1)+1e-6
    slices=[]; p=0
    for q0 in quota:
        q=int(q0); slices.append(slice(p,p+q)); p+=q
    feats=[]; scalars=[]
    for idx in library:
        ds=[]; lrs=[]; selmeans=[]
        for b,slc in enumerate(slices):
            s=x[idx[slc]]; sm=s.mean(0); sv=s.var(0)+1e-6
            ds.append(sm-fm[b]); lrs.append(np.log(np.clip(sv/fv[b],1e-4,1e4))); selmeans.append(sm)
        d=np.stack(ds); lr=np.stack(lrs); sm=np.stack(selmeans)
        g=sm.mean(0) # full global standardized mean is zero
        brms=np.sqrt(np.mean(d*d,axis=0))
        babs=np.mean(np.abs(d),axis=0)
        bmax=np.max(np.abs(d),axis=0)
        vlm=np.mean(lr,axis=0)
        vlr=np.sqrt(np.mean(lr*lr,axis=0))
        effort=np.mean(d*d/(np.exp(lr)*fv + .05*fv + 1e-8),axis=1)
        sc=np.array([
            np.mean(g*g), np.mean(brms), np.mean(babs), np.mean(bmax),
            np.mean(vlm), np.mean(vlr), np.mean(effort), np.std(effort),
            np.quantile(effort,.9), np.max(effort),
        ])
        feats.append(np.concatenate([g,brms,babs,bmax,vlm,vlr,sc]))
        scalars.append(sc)
    return np.asarray(feats,dtype=np.float32), np.asarray(scalars,dtype=np.float32)


def run(seed, asset, outdir, qcoords, library_count):
    t=time.time()
    with np.load(asset,allow_pickle=False) as a:
        chirps=a['chirps'].astype(np.float32); rotation=a['rotation'].astype(np.float32)
    quota=base.quotas(4096)
    rf=sl.fixed_random_library(library_count,quota,202607295)
    af=sl.affine_stratified_library(library_count,quota,202607295+99173)
    lib=np.concatenate([rf,af])
    W=base.gen_weights(seed); H=base.propagate_to_anchor(W,chirps,rotation,28)
    A=H
    for w in W[28:31]: A=base.relu(A@w)
    coords=sl.pilot_output_coordinates(A,W[31],qcoords)
    sketch=base.pair_average(base.relu(A@W[31][:,coords]))
    X,scalar=descriptors(sketch,lib,quota)
    Y=base.pair_average(base.relu(A@W[31])); full=Y.mean(0); OF=base.standardize(Y)
    labels=[]; minrel=[]; maxrel=[]; ess=[]
    for i,s in enumerate(lib):
        ow,info=base.calibrated_weights(OF[s],s,quota)
        labels.append(base.added_mse(Y,s,ow,full))
        minrel.append(info['min_relative']); maxrel.append(info['max_relative']); ess.append(info['ess_fraction'])
    labels=np.asarray(labels,dtype=np.float64)
    path=outdir/f'setlevel_seed{seed}_q{qcoords}_c{library_count}.npz'
    np.savez_compressed(path,features=X,scalar_features=scalar,labels=labels,
                        family=np.r_[np.zeros(library_count,np.int8),np.ones(library_count,np.int8)],
                        candidate=np.arange(2*library_count,dtype=np.int16),coords=coords,
                        min_relative=np.asarray(minrel),max_relative=np.asarray(maxrel),ess=np.asarray(ess))
    meta={'seed':seed,'path':str(path),'runtime_s':time.time()-t,'best':float(labels.min()),
          'median':float(np.median(labels)),'passes_1.1e-8':int(np.sum(labels<=1.1e-8)),
          'passes_2.2e-8':int(np.sum(labels<=2.2e-8)),'fixed0':float(labels[0]),
          'feature_dim':int(X.shape[1]),'qcoords':qcoords,'library_count_each':library_count}
    (outdir/f'setlevel_seed{seed}_meta.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta),flush=True)


def parse(s):
    if ':' in s:
        a,b=s.split(':'); return range(int(a),int(b))
    return [int(x) for x in s.split(',') if x]

ap=argparse.ArgumentParser();ap.add_argument('--asset',type=Path,required=True);ap.add_argument('--seeds',required=True)
ap.add_argument('--outdir',type=Path,required=True);ap.add_argument('--qcoords',type=int,default=128);ap.add_argument('--library-count',type=int,default=64)
a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
for seed in parse(a.seeds): run(seed,a.asset,a.outdir,a.qcoords,a.library_count)
