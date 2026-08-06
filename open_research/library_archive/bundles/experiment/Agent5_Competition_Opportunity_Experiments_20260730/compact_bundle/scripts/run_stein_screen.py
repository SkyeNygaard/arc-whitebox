#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, json, math, os, sys, time, hashlib
from pathlib import Path
import numpy as np
import torch

# Reuse the exact Kerdock and deterministic network construction from the audited T4 package.
T4_CODE=Path('/mnt/data/work/T4_legal_layer31_anchor_hedge_20260729_review/T4_legal_layer31_anchor_hedge_20260729/code')
sys.path.insert(0,str(T4_CODE))
import frozen_reference_impl as fr

D=256; DEPTH=32; RHO=fr.chi_mean(D); N_BASES=129; ROWS_PER_BASIS=512
OUT=Path('/mnt/data/work/new_opportunities/results'); OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(min(5,os.cpu_count() or 1)); torch.set_num_interop_threads(1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    out=np.empty_like(x,dtype=np.float64)
    pos=x>=0
    out[pos]=1/(1+np.exp(-x[pos])); e=np.exp(x[~pos]); out[~pos]=e/(1+e)
    return out


def sphere_stein(t: np.ndarray, family: str, k: float, b: float) -> np.ndarray:
    z=k*(t-b)
    if family=='softplus':
        psi=np.logaddexp(0.0,z)/k
        dpsi=sigmoid(z)
    elif family=='sigmoid':
        psi=sigmoid(z)
        dpsi=k*psi*(1-psi)
    elif family=='tanh':
        psi=np.tanh(z)
        dpsi=k*(1-psi*psi)
    else: raise ValueError(family)
    return (1-t*t)*dpsi-(D-1)*t*psi


def gegen_raw(t: np.ndarray,n:int) -> np.ndarray:
    a=(D-2)/2
    c0=np.ones_like(t,dtype=np.float64); s0=1.0
    if n==0:return c0
    c1=2*a*t; s1=2*a
    if n==1:return c1/s1
    for m in range(2,n+1):
        c2=(2*(m+a-1)*t*c1-(m+2*a-2)*c0)/m
        s2=(2*(m+a-1)*s1-(m+2*a-2)*s0)/m
        c0,c1=c1,c2; s0,s1=s1,s2
    return c1/s1


def highpass(t: np.ndarray, s: np.ndarray, maxdeg:int=5) -> np.ndarray:
    # Remove the exact low-degree zonal span. The Kerdock 5-design makes every
    # removed component mean-zero, so the control's exact expectation stays zero.
    Z=np.stack([gegen_raw(t,n) for n in range(1,maxdeg+1)],axis=1)
    coef=np.linalg.lstsq(Z,s,rcond=1e-12)[0]
    return s-Z@coef


def normalize_cols(G: np.ndarray) -> np.ndarray:
    scale=np.sqrt(np.mean(G*G,axis=0))
    keep=scale>1e-12
    G=G[:,keep]; scale=scale[keep]
    return G/scale


def effective_j(ws:list[torch.Tensor], gates:list[np.ndarray]) -> np.ndarray:
    j=np.eye(D,dtype=np.float64)
    for w,p in zip(ws,gates):
        j=(j@w.numpy().astype(np.float64))*p[None,:]
        norm=np.linalg.norm(j)
        if norm>1e30 or (norm and norm<1e-30): j/=norm
    return j


def direction_sets(ws:list[torch.Tensor], gates:list[np.ndarray], k:int=4) -> dict[str,np.ndarray]:
    w0=ws[0].numpy().astype(np.float64)
    Uj=np.linalg.svd(effective_j(ws,gates),full_matrices=False)[0][:,:k]
    Uw=np.linalg.svd(w0,full_matrices=False)[0][:,:k]
    # First-layer neuron directions ranked by downstream linearized path norm.
    P=np.eye(D,dtype=np.float64)
    for w,p in zip(ws[1:],gates[1:]):
        P=(P@w.numpy().astype(np.float64))*p[None,:]
        nrm=np.linalg.norm(P)
        if nrm>1e20 or (nrm and nrm<1e-20): P/=nrm
    idx=np.argsort(np.linalg.norm(P,axis=1))[::-1]
    cols=[]
    for j in idx:
        u=w0[:,j].copy(); n=np.linalg.norm(u)
        if n>1e-12:
            u/=n
            if not cols or max(abs(u@v) for v in cols)<0.98: cols.append(u)
        if len(cols)>=k: break
    Ud=np.stack(cols,axis=1)
    Ur=np.linalg.qr(np.random.default_rng(20260730).standard_normal((D,k)))[0][:,:k]
    return {'jac':Uj,'w1':Uw,'downstream':Ud,'random':Ur}


def build_features(xk: np.ndarray, dirs:dict[str,np.ndarray]) -> dict[str,np.ndarray]:
    ts={name:(xk@U)/RHO for name,U in dirs.items()}
    out={}
    # Frozen historical comparator: network-adaptive jac directions, degrees 6+8.
    tj=ts['jac']
    out['harmonic_jac_d68']=normalize_cols(np.concatenate([
        np.stack([gegen_raw(tj[:,q],n) for q in range(tj.shape[1])],axis=1)
        for n in (6,8)],axis=1))
    biases=(-0.10,0.0,0.10)
    for fam,k in [('softplus',8.0),('sigmoid',8.0),('tanh',6.0)]:
        bysrc={}
        for src,tmat in ts.items():
            cols=[]
            for q in range(tmat.shape[1]):
                t=tmat[:,q]
                for b in biases:
                    cols.append(highpass(t,sphere_stein(t,fam,k,b)))
            bysrc[src]=normalize_cols(np.stack(cols,axis=1))
        out[f'stein_{fam}_jac_hp']=bysrc['jac']
        out[f'stein_{fam}_all_hp']=normalize_cols(np.concatenate([bysrc[s] for s in ('jac','w1','downstream')],axis=1))
    out['stein_combo_all_hp']=normalize_cols(np.concatenate([
        out['stein_softplus_all_hp'],out['stein_sigmoid_all_hp'],out['stein_tanh_all_hp']],axis=1))
    return out


def ridge_beta(G:np.ndarray,Y:np.ndarray,ridge_mult:float=1e-8) -> np.ndarray:
    Gc=G-G.mean(0,keepdims=True); Yc=Y-Y.mean(0,keepdims=True)
    gram=Gc.T@Gc
    r=ridge_mult*max(float(np.trace(gram))/max(1,gram.shape[0]),1e-30)
    return np.linalg.solve(gram+r*np.eye(gram.shape[0]),Gc.T@Yc)


def estimate_full(G,Y):
    b=ridge_beta(G,Y)
    return Y.mean(0)-G.mean(0)@b,b


def estimate_cf(G,Y,bid,folds=4):
    vals=[]; weights=[]
    for f in range(folds):
        te=(bid%folds)==f; tr=~te
        b=ridge_beta(G[tr],Y[tr])
        vals.append(Y[te].mean(0)-G[te].mean(0)@b); weights.append(int(te.sum()))
    return np.average(np.stack(vals),axis=0,weights=weights)


def mse_obs(pred,truth): return float(np.mean((pred-truth)**2))
def mse_unb(pred,a,b): return float(np.mean((pred-a)*(pred-b)))

def geometry(base,pred,truth):
    e=base-truth; c=pred-base
    nn=float(c@c/D); inn=float(e@c/D); ee=float(e@e/D)
    cos=float(-inn/max(math.sqrt(max(ee*nn,0.0)),1e-30))
    alpha=float(-inn/max(nn,1e-30))
    op=base+alpha*c
    return {'correction_norm_sq':nn,'error_correction_inner':inn,'correction_cosine':cos,'oracle_alpha':alpha,'oracle_pred':op}


def calibration_stats(G:np.ndarray) -> dict:
    n=len(G); gm=G.mean(0); Gc=G-gm
    gram=Gc.T@Gc; r=1e-10*max(float(np.trace(gram))/max(1,gram.shape[0]),1e-30)
    a=np.linalg.solve(gram+r*np.eye(gram.shape[0]),gm)
    delta=-(Gc@a)
    w=1.0/n+delta
    return {'negative_mass':float(np.maximum(-w,0).sum()),'min_weight':float(w.min()),'max_weight':float(w.max()),'l1_weight':float(np.abs(w).sum()),'effective_support_inverse_l2':float(1/np.sum(w*w))}


def forward_kerdock(xk,ws):
    x=torch.from_numpy(xk.copy())
    gates=[]
    with torch.inference_mode():
        for w in ws:
            pre=x@w; gates.append((pre>0).double().mean(0).numpy()); x=torch.relu(pre)
    return x.numpy(),gates


def qmc_final(ws,n,seed):
    eng=torch.quasirandom.SobolEngine(D,scramble=True,seed=seed)
    total=np.zeros(D,dtype=np.float64); done=0; chunk=8192
    with torch.inference_mode():
        while done<n:
            b=min(chunk,n-done)
            u=eng.draw(b,dtype=torch.float32).clamp_(1e-7,1-1e-7)
            x=math.sqrt(2.0)*torch.erfinv(2*u-1)
            for w in ws:x=torch.relu(x@w)
            total+=x.double().sum(0).numpy(); done+=b
    return total/n


def run_one(network_id:int,nref:int,xk:np.ndarray,bid:np.ndarray):
    t0=time.time(); ws,whash,wseed=fr.make_weights(network_id)
    ta=qmc_final(ws,nref,900000+2*network_id); tb=qmc_final(ws,nref,900001+2*network_id); truth=.5*(ta+tb)
    Y,gates=forward_kerdock(xk,ws); Y=Y.astype(np.float64,copy=False); base=Y.mean(0)
    dirs=direction_sets(ws,gates); features=build_features(xk,dirs)
    row={'network_id':network_id,'weight_seed':wseed,'weight_sha256':whash,'nref_per_half':nref,
         'base_observed_mse':mse_obs(base,truth),'base_unbiased_mse':mse_unb(base,ta,tb),
         'reference_noise_mse':float(np.mean((ta-tb)**2)/4),'candidates':{}}
    for name,G in features.items():
        full,_=estimate_full(G,Y); cf4=estimate_cf(G,Y,bid,4)
        geo=geometry(base,full,truth); oracle=geo.pop('oracle_pred')
        row['candidates'][name]={
            'n_features':int(G.shape[1]),
            'full_observed_mse':mse_obs(full,truth),'full_unbiased_mse':mse_unb(full,ta,tb),
            'cf4_observed_mse':mse_obs(cf4,truth),'cf4_unbiased_mse':mse_unb(cf4,ta,tb),
            'oracle_scalar_observed_mse':mse_obs(oracle,truth),'oracle_scalar_unbiased_mse':mse_unb(oracle,ta,tb),
            **geo,**calibration_stats(G)}
    row['seconds']=time.time()-t0
    p=OUT/f'stein_network_{network_id}.json'; p.write_text(json.dumps(row,indent=2,sort_keys=True))
    print(json.dumps({'network':network_id,'seconds':row['seconds'],'base':row['base_observed_mse'],
                      'best_cf4':min((v['cf4_observed_mse'],k) for k,v in row['candidates'].items()),
                      'best_oracle':min((v['oracle_scalar_observed_mse'],k) for k,v in row['candidates'].items())},sort_keys=True),flush=True)
    del Y,features,dirs,ws; gc.collect()
    return row


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('networks',nargs='+',type=int); ap.add_argument('--nref',type=int,default=65536); args=ap.parse_args()
    xk,meta=fr.make_kerdock(); bid=np.repeat(np.arange(N_BASES,dtype=np.int64),ROWS_PER_BASIS)
    for n in args.networks: run_one(n,args.nref,xk,bid)

if __name__=='__main__': main()
