#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, resource, time
from pathlib import Path
from typing import Any
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

D=256; DEPTH=32; TARGET=29; ROWS_PER_BASIS=512; P=128; RIDGE=.1; AMP=.20
BASELINE_EFFECTIVE=175.62e9
ORIENTATION_SEEDS=[1005,1007,1011,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091]
COMMON_SEED=1193
REFERENCE_ROTATION_SEEDS=list(range(1170000,1170016))
PROTOCOL='priority6-coherent-orientation-codebook-v2-immutable-validation'
EPS=1e-30

def chi_mean(d=D):
    return math.sqrt(2)*math.exp(math.lgamma((d+1)/2)-math.lgamma(d/2))
RHO=chi_mean()

def haar(seed:int)->np.ndarray:
    rng=np.random.default_rng(seed); a=rng.standard_normal((D,D)); q,r=np.linalg.qr(a)
    s=np.sign(np.diag(r)); s[s==0]=1
    return (q*s[None,:]).astype(np.float32)

def fwht(v:np.ndarray)->np.ndarray:
    v=v.copy(); span=1
    while span<D:
        g=v.reshape(D//(2*span),2,span,D)
        left=g[:,0].copy(); right=g[:,1].copy()
        v=np.stack((left+right,left-right),axis=1).reshape(D,D)
        span*=2
    return v

def first_blocks(w0:np.ndarray, rotation:np.ndarray, blocks:list[int], chirps:np.ndarray, return_block_means=False):
    eff=np.asarray(rotation,np.float32)@w0
    out=[]; means=[]; scale=np.float32(RHO/math.sqrt(D))
    for b in blocks:
        if b<128:
            pre=fwht(chirps[b,:,None]*eff)*scale
            rows=np.stack((pre,-pre),axis=1).reshape(ROWS_PER_BASIS,D)
        else:
            rows=np.stack((np.float32(RHO)*eff,-np.float32(RHO)*eff),axis=1).reshape(ROWS_PER_BASIS,D)
        np.maximum(rows,0,out=rows); rows=rows.astype(np.float32,copy=False)
        out.append(rows)
        if return_block_means: means.append(rows.mean(0,dtype=np.float64))
    a=np.concatenate(out,axis=0)
    return (a,np.asarray(means)) if return_block_means else a

def propagate(a:np.ndarray, weights:list[np.ndarray], start_layer=1, target=TARGET, final=True):
    ht=None; buf=np.empty_like(a)
    for li in range(start_layer, DEPTH if final else target+1):
        np.matmul(a,weights[li],out=buf); np.maximum(buf,0,out=buf); a,buf=buf,a
        if li==target: ht=a.copy()
    if ht is None and target==0: ht=a.copy()
    return ht,a if final else ht


def complete_design_mean(weights:list[np.ndarray], chirps:np.ndarray, rotation:np.ndarray)->np.ndarray:
    a=first_blocks(weights[0],rotation,list(range(129)),chirps)
    _,y=propagate(a,weights,1,TARGET,True)
    return y.mean(0,dtype=np.float64)

def complete_rotation_reference(weights:list[np.ndarray], chirps:np.ndarray, seeds:list[int])->tuple[np.ndarray,np.ndarray,np.ndarray]:
    vals=[]
    for seed in seeds:
        vals.append(complete_design_mean(weights,chirps,haar(seed)))
    vals=np.asarray(vals,dtype=np.float64)
    mid=len(vals)//2
    a=vals[:mid].mean(0); b=vals[mid:].mean(0)
    return .5*(a+b),a,b

def qmc_final(weights:list[np.ndarray], n:int, seed:int, chunk=8192)->np.ndarray:
    eng=qmc.Sobol(d=D,scramble=True,seed=seed); total=np.zeros(D); done=0
    # n is power of two in production runs.
    while done<n:
        k=min(chunk,n-done)
        u=eng.random(k); x=ndtri(np.clip(u,1e-12,1-1e-12)).astype(np.float32)
        a=x; buf=np.empty_like(a)
        for w in weights:
            np.matmul(a,w,out=buf); np.maximum(buf,0,out=buf); a,buf=buf,a
        total+=a.sum(0,dtype=np.float64); done+=k
    return total/n

def make_weights(seed:int)->list[np.ndarray]:
    rng=np.random.default_rng(seed)
    arr=(rng.standard_normal((DEPTH,D,D))*math.sqrt(2/D)).astype(np.float32)
    return [arr[i] for i in range(DEPTH)]

def sample_anchor_matrix(H,m):
    raw=(H*H).T@H/len(H); M=H.T@H/len(H); m2=np.diag(M)
    return raw/(RHO*RHO)-D/(D+1)*m2[:,None]*m[None,:]/(RHO*RHO)-2*D/(D+1)*m[:,None]*M/(RHO*RHO)+2/(D+1)*(m*m)[:,None]*m[None,:]

def radial_features(H,m,L,R):
    h2=H*H; hm=H*m[None,:]
    return ((h2@L)*(H@R)/(RHO*RHO)
            -(h2@L)*(m@R)[None,:]*(D/(D+1))/(RHO*RHO)
            -(hm@L)*(H@R)*(2*D/(D+1))/(RHO*RHO)
            +(m*m@L)[None,:]*(H@R)*(2/(D+1)))

def fit_crossfit(X,Y,bid,folds=6,ridge=RIDGE):
    uniq=np.unique(bid); groups=np.array_split(uniq,folds)
    c=np.zeros(Y.shape[1]); B=np.zeros((X.shape[1],Y.shape[1]))
    for g in groups:
        take=~np.isin(bid,g); x=X[take]; y=Y[take]
        xm=x.mean(0); ym=y.mean(0); xc=x-xm; yc=y-ym
        G=xc.T@xc/len(xc); C=xc.T@yc/len(xc)
        lam=ridge*(np.trace(G)/max(len(G),1)+1e-12)
        b=np.linalg.solve(G+lam*np.eye(len(G)),C)
        c+=(ym-xm@b)/folds; B+=b/folds
    return c,B

def lower_anchor(mu,M,m,L,R):
    # Contraction-specific lower-order center defect, with free same-cloud pair moments.
    du=(m-mu)@L  # p, L generally coordinate columns
    dv=(m-mu)@R
    mu_u=mu@L; mu_v=mu@R
    m2uu=np.einsum('ir,ij,jr->r',L,M,L)
    m2uv=np.einsum('ir,ij,jr->r',L,M,R)
    return (-m2uu*dv-2*du*m2uv+4*mu_u*du*mu_v+2*du*du*mu_v)/(D+1)

def cos(a,b):
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+EPS))

def correction_from_mu(mu,M,m,L,R,B):
    return lower_anchor(mu,M,m,L,R)@B

def case_run(base_id:int, variant:int, truth:np.ndarray, truth_noise:float, chirps, main_rot, orientations, ref_n:int)->dict[str,Any]:
    t0=time.perf_counter(); weights=make_weights(730000+base_id)
    if variant:
        qv=haar(910000+1009*base_id+variant); weights[0]=qv@weights[0]
    exact_mu1=np.linalg.norm(weights[0].astype(np.float64),axis=0)/math.sqrt(2*math.pi)
    # Full 129 basis baseline; derive frozen 112 arm from it.
    a=first_blocks(weights[0],main_rot,list(range(129)),chirps)
    H,Y=propagate(a,weights,1,TARGET,True)
    basefull=Y.mean(0,dtype=np.float64)
    bid=np.repeat(np.arange(129),ROWS_PER_BASIS)
    keep=np.r_[np.arange(111),128]; take=np.isin(bid,keep)
    Hs=H[take].astype(np.float64); Ys=Y[take].astype(np.float64); bids=bid[take]
    m=Hs.mean(0); M=Hs.T@Hs/len(Hs)*(D/(RHO*RHO)); redbase=Ys.mean(0)
    Q=sample_anchor_matrix(Hs,m); ix=np.argsort(np.linalg.norm(Q,axis=1))[::-1][:P]
    L=np.eye(D)[:,ix]; rr=Q[ix].copy(); rr/=np.maximum(np.linalg.norm(rr,axis=1,keepdims=True),EPS); R=rr.T
    X=radial_features(Hs,m,L,R); c,B=fit_crossfit(X,Ys,bids); qmean=X.mean(0); y0=c+qmean@B
    base_mse=float(np.mean((basefull-truth)**2)); base_mse_nc=max(base_mse-truth_noise,1e-20)
    rows=[]; probe_vectors=[]
    orig1=Hs[bids==0].mean(0); orig2=Hs[np.isin(bids,[0,1])].mean(0)
    for oi,(oseed,orot) in enumerate(zip(ORIENTATION_SEEDS,orientations)):
        # One pass for all 17 bases; nested 1/2 probes retained and reused.
        ca,block_mu1=first_blocks(weights[0],orot,list(range(17)),chirps,True)
        # first-layer legal discrepancy from the first two probe bases
        first_err=float(np.linalg.norm(block_mu1[:2].mean(0)-exact_mu1)/(np.linalg.norm(exact_mu1)+EPS))
        h=ca; buf=np.empty_like(h); nested={}
        for li in range(1,TARGET+1):
            np.matmul(h,weights[li],out=buf); np.maximum(buf,0,out=buf); h,buf=buf,h
        hb=h.astype(np.float64).reshape(17,ROWS_PER_BASIS,D).mean(1)
        mu17=hb.mean(0); mu1=hb[0]; mu2=hb[:2].mean(0)
        c17=AMP*correction_from_mu(mu17,M,m,L,R,B)
        c1=AMP*correction_from_mu(mu1,M,m,L,R,B)
        c2=AMP*correction_from_mu(mu2,M,m,L,R,B)
        d1=(1/129.0)*(orig1-mu1); d2=(2/129.0)*(orig2-mu2)
        p1=correction_from_mu(m-d1,M,m,L,R,B)
        p2=correction_from_mu(m-d2,M,m,L,R,B)
        pred=y0+c17; mse=float(np.mean((pred-truth)**2)); mse_nc=max(mse-truth_noise,1e-20)
        err=truth-y0
        ip=float(err@c17); cn=float(np.linalg.norm(c17)); ec=cos(err,c17)
        opt=float(np.clip(ip/(c17@c17+EPS),-2,2)); optm=float(np.mean((y0+opt*c17-truth)**2))
        rows.append({
            'orientation':oi,'orientation_seed':oseed,'mse':mse,'mse_nc':mse_nc,'ratio':mse/base_mse,
            'ratio_nc':mse_nc/base_mse_nc,'correction':c17.tolist(),'p1':p1.tolist(),'p2':p2.tolist(),
            'c1':c1.tolist(),'c2':c2.tolist(),'correction_norm':cn,'error_correction_ip':ip,
            'correction_cosine':ec,'oracle_scale':opt,'oracle_scale_mse':optm,
            'p1_p2_cos':cos(p1,p2),'c1_c2_cos':cos(c1,c2),'c2_p2_cos':cos(c2,p2),
            'p2_norm':float(np.linalg.norm(p2)),'c2_norm':float(np.linalg.norm(c2)),
            'nested_rel':float(np.linalg.norm(c2-c1)/(np.linalg.norm(c2)+EPS)),'first_layer_relerr':first_err,
        }); probe_vectors.append(p2)
    # Common two-basis probe.
    ca=first_blocks(weights[0],haar(COMMON_SEED),[0,1],chirps)
    h=ca; buf=np.empty_like(h)
    for li in range(1,TARGET+1): np.matmul(h,weights[li],out=buf); np.maximum(buf,0,out=buf); h,buf=buf,h
    cm=h.astype(np.float64).mean(0); dc=(2/129.0)*(orig2-cm); pcommon=correction_from_mu(m-dc,M,m,L,R,B)
    P2=np.asarray(probe_vectors); consensus=P2.mean(0)
    for r,p2 in zip(rows,P2):
        r['p2_common_cos']=cos(p2,pcommon); r['p2_consensus_cos']=cos(p2,consensus)
    return {
      'base_id':base_id,'variant':variant,'case_id':f'{base_id}:{variant}','truth_noise_mse':truth_noise,
      'baseline_mse':base_mse,'baseline_mse_nc':base_mse_nc,'base_output':basefull.tolist(),'truth':truth.tolist(),
      'reduced_base_mse':float(np.mean((redbase-truth)**2)),'y0_mse':float(np.mean((y0-truth)**2)),
      'y0':y0.tolist(),'pcommon':pcommon.tolist(),'orientations':rows,'seconds':time.perf_counter()-t0,
      'peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }

def derive_seeds(n):
    out=[]; i=0
    while len(out)<n:
        h=int(hashlib.sha256(f'{PROTOCOL}:base:{i}'.encode()).hexdigest()[:8],16)
        if h>100000: out.append(h); i+=1
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--asset',type=Path,required=True); ap.add_argument('--outdir',type=Path,required=True)
    ap.add_argument('--bases',type=int,default=12); ap.add_argument('--variants',type=int,default=3); ap.add_argument('--orientations',type=int,default=16)
    ap.add_argument('--ref-n',type=int,default=32768); ap.add_argument('--ref-rotations',type=int,default=16); ap.add_argument('--resume',action='store_true')
    a=ap.parse_args(); a.outdir.mkdir(parents=True,exist_ok=True)
    z=np.load(a.asset); chirps=z['chirps'].astype(np.float32); main_rot=z['rotation'].astype(np.float32)
    orientations=[haar(s) for s in ORIENTATION_SEEDS[:a.orientations]]
    seeds=derive_seeds(a.bases)
    manifest={'protocol':PROTOCOL,'base_ids':seeds,'variants':a.variants,'orientation_seeds':ORIENTATION_SEEDS[:a.orientations],
              'common_seed':COMMON_SEED,'reference_rotation_seeds':REFERENCE_ROTATION_SEEDS[:a.ref_rotations],
              'reference_complete_designs_per_group':a.ref_rotations//2,'target':TARGET,'amplitude':AMP,'ridge':RIDGE,
              'split_by_base':{'validation':seeds}}
    (a.outdir/'freeze_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for bi,base_id in enumerate(seeds):
        truth_path=a.outdir/f'truth_{base_id}.npz'
        weights=make_weights(730000+base_id)
        if truth_path.exists() and a.resume:
            zz=np.load(truth_path); truth=zz['truth']; t1=zz['t1']; t2=zz['t2']
        else:
            truth,t1,t2=complete_rotation_reference(weights,chirps,REFERENCE_ROTATION_SEEDS[:a.ref_rotations])
            np.savez_compressed(truth_path,truth=truth,t1=t1,t2=t2)
        noise=.25*float(np.mean((t1-t2)**2))
        for v in range(a.variants):
            p=a.outdir/f'case_{base_id}_{v}.json'
            if p.exists() and a.resume: continue
            rec=case_run(base_id,v,truth,noise,chirps,main_rot,orientations,a.ref_n)
            p.write_text(json.dumps(rec)+'\n')
            print(json.dumps({'done':f'{bi+1}/{len(seeds)}','base':base_id,'variant':v,'seconds':rec['seconds'],
                              'fixed0':rec['orientations'][0]['ratio_nc'],'oracle':min(x['ratio_nc'] for x in rec['orientations'])}),flush=True)
if __name__=='__main__': main()
