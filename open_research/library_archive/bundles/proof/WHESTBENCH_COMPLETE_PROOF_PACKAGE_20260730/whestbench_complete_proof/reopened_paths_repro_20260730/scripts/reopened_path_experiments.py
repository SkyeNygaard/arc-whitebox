from __future__ import annotations
import json, math, os, sys, time, hashlib
from pathlib import Path
import numpy as np
from scipy.special import ndtr, roots_jacobi

ROOT=Path('/mnt/data/whest_reopened')
A9=ROOT/'agent9_10_oracle_bundle/agent9_10_oracle_bundle'
sys.path.insert(0,str(A9))
import arc_experiments as ae
from agent9_10_oracle_screen import gradient_directions

OUT=ROOT/'reopened_path_results'
OUT.mkdir(parents=True,exist_ok=True)
SEEDS=list(range(3000,3008))
TRAIN=set(range(3000,3004))
TEST=set(range(3004,3008))
REF_ROTS=(20,21,22,23)
BASE_ROT=3
D=ae.D
N=ae.N
NB=129
RB=512
LAMBDAS=(1e-4,1e-3,1e-2,1e-1,1.0,10.0)

# Correct basis ids for the asset ordering: positive chirps, negative chirps, coordinate +/-.
BASIS_IDS=np.concatenate([
    np.repeat(np.arange(128),256),
    np.repeat(np.arange(128),256),
    np.full(512,128),
]).astype(np.int16)
# Deterministic grouped folds.
_basis_perm=np.random.default_rng(2026073001).permutation(NB)
_basis_fold=np.empty(NB,dtype=np.int8)
for fi,g in enumerate(np.array_split(_basis_perm,6)):_basis_fold[g]=fi
FOLD_IDS=_basis_fold[BASIS_IDS]


def make_kerdock_directions(rot_seed:int)->np.ndarray:
    R=ae.rotation(rot_seed)
    weighted=ae.CHIRPS[:,:,None]*R[None,:,:]
    pre=ae.fwht_axis1_inplace(weighted.copy())
    pre*=np.float32(1/math.sqrt(D))
    flat=pre.reshape(128*D,D)
    U=np.empty((N,D),dtype=np.float32)
    U[:128*D]=flat
    U[128*D:2*128*D]=-flat
    U[2*128*D:2*128*D+D]=R
    U[2*128*D+D:]=-R
    return U


def final_rows(weights:list[np.ndarray],rot_seed:int)->np.ndarray:
    A=ae.first_activation(weights[0],rot_seed)
    B=np.empty_like(A)
    for W in weights[1:]:
        np.matmul(A,W,out=B);np.maximum(B,0,out=B);A,B=B,A
    return A


def forward_points(X:np.ndarray,weights:list[np.ndarray])->np.ndarray:
    A=np.asarray(X,dtype=np.float32)
    for W in weights:
        A=A@W;np.maximum(A,0,out=A)
    return A


def refs(weights):
    vals=[ae.final_for_rotation(weights,r) for r in REF_ROTS]
    a=0.5*(vals[0]+vals[1]);b=0.5*(vals[2]+vals[3]);m=0.5*(a+b)
    return m,a,b


def orthonormal_network_directions(weights,seed,max_dirs=6):
    raw=[]
    for name,v in gradient_directions(weights,seed+123,n=768):
        if name!='random':raw.append((name,np.asarray(v,float)))
    # Weight-only input singular directions.
    u,_,_=np.linalg.svd(weights[0].astype(np.float64),full_matrices=False)
    raw.extend([(f'w0_sv{i+1}',u[:,i]) for i in range(4)])
    out=[]
    for name,v in raw:
        q=v.copy()
        for _,z in out:q-=z*np.dot(z,q)
        n=np.linalg.norm(q)
        if n>1e-8:out.append((name,q/n))
        if len(out)>=max_dirs:break
    return out


def centered_stats(H,F,mask):
    X=H[mask].astype(np.float64,copy=False);Y=F[mask].astype(np.float64,copy=False)
    n=len(X);sx=X.sum(0);sy=Y.sum(0);xx=X.T@X;xy=X.T@Y
    return n,sx,sy,xx,xy


def crossfit_control_predictions(H,F,exact_mean,lambdas=LAMBDAS):
    H=np.asarray(H,np.float64);F=np.asarray(F,np.float64);e=np.asarray(exact_mean,np.float64)
    K=H.shape[1];nall=len(H);base=F.mean(0)
    n0,sx0,sy0,xx0,xy0=centered_stats(H,F,np.ones(nall,dtype=bool))
    fold_stats=[]
    for fi in range(6):fold_stats.append(centered_stats(H,F,FOLD_IDS==fi))
    out={lam:np.zeros(D,float) for lam in lambdas}
    for fi,(nh,sxh,syh,xxh,xyh) in enumerate(fold_stats):
        nt=n0-nh;sx=sx0-sxh;sy=sy0-syh;xx=xx0-xxh;xy=xy0-xyh
        hm=sx/nt;fm=sy/nt
        G=xx-nt*np.outer(hm,hm);C=xy-nt*np.outer(hm,fm)
        scale=max(np.trace(G)/max(K,1),1e-12)
        hh=sxh/nh;fh=syh/nh
        for lam in lambdas:
            B=np.linalg.solve(G+lam*scale*np.eye(K),C)
            out[lam]+=(nh/nall)*(fh+(e-hh)@B)
    return base,out


def poisson_features(U,dirs):
    T=U@np.stack([v for _,v in dirs],axis=1).astype(np.float32)
    rs=(0.03,0.06,0.10,0.15,0.20)
    cols=[];names=[]
    for j,(name,_) in enumerate(dirs):
        t=T[:,j].astype(np.float64)
        for r in rs:
            lp=math.log1p(-r*r)-(D/2)*np.log(1-2*r*t+r*r)
            lm=math.log1p(-r*r)-(D/2)*np.log(1+2*r*t+r*r)
            g=0.5*(np.exp(np.clip(lp,-80,80))+np.exp(np.clip(lm,-80,80)))
            cols.append(g);names.append(f'{name}_r{r:.2f}')
    G=np.stack(cols,axis=1)
    return G,np.ones(G.shape[1]),names


_JAC_NODES,_JAC_WEIGHTS=roots_jacobi(384,(D-3)/2,(D-3)/2)
_JAC_WEIGHTS=np.asarray(_JAC_WEIGHTS,dtype=np.float64)
_JAC_WEIGHTS/=_JAC_WEIGHTS.sum()

def angular_relu_exact_mean(b):
    """Exact angular mean for the radially reduced, biased ReLU control."""
    b=np.asarray(b,dtype=np.float64)
    z=ae.RADIUS*_JAC_NODES[:,None]+b[None,:]
    return _JAC_WEIGHTS@np.maximum(z,0.0)


def nonlinear_features(X,dirs,k,m,bias_scale,seed):
    P=np.stack([v for _,v in dirs[:k]],axis=1)
    Z=X@P
    rng=np.random.default_rng(seed+1000*k+10*m+int(100*bias_scale))
    A=rng.standard_normal((m,k));A/=np.linalg.norm(A,axis=1,keepdims=True)
    # Ensure axes and pairwise combinations are represented.
    q=0
    for j in range(min(k,m)):
        A[q]=0;A[q,j]=1;q+=1
        if q<m:A[q]=0;A[q,j]=-1;q+=1
    if bias_scale==0:
        b=np.zeros(m)
    else:
        # Symmetric deterministic spread, avoiding all features sharing one radial mismatch.
        b=bias_scale*np.linspace(-2.0,2.0,m)
        rng.shuffle(b)
    H=np.maximum(Z@A.T+b[None,:],0)
    EH=angular_relu_exact_mean(b) # exact fixed-radius angular mean after radial reduction
    return H,EH,dict(k=k,m=m,bias_scale=bias_scale,A=A.tolist(),b=b.tolist(),dir_names=[x[0] for x in dirs[:k]])


def signed_probe_basis(weights,dirs,seed):
    V=np.stack([v for _,v in dirs],axis=0)
    centers=[];tangents=[];labels=[]
    for i in range(len(V)):
        for j in range(len(V)):
            if i==j:continue
            c=V[i]+0.35*V[j];c/=np.linalg.norm(c)
            v=V[j]-c*np.dot(c,V[j]);v/=np.linalg.norm(v)
            centers.append(c);tangents.append(v);labels.append(f'{dirs[i][0]}__{dirs[j][0]}')
            if len(centers)>=16:break
        if len(centers)>=16:break
    deltas=(0.02,0.05,0.10)
    points=[];basis_labels=[]
    for delta in deltas:
        for c,v,lbl in zip(centers,tangents,labels):
            up=c+delta*v;um=c-delta*v;up/=np.linalg.norm(up);um/=np.linalg.norm(um)
            points.extend([ae.RADIUS*up,ae.RADIUS*um]);basis_labels.append(f'{lbl}_d{delta:.2f}')
    # Random control family, same number at delta .05.
    rng=np.random.default_rng(seed+909)
    random_labels=[]
    for j in range(16):
        c=rng.standard_normal(D);c/=np.linalg.norm(c)
        v=rng.standard_normal(D);v-=c*np.dot(c,v);v/=np.linalg.norm(v)
        delta=.05;up=c+delta*v;um=c-delta*v;up/=np.linalg.norm(up);um/=np.linalg.norm(um)
        points.extend([ae.RADIUS*up,ae.RADIUS*um]);random_labels.append(f'random_{j}_d0.05')
    Y=forward_points(np.asarray(points,np.float32),weights).astype(np.float64)
    M=len(basis_labels)
    Dnet=(Y[0:2*M:2]-Y[1:2*M:2]) # raw signed pair differences
    off=2*M
    Drand=(Y[off::2]-Y[off+1::2])
    return Dnet,Drand,basis_labels,random_labels


def raw_mse(pred,ref):return float(np.mean((pred-ref)**2))
def cross_mse(pred,a,b):return float(np.mean((pred-a)*(pred-b)))

def run_network(seed,U):
    t0=time.time();print(json.dumps({'start_seed':seed}),flush=True)
    w=ae.make_weights(seed)
    ref,ra,rb=refs(w)
    F=final_rows(w,BASE_ROT).astype(np.float64)
    base=F.mean(0)
    dirs=orthonormal_network_directions(w,seed,max_dirs=6)
    result={'seed':seed,'dir_names':[x[0] for x in dirs],
            'baseline_raw':raw_mse(base,ref),'baseline_cross':cross_mse(base,ra,rb)}
    arrays={'base':base,'ref':ref,'ref_a':ra,'ref_b':rb}

    # Branch 2: non-polynomial exact-mean Poisson controls.
    G,EG,gnames=poisson_features(U,dirs[:3])
    _,pp_all=crossfit_control_predictions(G,F,EG)
    # nested subsets: each r across directions, all features, low-r only.
    subsets={'all':np.arange(G.shape[1]),'low_r':np.array([i for i,n in enumerate(gnames) if ('r0.03' in n or 'r0.06' in n)]),
             'mid_r':np.array([i for i,n in enumerate(gnames) if ('r0.10' in n or 'r0.15' in n)])}
    pois={}
    for sn,ix in subsets.items():
        _,preds=crossfit_control_predictions(G[:,ix],F,EG[ix])
        pois[sn]={str(l):{'raw':raw_mse(p,ref),'cross':cross_mse(p,ra,rb)} for l,p in preds.items()}
        for l,p in preds.items():arrays[f'poisson_{sn}_{l}']=p
    result['poisson']={'feature_names':gnames,'mean_deviation':(G.mean(0)-1).tolist(),
                       'feature_std':G.std(0).tolist(),'subsets':pois}
    del G,pp_all

    # Branch 3: exactly integrable nonlinear projected ReLU surrogates.
    X=ae.RADIUS*U.astype(np.float64)
    nlcfgs=[(2,32,1.0),(4,64,1.0),(4,64,0.0),(4,64,2.0)]
    nlr={}
    for k,m,bs in nlcfgs:
        H,EH,meta=nonlinear_features(X,dirs,k,m,bs,seed)
        _,preds=crossfit_control_predictions(H,F,EH)
        key=f'k{k}_m{m}_b{bs:g}'
        nlr[key]={'meta':meta,'metrics':{str(l):{'raw':raw_mse(p,ref),'cross':cross_mse(p,ra,rb)} for l,p in preds.items()},
                  'mean_feature_abs_error':float(np.mean(np.abs(H.mean(0)-EH)))}
        for l,p in preds.items():arrays[f'nonlinear_{key}_{l}']=p
        del H
    result['nonlinear_surrogate']=nlr

    # Branch 4: outside-universe signed near-collision probes.
    Dnet,Drand,labels,rlabels=signed_probe_basis(w,dirs,seed)
    arrays['signed_network']=Dnet;arrays['signed_random']=Drand
    result['signed']={'network_labels':labels,'random_labels':rlabels,
                      'network_norms':np.linalg.norm(Dnet,axis=1).tolist(),
                      'random_norms':np.linalg.norm(Drand,axis=1).tolist()}
    arrays['directions']=np.stack([v for _,v in dirs])
    np.savez_compressed(OUT/f'network_{seed}.npz',**arrays)
    result['elapsed']=time.time()-t0
    (OUT/f'network_{seed}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'done_seed':seed,'elapsed':result['elapsed'],'base_raw':result['baseline_raw'],'base_cross':result['baseline_cross']}),flush=True)
    return result


def fit_scalar(train_pairs):
    num=sum(float(np.dot(p,e)) for p,e in train_pairs);den=sum(float(np.dot(p,p)) for p,e in train_pairs)
    return num/max(den,1e-30)


def metric_summary(preds,arrs,seeds):
    raw=[];cross=[];base_raw=[];base_cross=[]
    for s,p in zip(seeds,preds):
        a=arrs[s];raw.append(raw_mse(p,a['ref']));cross.append(cross_mse(p,a['ref_a'],a['ref_b']))
        base_raw.append(raw_mse(a['base'],a['ref']));base_cross.append(cross_mse(a['base'],a['ref_a'],a['ref_b']))
    raw=np.array(raw);cross=np.array(cross);br=np.array(base_raw);bc=np.array(base_cross)
    return {'pooled_raw_ratio':float(raw.sum()/br.sum()),'pooled_cross_ratio':float(cross.sum()/bc.sum()),
            'mean_raw_ratio':float(np.mean(raw/br)),'wins_raw':int(np.sum(raw<br)),'worst_raw_ratio':float(np.max(raw/br)),
            'per_seed_raw_ratio':{str(s):float(x) for s,x in zip(seeds,raw/br)},
            'candidate_raw':raw.tolist(),'candidate_cross':cross.tolist()}


def aggregate():
    arrs={s:dict(np.load(OUT/f'network_{s}.npz')) for s in SEEDS}
    train=sorted(TRAIN);test=sorted(TEST)
    summary={'seeds':SEEDS,'train':train,'test':test}

    # Branches 2/3: select candidate+lambda and one scalar shrink on training only.
    families={}
    keys=[k for k in arrs[SEEDS[0]] if k.startswith('poisson_') or k.startswith('nonlinear_')]
    for key in keys:
        pairs=[(arrs[s][key]-arrs[s]['base'],arrs[s]['ref']-arrs[s]['base']) for s in train]
        alpha=float(np.clip(fit_scalar(pairs),-2,2))
        trpred=[arrs[s]['base']+alpha*(arrs[s][key]-arrs[s]['base']) for s in train]
        tepred=[arrs[s]['base']+alpha*(arrs[s][key]-arrs[s]['base']) for s in test]
        families[key]={'alpha':alpha,'train':metric_summary(trpred,arrs,train),'test':metric_summary(tepred,arrs,test)}
    pois={k:v for k,v in families.items() if k.startswith('poisson_')}
    nonlin={k:v for k,v in families.items() if k.startswith('nonlinear_')}
    summary['poisson_ranked']=dict(sorted(pois.items(),key=lambda kv:kv[1]['train']['pooled_raw_ratio']))
    summary['nonlinear_ranked']=dict(sorted(nonlin.items(),key=lambda kv:kv[1]['train']['pooled_raw_ratio']))

    # Signed probes: global coefficient vectors fit on train; ridge and mass-cap sweep.
    signed={}
    for fam,akey in [('network','signed_network'),('random','signed_random')]:
        M=arrs[train[0]][akey].shape[0]
        G=np.zeros((M,M));b=np.zeros(M)
        for s in train:
            Dm=arrs[s][akey];err=arrs[s]['ref']-arrs[s]['base'];G+=Dm@Dm.T;b+=Dm@err
        scale=max(np.trace(G)/M,1e-30)
        candidates={}
        for lam in LAMBDAS:
            w=np.linalg.solve(G+lam*scale*np.eye(M),b)
            beta=float(np.sum(np.abs(w)))
            for cap in [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1.0,10.0]:
                wc=w*min(1.0,cap/max(beta,1e-30));key=f'lam{lam}_cap{cap}'
                trpred=[arrs[s]['base']+wc@arrs[s][akey] for s in train]
                tepred=[arrs[s]['base']+wc@arrs[s][akey] for s in test]
                candidates[key]={'ridge':lam,'cap':cap,'negative_mass':float(np.sum(np.abs(wc))),
                                 'train':metric_summary(trpred,arrs,train),'test':metric_summary(tepred,arrs,test)}
        # Per-network oracle ceiling and required signed mass.
        oracle={}
        for s in SEEDS:
            Dm=arrs[s][akey];err=arrs[s]['ref']-arrs[s]['base'];GG=Dm@Dm.T;sc=max(np.trace(GG)/M,1e-30)
            w=np.linalg.solve(GG+1e-8*sc*np.eye(M),Dm@err)
            pred=arrs[s]['base']+w@Dm
            oracle[str(s)]={'raw_ratio':raw_mse(pred,arrs[s]['ref'])/raw_mse(arrs[s]['base'],arrs[s]['ref']),
                            'negative_mass':float(np.sum(np.abs(w)))}
        signed[fam]={'ranked':dict(sorted(candidates.items(),key=lambda kv:kv[1]['train']['pooled_raw_ratio'])),'oracle':oracle}
    summary['signed']=signed
    (OUT/'FULL_WIDTH_SUMMARY.json').write_text(json.dumps(summary,indent=2))
    return summary


def main():
    U=make_kerdock_directions(BASE_ROT)
    norms=np.linalg.norm(U,axis=1)
    print(json.dumps({'kerdock_norm_min':float(norms.min()),'max':float(norms.max()),'mean':float(norms.mean())}),flush=True)
    np.save(OUT/'kerdock_directions_seed3.npy',U)
    for s in SEEDS:
        if not (OUT/f'network_{s}.npz').exists():run_network(s,U)
    sm=aggregate()
    print(json.dumps({'summary_path':str(OUT/'FULL_WIDTH_SUMMARY.json'),
                      'best_poisson':next(iter(sm['poisson_ranked'].items())),
                      'best_nonlinear':next(iter(sm['nonlinear_ranked'].items())),
                      'best_signed_network':next(iter(sm['signed']['network']['ranked'].items()))},indent=2),flush=True)

if __name__=='__main__':main()
