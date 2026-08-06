#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy.linalg import hadamard
from scipy.special import roots_jacobi

D=256; B=129; N=2*B*D; L=10
LAM_C=np.array([64293.0127076907740,0.0199556322060796393,0.0251746480145940471,0.796825266924534112,0.0219825945011625946])
LAM_D=np.array([62584.6253844085647,2.9882e-14,4.2248036872e-10,9.1827509366e-6,2.7655590677e-12])
ROOT=Path(__file__).resolve().parents[1]
asset=np.load(ROOT/'assets'/'kerdock_mub5_seed3.npz')
chirps=asset['chirps'].astype(np.float64)
H=hadamard(D,dtype=np.int8).astype(np.float64)/16
bases=np.empty((B,D,D),dtype=np.float64); bases[0]=np.eye(D)
for u in range(128): bases[u+1]=H*chirps[u][None,:]
R=bases.reshape(-1,D)

def kappa(t):
    t=np.clip(t,-1,1)
    return (np.sqrt(np.maximum(0,1-t*t))+t*(np.pi-np.arccos(t)))/np.pi
def Cfun(t):
    y=np.asarray(t,dtype=np.float64)
    for _ in range(31): y=kappa(y)
    return y
def hdim(l):
    return math.comb(D+l-1,l)-(math.comb(D+l-3,l-2) if l>=2 else 0)
def gseq(t):
    out=[np.ones_like(t),t.copy()]
    for n in range(1,L): out.append(((2*n+D-2)/(n+D-2))*t*out[n]-(n/(n+D-2))*out[n-1])
    return out
qx,qw=roots_jacobi(800,(D-3)/2,(D-3)/2); qw/=qw.sum(); gg=gseq(qx); cv=Cfun(qx)
coeff=np.array([hdim(l)*np.sum(qw*cv*gg[l]) for l in range(L+1)])
def Dfun(t):
    gs=gseq(np.asarray(t,dtype=np.float64)); out=np.zeros_like(gs[0])
    for l in range(L+1): out+=(coeff[l]**2/hdim(l))*gs[l]
    return out
SQ2=np.sqrt(2.0)
def split(v): return (v[:,:,0,...]+v[:,:,1,...])/SQ2,(v[:,:,0,...]-v[:,:,1,...])/SQ2
def merge(e,o): return np.stack([(e+o)/SQ2,(e-o)/SQ2],axis=2)
def apply_sectors(v,coef):
    e,o=split(v); bm=e.mean(axis=1,keepdims=True); overall=bm.mean(axis=0,keepdims=True)
    eg=np.broadcast_to(overall,e.shape); eb=np.broadcast_to(bm-overall,e.shape); ew=e-bm
    of=o.reshape(B*D,*o.shape[2:]); tmp=np.tensordot(R.T,of,axes=(1,0)); orow=np.tensordot(R,tmp,axes=(1,0))/B
    oo=(coef[3]*orow+coef[4]*(of-orow)).reshape(o.shape)
    return merge(coef[0]*eg+coef[1]*eb+coef[2]*ew,oo)
def kernel_vectors(X,fun):
    t=R@X.T
    return np.stack([fun(t),fun(-t)],axis=1).reshape(B,D,2,X.shape[0])
def assoc(vals):
    f1,fm1,f0,fa,fma=vals; fe1=(f1+fm1)/2; fo1=(f1-fm1)/2; fea=(fa+fma)/2; foa=(fa-fma)/2
    return np.array([2*(fe1+(D-1)*f0+(B-1)*D*fea),2*(fe1+(D-1)*f0-D*fea),2*(fe1-f0),2*(fo1+2048*foa),2*(fo1-16*foa)])
pts=np.array([1.,-1.,0.,1/16,-1/16]); gp=gseq(pts); lamE=np.zeros(5)
for l in range(L+1): lamE+=(coeff[l]**3/hdim(l)**2)*assoc(gp[l])
lamH=lamE-LAM_D**2/LAM_C
score_norm=lamH[0]/LAM_C[0]; ratio0=LAM_D[0]/LAM_C[0]
def orth(a,b):
    a=a/np.linalg.norm(a); b=b-a*np.dot(a,b); b=b/np.linalg.norm(b); return np.stack([a,b])
def capture(U,m=64):
    th=np.arange(m)*2*np.pi/m; X=np.cos(th)[:,None]*U[0]+np.sin(th)[:,None]*U[1]
    k=kernel_vectors(X,Cfun); dv=kernel_vectors(X,Dfun); flat=k.reshape(-1,m)
    kinv=apply_sectors(k,1/LAM_C).reshape(-1,m)
    S=Cfun(np.clip(X@X.T,-1,1))-flat.T@kinv; S=(S+S.T)/2
    s=(dv.reshape(-1,m).sum(axis=0)-ratio0*flat.sum(axis=0))/np.sqrt(N*LAM_C[0])
    ev,Q=np.linalg.eigh(S); a=Q.T@s
    # Numerical cancellation affects four already-observed modes at ~2e-8 or below.
    floor=1e-9
    value=float(np.sum(a*a/np.maximum(ev,floor)))
    return {'rays':m,'capture':value/score_norm,'eigenvalue_min':float(ev.min()),'eigenvalue_max':float(ev.max()),'score_norm':score_norm,'regularization_floor':floor}
res={
 'title':'Two-plane capture of the global posterior-score direction',
 'same_basis':capture(orth(bases[0,0],bases[0,1])),
 'cross_basis':capture(orth(bases[1,0],bases[2,0])),
 'scope':'Limiting depth-31 preactivation Gaussian model; exact point observations; diagnostic, not CLAF physical-span theorem.',
}
res['passed']=bool(res['same_basis']['capture']<1e-3 and res['cross_basis']['capture']<1e-3)
(ROOT/'results'/'plane_capture_diagnostic.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps(res,indent=2))
if not res['passed']: raise SystemExit(1)
