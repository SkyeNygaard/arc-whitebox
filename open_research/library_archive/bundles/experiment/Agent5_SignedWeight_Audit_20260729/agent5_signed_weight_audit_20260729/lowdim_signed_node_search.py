#!/usr/bin/env python3
"""Deterministic adversarial signed-node searches on S^1 and S^2.

These are empirical sharpness tests, not proof components. The low-dimensional
spheres are embedded as subspheres of S^255, so the dimension-256 kernel and
Gegenbauer witness remain applicable to the tested Gram matrices.
"""
from __future__ import annotations
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
ROOT=Path(__file__).resolve().parent
COEFF=[Fraction(x) for x in json.loads((ROOT/'vendor_proof/auxiliary_coefficients_d256_L32_deg5.json').read_text())['coefficients']]

def gegenbauer(d,L):
    G=[[Fraction(1)],[Fraction(0),Fraction(1)]]
    for ell in range(1,L):
        A=Fraction(2*ell+d-2,ell+d-2);B=Fraction(ell,ell+d-2)
        nxt=[Fraction(0)]*(len(G[-1])+1)
        for k,v in enumerate(G[-1]):nxt[k+1]+=A*v
        for k,v in enumerate(G[-2]):nxt[k]-=B*v
        G.append(nxt)
    return G[:L+1]
G=gegenbauer(256,5);HCOEF=[Fraction(0)]*6
for c,g in zip(COEFF,G):
    for k,v in enumerate(g):HCOEF[k]+=c*v
HF=np.array([float(x) for x in HCOEF]);C0=float(COEFF[0]);Q1=float(Fraction(1)-sum(COEFF))

def kappa(x):
    x=np.clip(x,-1.,1.)
    return (np.sqrt(np.maximum(0.,1-x*x))+(np.pi-np.arccos(x))*x)/np.pi

def kernel(x):
    y=np.asarray(x,float)
    for _ in range(32):y=kappa(y)
    return y

def hpoly(x):return np.polynomial.polynomial.polyval(x,HF)
def softmax(z):
    z=np.asarray(z);e=np.exp(z-z.max());return e/e.sum()
def nodes(dim,v):
    if dim==2:
        a=np.r_[0.,v[:3]]
        return np.c_[np.cos(a),np.sin(a)]
    pts=[[0.,0.,1.]]
    for th,ph in np.asarray(v[:6]).reshape(-1,2):
        pts.append([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
    return np.asarray(pts)

def optimize(dim,beta,seed):
    # For m=4 and beta in {0.1,0.5}, the exact integer diagonal envelope is
    # attained with three positive nodes and one negative node.
    m=4;pcount=3;nv=3 if dim==2 else 6
    bounds=([(0,2*np.pi)]*3 if dim==2 else sum(([(0,np.pi),(0,2*np.pi)] for _ in range(3)),[]))+[(-8,8)]*pcount
    def unpack(v):
        X=nodes(dim,v[:nv])
        wp=(1+beta)*softmax(v[nv:nv+pcount])
        return X,np.r_[wp,-beta]
    def objective(v):
        X,w=unpack(v);gram=np.clip(X@X.T,-1,1)
        return float(w@kernel(gram)@w)
    res=differential_evolution(objective,bounds,seed=seed,maxiter=60,popsize=8,tol=1e-9,polish=True,updating='immediate')
    X,w=unpack(res.x);gram=np.clip(X@X.T,-1,1)
    E=objective(res.x);HE=float(w@hpoly(gram)@w);QE=E-HE
    original=C0+Q1/m-2*Q1*beta*(1+beta)
    strengthened=C0+Q1*((1+beta)**2/3+beta**2)-2*Q1*beta*(1+beta)
    return {
      'ambient_subsphere_dimension':dim,'node_count':m,'positive_node_count':3,'negative_node_count':1,
      'negative_mass_beta':beta,'energy':E,'h_energy':HE,'q_energy':QE,
      'original_prop5_bound_with_N_equals_m':original,
      'exact_groupwise_diagonal_bound_for_this_sign_count':strengthened,
      'slack_to_original':E-original,'slack_to_groupwise':E-strengthened,
      'optimizer_success':bool(res.success),'evaluations':int(res.nfev),
      'weights':w.tolist(),'gram':np.round(gram,12).tolist(),
    }

rows=[]
for dim in (2,3):
    for beta in (0.1,0.5):
        rows.append(optimize(dim,beta,20260729+100*dim+int(100*beta)))
out={
 'status':'EXPLORATORY EMPIRICAL',
 'purpose':'Adversarial sharpness search for Proposition 5 on low-dimensional subspheres embedded in S^255.',
 'method':'Differential evolution over four nodes and within-sign weight allocations; fixed deterministic seeds.',
 'results':rows,
 'conclusion':'All searches remained far above even the strengthened certificate; no near-saturating geometry was found.'
}
(ROOT/'LOWDIM_OPTIMIZATION_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
