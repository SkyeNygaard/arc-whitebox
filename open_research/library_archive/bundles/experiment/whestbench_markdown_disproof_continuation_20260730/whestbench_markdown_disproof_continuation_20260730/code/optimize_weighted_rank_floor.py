#!/usr/bin/env python3
import sys, json, math, time
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution, minimize
ROOT=Path('/mnt/data/continued_review_inputs/oracle_v18/WHestBench_Oracle_Proof_Completion_v18_20260730')
sys.path.insert(0,str(ROOT/'code'))
from verify_oracle_proof_completion import harmonic_dimension, normalized_gegenbauer, projection_of_monomial, kernel_maclaurin_jet, interval_endpoints, mp, ORDER, NODES, T
import sympy as sp

L=int(sys.argv[1]) if len(sys.argv)>1 else 18
print('precompute L',L,flush=True)
dims=np.array([harmonic_dimension(l) for l in range(L+1)],dtype=object)
# polynomial coefficient q[l,p] for d_l G_l(t), increasing power
q=np.zeros((L+1,L+1),dtype=np.float64)
for l in range(L+1):
    poly=sp.Poly(sp.Integer(int(dims[l]))*normalized_gegenbauer(l),T)
    for (p,),coef in poly.terms(): q[l,p]=float(coef)
# projection P[p,r]
P=np.zeros((2*L+1,2*L+1),dtype=np.float64)
for p in range(2*L+1):
    for r in range(p+1):
        v=projection_of_monomial(p,r)
        P[p,r]=float(v)
print('projection done',flush=True)
# rigorous k lower from interval jet (float for discovery)
jet=kernel_maclaurin_jet(); kl=[]
for r in range(2*L+1):
    lo=mp.mpf('0')
    for p in range(r,ORDER+1):
        pr=projection_of_monomial(p,r)
        if pr:
            a,_=interval_endpoints(jet[p]); lo += a*mp.mpf(int(pr.p))/int(pr.q)
    kl.append(float(lo))
kl=np.array(kl)
print('k done',flush=True)

def bcoeff(w):
    poly=w@q
    sq=np.convolve(poly,poly)
    return sq@P

def rankdef(w):
    # eigenvalues w_l repeated dims_l; select N largest; formula exact analogue
    order=np.argsort(-w)
    rem=NODES; tsum=0.; tsq=0.
    for l in order:
        take=min(rem,int(dims[l])); rem-=take
        tail=int(dims[l])-take
        tsum += tail*w[l]
        tsq += tail*w[l]*w[l]
    if rem: return 0.
    return tsq+tsum*tsum/NODES

def evaluate_log(x):
    # normalize max log to 0 for scale invariance
    lx=np.r_[x[:3],0.0,x[3:]] # fix degree3 weight=1
    lx=lx-np.max(lx[:4]) # avoid excessive scale; scale invariant
    w=np.exp(lx)
    b=bcoeff(w)
    if np.any(b[1:]<=0) or not np.all(np.isfinite(b)): return -1e300,w,b,None
    ratios=kl[1:]/b[1:]
    gamma=np.min(ratios); bind=1+int(np.argmin(ratios)); rd=rankdef(w)
    return gamma*rd,w,b,bind

def obj(x):
    val,*_=evaluate_log(x)
    return -math.log(max(val,1e-300))

# existing start, extended geometrically
old=np.array([1,1,1,1,0.007971493217727095,9.638797005852535e-5,1.6379172467841997e-6,3.209923511000207e-8,6.658135190281046e-10,2.8585879019099307e-11,7.795660364439458e-13,2.5373546946236714e-14,9.28902436342902e-16,5.496195450219314e-17,2.6771772862001115e-18,1.382308855215168e-19],float)
if L+1>len(old):
    ratio=old[-1]/old[-2]
    old=np.r_[old,[old[-1]*ratio**i for i in range(1,L+2-len(old))]]
else: old=old[:L+1]
x0=np.log(old); x0=np.r_[x0[:3],x0[4:]]
val0,w0,b0,bind0=evaluate_log(x0); print('start',val0,bind0,w0,flush=True)
# several local methods and random perturb starts
best=(val0,x0,w0,bind0)
rng=np.random.default_rng(20260730)
starts=[x0]
for scale in [0.15,0.4,0.8,1.5]:
    for _ in range(5): starts.append(x0+rng.normal(0,scale,size=x0.shape))
# bounds: low degrees moderately variable; high log weights [-60,2]
bounds=[(-5,2)]*3+[(-60,0)]*(L-3)
for i,s in enumerate(starts):
    s=np.clip(s,[b[0] for b in bounds],[b[1] for b in bounds])
    res=minimize(obj,s,method='Nelder-Mead',options={'maxiter':2500,'xatol':1e-8,'fatol':1e-10})
    val,w,b,bind=evaluate_log(res.x)
    if val>best[0]:
        best=(val,res.x,w,bind); print('NEW',i,val,bind,w,flush=True)
# polish Powell bounded
res=minimize(obj,best[1],method='Powell',bounds=bounds,options={'maxiter':3000,'xtol':1e-10,'ftol':1e-12})
val,w,b,bind=evaluate_log(res.x)
if val>best[0]: best=(val,res.x,w,bind)
val,x,w,bind=best
rat=kl[1:]/bcoeff(w)[1:]
result={'L':L,'discovery_floor':val,'binding_degree':bind,'weights':[format(v,'.18g') for v in w],
        'rank_defect':rankdef(w),'gamma':float(np.min(rat)),
        'ratios':{str(i+1):float(v) for i,v in enumerate(rat)},
        'fraction_of_kerdock':val/2.433660357543006e-7,'improvement_cap':2.433660357543006e-7/val,
        'start_floor':val0,'improvement_over_t47':val/1.2294295437956858e-7}
out=Path('/mnt/data/continued_review_outputs')/f'WEIGHTED_RANK_DISCOVERY_L{L}.json'; out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
