from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy.linalg import hadamard
from scipy.special import roots_jacobi

ROOT=Path(__file__).resolve().parents[1]
ASSET=ROOT/'sources'/'kerdock_mub5_seed3.npz'

def fwht_int(x: np.ndarray) -> np.ndarray:
    y=np.asarray(x,dtype=np.int64).copy()
    h=1
    while h<len(y):
        z=y.reshape(-1,2*h)
        a=z[:,:h].copy(); b=z[:,h:].copy()
        z[:,:h]=a+b; z[:,h:]=a-b
        h*=2
    return y

def relu_kernel(t):
    t=np.clip(np.asarray(t,dtype=np.float64),-1.0,1.0)
    return (np.sqrt(np.maximum(1-t*t,0.0))+t*(np.pi-np.arccos(t)))/np.pi

def deep_kernel(t,depth=32):
    v=np.asarray(t,dtype=np.float64)
    for _ in range(depth): v=relu_kernel(v)
    return v

def spherical_mean(depth=32,d=256,nq=1024):
    a=(d-3)/2
    x,w=roots_jacobi(nq,a,a)
    return float(np.dot(w,deep_kernel(x,depth))/w.sum())

with np.load(ASSET,allow_pickle=False) as z:
    chirps=np.asarray(z['chirps'])
    rotation=np.asarray(z['rotation'],dtype=np.float64)
D=rotation.shape[0]
assert chirps.shape==(128,D)
unique=np.unique(chirps)
assert set(unique.tolist())<=set([-1.0,1.0])
# Exhaustive exact MUB check via Walsh spectra of pairwise chirp products.
violations=[]; max_abs_dev=0; checked_pairs=0
for i in range(len(chirps)):
    ci=chirps[i].astype(np.int64)
    for j in range(i+1,len(chirps)):
        spec=fwht_int(ci*chirps[j].astype(np.int64))
        dev=int(np.max(np.abs(np.abs(spec)-int(math.isqrt(D)))))
        max_abs_dev=max(max_abs_dev,dev); checked_pairs+=1
        if dev: violations.append((i,j,dev,int(np.min(np.abs(spec))),int(np.max(np.abs(spec)))))
# Rotation orthogonality and representative numerical reconstruction.
rot_err=float(np.max(np.abs(rotation@rotation.T-np.eye(D))))
H=hadamard(D,dtype=np.float64)/math.sqrt(D)
bases=np.empty((129,D,D),dtype=np.float64)
for i,c in enumerate(chirps): bases[i]=(H*c[None,:])@rotation
bases[-1]=rotation
orth_err=float(np.max(np.abs(bases@np.swapaxes(bases,-1,-2)-np.eye(D))))
rng=np.random.default_rng(20260730)
pair_ix=rng.choice(128*127//2,size=256,replace=False)
pairs=[]
k=0
for i in range(128):
    for j in range(i+1,128):
        if k in set(pair_ix.tolist()): pairs.append((i,j))
        k+=1
cross_dev=0.0
for i,j in pairs:
    cross_dev=max(cross_dev,float(np.max(np.abs(np.abs(bases[i]@bases[j].T)-1/16))))
std_dev=float(np.max(np.abs(np.abs(bases[:128]@bases[-1].T)-1/16)))
# Kernel numbers and global scale.
A0=spherical_mean()
vals={t:float(deep_kernel(np.array([t]))[0]) for t in [1.0,-1.0,0.0,1/16,-1/16]}
N=66048
rowavg=(vals[1.0]+vals[-1.0]+510*vals[0.0]+32768*vals[1/16]+32768*vals[-1/16])/N
R=rowavg-A0
alpha=A0/rowavg
risk_scaled=A0*R/rowavg
# Rank-one ReLU-of-GP nonlinear counterexample.
d=16
rng=np.random.default_rng(314159)
trials=10000
a=rng.standard_normal((trials,d))
obs_pos=np.maximum(a,0); obs_neg=np.maximum(-a,0)
abscoords=obs_pos+obs_neg
# Spherical mean of ReLU(a.u) = c_d ||a||; linear equal-weight rule proportional to L1.
# c_d = E[U_1^+] for U uniform sphere.
c_d=math.gamma(d/2)/(2*math.sqrt(math.pi)*math.gamma((d+1)/2))
theta=c_d*np.linalg.norm(a,axis=1)
linear=(obs_pos.sum(1)+obs_neg.sum(1))/(2*d) # sphere sample mean over +-basis
nonlinear=c_d*np.sqrt((abscoords**2).sum(1))
res={
 'test0_architecture_assumed_from_source': {'D':256,'depth':32,'bias_free':True,'weight_sd':math.sqrt(2/256)},
 'test1a': {
  'chirp_shape':list(chirps.shape),'chirp_values':unique.tolist(),'all_chirp_pairs_checked':checked_pairs,
  'integer_walsh_mub_violations':violations[:10],'max_integer_abs_deviation_from_16':max_abs_dev,
  'rotation_orthogonality_max_abs':rot_err,'all_basis_orthogonality_max_abs':orth_err,
  'sampled_cross_basis_abs_inner_product_max_deviation':cross_dev,
  'standard_vs_chirp_abs_inner_product_max_deviation':std_dev,
  'certified_row_profile':{'self':1,'antipode':1,'orthogonal':510,'cross_abs_1_over_16':65536}
 },
 'test1b_infinite_width': {
  'A0_quadrature':A0,'row_average_kernel':rowavg,'baseline_linear_risk':R,
  'alpha_star_unconstrained_linear':alpha,'alpha_minus_one':alpha-1,
  'scaled_linear_risk':risk_scaled,'relative_risk_reduction':(R-risk_scaled)/R,
  'absolute_risk_reduction':R-risk_scaled,'kernel_values':{str(k):v for k,v in vals.items()}
 },
 'nonlinear_counterexample': {
  'dimension':d,'trials':trials,'c_d':c_d,
  'nonlinear_exact_max_abs_error':float(np.max(np.abs(nonlinear-theta))),
  'linear_mse':float(np.mean((linear-theta)**2)),
  'nonlinear_mse':float(np.mean((nonlinear-theta)**2)),
  'linear_relative_bias':float(np.mean(linear-theta)/np.mean(theta)),
  'statement':'For f_a(u)=ReLU(a^T u), antipodal orthonormal-basis observations recover |a_i|, and nonlinear L2 postprocessing recovers the exact spherical integral. The equal-weight linear rule uses L1 instead and is not Bayes-exact.'
 }
}
(ROOT/'results'/'MATH_DESIGN_AUDIT.json').write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
