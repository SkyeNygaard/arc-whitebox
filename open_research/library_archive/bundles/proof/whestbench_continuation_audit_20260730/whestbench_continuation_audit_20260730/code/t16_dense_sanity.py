from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import mpmath as mp
from scipy.special import eval_gegenbauer
CERT=Path('/mnt/data/whestbench_continuation_20260730/closure/whestbench_closure_20260730/T16_PRIMAL_DUAL_CERTIFICATE.json')
d=json.load(open(CERT)); cc=d['hermite_coefficient_certificate']['coefficient_intervals']
co=np.array([(float(a)+float(b))/2 for a,b in cc])
roots=np.sort(np.roots([22102,21930,-87,-85]).real)
lam=127.0
norm=np.array([eval_gegenbauer(l,lam,1.0) for l in range(6)])
def kappa(x):
    x=np.clip(x,-1,1)
    return (np.sqrt(np.maximum(0,1-x*x))+(np.pi-np.arccos(x))*x)/np.pi
def K(x):
    y=np.asarray(x,dtype=float)
    for _ in range(32): y=kappa(y)
    return y
def h(x):
    x=np.asarray(x,dtype=float)
    return sum(co[l]*eval_gegenbauer(l,lam,x)/norm[l] for l in range(6))
# Dense grid plus root neighborhoods.
x=np.linspace(-1,1,400001)
g=K(x)-h(x)
mask=np.ones(len(x),bool)
for r in roots: mask &= np.abs(x-r)>2e-5
# Contact evaluation and derivative sanity.
contact_gap=(K(roots)-h(roots)).tolist()
# Finite derivative at contacts.
eps=1e-6
contact_dgap=((K(roots+eps)-K(roots-eps))/(2*eps)-(h(roots+eps)-h(roots-eps))/(2*eps)).tolist()
# high precision sixth derivative at a set of points.
mp.mp.dps=60
def kapm(t): return (mp.sqrt(1-t*t)+(mp.pi-mp.acos(t))*t)/mp.pi
def Km(t):
    y=t
    for _ in range(32): y=kapm(y)
    return y
pts=[mp.mpf('-0.999'),mp.mpf('-0.95'),mp.mpf('-0.8'),mp.mpf('-0.5'),mp.mpf('-0.35'),mp.mpf('-0.1'),mp.mpf('0'),mp.mpf('0.1'),mp.mpf('0.5'),mp.mpf('0.9'),mp.mpf('0.999')]
d6=[mp.diff(Km,p,6) for p in pts]
out={
 'roots':roots.tolist(),'coefficients_mid':co.tolist(),
 'dense_grid_points':len(x),'global_min_gap_double':float(g.min()),'global_min_location':float(x[g.argmin()]),
 'min_gap_away_2e-5_from_contacts':float(g[mask].min()),
 'endpoint_gaps':[float(g[0]),float(g[-1])],
 'contact_gaps_double':contact_gap,'contact_derivative_gaps_fd':contact_dgap,
 'sixth_derivative_points':[str(p) for p in pts], 'sixth_derivative_values':[mp.nstr(v,30) for v in d6],
 'all_sampled_sixth_derivatives_positive':all(v>0 for v in d6)
}
P=Path('/mnt/data/whestbench_continuation_20260730/local_verification/t16_dense_sanity.json');P.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
