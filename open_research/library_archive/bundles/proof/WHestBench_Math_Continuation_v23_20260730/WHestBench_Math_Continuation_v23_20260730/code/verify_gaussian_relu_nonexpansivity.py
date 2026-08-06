#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent

def kappa(r: float)->float:
    r=max(-1.0,min(1.0,r))
    return (math.sqrt(max(0.0,1-r*r))+(math.pi-math.acos(r))*r)/math.pi

# Dense deterministic grid checks the analytic sign identity numerically.
grid=np.linspace(-1.0,1.0,20001)
gaps=np.array([kappa(float(r))-float(r) for r in grid])
assert gaps.min() >= -2e-15

# Monte Carlo checks the exact one-layer identity using the exact bivariate
# Gaussian law of a single row, avoiding large matrix allocation.
rng=np.random.default_rng(20260730)
m=256
checks=[]
for case in range(8):
    u=rng.normal(size=m);v=rng.normal(size=m)
    if case==0: v=1.7*u
    if case==1: v=-0.8*u+0.02*rng.normal(size=m)
    nu=float(np.linalg.norm(u));nv=float(np.linalg.norm(v))
    rho=float(u@v/(nu*nv))
    exact=float(nu*nu+nv*nv-2*nu*nv*kappa(rho))
    n=400000
    z1=rng.normal(size=n);z2=rng.normal(size=n)
    x=math.sqrt(2/m)*nu*z1
    y=math.sqrt(2/m)*nv*(rho*z1+math.sqrt(max(0.0,1-rho*rho))*z2)
    row=(np.maximum(x,0)-np.maximum(y,0))**2
    # Full-layer norm is a sum of m iid row contributions.
    mean=float(m*row.mean());se=float(m*row.std(ddof=1)/math.sqrt(n))
    assert abs(mean-exact) <= 6*se + 2e-9
    assert exact <= float(np.dot(u-v,u-v))+1e-10
    checks.append({'case':case,'rho':rho,'exact':exact,'mc_mean':mean,'mc_se':se})

out={
 'status':'PASS',
 'grid_min_kappa_minus_identity':float(gaps.min()),
 'grid_argmin':float(grid[int(gaps.argmin())]),
 'monte_carlo_checks':checks,
 'theorem':'E||ReLU(Wu)-ReLU(Wv)||^2 = ||u||^2+||v||^2-2||u||||v||kappa(rho) <= ||u-v||^2',
 'scope':'W_ij iid N(0,2/m), equal width, checkpoint states independent of W',
}
(ROOT/'results'/'GAUSSIAN_RELU_SUFFIX_NONEXPANSIVITY_VERIFICATION.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'status':out['status'],'min_gap':out['grid_min_kappa_minus_identity'],'cases':len(checks)},indent=2))
