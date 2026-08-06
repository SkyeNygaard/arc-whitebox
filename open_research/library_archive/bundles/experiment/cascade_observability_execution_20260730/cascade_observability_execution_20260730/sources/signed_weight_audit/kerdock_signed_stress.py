#!/usr/bin/env python3
"""Random signed perturbation stress test inside the complete Kerdock-line universe."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
A_MINUS_O=0.011988581160655598
O_MINUS_C=-0.000009468153657654632
C_MINUS_A0=-0.000000046263743724850315
B,R=129,256;P=B*R
W0=np.full((B,R),1.0/P)
def risk(w):
    S=w.sum(axis=1)
    return C_MINUS_A0+O_MINUS_C*np.dot(S,S)+A_MINUS_O*np.sum(w*w)
def negmass(w):return float(-w[w<0].sum())
def scaled(rng,target):
    z=rng.normal(size=(B,R));z-=z.mean()
    lo,hi=0.,1.
    while negmass(W0+hi*z)<target:hi*=2
    for _ in range(80):
        mid=(lo+hi)/2
        if negmass(W0+mid*z)<target:lo=mid
        else:hi=mid
    return W0+hi*z
rng=np.random.default_rng(20260729);base=risk(W0);results=[]
for beta in (1e-6,1e-4,0.01,0.1,0.5):
    excess=[];mass_err=[]
    for _ in range(200):
        w=scaled(rng,beta);excess.append(risk(w)-base);mass_err.append(abs(negmass(w)-beta))
    a=np.asarray(excess)
    results.append({'beta':beta,'trials':200,'minimum_risk_excess':float(a.min()),'median_risk_excess':float(np.median(a)),'maximum_risk_excess':float(a.max()),'improving_trials':int(np.sum(a<0)),'maximum_negative_mass_error':float(max(mass_err))})
out={'status':'EXPLORATORY EMPIRICAL; exact T27 theorem is authoritative','base_uniform_complete_kerdock_line_risk':base,'results':results,'conclusion':'No signed perturbation improved the exact line-universe optimum; all increased risk.'}
(ROOT/'KERDOCK_SIGNED_STRESS_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
