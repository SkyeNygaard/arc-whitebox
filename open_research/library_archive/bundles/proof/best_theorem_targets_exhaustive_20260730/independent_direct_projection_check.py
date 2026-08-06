#!/usr/bin/env python3
import json, math
import numpy as np
from scipy.special import roots_jacobi, eval_gegenbauer

d=256; lam=127.; alpha=126.5; depth=32

def kappa(t):
    t=np.clip(t,-1.,1.)
    return (np.sqrt(np.maximum(0.,1-t*t))+(np.pi-np.arccos(t))*t)/np.pi

def coeffs(nq):
    x,w=roots_jacobi(nq,alpha,alpha); w=w/w.sum()
    K=x.copy()
    for _ in range(depth): K=kappa(K)
    out=[]
    for l in range(7):
        G=eval_gegenbauer(l,lam,x)/eval_gegenbauer(l,lam,1.)
        out.append(float(np.sum(w*K*G)/np.sum(w*G*G)))
    return out
runs={str(n):coeffs(n) for n in (200,400,800,1200)}
# Exact T16 central values, only for independent sanity comparison.
h=[0.9747299751309444,0.002796473061541184,0.0024362952737152224,0.0018037348551971006,0.0010317284867674261,0.00017989892346364459]
last=runs['1200']
margins=[last[i]-h[i] for i in range(6)]
# Compare independent direct values to interval-jet lower bounds.
cert=json.load(open('/mnt/data/theorem_targets/best_theorem_targets_verification.json'))
lower=[float(cert['coefficient_checks'][i]['K_coefficient_lower_from_degree_11_jet']) for i in range(7)]
out={'quadrature_runs':runs,'direct_minus_h_margins':margins,'direct_minus_interval_lower':[last[i]-lower[i] for i in range(7)]}
json.dump(out,open('/mnt/data/theorem_targets/independent_direct_projection_check.json','w'),indent=2)
assert min(margins)>0
assert min(out['direct_minus_interval_lower'])>-2e-10
print(json.dumps({'min_direct_h_margin':min(margins),'degree6':last[6],'max_spread':max(max(runs[str(n)][i] for n in (200,400,800,1200))-min(runs[str(n)][i] for n in (200,400,800,1200)) for i in range(7))},indent=2))
