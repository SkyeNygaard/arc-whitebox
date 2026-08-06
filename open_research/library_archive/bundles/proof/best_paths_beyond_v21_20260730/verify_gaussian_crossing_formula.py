#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

ROOT=Path(__file__).resolve().parent

def theory(rho,sigy=1.0):
    rho=max(-1.0,min(1.0,rho))
    er=sigy*(1-rho)/math.sqrt(2*math.pi)
    er2=sigy*sigy*(math.acos(rho)-rho*math.sqrt(max(0.0,1-rho*rho)))/math.pi
    return er,er2

def quadrature(rho,sigy=1.0):
    if abs(rho)>=1:
        return theory(rho,sigy)
    a=rho/math.sqrt(1-rho*rho)
    # Symmetry: y>0 and X<0 plus y<0 and X>0.
    i1=2*quad(lambda y: y*norm.pdf(y)*norm.cdf(-a*y),0,np.inf,epsabs=2e-13,epsrel=2e-13,limit=400)[0]
    i2=2*quad(lambda y: y*y*norm.pdf(y)*norm.cdf(-a*y),0,np.inf,epsabs=2e-13,epsrel=2e-13,limit=400)[0]
    return sigy*i1,sigy*sigy*i2

def monte_carlo(rho,n=2_000_000,seed=123):
    rng=np.random.default_rng(seed)
    x=rng.standard_normal(n)
    e=rng.standard_normal(n)
    y=rho*x+math.sqrt(max(0,1-rho*rho))*e
    r=np.abs(y)*(x*y<0)
    return float(r.mean()),float(np.mean(r*r))

def main():
    rows=[]
    for j,rho in enumerate([-0.9,-0.5,0.0,0.3,0.8,0.99]):
        th=theory(rho); q=quadrature(rho); mc=monte_carlo(rho,seed=123+j)
        rows.append({
            'rho':rho,
            'theory_E_R':th[0],'quadrature_E_R':q[0],'mc_E_R':mc[0],
            'theory_E_R2':th[1],'quadrature_E_R2':q[1],'mc_E_R2':mc[1],
            'quadrature_abs_error_R':abs(th[0]-q[0]),
            'quadrature_abs_error_R2':abs(th[1]-q[1]),
            'mc_abs_error_R':abs(th[0]-mc[0]),
            'mc_abs_error_R2':abs(th[1]-mc[1]),
        })
    small=[]
    for theta in [1e-1,5e-2,2e-2,1e-2,5e-3]:
        rho=math.cos(theta)
        psi=(theta-math.sin(theta)*math.cos(theta))/math.pi
        lead=2*theta**3/(3*math.pi)
        small.append({'theta':theta,'psi':psi,'leading':lead,'ratio':psi/lead})
    out={
        'status':'PASS',
        'theorem':{
            'setting':'Condition on deterministic vectors a,b before drawing a current Gaussian weight row w~N(0,I/d). Put z=w^T a and z_prime=w^T(a+b).',
            'rho':'a^T(a+b)/(||a|| ||a+b||)',
            'E_R':'||a+b||/sqrt(d) * (1-rho)/sqrt(2*pi)',
            'E_R2':'||a+b||^2/d * (acos(rho)-rho*sqrt(1-rho^2))/pi',
            'remainder':'R=|z_prime| 1_{z z_prime<0}',
            'small_angle':'E_R2/(||a+b||^2/d) = 2 theta^3/(3 pi)+O(theta^5), rho=cos(theta).',
        },
        'checks':rows,
        'small_angle_checks':small,
        'scope_limitations':[
            'Exact under conditioning only when the perturbation vectors are fixed independently of the current Gaussian row.',
            'A source using the current or future weight row requires leave-one-row decoupling, a conditional Gaussian argument, or a separate dependence bound.',
            'The formula controls gate crossing exactly but does not by itself control downstream operator amplification.'
        ]
    }
    assert max(x['quadrature_abs_error_R'] for x in rows)<2e-11
    assert max(x['quadrature_abs_error_R2'] for x in rows)<2e-11
    (ROOT/'gaussian_crossing_formula_verification.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':'PASS','max_quad_R':max(x['quadrature_abs_error_R'] for x in rows),'max_quad_R2':max(x['quadrature_abs_error_R2'] for x in rows)},indent=2))
if __name__=='__main__': main()
