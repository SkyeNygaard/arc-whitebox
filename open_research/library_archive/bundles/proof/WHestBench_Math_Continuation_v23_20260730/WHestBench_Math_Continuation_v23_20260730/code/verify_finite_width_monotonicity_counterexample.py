#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import sympy as sp

ROOT=Path(__file__).resolve().parent
t=sp.symbols('t')
pi=sp.pi

def kappa(x):
    return (sp.sqrt(1-x*x)+(pi-sp.acos(x))*x)/pi

def main():
    kap=kappa(t)
    s1=sp.series(kap,t,0,5).removeO().expand()
    c=sp.Rational(1,1)/pi
    a=sp.Rational(1,2)
    b=sp.Rational(1,2)/pi
    k2=sp.diff(kappa(t),t,2).subs(t,c)
    k3=sp.diff(kappa(t),t,3).subs(t,c)
    cubic=sp.simplify(k2*a*b+k3*a**3/sp.Integer(6))
    composed_series=sp.series(kappa(kap),t,0,4).removeO().expand()
    direct_cubic=sp.simplify(composed_series.coeff(t,3))
    assert sp.simplify(cubic-direct_cubic)==0
    assert cubic>0
    width1_cubic=sp.simplify(s1.coeff(t,3))
    assert width1_cubic==0
    out={
        'status':'PASS',
        'normalized_relu_kernel_series':str(s1),
        'width1_deep_kernel':'kappa(t) at every later layer, because ReLU(w*a(x))=ReLU(w)*a(x) for scalar nonnegative a(x).',
        'infinite_width_depth2_kernel':'kappa(kappa(t))',
        'width1_cubic_coefficient':str(width1_cubic),
        'infinite_width_depth2_cubic_coefficient_exact':str(cubic),
        'infinite_width_depth2_cubic_coefficient_numeric':float(sp.N(cubic,30)),
        'conclusion':'Coefficientwise finite-width >= infinite-width monotonicity is false even in the standard Gaussian ReLU family. Width-specific lower bounds require quantitative architecture-specific work.',
        'scope':'This is a width-1 counterexample to a universal shortcut. It does not show that any particular width-256 coefficient is below its infinite-width value.'
    }
    (ROOT/'finite_width_monotonicity_counterexample.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
