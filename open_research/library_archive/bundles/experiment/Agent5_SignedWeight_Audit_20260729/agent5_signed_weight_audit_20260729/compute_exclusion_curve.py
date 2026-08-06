#!/usr/bin/env python3
"""Exact-rational negative-mass exclusion curves with certified root brackets."""
from __future__ import annotations
import csv,json
from decimal import Decimal,localcontext,ROUND_FLOOR,ROUND_CEILING
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parent
cert=json.loads((ROOT/'M_CERTIFICATE.json').read_text())
theorem=json.loads((ROOT/'FORMAL_NEAR_OPTIMALITY_THEOREM_D256_L32.json').read_text())
M=Fraction(cert['M_exact']);q1=Fraction(cert['q1_exact']);N=int(theorem['node_budget'])
L=Fraction(theorem['optimum_mse_lower_bound']['lower'])
U=Fraction(theorem['kerdock_mse']['upper'])

def dec_bound(x:Fraction,upper:bool=False,prec:int=80)->str:
    with localcontext() as c:
        c.prec=prec;c.rounding=ROUND_CEILING if upper else ROUND_FLOOR
        return str(Decimal(x.numerator)/Decimal(x.denominator))

def original_allowance(beta:Fraction)->Fraction:
    return 2*M*beta*(1+beta)

def diag_min(beta:Fraction):
    if beta<=0:return Fraction(1,N),0
    nstar=Fraction(N)*beta/(1+2*beta)
    k=nstar.numerator//nstar.denominator
    cand={1,N-1}
    for n in (k-1,k,k+1,k+2):
        if 1<=n<=N-1:cand.add(n)
    vals=[(((1+beta)**2/Fraction(N-n))+(beta**2/Fraction(n)),n) for n in cand]
    return min(vals)

def integer_allowance(beta:Fraction)->Fraction:
    D,_=diag_min(beta)
    return 2*M*beta*(1+beta)-q1*(D-Fraction(1,N))

def root_bracket(delta:Fraction,fn):
    if delta<=0:return Fraction(0),Fraction(0)
    lo=Fraction(0);hi=Fraction(1,10**12)
    while fn(hi)<delta:hi*=2
    for _ in range(280):
        mid=(lo+hi)/2
        if fn(mid)<delta:lo=mid
        else:hi=mid
    assert fn(lo)<delta<=fn(hi)
    return lo,hi

percentages=[('0.01',Fraction(1,10000)),('0.1',Fraction(1,1000)),('1',Fraction(1,100)),('5',Fraction(1,20)),('10',Fraction(1,10))]
rows=[]
for label,p in percentages:
    delta_pos=p*L
    delta_k=max(Fraction(0),L-(1-p)*U)
    o_lo,o_hi=root_bracket(delta_pos,original_allowance)
    i_lo,i_hi=root_bracket(delta_pos,integer_allowance)
    ko_lo,ko_hi=root_bracket(delta_k,original_allowance)
    ki_lo,ki_hi=root_bracket(delta_k,integer_allowance)
    rows.append({
      'target_improvement_percent':label,
      'delta_below_positive_certificate_lower':dec_bound(delta_pos),
      'beta_original_positive_lower':dec_bound(o_lo),
      'beta_original_positive_upper':dec_bound(o_hi,True),
      'beta_integer_positive_lower':dec_bound(i_lo),
      'beta_integer_positive_upper':dec_bound(i_hi,True),
      'integer_negative_nodes_at_upper':diag_min(i_hi)[1] if i_hi else 0,
      'delta_kerdock_relative_lower':dec_bound(delta_k),
      'beta_original_kerdock_lower':dec_bound(ko_lo),
      'beta_original_kerdock_upper':dec_bound(ko_hi,True),
      'beta_integer_kerdock_lower':dec_bound(ki_lo),
      'beta_integer_kerdock_upper':dec_bound(ki_hi,True),
      'kerdock_integer_negative_nodes_at_upper':diag_min(ki_hi)[1] if ki_hi else 0,
    })
meta={
 'arithmetic':'Exact Fraction evaluation and 280-step dyadic bisection; each beta is enclosed by lower/upper decimal bounds.',
 'M_exact':str(M),'q1_exact':str(q1),'N':N,
 'positive_mse_certificate_lower_exact_decimal':theorem['optimum_mse_lower_bound']['lower'],
 'kerdock_mse_upper_exact_decimal':theorem['kerdock_mse']['upper'],
 'rows':rows,
}
(ROOT/'NEGATIVE_MASS_EXCLUSION_CURVE.json').write_text(json.dumps(meta,indent=2)+'\n')
with (ROOT/'NEGATIVE_MASS_EXCLUSION_CURVE.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(meta,indent=2))
