#!/usr/bin/env python3
"""Generate a rigorous multi-rank lower bound for arbitrary signed cubature.
The LP was used only to discover weights. This script verifies a fixed conservative
rational combination using interval lower bounds and exact rational harmonic algebra.
"""
import json, math
from fractions import Fraction
from pathlib import Path
import mpmath as mp
import sympy as sp

mp.mp.dps=90
iv=mp.iv; iv.dps=90
DIM=256; N=66048; DEPTH=32; ORDER=27; MAXL=26
t=sp.symbols('t')

ACTIVE=[
 ([0,1,2,3], '1.6246643258268957e-8'),
 ([0,1,2,3,4], '1.2700892655042479e-8'),
 ([0,1,2,4,5], '1.0129781921735312e-8'),
 ([0,3,5,6], '8.325893175139532e-9'),
 ([0,3,4,6,7], '6.996958286710683e-9'),
 ([0,1,3,5,7,8], '5.982074123331497e-9'),
 ([0,2,4,6,8,9], '5.1848295401305905e-9'),
 ([0,4,5,7,9,10], '4.539240015137317e-9'),
 ([0,6,8,10,11], '3.960091281821185e-9'),
 ([0,5,7,9,11,12], '3.196044832656374e-9'),
 ([0,6,8,10,12,13], '1.7537812316179916e-9'),
]
SAFETY=Fraction(999999,1000000)

def frac_decimal(s): return Fraction(s)
def moment(k):
 if k%2:return sp.Rational(0)
 m=k//2;num=sp.Integer(1) if m==0 else sp.factorial2(2*m-1);den=sp.Integer(1)
 for j in range(m):den*=DIM+2*j
 return sp.Rational(num,den)
def hdim(l):
 if l==0:return 1
 if l==1:return DIM
 return math.comb(DIM+l-1,l)-math.comb(DIM+l-3,l-2)
def G(l):
 p=sp.gegenbauer(l,sp.Rational(DIM-2,2),t);return sp.expand(p/p.subs(t,1))
Gs=[G(l) for l in range(MAXL+1)]
def expectation(poly):return sum(c*moment(mon[0]) for mon,c in sp.Poly(sp.expand(poly),t).terms())
norm=[expectation(g*g) for g in Gs]
def projection(n,l):return sp.factor(expectation(t**n*Gs[l])/norm[l])

# Interval Taylor jet for K_32 around zero.
k=(sp.sqrt(1-t**2)+(sp.pi-sp.acos(t))*t)/sp.pi
der=[sp.diff(k,t,n)/sp.factorial(n) for n in range(ORDER+1)]
mods=[{'sqrt':iv.sqrt,'acos':lambda z:iv.atan2(iv.sqrt(1-z*z),z),'pi':iv.pi}]
fd=[sp.lambdify(t,e,modules=mods) for e in der]
def z():return iv.mpf([0,0])
def conv(a,b):
 c=[z() for _ in range(ORDER+1)]
 for i,ai in enumerate(a):
  for j,bj in enumerate(b):
   if i+j>ORDER:break
   c[i+j]+=ai*bj
 return c
def compose(p):
 p0=p[0];delta=list(p);delta[0]=z();out=[z() for _ in range(ORDER+1)]
 power=[z() for _ in range(ORDER+1)];power[0]=iv.mpf([1,1])
 for n in range(ORDER+1):
  dn=fd[n](p0)
  for i in range(ORDER+1):out[i]+=dn*power[i]
  power=conv(power,delta)
 return out
jet=[z() for _ in range(ORDER+1)];jet[1]=iv.mpf([1,1])
for _ in range(DEPTH):jet=compose(jet)
def endpoint_strings(x):
 s=str(x);return s[1:s.index(',')],s[s.index(',')+1:-1]

# Every omitted Maclaurin coefficient contributes nonnegatively to every
# Gegenbauer coefficient, so the truncated projected jet is a rigorous lower bound.
Kcap=[]
for l in range(MAXL+1):
 total=Fraction(0)
 terms=[]
 for n in range(l,ORDER+1):
  p=projection(n,l)
  if p==0:continue
  lo,_=endpoint_strings(jet[n]); term=frac_decimal(lo)*Fraction(int(p.p),int(p.q))
  total+=term
  terms.append((n,str(p),str(term)))
 Kcap.append(total)

certs=[]
used=[Fraction(0) for _ in range(MAXL+1)]
objective=Fraction(0)
for S,y_raw in ACTIVE:
 y=frac_decimal(y_raw)*SAFETY
 D=sum(hdim(i) for i in S)
 floor=Fraction(D*D,N)-D
 Lpoly=sum(sp.Integer(hdim(i))*Gs[i] for i in S)
 b=[]
 for l in range(MAXL+1):
  q=sp.factor(expectation(Lpoly**2*Gs[l])/norm[l])
  bf=Fraction(int(q.p),int(q.q));b.append(bf)
  used[l]+=y*bf/floor
 lam=y/floor
 objective+=y
 certs.append({
  'feature_degrees':S,'feature_dimension':D,
  'rank_floor':{'numerator':floor.numerator,'denominator':floor.denominator},
  'objective_contribution':str(y),'lambda':str(lam),
  'squared_kernel_coefficients':{str(l):str(b[l]) for l in range(1,MAXL+1) if b[l]}
 })

aud=[]
for l in range(1,MAXL+1):
 slack=Kcap[l]-used[l]
 aud.append({'degree':l,'capacity':str(Kcap[l]),'used':str(used[l]),'slack':str(slack),'passes':slack>=0})
assert all(x['passes'] for x in aud)
kmse_lo=Fraction('2.4336603575430029389091338017406054668573276382630724978671590845071104120856063e-7')
kmse_hi=Fraction('2.4336603575430052276094665026697645914811206370055599695108464279151347033914533e-7')
# Lower-bound fraction uses the Kerdock upper endpoint; max-improvement uses same.
fraction=objective/kmse_hi
factor=kmse_hi/objective
out={
 'title':'Certified multi-rank obstruction for arbitrary signed spherical cubature',
 'scope':{'dimension':DIM,'depth':DEPTH,'node_budget':N,'weights':'arbitrary real, summing to one','nodes':'arbitrary points on S^255','kernel':'infinite-width normalized depth-32 ReLU kernel'},
 'method':'nonnegative combination of squared reproducing kernels, exact rank/trace floors, and coefficientwise Gegenbauer domination',
 'interval_jet_order':ORDER,'safety_factor':str(SAFETY),
 'kernel_coefficient_lower_bounds':{str(l):str(Kcap[l]) for l in range(1,MAXL+1)},
 'active_rank_certificates':certs,'constraint_audit':aud,
 'signed_rule_mse_lower_bound':str(objective),
 'signed_rule_mse_lower_bound_decimal':mp.nstr(mp.mpf(objective.numerator)/objective.denominator,70),
 'fraction_of_kerdock_mse_lower_bound':mp.nstr(mp.mpf(fraction.numerator)/fraction.denominator,60),
 'maximum_permitted_improvement_factor_vs_kerdock':mp.nstr(mp.mpf(factor.numerator)/factor.denominator,60),
 'interpretation':'Every static mass-one signed rule with at most 66,048 arbitrary nodes retains at least the stated fraction of complete-Kerdock limiting-kernel MSE. This is not signed near-optimality.',
}
path=Path(__file__).with_name('MULTIRANK_SIGNED_NODE_CERTIFICATE.json')
path.write_text(json.dumps(out,indent=2))
print(json.dumps({k:out[k] for k in ['signed_rule_mse_lower_bound_decimal','fraction_of_kerdock_mse_lower_bound','maximum_permitted_improvement_factor_vs_kerdock']},indent=2))
