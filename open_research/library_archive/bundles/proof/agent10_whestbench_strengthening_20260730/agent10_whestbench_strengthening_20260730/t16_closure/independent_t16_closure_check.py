#!/usr/bin/env python3
"""Independent non-proof numerical check of the T16 primal closure."""
from __future__ import annotations
import json, math
from pathlib import Path
import mpmath as mp
import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar

mp.mp.dps=80
HERE=Path(__file__).resolve().parent
x=sp.symbols('x')
roots=[mp.mpf(str(r)) for r in sp.nroots(22102*x**3+21930*x**2-87*x-85,n=80,maxsteps=200)]

def kap(t): return (mp.sqrt(1-t*t)+(mp.pi-mp.acos(t))*t)/mp.pi
def kapp(t): return (mp.pi-mp.acos(t))/mp.pi
def K(t):
 y=mp.mpf(t)
 for _ in range(32):y=kap(y)
 return y
def Kp(t):
 y=mp.mpf(t);p=mp.mpf(1)
 for _ in range(32):p*=kapp(y);y=kap(y)
 return p
A=mp.matrix(6,6);b=mp.matrix(6,1)
for j,r in enumerate(roots):
 for k in range(6):A[2*j,k]=r**k
 for k in range(6):A[2*j+1,k]=0 if k==0 else k*r**(k-1)
 b[2*j]=K(r);b[2*j+1]=Kp(r)
mono=mp.lu_solve(A,b)

def hp(t):return mp.fsum(mono[k]*mp.mpf(t)**k for k in range(6))
# high-precision checks at representative points and contacts
contacts=[]
for r in roots:contacts.append({'t':mp.nstr(r,60),'value_gap':mp.nstr(K(r)-hp(r),70),'derivative_gap':mp.nstr(Kp(r)-mp.diff(hp,r),70)})
points=[mp.mpf(-1),mp.mpf('-.999'),mp.mpf('-.8'),mp.mpf('-.4'),mp.mpf(0),mp.mpf('.4'),mp.mpf('.8'),mp.mpf('.999'),mp.mpf(1)]
gaps=[{'t':str(z),'K_minus_h':mp.nstr(K(z)-hp(z),60)} for z in points]

# float derivative jets through order 7, independent from interval proof
kap_expr=(sp.sqrt(1-x*x)+(sp.pi-sp.acos(x))*x)/sp.pi
kd=[sp.lambdify(x,sp.diff(kap_expr,x,n),'numpy') for n in range(8)]
def compose(j,f):
 order=len(j)-1
 a=np.array([j[n]/math.factorial(n) for n in range(order+1)])
 delta=a.copy();delta[0]=0
 out=np.zeros(order+1);power=np.zeros(order+1);power[0]=1
 for k in range(order+1):
  if k:power=np.convolve(power,delta)[:order+1]
  out+=f[k]/math.factorial(k)*power
 return np.array([out[n]*math.factorial(n) for n in range(order+1)])
def jets(z,order=7):
 j=np.zeros(order+1);j[0]=z;j[1]=1
 for _ in range(32):j=compose(j,np.array([kd[n](j[0]) for n in range(order+1)],float))
 return j
res=minimize_scalar(lambda z:jets(z)[6],bounds=(-.9,.2),method='bounded',options={'xatol':1e-13})
output={
 'status':'PASS' if res.fun>0 and all(mp.mpf(row['K_minus_h'])>=0 for row in gaps) else 'FAIL',
 'role':'Independent high-precision/non-interval numerical check; not a proof.',
 'contacts':contacts,
 'representative_gaps':gaps,
 'sixth_derivative_numerical_minimum':{'t':repr(float(res.x)),'value':repr(float(res.fun))},
}
(HERE/'T16_PRIMAL_DUAL_INDEPENDENT_CHECK.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
if output['status']!='PASS':raise SystemExit(1)
