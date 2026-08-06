#!/usr/bin/env python3
from __future__ import annotations
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path
import importlib.util, json, math
import numpy as np
from scipy.optimize import linprog

getcontext().prec=100
ROOT=Path(__file__).resolve().parent.parent
src=ROOT/'source_v21'/'verify_signed_near_optimality_certificate_blocktrace_order320.py'
spec=importlib.util.spec_from_file_location('v21verify',src)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
cert=json.loads((ROOT/'source_v21'/'SIGNED_NEAR_OPTIMALITY_CERTIFICATE_BLOCKTRACE_ORDER320.json').read_text())
jet=json.loads((ROOT/'source_v21'/'K32_MACLAURIN_INTERVALS_ORDER320.json').read_text())
G=v.normalized_gegenbauer_polynomials(v.MAX_DEG)
mon=[]
for n in range(v.MAX_DEG+1):
 p=[Fraction(0)]*(n+1);p[n]=1;mon.append(v.gegenbauer_decomposition(p,G))
klow=[Fraction(0)]*(v.MAX_DEG+1)
for n,row in enumerate(jet['maclaurin_intervals']):
 a=Fraction(Decimal(row[0]))
 for l,c in enumerate(mon[n]):
  if c: klow[l]+=a*c
active_s=sorted({int(r['s']) for r in cert['components']})
sq={};cross={}
for s in active_s:
 sq[s]=v.gegenbauer_decomposition(v.polynomial_product(G[s],G[s]),G)
 sq.setdefault(s+1,v.gegenbauer_decomposition(v.polynomial_product(G[s+1],G[s+1]),G))
 cross[s]=v.gegenbauer_decomposition(v.polynomial_product(G[s],G[s+1]),G)
cols=[];factors=[];oldy=[]
for row in cert['components']:
 s=int(row['s']);r=Fraction(row['r']);ds,dt=v.harmonic_dim(s),v.harmonic_dim(s+1)
 B=(ds+r*dt)**2/v.N-ds-r*r*dt
 col=[]
 for l in range(1,v.MAX_DEG+1):
  raw=Fraction(ds*ds)*sq[s][l]+2*r*ds*dt*cross[s][l]+r*r*dt*dt*sq[s+1][l]
  col.append(float(raw/B))
 cols.append(col)
 delta=(ds+r*dt)**2/Fraction(v.N*(v.N-1),1)
 factors.append(float(1+delta/B))
 oldy.append(float(Decimal(row['y'])))
C=np.array(cols).T
b=np.array([float(klow[l]) for l in range(1,v.MAX_DEG+1)])
f=np.array(factors);y0=np.array(oldy)
# keep a safety factor against floating LP edge issues; discovery only.
mask=(b>1e-290)
SCALE=1e-7
Cs=C[mask]*SCALE/b[mask,None]
bs=np.ones(mask.sum())
print('scaled shape',Cs.shape,'range',np.nanmin(Cs),np.nanmax(Cs),'finite',np.isfinite(Cs).all())
res=linprog(-f*SCALE,A_ub=Cs,b_ub=bs,bounds=(0,None),method='highs',options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
if res.success: res.x=res.x*SCALE
print('success',res.success,res.message)
print('old objective',f@y0,'old base',y0.sum())
if res.success:
 print('new objective',f@res.x,'base part',res.x.sum(),'active',np.count_nonzero(res.x>1e-18))
 print('max violation',np.max(C@res.x-b),'min slack',np.min(b-C@res.x),'max normalized',np.max((C@res.x-b)[mask]/b[mask]))
 out={'success':True,'exploratory_only':True,'old_inertia_objective':float(f@y0),'lp_inertia_objective':float(f@res.x),'lp_base_objective':float(res.x.sum()),'active_variables':int(np.count_nonzero(res.x>1e-18)),'max_float_violation':float(np.max(C@res.x-b)),'solution':[float(z) for z in res.x]}
else: out={'success':False,'message':res.message}
(ROOT/'results'/'INERTIA_REOPTIMIZED_LP_EXPLORATORY.json').write_text(json.dumps(out,indent=2))
