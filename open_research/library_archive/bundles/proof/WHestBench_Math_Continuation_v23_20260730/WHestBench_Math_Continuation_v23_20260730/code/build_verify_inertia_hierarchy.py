#!/usr/bin/env python3
from __future__ import annotations
from decimal import Decimal, ROUND_FLOOR, getcontext
from fractions import Fraction
from pathlib import Path
import importlib.util, json, sys
import numpy as np
from scipy.optimize import linprog

getcontext().prec=180
sys.set_int_max_str_digits(0)
ROOT=Path(__file__).resolve().parent.parent
src=ROOT/'source_v21'/'verify_signed_near_optimality_certificate_blocktrace_order320.py'
spec=importlib.util.spec_from_file_location('v21verify',src)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
base=json.loads((ROOT/'source_v21'/'SIGNED_NEAR_OPTIMALITY_CERTIFICATE_BLOCKTRACE_ORDER320.json').read_text())
jet=json.loads((ROOT/'source_v21'/'K32_MACLAURIN_INTERVALS_ORDER320.json').read_text())
G=v.normalized_gegenbauer_polynomials(v.MAX_DEG)
mon=[]
for n in range(v.MAX_DEG+1):
    p=[Fraction(0)]*(n+1);p[n]=1
    mon.append(v.gegenbauer_decomposition(p,G))
klow=[Fraction(0)]*(v.MAX_DEG+1)
for n,row in enumerate(jet['maclaurin_intervals']):
    a=Fraction(Decimal(row[0])); assert a>=0
    for l,c in enumerate(mon[n]):
        if c: klow[l]+=a*c
active_s=sorted({int(r['s']) for r in base['components']})
sq={};cross={}
for s in active_s:
    sq[s]=v.gegenbauer_decomposition(v.polynomial_product(G[s],G[s]),G)
    sq.setdefault(s+1,v.gegenbauer_decomposition(v.polynomial_product(G[s+1],G[s+1]),G))
    cross[s]=v.gegenbauer_decomposition(v.polynomial_product(G[s],G[s+1]),G)

raws=[];Blist=[];Tlist=[];S2list=[]
for row in base['components']:
    s=int(row['s']);r=Fraction(row['r']);ds,dt=v.harmonic_dim(s),v.harmonic_dim(s+1)
    T=Fraction(ds,1)+r*dt;S2=Fraction(ds,1)+r*r*dt
    B=T*T/Fraction(v.N,1)-S2
    raw=[]
    for l in range(1,v.MAX_DEG+1):
        raw.append(Fraction(ds*ds)*sq[s][l]+2*r*ds*dt*cross[s][l]+r*r*dt*dt*sq[s+1][l])
    raws.append(raw);Blist.append(B);Tlist.append(T);S2list.append(S2)

C=np.array([[float(raws[j][l]/Blist[j]) for j in range(len(raws))] for l in range(v.MAX_DEG)])
b=np.array([float(klow[l]) for l in range(1,v.MAX_DEG+1)])
mask=b>1e-290
SCALE=1e-7
Cs=C[mask]*SCALE/b[mask,None]
bs=np.ones(mask.sum())
kerdock=Fraction(Decimal(base['certified_result']['kerdock_mse_upper_bound']))
SHRINK=Decimal('0.9999995');QUANT=Decimal('1e-30')

def dec(q:Fraction,d=65):
    return format(Decimal(q.numerator)/Decimal(q.denominator),f'.{d}E')

rows_out=[]
for qneg in [1, 2, 16, 64, 256, 1024, 1050, 1072, 1152, 2048, 4096, 4160, 4224, 8192]:
    p=v.N-qneg
    factors=np.array([float((T*T/Fraction(p,1)-S2)/B) for T,S2,B in zip(Tlist,S2list,Blist)])
    res=linprog(-factors*SCALE,A_ub=Cs,b_ub=bs,bounds=(0,None),method='highs',
                options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
    assert res.success,res.message
    x=res.x*SCALE
    ys=[]
    for z in x:
        d=(Decimal(str(z))*SHRINK).quantize(QUANT,rounding=ROUND_FLOOR)
        ys.append(Fraction(d))
    used=[Fraction(0)]*v.MAX_DEG
    objective=Fraction(0);active=0;component_rows=[]
    for j,y in enumerate(ys):
        if y==0: continue
        active+=1
        source=base['components'][j]
        component_rows.append({'component_index':j,'s':int(source['s']),'r':source['r'],'y':str(y)})
        for l in range(v.MAX_DEG): used[l]+=y*raws[j][l]/Blist[j]
        Bp=Tlist[j]*Tlist[j]/Fraction(p,1)-S2list[j]
        objective+=y*Bp/Blist[j]
    slacks=[klow[l+1]-used[l] for l in range(v.MAX_DEG)]
    assert min(slacks)>=0,(qneg,min(slacks),slacks.index(min(slacks))+1)
    rows_out.append({
        'minimum_negative_weight_count':qneg,
        'maximum_positive_weight_count':p,
        'active_components':active,
        'mse_lower_bound':dec(objective),
        'mse_lower_bound_exact':str(objective),
        'fraction_of_kerdock_upper':dec(objective/kerdock),
        'fraction_of_kerdock_upper_exact':str(objective/kerdock),
        'same_cost_improvement_cap':dec(kerdock/objective),
        'same_cost_improvement_cap_exact':str(kerdock/objective),
        'minimum_slack':dec(min(slacks)),
        'minimum_slack_degree':slacks.index(min(slacks))+1,
        'components':component_rows,
    })

out={
 'title':'Sign-count inertia hierarchy for static arbitrary-signed cubature',
 'date':'2026-07-30',
 'status':'exact-rational verification over the v21 order-320 kernel lower endpoints; float LP used only for discovery then downward shrink/rounding',
 'scope':base['scope'],
 'theorem':'If a rule has at least q nonzero negative weights among at most N nodes, every comparison moment matrix has at most N-q positive eigenvalues and ||M||_F^2 >= T^2/(N-q).',
 'rows':rows_out,
}
(ROOT/'results'/'SIGNED_INERTIA_SIGN_COUNT_HIERARCHY.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'status':'PASS','rows':rows_out},indent=2))
