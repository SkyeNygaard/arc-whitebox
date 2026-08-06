#!/usr/bin/env python3
from __future__ import annotations
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path
import importlib.util, json, sys

getcontext().prec=190
sys.set_int_max_str_digits(0)
ROOT=Path(__file__).resolve().parent.parent
src=ROOT/'source_v21'/'verify_signed_near_optimality_certificate_blocktrace_order320.py'
spec=importlib.util.spec_from_file_location('v21verify',src)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
base=json.loads((ROOT/'source_v21'/'SIGNED_NEAR_OPTIMALITY_CERTIFICATE_BLOCKTRACE_ORDER320.json').read_text())
jet=json.loads((ROOT/'source_v21'/'K32_MACLAURIN_INTERVALS_ORDER320.json').read_text())
headline=json.loads((ROOT/'results'/'SIGNED_INERTIA_CERTIFICATE_ORDER320.json').read_text())
hierarchy=json.loads((ROOT/'results'/'SIGNED_INERTIA_SIGN_COUNT_HIERARCHY.json').read_text())

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

def profile(source):
    s=int(source['s']);r=Fraction(source['r']);ds,dt=v.harmonic_dim(s),v.harmonic_dim(s+1)
    T=Fraction(ds,1)+r*dt;S2=Fraction(ds,1)+r*r*dt
    B=T*T/Fraction(v.N,1)-S2
    raw=[Fraction(ds*ds)*sq[s][l]+2*r*ds*dt*cross[s][l]+r*r*dt*dt*sq[s+1][l] for l in range(v.MAX_DEG+1)]
    return T,S2,B,raw

profiles=[profile(s) for s in base['components']]
kerdock=Fraction(Decimal(base['certified_result']['kerdock_mse_upper_bound']))

def verify_components(rows,p):
    used=[Fraction(0)]*(v.MAX_DEG+1); objective=Fraction(0)
    for row in rows:
        idx=row.get('component_index')
        if idx is None:
            # Headline rows retain s/r rather than index. Resolve exactly.
            idx=next(i for i,b in enumerate(base['components']) if int(b['s'])==int(row['s']) and Fraction(b['r'])==Fraction(row['r']))
        y=Fraction(row['y']);T,S2,B,raw=profiles[idx]
        Bp=T*T/Fraction(p,1)-S2
        objective+=y*Bp/B
        for l in range(1,v.MAX_DEG+1): used[l]+=y*raw[l]/B
    slacks=[klow[l]-used[l] for l in range(1,v.MAX_DEG+1)]
    assert min(slacks)>=0
    return objective,slacks

obj,slacks=verify_components(headline['components'],v.N-1)
recorded=Fraction(headline['certified_result']['inertia_strengthened_mse_lower_bound_exact'])
assert obj==recorded,(obj,recorded)
assert Fraction(headline['certified_result']['fraction_of_kerdock_upper_exact'])==obj/kerdock

hier_checks=[]
for row in hierarchy['rows']:
    q=int(row['minimum_negative_weight_count']);p=v.N-q
    obj2,sl2=verify_components(row['components'],p)
    rec_dec=Decimal(row['mse_lower_bound'])
    obj2_dec=Decimal(obj2.numerator)/Decimal(obj2.denominator)
    assert abs(obj2_dec-rec_dec) < Decimal('1e-70'),(q,obj2_dec,rec_dec)
    frac_dec=Decimal((obj2/kerdock).numerator)/Decimal((obj2/kerdock).denominator)
    assert abs(frac_dec-Decimal(row['fraction_of_kerdock_upper'])) < Decimal('1e-65')
    hier_checks.append({'q_negative':q,'verified':True,'minimum_slack_degree':sl2.index(min(sl2))+1})

out={
 'status':'PASS',
 'headline':{
   'mse_lower_bound':format(Decimal(obj.numerator)/Decimal(obj.denominator),'.80E'),
   'fraction_of_kerdock_upper':format(Decimal((obj/kerdock).numerator)/Decimal((obj/kerdock).denominator),'.80E'),
   'minimum_slack':format(Decimal(min(slacks).numerator)/Decimal(min(slacks).denominator),'.80E'),
   'minimum_slack_degree':slacks.index(min(slacks))+1,
 },
 'hierarchy_rows_verified':hier_checks,
 'trust_note':'Exact Fraction replay of rational witnesses against v21 directed kernel lower endpoints; does not independently reproduce those interval endpoints.',
}
(ROOT/'results'/'INERTIA_CERTIFICATES_EXACT_REPLAY.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'status':'PASS','headline_fraction':format(Decimal((obj/kerdock).numerator)/Decimal((obj/kerdock).denominator),'.25E'),'hierarchy_rows':len(hier_checks)},indent=2))
