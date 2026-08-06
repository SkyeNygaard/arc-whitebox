#!/usr/bin/env python3
from __future__ import annotations
from decimal import Decimal, ROUND_FLOOR, getcontext
from fractions import Fraction
from pathlib import Path
import importlib.util, json, sys

getcontext().prec=160
sys.set_int_max_str_digits(0)
ROOT=Path(__file__).resolve().parent.parent
src=ROOT/'source_v21'/'verify_signed_near_optimality_certificate_blocktrace_order320.py'
spec=importlib.util.spec_from_file_location('v21verify',src)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
base=json.loads((ROOT/'source_v21'/'SIGNED_NEAR_OPTIMALITY_CERTIFICATE_BLOCKTRACE_ORDER320.json').read_text())
jet=json.loads((ROOT/'source_v21'/'K32_MACLAURIN_INTERVALS_ORDER320.json').read_text())
explore=json.loads((ROOT/'results'/'INERTIA_REOPTIMIZED_LP_EXPLORATORY.json').read_text())
assert explore['success'] and len(explore['solution'])==len(base['components'])
SHRINK=Decimal('0.9999999'); QUANT=Decimal('1e-30')
ys=[]
for z in explore['solution']:
    d=(Decimal(str(z))*SHRINK).quantize(QUANT,rounding=ROUND_FLOOR)
    ys.append(Fraction(d))

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
used=[Fraction(0)]*(v.MAX_DEG+1)
objective=Fraction(0);base_objective=Fraction(0);constant_used=Fraction(0);inertia_quadratic=Fraction(0);rows=[]
for source,y in zip(base['components'],ys):
    s=int(source['s']);r=Fraction(source['r']);ds,dt=v.harmonic_dim(s),v.harmonic_dim(s+1)
    B=(ds+r*dt)**2/v.N-ds-r*r*dt
    Bminus=(ds+r*dt)**2/Fraction(v.N-1,1)-ds-r*r*dt
    factor=Bminus/B
    if y==0: continue
    for l in range(1,v.MAX_DEG+1):
        raw=Fraction(ds*ds)*sq[s][l]+2*r*ds*dt*cross[s][l]+r*r*dt*dt*sq[s+1][l]
        used[l]+=y*raw/B
    base_objective+=y
    objective+=y*factor
    constant_used += y*(Fraction(ds,1)+r*r*dt)/B
    inertia_quadratic += y*(Fraction(ds+r*dt,1)**2/Fraction(v.N-1,1))/B
    rows.append({'s':s,'r':str(r),'y':str(y),'inertia_factor':str(factor)})
slacks=[klow[l]-used[l] for l in range(1,v.MAX_DEG+1)]
assert min(slacks)>=0, (min(slacks),slacks.index(min(slacks))+1)
old=Fraction(Decimal(base['certified_result']['mse_lower_bound']))
kerdock=Fraction(Decimal(base['certified_result']['kerdock_mse_upper_bound']))
# Original v21 weights with the exact inertia increment, for comparison.
old_inertia=Fraction(0)
for source in base['components']:
    s=int(source['s']);r=Fraction(source['r']);y=Fraction(Decimal(source['y']))
    ds,dt=v.harmonic_dim(s),v.harmonic_dim(s+1)
    B=(ds+r*dt)**2/v.N-ds-r*r*dt
    Bminus=(ds+r*dt)**2/Fraction(v.N-1,1)-ds-r*r*dt
    old_inertia+=y*Bminus/B
assert objective>old_inertia>old
# Arbitrary-total-mass extension. For the signed/inertia branch with total mass s>0,
# the comparison lower bound is B0(1-2s)+Aminus*s^2. The unused
# degree-zero kernel coefficient contributes (k0-B0)(1-s)^2. For s<=0
# that residual alone is enormous relative to the claimed floor. The exactly-N
# positive branch is handled separately by the positive T22 theorem.
k0=klow[0]
assert k0>constant_used
quad_coeff=inertia_quadratic+k0-constant_used
s_star=k0/quad_coeff
arbitrary_mass_inertia=k0-k0*k0/quad_coeff
assert 0<s_star<1
# Positive arbitrary-mass branch: if Q=s Q0 with Q0 a positive probability
# rule, orthogonality of the constant mode gives k0(1-s)^2+s^2 L_plus.
canon=json.loads((ROOT/'source_v21'/'FORMAL_CANONICAL_THEOREM_RECORD_V5_2.json').read_text())
Lplus=Fraction(Decimal(canon['primary_static_theorem']['auxiliary_optimum_mse_interval'][0]))
positive_arbitrary_mass=k0*Lplus/(k0+Lplus)
assert positive_arbitrary_mass>arbitrary_mass_inertia

def dec(q,d=70):
    return format(Decimal(q.numerator)/Decimal(q.denominator),f'.{d}E')
active=[l for l in range(1,v.MAX_DEG+1) if used[l]>0]
minall=min(range(1,v.MAX_DEG+1),key=lambda l:slacks[l-1])
mina=min(active,key=lambda l:slacks[l-1])
out={
 'title':'Inertia-strengthened degree-280 signed atomic cubature certificate',
 'date':'2026-07-30',
 'status':'exact-rational verification over v21 directed kernel lower endpoints; independent interval stack still required for public release',
 'scope':base['scope'],
 'dichotomy':'If all nonzero weights are positive, apply T22. Otherwise every profile moment matrix has at most N-1 positive eigenvalues, so its Frobenius rank floor is T^2/(N-1)-S2.',
 'construction':{'source_component_grid':'v21 146-profile grid','active_components':len(rows),'float_discovery_shrink':str(SHRINK),'quantum':str(QUANT)},
 'certified_result':{
   'verified':True,
   'base_rank_floor_part':dec(base_objective),
   'inertia_strengthened_mse_lower_bound':dec(objective),
   'inertia_strengthened_mse_lower_bound_exact':str(objective),
   'previous_v21_mse_lower_bound':dec(old),
   'previous_weights_with_inertia':dec(old_inertia),
   'strict_improvement_over_previous_inertia':dec(objective-old_inertia),
   'fraction_of_kerdock_upper':dec(objective/kerdock),
   'fraction_of_kerdock_upper_exact':str(objective/kerdock),
   'same_cost_improvement_cap':dec(kerdock/objective),
   'same_cost_improvement_cap_exact':str(kerdock/objective),
   'minimum_all_checked_slack':dec(slacks[minall-1]),
   'minimum_all_checked_degree':minall,
   'minimum_active_slack':dec(slacks[mina-1]),
   'minimum_active_slack_degree':mina,
   'arbitrary_total_mass':{
      'constant_kernel_lower':dec(k0),
      'comparison_degree0_used':dec(constant_used),
      'signed_branch_quadratic_coefficient':dec(quad_coeff),
      'minimizing_total_mass':dec(s_star),
      'signed_branch_floor':dec(arbitrary_mass_inertia),
      'positive_branch_floor':dec(positive_arbitrary_mass),
      'universal_floor':dec(arbitrary_mass_inertia),
      'fraction_of_kerdock_upper':dec(arbitrary_mass_inertia/kerdock),
      'same_cost_improvement_cap':dec(kerdock/arbitrary_mass_inertia),
   },
 },
 'components':rows,
}
(ROOT/'results'/'SIGNED_INERTIA_CERTIFICATE_ORDER320.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out['certified_result'],indent=2))
