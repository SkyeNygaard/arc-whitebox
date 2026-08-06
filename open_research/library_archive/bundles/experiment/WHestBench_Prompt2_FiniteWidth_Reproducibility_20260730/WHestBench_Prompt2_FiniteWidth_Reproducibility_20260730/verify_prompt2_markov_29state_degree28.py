#!/usr/bin/env python3
from __future__ import annotations
import contextlib, io, json, runpy, sys
from fractions import Fraction
from pathlib import Path
if hasattr(sys,'set_int_max_str_digits'): sys.set_int_max_str_digits(0)
import prompt2_full_hermite_core as c

MAX=28
STATES=list(range(MAX+1))
DEPTH=32
ROUND_DIGITS=75
OUT=Path('/mnt/data/prompt2_markov_29state_degree28_certificate.json')


def floor_decimal(q: Fraction, digits: int=ROUND_DIGITS) -> Fraction:
    if q < 0: raise ValueError(q)
    scale=10**digits
    return Fraction(q.numerator*scale//q.denominator, scale)

# Load exact/lower-bound transition entries assembled from independently generated
# row records. The source script only uses max across alternative lower bounds and
# sums declared orthogonal equality classes.
with contextlib.redirect_stdout(io.StringIO()):
    ns=runpy.run_path('/mnt/data/explore_markov_29state.py')
D=ns['D']
source=ns['source']

# Exact nonnegative propagation with outward (downward) rational rounding after
# each layer to control denominator growth.
v=[Fraction(0) for _ in STATES]
v[1]=Fraction(1)
layer_masses=[]
for layer in range(DEPTH):
    nv=[]
    for k in STATES:
        s=sum((v[i]*D[(i,k)] for i in STATES), Fraction(0))
        nv.append(floor_decimal(s))
    v=nv
    layer_masses.append(sum(v,Fraction(0)))
assert all(x>=0 for x in v)
assert layer_masses[-1] <= 1

# Exact monomial -> normalized Gegenbauer conversion.
G=c.gegenbauer_normalized(MAX,(c.D-2)//2)
mon={n:c.expand_in_basis([Fraction(0)]*n+[Fraction(1)],G) for n in STATES}
K=[Fraction(0) for _ in STATES]
for n in range(1,MAX+1):
    for ell in range(1,n+1):
        q=mon[n][ell]
        assert q>=0
        K[ell]+=v[n]*q

# Rationalized discovery weights from the 29-state degree-28 optimization.
WEIGHTS=[
 Fraction('0.000000439814440'),
 Fraction('0.000000450949816'),
 Fraction(1),
 Fraction('0.999899927'),
 Fraction('0.00443739376'),
 Fraction('0.0000447886438'),
 Fraction('0.000000170462504'),
 Fraction('0.00000000659272692'),
 Fraction('0.0000000000593189800'),
 Fraction('0.00000000000219218817'),
 Fraction('0.0000000000000378437511'),
 Fraction('0.00000000000000197290296'),
 Fraction('0.00000000000000000461656542'),
 Fraction('0.00000000000000000162787402'),
 Fraction('0.0000000000000000000582538166'),
]
L=len(WEIGHTS)-1
lpoly=[Fraction(0)]
for ell,w in enumerate(WEIGHTS):
    lpoly=c.poly_add(lpoly,c.poly_scale(G[ell],w*c.harmonic_dim(ell)))
B=c.expand_in_basis(c.poly_mul(lpoly,lpoly),G)
ratios={n:K[n]/B[n] for n in range(1,MAX+1) if B[n]>0}
min_degree=min(ratios,key=ratios.get)
min_ratio=ratios[min_degree]
# Freeze gamma 0.01% below the exact minimum ratio.
GAMMA=floor_decimal(min_ratio*Fraction(9999,10000), 80)
assert GAMMA>0
margins={n:K[n]-GAMMA*B[n] for n in ratios}
assert all(x>0 for x in margins.values())
F=c.rank_floor(WEIGHTS)
floor=GAMMA*F

# Exact Kerdock risk of selected polynomial subkernel (absolute ceiling for this component).
def sphere_moment(n:int)->Fraction:
    if n%2:return Fraction(0)
    x=Fraction(1)
    for j in range(1,n//2+1):x*=Fraction(2*j-1,c.D+2*j-2)
    return x

def kerdock_delta(n:int)->Fraction:
    if n%2:return Fraction(0)
    if n==0:return Fraction(0)
    x=Fraction(2,c.NODES)+Fraction(65_536,c.NODES*16**n)-sphere_moment(n)
    assert x>=0
    return x
route=sum((v[n]*kerdock_delta(n) for n in range(6,MAX+1)),Fraction(0))

binding=sorted(ratios,key=ratios.get)[:15]
used_sources={}
for (i,k),label in source.items():
    if D[(i,k)]>0:
        used_sources[label]=used_sources.get(label,0)+1

rec={
 'status':'PASS',
 'decision':'PROVED_UNDER_EXPLICIT_MODEL_AND_DECLARED_ROW_COMPONENTS',
 'architecture':{'dimension':c.D,'width':c.M,'depth':DEPTH,'node_budget':c.NODES,'post_relu_output':True},
 'scope':'static realized-network-independent mass-one arbitrary-signed linear cubature; ensemble one-coordinate MSE',
 'method':'exact Hermite tensor-degree Markov chain; 29-state lower transition matrix; 32-step downward-rational propagation; degree-28 weighted-rank certificate',
 'rounding':{'digits':ROUND_DIGITS,'rule':'floor every nonnegative propagated state mass after each layer'},
 'selected_kernel_mass':c.frac_record(sum(v,Fraction(0)),45),
 'layer_mass_final_5':[c.frac_record(x,35) for x in layer_masses[-5:]],
 'monomial_coefficients':{str(i):c.frac_record(v[i],40) for i in STATES if v[i]},
 'gegenbauer_lower_bounds':{str(i):c.frac_record(K[i],40) for i in range(1,MAX+1)},
 'comparison':{
   'weights':[f'{x.numerator}/{x.denominator}' for x in WEIGHTS],
   'gamma':c.frac_record(GAMMA,50),
   'exact_min_ratio_before_safety':c.frac_record(min_ratio,50),
   'minimum_degree':min_degree,
   'rank_floor_F_N':c.frac_record(F,45),
   'binding_degrees':binding,
   'binding_ratios':{str(i):c.frac_record(ratios[i],45) for i in binding},
   'minimum_margin':c.frac_record(min(margins.values()),45),
 },
 'risk_floor_normalized':c.frac_record(floor,55),
 'component_kerdock_risk':c.frac_record(route,55),
 'component_ceiling_over_floor':c.frac_record(route/floor,40),
 'transition_source_entry_counts':used_sources,
 'limitations':[
   'The theorem lower-bounds the exact ensemble kernel by the declared transition-row components; it is not a per-realized-network statement.',
   'Generated row records and their orthogonality/equality-class interpretation remain in the proof trust base.',
   'No independent clean-room implementation of the degree-28 assembly has yet been run.',
   'The result is an absolute normalized risk floor, not yet a finite-width ratio to exact finite-width Kerdock risk.'
 ]
}
OUT.write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps({
 'status':'PASS',
 'floor_bounds':rec['risk_floor_normalized']['bounds'],
 'component_kerdock_bounds':rec['component_kerdock_risk']['bounds'],
 'ceiling_factor_bounds':rec['component_ceiling_over_floor']['bounds'],
 'selected_mass_bounds':rec['selected_kernel_mass']['bounds'],
 'binding_degrees':binding,
 'output':str(OUT)
},indent=2))
