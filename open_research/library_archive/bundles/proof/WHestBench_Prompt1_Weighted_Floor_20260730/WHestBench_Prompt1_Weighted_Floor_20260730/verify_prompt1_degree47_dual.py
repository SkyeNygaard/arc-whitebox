#!/usr/bin/env python3
"""Directed verifier for the Prompt-1 degree-47 primal/dual certificate.

Trust boundary:
  * exact Python Fraction harmonic and rank algebra;
  * direct-C MPFR outward intervals for the order-511 deep-kernel jet;
  * mpmath.iv only for recombining already-directed endpoints and checking margins.
The analytic proof supplies positivity of all Maclaurin coefficients and K(1)=1.
"""
from fractions import Fraction
import json, math
from pathlib import Path
import mpmath as mp
mp.mp.dps=115; mp.iv.dps=115
iv=mp.iv
ROOT=Path(__file__).resolve().parent
cert=json.load(open(ROOT/'SIGNED_RANK_DEGREE47_CERTIFICATE.json'))
dual=json.load(open(ROOT/'PROMPT1_DEGREE47_ENTRYWISE_DUAL_CERTIFICATE.json'))
jet=json.load(open(ROOT/'MPFR_KERNEL_JET_511.json'))
D=256;L=47;R=94;N=66048;M=jet['order']
assert M==511 and jet['depth']==32

def point_frac(x:Fraction): return iv.mpf([str(x.numerator),str(x.numerator)])/x.denominator
def point_str(s:str): return iv.mpf([s,s])
def endpoints(x):
    s=str(x); c=s.index(','); return mp.mpf(s[1:c]),mp.mpf(s[c+1:-1])
def lower(x): return endpoints(x)[0]
def hd(l): return math.comb(D+l-1,l)-(math.comb(D+l-3,l-2) if l>=2 else 0)

# Exact monomial -> normalized-Gegenbauer recurrence through M+2.
P=[[Fraction(0)]*(R+1) for _ in range(M+3)];P[0][0]=Fraction(1)
for p in range(M+2):
    for l,v in enumerate(P[p]):
        if not v: continue
        if l+1<=R: P[p+1][l+1]+=v*Fraction(l+D-2,2*l+D-2)
        if l: P[p+1][l-1]+=v*Fraction(l,2*l+D-2)
# Directed full kernel-coefficient upper bounds. Since alpha_p>=0 and sum alpha_p=K(1)=1,
# tail mass <= 1-sum lower(alpha_p). For p>M, P[p,r] is unimodal by the exact ratio.
alpha_lo=[point_str(row['lo']) for row in jet['coefficients']]
alpha_hi=[point_str(row['hi']) for row in jet['coefficients']]
assert all(lower(x)>=0 for x in alpha_lo)
tail_mass=point_str('1')-sum(alpha_lo,point_str('0'))
kup=[]; tail_sup=[]
for r in range(R+1):
    trunc=point_str('0')
    for p in range(r,M+1):
        if P[p][r]: trunc += alpha_hi[p]*point_frac(P[p][r])
    p0=M+1 if ((M+1-r)&1)==0 else M+2
    sup=P[p0][r]
    # Ratio after p0 is <1; verify exact one-crossing condition at p0.
    ratio=Fraction((p0+2)*(p0+1),(p0+2-r)*(D+p0+r))
    assert ratio<1
    full=trunc+tail_mass*point_frac(sup)
    # Explicit extra outward inflation used by the frozen dual generator.
    full=full*(1+point_str('1e-75'))+point_str('1e-120')
    kup.append(full); tail_sup.append(sup)

# Exact normalized Gegenbauer polynomials and square-kernel linearization matrices on demand.
G=[[Fraction(1)],[Fraction(0),Fraction(1)]]
for l in range(1,L):
    A=Fraction(2*(l+127),l+254); B=Fraction(l,l+254)
    out=[Fraction(0)]*(l+2)
    for p,v in enumerate(G[l]): out[p+1]+=A*v
    for p,v in enumerate(G[l-1]): out[p]-=B*v
    G.append(out)
dims=[hd(l) for l in range(L+1)]
q=[[v*dims[l] for v in G[l]] for l in range(L+1)]
weights=[Fraction(x) for x in cert['weights']]
sel=dual['fixed_selection']; tail=[dims[l]-sel[l] for l in range(L+1)]
assert sum(sel)==N and all(0<=sel[l]<=dims[l] for l in range(L+1))
rankdef=sum(Fraction(tail[l])*weights[l]*weights[l] for l in range(L+1))
rankdef += sum(Fraction(tail[l])*weights[l] for l in range(L+1))**2/Fraction(N)
assert str(rankdef)==dual['rankdef_exact']==cert['rank_defect_exact']
g=Fraction(cert['gamma_lower']); giv=point_frac(g); rd=point_frac(rankdef)
y=[point_str(s) for s in dual['dual_weights']]
min_margin=None;min_pair=None;checked=0
for i in range(L+1):
    for j in range(i,L+1):
        Hij=Fraction(tail[i] if i==j else 0)+Fraction(tail[i]*tail[j],N)
        if Hij==0: continue
        prod=[Fraction(0)]*(i+j+1)
        for p,x in enumerate(q[i]):
            for s,z in enumerate(q[j]): prod[p+s]+=x*z
        lhs=point_str('0')
        for r in range(1,R+1):
            if lower(y[r-1])==0: continue
            Cr=sum((prod[p]*P[p][r] for p in range(r,len(prod))),Fraction(0))
            if Cr: lhs += y[r-1]*giv*point_frac(Cr)/kup[r]
        rhs=point_frac(Hij)/rd
        margin=lhs-rhs; ml=lower(margin)
        if not (ml>0): raise AssertionError(f'nonpositive entry margin {(i,j)}: {margin}')
        if min_margin is None or ml<min_margin: min_margin=ml;min_pair=(i,j)
        checked+=1
U=sum(mp.mpf(s) for s in dual['dual_weights'])
f0=mp.mpf(cert['fraction_kerdock']); upper_fraction=U*f0
result={
 'status':'PASS','checked_positive_entries':checked,
 'minimum_directed_entry_margin':mp.nstr(min_margin,55),'minimum_margin_pair':list(min_pair),
 'dual_objective_upper_factor_over_D47':mp.nstr(U,55),
 'D47_lower_fraction_of_kerdock':cert['fraction_kerdock'],
 'degree47_family_upper_fraction_of_kerdock':mp.nstr(upper_fraction,55),
 'degree47_family_floor_window':[cert['fraction_kerdock'],mp.nstr(upper_fraction,55)],
 'target_0_909091_ruled_out':bool(upper_fraction<mp.mpf('0.909091')),
 'kernel_jet_order':M,'kernel_jet_precision_bits':jet['precision_bits'],
 'maclaurin_tail_mass_upper':mp.nstr(endpoints(tail_mass)[1],55),
 'maximum_tail_projection_sup_r1_to_r94':mp.nstr(max(mp.mpf(x.numerator)/x.denominator for x in tail_sup[1:]),55),
 'trust_note':'Exact Fraction algebra; direct-C MPFR directed order-511 jet; mpmath.iv endpoint recombination. Human review still required.'
}
open(ROOT/'PROMPT1_DEGREE47_DUAL_VERIFICATION.json','w').write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
