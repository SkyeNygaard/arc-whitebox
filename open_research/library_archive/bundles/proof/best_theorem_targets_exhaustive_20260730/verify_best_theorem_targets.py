#!/usr/bin/env python3
import json, math
from pathlib import Path
import mpmath as mp
import sympy as sp

mp.mp.dps = 80
iv = mp.iv
iv.dps = 80

d = 256
N = 66048
L = 32
ORDER = 11

t = sp.symbols('t')
kappa_expr = (sp.sqrt(1-t**2) + (sp.pi-sp.acos(t))*t)/sp.pi
# Taylor coefficients kappa^(n)/n!
derivs = [sp.diff(kappa_expr,t,n)/sp.factorial(n) for n in range(ORDER+1)]
mods = [{'sqrt':iv.sqrt, 'acos':lambda z: iv.atan2(iv.sqrt(1-z*z),z), 'pi':iv.pi}]
fders = [sp.lambdify(t,e,modules=mods) for e in derivs]

def z(): return iv.mpf([0,0])
def iconv(a,b):
    c=[z() for _ in range(ORDER+1)]
    for i,ai in enumerate(a):
        for j,bj in enumerate(b):
            if i+j>ORDER: break
            c[i+j] += ai*bj
    return c

def compose(p):
    p0=p[0]
    delta=list(p); delta[0]=z()
    out=[z() for _ in range(ORDER+1)]
    power=[z() for _ in range(ORDER+1)]; power[0]=iv.mpf([1,1])
    for n in range(ORDER+1):
        dn=fders[n](p0)
        for i in range(ORDER+1): out[i] += dn*power[i]
        power=iconv(power,delta)
    return out

jet=[z() for _ in range(ORDER+1)]; jet[1]=iv.mpf([1,1])
for _ in range(L): jet=compose(jet)

def endpoints(x):
    s=str(x)
    lo=mp.mpf(s[1:s.index(',')])
    hi=mp.mpf(s[s.index(',')+1:-1])
    return lo,hi

def moment(k):
    if k%2: return sp.Rational(0)
    m=k//2
    num=sp.Integer(1) if m==0 else sp.factorial2(2*m-1)
    den=sp.Integer(1)
    for j in range(m): den *= d+2*j
    return sp.Rational(num,den)

def harmonic_dim(l):
    if l==0: return 1
    if l==1: return d
    return math.comb(d+l-1,l)-math.comb(d+l-3,l-2)

def G(l):
    p=sp.gegenbauer(l,sp.Rational(d-2,2),t)
    return sp.expand(p/p.subs(t,1))

def expectation(poly):
    return sum(c*moment(mon[0]) for mon,c in sp.Poly(sp.expand(poly),t).terms())

def projection(n,l):
    gl=G(l)
    return sp.factor(expectation(t**n*gl)/expectation(gl**2))

# Certified T16 Hermite coefficients from complete proof package.
h_intervals = [
('0.97472997513094444136665930858028707859238690682343487472283238278005349338757190','0.97472997513094444136665930858028707859238690682343487472283238278005350661242810'),
('0.0027964730615411841661658602352601821301693853633433268680467387544975333875719043','0.0027964730615411841661658602352601821301693853633433268680467387544975466124280957'),
('0.0024362952737152224244706806097631082956725787352544274932020326217268633875719043','0.0024362952737152224244706806097631082956725787352544274932020326217268766124280957'),
('0.0018037348551971006089123342400015767220307118987410296501926650616942633875719043','0.0018037348551971006089123342400015767220307118987410296501926650616942766124280957'),
('0.0010317284867674261481582137477767852671420383283799842341609475791693633875719043','0.0010317284867674261481582137477767852671420383283799842341609475791693766124280957'),
('0.00017989892346364458549448698909864663853047158683039399322157885175165338757190436','0.00017989892346364458549448698909864663853047158683039399322157885175166661242809564')]

coef_checks=[]
K_lower=[]
for l in range(7):
    s=mp.mpf('0')
    terms=[]
    for n in range(l,ORDER+1):
        p=projection(n,l)
        if p==0: continue
        lo,_=endpoints(jet[n])
        val=lo*mp.mpf(int(p.p))/mp.mpf(int(p.q))
        s += val
        terms.append({'power':n,'projection':str(p),'lower_contribution':mp.nstr(val,50)})
    K_lower.append(s)
    row={'degree':l,'K_coefficient_lower_from_degree_11_jet':mp.nstr(s,60),'terms':terms}
    if l<=5:
        hu=mp.mpf(h_intervals[l][1])
        margin=s-hu
        row.update({'h_coefficient_upper':str(hu),'margin_lower':mp.nstr(margin,60),'passes':margin>0})
    coef_checks.append(row)

# Rank obstruction using the full spherical-polynomial feature space through degree 3.
feature_degrees=list(range(4))
D3=sum(harmonic_dim(j) for j in feature_degrees)
L3=sum(sp.Integer(harmonic_dim(j))*G(j) for j in feature_degrees)
b=[]
for l in range(7):
    gl=G(l)
    b.append(sp.factor(expectation(L3**2*gl)/expectation(gl**2)))
active=[2,4,6]
gamma_candidates=[]
for l in active:
    gamma_candidates.append((K_lower[l]/(mp.mpf(int(b[l].p))/mp.mpf(int(b[l].q))),l))
gamma,lstar=min(gamma_candidates)
rank_defect=mp.mpf(D3)*D3/N-D3
signed_lb=gamma*rank_defect
kerdock_mse=mp.mpf('2.433660357543006e-7')

out={
 'settings':{'dimension':d,'nodes':N,'depth':L,'jet_order':ORDER,'interval_dps':80},
 'jet_intervals':{str(i):[str(endpoints(jet[i])[0]),str(endpoints(jet[i])[1])] for i in range(ORDER+1)},
 'coefficient_checks':coef_checks,
 'residual_psd_conclusion': all(r.get('passes',True) for r in coef_checks[:6]),
 'rank_obstruction':{
   'feature_space':'all spherical harmonics through degree 3',
   'dimension_D3':D3,
   'squared_kernel_coefficients':{str(l):str(b[l]) for l in range(7)},
   'active_degrees':active,
   'gamma_lower':mp.nstr(gamma,60),
   'binding_degree':lstar,
   'rank_defect_D2_over_N_minus_D':mp.nstr(rank_defect,60),
   'signed_rule_mse_lower_bound':mp.nstr(signed_lb,60),
   'fraction_of_kerdock_mse_lower_bound':mp.nstr(signed_lb/kerdock_mse,40),
   'maximum_permitted_improvement_factor_vs_kerdock':mp.nstr(kerdock_mse/signed_lb,40)
 }
}
path=Path('/mnt/data/theorem_targets/best_theorem_targets_verification.json')
path.write_text(json.dumps(out,indent=2))
assert out['residual_psd_conclusion']
assert signed_lb>0
print(json.dumps({
 'residual_psd':out['residual_psd_conclusion'],
 'smallest_margin_degree_0_to_5':min(mp.mpf(r['margin_lower']) for r in coef_checks[:6]),
 'signed_rule_lower_bound':out['rank_obstruction']['signed_rule_mse_lower_bound'],
 'fraction_of_kerdock':out['rank_obstruction']['fraction_of_kerdock_mse_lower_bound'],
 'max_improvement_factor':out['rank_obstruction']['maximum_permitted_improvement_factor_vs_kerdock']
},indent=2,default=str))
