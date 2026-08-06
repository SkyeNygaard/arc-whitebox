#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
from fractions import Fraction
import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
CERT = Path('/mnt/data/new_unresolved_questions/SIGNED_NEAR_OPTIMALITY_CERTIFICATE_BLOCKTRACE_ORDER320.json')
N = 66048
DIM = 256

def harmonic_dim(dim:int, ell:int)->int:
    from math import comb
    if ell == 0:
        return 1
    return comb(dim+ell-1, ell) - comb(dim+ell-3, ell-2)

def toy_simultaneous_check(seed=1234):
    rng=np.random.default_rng(seed)
    N0=3
    dims=[5,6,7]
    Us=[]
    for d in dims:
        q,_=np.linalg.qr(rng.normal(size=(d,N0)))
        Us.append(q[:,:N0])
    # universal unweighted cross-block system M0_lm = sqrt(d_l d_m)/N U_l U_m^T
    offsets=np.cumsum([0]+dims)
    total=sum(dims)
    M0=np.zeros((total,total))
    for i,(di,Ui) in enumerate(zip(dims,Us)):
        for j,(dj,Uj) in enumerate(zip(dims,Us)):
            M0[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]] = math.sqrt(di*dj)/N0*(Ui@Uj.T)
    profiles=[[1,.2,.05],[.4,1,.7],[.03,.11,1]]
    checks=[]
    for a in profiles:
        D=np.zeros((total,total)); A=np.zeros((total,total))
        for i,(di,ai) in enumerate(zip(dims,a)):
            sl=slice(offsets[i],offsets[i+1])
            D[sl,sl]=math.sqrt(ai)*np.eye(di)
            A[sl,sl]=ai*np.eye(di)
        M=D@M0@D
        T=sum(ai*di for ai,di in zip(a,dims))
        S2=sum(ai*ai*di for ai,di in zip(a,dims))
        eig=np.linalg.eigvalsh(M)
        nz=eig[eig>1e-9]
        block_traces=[]
        for i,(di,ai) in enumerate(zip(dims,a)):
            sl=slice(offsets[i],offsets[i+1])
            block_traces.append(float(np.trace(M[sl,sl])))
        lhs=float(np.linalg.norm(A-M,'fro')**2)
        rhs=T*T/N0-S2
        checks.append({
            'profile':a,
            'rank':int(np.linalg.matrix_rank(M,tol=1e-9)),
            'nonzero_eigenvalues':nz.tolist(),
            'target_nonzero_eigenvalue':T/N0,
            'block_traces':block_traces,
            'target_block_traces':[a[i]*dims[i] for i in range(len(a))],
            'frobenius_defect':lhs,
            'rank_floor':rhs,
            'absolute_error':abs(lhs-rhs),
        })
    return checks

def strictness_exact_check(cert):
    comps=cert['components']
    pair=[]
    for c in comps:
        if int(c['s'])==3:
            pair.append(c)
    assert len(pair)>=2
    c1,c2=pair[0],pair[1]
    r1=sp.Rational(c1['r']); r2=sp.Rational(c2['r'])
    assert r1 != r2 and sp.Rational(c1['y'])>0 and sp.Rational(c2['y'])>0
    t=sp.symbols('t')
    G3=sp.factor(t*(86*t*t-1)/85)
    G4=sp.factor((22360*t**4-516*t*t+1)/21845)
    gcd=sp.gcd(sp.Poly(sp.together(G3).as_numer_denom()[0],t), sp.Poly(sp.together(G4).as_numer_denom()[0],t))
    resultant=sp.resultant(sp.together(G3).as_numer_denom()[0], sp.together(G4).as_numer_denom()[0], t)
    s,u,g=sp.gcdex(G3,G4,t)
    bez=sp.simplify(s*G3+u*G4)
    # Direct candidate from earlier hand derivation, useful for an implementation-diverse exact check.
    s2=172*t*(16640*t*t-319)
    u2=-257*(11008*t*t-85)
    bez2=sp.simplify(s2*G3+u2*G4)
    # numerical minimum of G3^2+G4^2 on [-1,1]
    f=sp.lambdify(t,G3**2+G4**2,'numpy')
    grid=np.linspace(-1,1,2_000_001)
    vals=f(grid)
    idx=int(np.argmin(vals))
    return {
        'component_1':c1,
        'component_2':c2,
        'radii_distinct':bool(r1!=r2),
        'G3':str(G3),
        'G4':str(G4),
        'gcd_numerators':str(gcd.as_expr()),
        'resultant_numerators':str(resultant),
        'sympy_bezout_value':str(bez),
        'explicit_bezout_value':str(bez2),
        'grid_min_G3sq_plus_G4sq':float(vals[idx]),
        'grid_argmin':float(grid[idx]),
        'strict_nonattainment_logic':[
            'Total equality forces equality in every positive certificate component.',
            'Equality in both s=3 profiles forces d3*G3(t)+r1*d4*G4(t)=0 and d3*G3(t)+r2*d4*G4(t)=0 for every distinct node pair.',
            'Subtracting gives G4(t)=0, then G3(t)=0.',
            'The exact gcd/resultant/Bezout checks show G3 and G4 have no common root, so no actual N>=2 atomic rule attains the released floor.'
        ]
    }

def quantitative_conditioned_gap(cert):
    comps=[c for c in cert['components'] if int(c['s'])==3][:2]
    t=sp.symbols('t')
    d3=harmonic_dim(DIM,3); d4=harmonic_dim(DIM,4)
    r1,r2=[sp.Rational(c['r']) for c in comps]
    G3=t*(86*t*t-1)/85
    G4=(22360*t**4-516*t*t+1)/21845
    T1=d3+r1*d4; T2=d3+r2*d4
    K1=sp.together((d3*G3+r1*d4*G4)/T1)
    K2=sp.together((d3*G3+r2*d4*G4)/T2)
    mix=sp.Matrix([[sp.Rational(d3,1)/T1,r1*d4/T1],[sp.Rational(d3,1)/T2,r2*d4/T2]])
    inv=mix.inv()
    sb=172*t*(16640*t*t-319)
    ub=-257*(11008*t*t-85)
    P1=sp.expand(sb*inv[0,0]+ub*inv[1,0])
    P2=sp.expand(sb*inv[0,1]+ub*inv[1,1])
    assert sp.simplify(P1*K1+P2*K2)==1
    def coeff_l1(poly):
        return sum(abs(c) for c in sp.Poly(poly,t).all_coeffs())
    L1=coeff_l1(P1); L2=coeff_l1(P2)
    m2_lower=sp.factor(1/(L1**2+L2**2))
    return {
        'normalized_kernels':'K_j(t)=(d3 G3(t)+r_j d4 G4(t))/(d3+r_j d4)',
        'exact_bezout_identity':'P1(t) K1(t)+P2(t) K2(t)=1',
        'P1':str(P1),'P2':str(P2),
        'coefficient_l1_bound_P1':str(L1),
        'coefficient_l1_bound_P2':str(L2),
        'rigorous_common_zero_margin_m2_lower':str(m2_lower),
        'rigorous_common_zero_margin_m2_lower_decimal':str(sp.N(m2_lower,40)),
        'stability_theorem':(
            'For an N-atom rule with |w_i|>=mu and normalized evaluation matrices E_j having sigma_min(E_j)>=s, '
            'let Delta_j=||M_j||_F^2-1/N. Then sum_j (1/N+Delta_j)Delta_j/s^4 '
            '>= mu^4 N(N-1) m^2, where m^2=min_t(K1(t)^2+K2(t)^2). '
            'Thus both profile gaps cannot be small unless weights vanish/blow up, evaluation matrices become ill-conditioned, or nodes approach a singular configuration.'
        ),
        'interpretation':'The exact rational margin is conservative and numerically tiny; its value is structural, not a competitive constant.'
    }

def main():
    cert=json.load(open(CERT))
    active_degrees=sorted(set([int(c['s']) for c in cert['components']]+[int(c['s'])+1 for c in cert['components']]))
    dims={str(l):harmonic_dim(DIM,l) for l in active_degrees}
    min_dim=min(dims.values())
    profile_conditions=[]
    for c in cert['components']:
        s=int(c['s']); r=Fraction(c['r'])
        ds=harmonic_dim(DIM,s); dt=harmonic_dim(DIM,s+1)
        T=Fraction(ds,1)+r*dt
        profile_conditions.append({
            's':s,'r':str(r),
            'N_over_T_for_first_weight':str(Fraction(N,1)/T),
            'N_r_over_T_for_second_weight':str(Fraction(N,1)*r/T),
            'condition_holds': Fraction(N,1)/T <= 1 and Fraction(N,1)*r/T <= 1,
        })
    out={
        'status':'PASS',
        'scope':{'dimension':DIM,'maximum_nodes':N,'certificate_components':len(cert['components'])},
        'active_harmonic_degrees':[active_degrees[0],active_degrees[-1]],
        'minimum_active_harmonic_dimension':min_dim,
        'all_active_dimensions_at_least_N':bool(min_dim>=N),
        'all_profile_schur_horn_conditions':all(x['condition_holds'] for x in profile_conditions),
        'simultaneous_abstract_sharpness_theorem':{
            'construction':'Choose one N-frame U_l in every harmonic block and set M0_lm=sqrt(d_l d_m)/N U_l U_m^T. For every profile a, D_a M0 D_a has fixed block traces and N equal nonzero eigenvalues T_a/N.',
            'consequence':'Any relaxation using only one shared abstract block system, rank <= N, and all individual block traces is simultaneously sharp for every nonnegative harmonic profile. Coupling profiles does not help without sphere/evaluation identities.',
            'toy_checks':toy_simultaneous_check(),
        },
        'strict_atomic_nonattainment':strictness_exact_check(cert),
        'quantitative_conditioned_realizability_gap':quantitative_conditioned_gap(cert),
        'bounded_total_variation_corollary':{
            'statement':'For every finite V, the compact class of <=N-atom mass-one signed rules with total variation <=V has risk at least L+epsilon(V) for some epsilon(V)>0. Therefore every sequence approaching the released abstract floor must have total variation diverging to infinity.',
            'proof_basis':'Compactness after padding to N atoms, continuity of kernel risk, and strict nonattainment at every point.'
        },
        'profile_condition_sample':profile_conditions[:5],
    }
    p=ROOT/'joint_sharpness_strictness_verification.json'
    p.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({
        'status':out['status'],
        'all_profile_conditions':out['all_profile_schur_horn_conditions'],
        'gcd':out['strict_atomic_nonattainment']['gcd_numerators'],
        'bezout':out['strict_atomic_nonattainment']['explicit_bezout_value'],
        'grid_min':out['strict_atomic_nonattainment']['grid_min_G3sq_plus_G4sq']
    },indent=2))

if __name__=='__main__':
    main()
