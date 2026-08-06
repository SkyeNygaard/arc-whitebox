#!/usr/bin/env python3
from __future__ import annotations

import json, sys
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE/'arc_cubature_proof_v5_1'
if not BASE.exists():
    BASE=Path('/mnt/data/work/proofv5/arc_cubature_proof_v5')
sys.path.insert(0,str(BASE))
from formal_interval_certificate import (
    Directed,I,pi_bounds,kappa_pair_interval,deep_kernel_and_prime_fraction,
    gegenbauer_fraction_polynomials,polynomial_interval_decimal,derivative
)

D=256; DEPTH=32; N=66048; PREC=80

def P(x:Fraction)->Fraction:
    return 22102*x**3+21930*x**2-87*x-85

def refine_root(a:Fraction,b:Fraction,bits:int=220)->tuple[Fraction,Fraction]:
    fa=P(a); fb=P(b)
    assert fa*fb<0
    for _ in range(bits):
        m=(a+b)/2; fm=P(m)
        if fm==0:return m,m
        if fa*fm<0:b,fb=m,fm
        else:a,fa=m,fm
    return a,b

def abs_hi(x:I)->Decimal:return max(abs(x.lo),abs(x.hi))

def point_dec(dr:Directed,s:str)->I:
    f=Fraction(s); return dr.frac_interval(f)

def matmul(A,B,dr):
    n=len(A); k=len(A[0]); m=len(B[0])
    out=[[I(Decimal(0),Decimal(0)) for _ in range(m)] for __ in range(n)]
    for i in range(n):
        for j in range(m):
            z=I(Decimal(0),Decimal(0))
            for q in range(k):z=dr.add(z,dr.mul(A[i][q],B[q][j]))
            out[i][j]=z
    return out

def matsub(A,B,dr):return [[dr.sub(A[i][j],B[i][j]) for j in range(len(A[0]))] for i in range(len(A))]

def vec_as_col(v):return [[x] for x in v]

def col_as_vec(v):return [x[0] for x in v]

def kpp_interval(y:I,dr:Directed,pi:I)->I:
    one=dr.integer(1)
    s=dr.sqrt(dr.sub(one,dr.mul(y,y)))
    return dr.div(one,dr.mul(pi,s))

def prove_outer_log_derivative(dr:Directed,pi:I):
    # For F=kappa^31, prove 0 <= F''/F' < 3 on [0,0.319].
    upper=Fraction(319,1000); parts=1
    maxR=Decimal(0); worst=None
    for i in range(parts):
        a=upper*i/parts;b=upper*(i+1)/parts
        y=I(dr.Dlo(a),dr.Dhi(b)); deriv=dr.integer(1); R=dr.integer(0)
        for _ in range(31):
            yn,kp=kappa_pair_interval(y,dr,pi)
            kpp=kpp_interval(y,dr,pi)
            R=dr.add(R,dr.mul(dr.div(kpp,kp),deriv))
            deriv=dr.mul(deriv,kp);y=yn
        if R.hi>maxR:maxR=R.hi;worst=(a,b,R)
    assert maxR<Decimal(3),(maxR,worst)
    return {'domain':['0','319/1000'],'parts':parts,'max_upper':str(maxR),
            'worst_interval':[str(worst[0]),str(worst[1])],
            'proof':'directed interval propagation of y, F_prime, and F_second/F_prime through 31 kappa compositions'}

def asin_phi_interval(tlo:Fraction,thi:Fraction,dr:Directed,pi:I)->I:
    # phi=pi/2+asin(t), monotone. Recover asin from kappa prime: kappa'=1/2+asin/pi.
    _,plo=kappa_pair_interval(dr.frac_interval(tlo),dr,pi)
    _,phi=kappa_pair_interval(dr.frac_interval(thi),dr,pi)
    kp=I(plo.lo,phi.hi)
    return dr.mul(pi, kp)

def prove_u6_plus_3B2(dr:Directed,pi:I):
    # H(t)>0 where pi^2 s^9 (u6+3B2)=3 H(t), t in (-1,0).
    # c=-t, s=sqrt(1-t^2), phi=pi/2+asin(t).
    parts=20; minlo=None; worst=None
    for i in range(parts):
        a=Fraction(-1)+Fraction(i,parts);b=Fraction(-1)+Fraction(i+1,parts)
        T=I(dr.Dlo(a),dr.Dhi(b)); c=dr.neg(T)
        # s is monotone increasing on [-1,0]
        sa=dr.sqrt(dr.sub(dr.integer(1),dr.mul(dr.frac_interval(a),dr.frac_interval(a))))
        sb=dr.sqrt(dr.sub(dr.integer(1),dr.mul(dr.frac_interval(b),dr.frac_interval(b))))
        s=I(sa.lo,sb.hi)
        phi=asin_phi_interval(a,b,dr,pi)
        c2=dr.mul(c,c);c4=dr.mul(c2,c2);s2=dr.mul(s,s);s3=dr.mul(s2,s)
        term1=dr.mul(pi,dr.add(dr.integer(3),dr.add(dr.mul(dr.integer(24),c2),dr.mul(dr.integer(8),c4))))
        term2=dr.mul(dr.integer(18),dr.mul(phi,dr.mul(c,dr.mul(dr.add(dr.integer(3),dr.mul(dr.integer(2),c2)),s2))))
        term3=dr.mul(dr.add(dr.integer(15),dr.mul(dr.integer(40),c2)),s3)
        H=dr.add(dr.sub(term1,term2),term3)
        if minlo is None or H.lo<minlo:minlo=H.lo;worst=(a,b,H)
        assert H.lo>0,(a,b,H)
    return {'domain':['-1','0'],'parts':parts,'minimum_lower':str(minlo),
            'worst_interval':[str(worst[0]),str(worst[1])],
            'identity':'pi^2*(1-t^2)^(9/2)*(kappa^(6)+3*B_{6,2}) = 3*H(t)',
            'proof':'directed interval certificate H>0'}

def root_and_weight_data():
    roots=[(Fraction(-992278935,10**9),Fraction(-992278934,10**9)),
           (Fraction(-62224856,10**9),Fraction(-62224855,10**9)),
           (Fraction(62285891,10**9),Fraction(62285892,10**9))]
    roots=[refine_root(*x) for x in roots]
    return roots

def coefficient_certificate(dr:Directed,pi:I,roots):
    G=gegenbauer_fraction_polynomials(D,5); Gp=[derivative(g) for g in G]
    A=[[None]*6 for _ in range(6)]; b=[None]*6
    for j,(lo,hi) in enumerate(roots):
        X=I(dr.Dlo(lo),dr.Dhi(hi))
        for n in range(6):
            A[2*j][n]=polynomial_interval_decimal([dr.frac_interval(z) for z in G[n]],X,dr)
            A[2*j+1][n]=polynomial_interval_decimal([dr.frac_interval(z) for z in Gp[n]],X,dr)
        Klo,Kplo=deep_kernel_and_prime_fraction(lo,DEPTH,dr,pi)
        Khi,Kphi=deep_kernel_and_prime_fraction(hi,DEPTH,dr,pi)
        b[2*j]=I(Klo.lo,Khi.hi);b[2*j+1]=I(Kplo.lo,Kphi.hi)

    # 70-digit point approximations generated independently with mpmath.
    x0s=[
      '0.9747299751309444413666593085802870785923869068234348747228323827800535',
      '0.00279647306154118416616586023526018213016938536334332686804673875449754',
      '0.00243629527371522242447068060976310829567257873525442749320203262172687',
      '0.00180373485519710060891233424000157672203071189874102965019266506169427',
      '0.00103172848676742614815821374777678526714203832837998423416094757916937',
      '0.000179898923463644585494486989098646638530471586830393993221578851751660',
    ]
    # Approximate inverse of the midpoint Hermite matrix, generated below if cache absent.
    import mpmath as mp
    mp.mp.dps=100
    mids=[mp.mpf(((lo+hi)/2).numerator)/mp.mpf(((lo+hi)/2).denominator) for lo,hi in roots]
    def gpoly(n,x):
        return sum(mp.mpf(z.numerator)/z.denominator*x**k for k,z in enumerate(G[n]))
    def gdpoly(n,x):
        return sum(mp.mpf(z.numerator)/z.denominator*x**k for k,z in enumerate(Gp[n]))
    Am=mp.matrix(6,6)
    for j,x in enumerate(mids):
        for n in range(6):Am[2*j,n]=gpoly(n,x);Am[2*j+1,n]=gdpoly(n,x)
    Rm=Am**-1
    R=[[dr.frac_interval(Fraction(mp.nstr(Rm[i,j],85))) for j in range(6)] for i in range(6)]
    x0=[dr.frac_interval(Fraction(s)) for s in x0s]
    # z=R(b-Ax0), E=I-RA
    Ax=col_as_vec(matmul(A,vec_as_col(x0),dr));res=[dr.sub(b[i],Ax[i]) for i in range(6)]
    z=col_as_vec(matmul(R,vec_as_col(res),dr))
    RA=matmul(R,A,dr)
    Id=[[dr.integer(1 if i==j else 0) for j in range(6)] for i in range(6)]
    E=matsub(Id,RA,dr)
    rho=max(sum(abs_hi(E[i][j]) for j in range(6)) for i in range(6))
    zmax=max(abs_hi(v) for v in z)
    with localcontext(dr.up):err=zmax/(Decimal(1)-rho)
    lower=[];upper=[]
    for v in x0:
        with localcontext(dr.down):lower.append(v.lo-err)
        with localcontext(dr.up):upper.append(v.hi+err)
    assert rho<1
    assert all(lower[i]>0 for i in range(1,6)),(rho,err,lower)
    # Objective Phi=1/N+(1-1/N)c0-(1/N)sum_{l=1}^5 c_l.
    invN=dr.frac_interval(Fraction(1,N));q0=dr.frac_interval(Fraction(N-1,N))
    obj=dr.add(invN,dr.mul(q0,I(lower[0],upper[0])))
    for i in range(1,6): obj=dr.sub(obj,dr.mul(invN,I(lower[i],upper[i])))
    # Subtract the independently certified spherical kernel mean.
    km=json.loads((BASE/'results/FORMAL_KERNEL_MEAN_D256_L32.json').read_text())
    a0=I(Decimal(km['A0_certified']['lower']),Decimal(km['A0_certified']['upper']))
    mse=dr.sub(obj,a0)
    th=json.loads((BASE/'results/FORMAL_NEAR_OPTIMALITY_THEOREM_D256_L32.json').read_text())
    kmse=I(Decimal(th['kerdock_mse_interval']['lower']),Decimal(th['kerdock_mse_interval']['upper']))
    ratio=dr.div(kmse,mse)
    excess=dr.sub(ratio,dr.integer(1)); percent=dr.mul(excess,dr.integer(100))
    return {'root_intervals':[[str(a),str(b)] for a,b in roots],
            'contraction_norm_upper':str(rho),'coefficient_error_upper':str(err),
            'coefficient_intervals':[[str(lower[i]),str(upper[i])] for i in range(6)],
            'positive_nonconstant_coefficients':True,
            'optimal_energy_interval':[str(obj.lo),str(obj.hi)],
            'optimal_mse_interval':[str(mse.lo),str(mse.hi)],
            'kerdock_over_auxiliary_optimum_ratio_interval':[str(ratio.lo),str(ratio.hi)],
            'kerdock_relative_excess_percent_interval':[str(percent.lo),str(percent.hi)]}

def main():
    dr=Directed(PREC);plo,phi=pi_bounds(PREC+40);pi=I(dr.Dlo(plo),dr.Dhi(phi))
    assert dr.div(dr.integer(1),pi).hi < Decimal('0.319')
    roots=root_and_weight_data()
    outer=prove_outer_log_derivative(dr,pi)
    hpos=prove_u6_plus_3B2(dr,pi)
    coeff=coefficient_certificate(dr,pi,roots)
    out={
      'claim':'The degree-5 Hermite interpolant of K_32 at the exact T16 dual nodes is a feasible primal optimizer; together with all-degree reduced-cost negativity this proves full finite-polynomial auxiliary-LP optimality.',
      'dimension':D,'depth':DEPTH,'N':N,'precision':PREC,
      'orthogonal_cubic':'22102*t^3+21930*t^2-87*t-85',
      'outer_log_derivative_certificate':outer,
      'kappa_sixth_combination_certificate':hpos,
      'hermite_coefficient_certificate':coeff,
      'analytic_lemmas':[
        'For t>=0 all kappa derivatives through order 6 and all outer F derivatives are nonnegative.',
        'For t<0, Bell terms B_{6,3}, B_{6,4}, B_{6,5}, B_{6,6} are positive; if B_{6,2}<0, F_prime*(kappa6+(F_second/F_prime)*B_{6,2}) >= F_prime*(kappa6+3*B_{6,2})>0.',
        'Therefore K_32^(6)(t)>0 on (-1,1).',
        'Hermite interpolation error gives K_32(t)-h(t)=K_32^(6)(xi)/6!*product_j(t-t_j)^2 >=0.',
        'Exact dual moment equalities through degree 5 and exact contact imply primal objective equals dual objective.',
        'Strict reduced-cost negativity for every degree >=6 forces every all-degree finite-polynomial optimizer to have zero higher coefficients.'
      ],
      'status':'COMPUTER-ASSISTED CERTIFIED, subject to the stated Decimal/libmpdec trust base',
    }
    p=Path(__file__).with_name('T16_PRIMAL_DUAL_CERTIFICATE.json');p.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
