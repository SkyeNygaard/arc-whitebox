from __future__ import annotations
import json
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from formal_interval_certificate import (
    Directed,I,pi_bounds,auxiliary_monomials,derivative,bernstein_on_minus1_1,
    deep_kernel_and_prime_fraction,kappa_pair_interval,polynomial_interval_decimal,
    polyval_fraction
)


def kappa_second_interval(x:I,dr:Directed,pi:I)->I:
    # kappa''(x)=1/(pi*sqrt(1-x^2)); positive and increasing with |x|.
    maxabs=max(x.lo.copy_abs(),x.hi.copy_abs())
    if maxabs >= 1:
        raise ValueError('endpoint singularity')
    minabs=Decimal(0) if x.lo<=0<=x.hi else min(x.lo.copy_abs(),x.hi.copy_abs())
    minarg=dr.sub(dr.integer(1),dr.mul(dr.point(minabs),dr.point(minabs)))
    maxarg=dr.sub(dr.integer(1),dr.mul(dr.point(maxabs),dr.point(maxabs)))
    # denominator is largest at minabs, smallest at maxabs.
    den_hi=dr.mul(pi,dr.sqrt(minarg))
    den_lo=dr.mul(pi,dr.sqrt(maxarg))
    return dr.div(dr.integer(1),I(den_lo.lo,den_hi.hi))


def deep_second_interval(a:Fraction,b:Fraction,depth:int,dr:Directed,pi:I)->I:
    z=I(dr.Dlo(a),dr.Dhi(b))
    p=dr.integer(1); q=I(Decimal(0),Decimal(0))
    for _ in range(depth):
        kp2=kappa_second_interval(z,dr,pi)
        znew,kp=kappa_pair_interval(z,dr,pi)
        qnew=dr.add(dr.mul(kp2,dr.mul(p,p)),dr.mul(kp,q))
        p=dr.mul(kp,p); q=qnew; z=znew
    return q


def prove(coeff_file,roots_file,d=256,depth=32,prec=70,root_radius='1e-9',end_radius='1e-8'):
    dr=Directed(prec)
    pl,ph=pi_bounds(prec+30); pi=I(dr.Dlo(pl),dr.Dhi(ph))
    coeff,h=auxiliary_monomials(coeff_file,d)
    hp=derivative(h); hpp=derivative(hp)
    bern=bernstein_on_minus1_1(hp)
    assert all(x>0 for x in bern)
    assert all(x>=0 for x in coeff[1:])
    hI=[dr.frac_interval(x) for x in h]
    hpI=[dr.frac_interval(x) for x in hp]
    hppI=[dr.frac_interval(x) for x in hpp]

    roots=json.loads(Path(roots_file).read_text())['roots']
    centers=[(Fraction(r['left'])+Fraction(r['right']))/2 for r in roots]
    rr=Fraction(root_radius); er=Fraction(end_radius)

    @lru_cache(maxsize=None)
    def state(x:Fraction):
        return deep_kernel_and_prime_fraction(x,depth,dr,pi)

    def hpoint(polyI,x):
        X=I(dr.Dlo(x),dr.Dhi(x))
        return polynomial_interval_decimal(polyI,X,dr)

    # Direct boxes at endpoints and around every stationary point.
    direct=[(Fraction(-1),Fraction(-1)+er,'left_endpoint_box')]
    for c in centers: direct.append((c-rr,c+rr,'stationary_box'))
    direct.append((Fraction(1)-er,Fraction(1),'right_endpoint_box'))
    direct.sort()
    regions=[]
    for a,b,k in direct: regions.append((a,b,k,0))
    # Add derivative regions between direct boxes.
    for (_,b,_),(a,_,_) in zip(direct[:-1],direct[1:]):
        regions.append((b,a,'derivative_region',0))

    accepted=[]; work=regions[:]; processed=0; largest=None
    while work:
        a,b,kind,lev=work.pop(); processed+=1
        if not a<b: continue
        if kind!='derivative_region':
            Ka,_=state(a)
            hb=hpoint(hI,b).hi
            with localcontext(dr.up): ub=hb-Ka.lo
            if ub>=0:
                raise RuntimeError(f'direct box failed {float(a),float(b)} ub={ub}')
            sign=kind
        else:
            m=(a+b)/2
            _,Kpm=state(m)
            hpm=hpoint(hpI,m)
            gp=dr.sub(hpm,Kpm)
            # Rigorous |g''| bound from h'' and K''.
            X=I(dr.Dlo(a),dr.Dhi(b))
            hppR=polynomial_interval_decimal(hppI,X,dr)
            try:
                KppR=deep_second_interval(a,b,depth,dr,pi)
            except ValueError:
                if lev>=40: raise
                mid=(a+b)/2
                work += [(a,mid,kind,lev+1),(mid,b,kind,lev+1)]
                continue
            gpp=dr.sub(hppR,KppR)
            M=max(gpp.lo.copy_abs(),gpp.hi.copy_abs())
            with localcontext(dr.up): rad=M*dr.Dhi((b-a)/2)
            
            with localcontext(dr.down): gp_lo=gp.lo-rad
            with localcontext(dr.up): gp_hi=gp.hi+rad
            if gp_lo>0:
                sign='positive'; Kb,_=state(b); ub=hpoint(hI,b).hi-Kb.lo
            elif gp_hi<0:
                sign='negative'; Ka,_=state(a); ub=hpoint(hI,a).hi-Ka.lo
            else:
                if lev>=35:
                    raise RuntimeError(f'mean value sign failed {float(a),float(b)} gp={gp} M={M}')
                mid=(a+b)/2
                work += [(a,mid,kind,lev+1),(mid,b,kind,lev+1)]
                continue
            if ub>=0:
                raise RuntimeError(f'derivative region inequality failed {float(a),float(b)} ub={ub}')
        row={'left':str(a),'right':str(b),'kind':sign,'upper_bound':str(ub),'level':lev}
        accepted.append(row)
        if largest is None or ub>largest[0]:largest=(ub,a,b)

    return {
      'dimension':d,'depth':depth,'coefficient_file':str(coeff_file),'precision':prec,
      'pi_lower':str(pi.lo),'pi_upper':str(pi.hi),
      'gegenbauer_coefficients_nonnegative':all(x>=0 for x in coeff[1:]),
      'hprime_bernstein_coefficients':[str(x) for x in bern],
      'h_strictly_increasing':all(x>0 for x in bern),
      'root_radius':str(rr),'endpoint_radius':str(er),
      'processed_intervals':processed,'accepted_intervals':len(accepted),
      'unique_point_evaluations':state.cache_info().currsize,
      'global_upper_bound':str(largest[0]),
      'global_upper_interval':[str(largest[1]),str(largest[2])],
      'intervals':sorted(accepted,key=lambda r:Fraction(r['left'])),
      'passed':True,
    }

if __name__=='__main__':
    base=Path(__file__).resolve().parent
    out=prove(base/'auxiliary_coefficients_d256_L32_deg5.json',base/'stationary_point_hints_d256_L32.json')
    path=base/'results/formal_meanvalue_certificate_d256_L32.json'
    path.write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
