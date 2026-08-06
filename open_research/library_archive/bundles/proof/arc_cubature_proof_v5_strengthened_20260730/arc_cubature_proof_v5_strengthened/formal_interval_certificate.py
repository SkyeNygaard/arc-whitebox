from __future__ import annotations

"""Directed-rounding certificate for h(t) <= K_depth(t) on [-1,1].

This script intentionally avoids mpmath interval arithmetic.  It uses:
  * exact Fraction arithmetic for the auxiliary polynomial and its Bernstein proof;
  * exact rational Machin-series bounds for pi;
  * Decimal arithmetic with explicit ROUND_FLOOR / ROUND_CEILING;
  * a positive-term asin series with a geometric tail bound;
  * monotonicity of both h and the deep ReLU kernel.

The result is a self-contained computer-assisted proof relative only to Python's
specified Decimal directed-rounding semantics.
"""

import argparse
import json
from dataclasses import dataclass, asdict
from decimal import Decimal, Context, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction
from functools import lru_cache
from math import comb
from pathlib import Path


@dataclass(frozen=True)
class I:
    lo: Decimal
    hi: Decimal
    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError((self.lo, self.hi))


class Directed:
    def __init__(self, prec: int = 120):
        self.prec = prec
        self.down = Context(prec=prec, rounding=ROUND_FLOOR, Emin=-999999999, Emax=999999999)
        self.up = Context(prec=prec, rounding=ROUND_CEILING, Emin=-999999999, Emax=999999999)

    def Dlo(self, f: Fraction) -> Decimal:
        with localcontext(self.down):
            return Decimal(f.numerator) / Decimal(f.denominator)

    def Dhi(self, f: Fraction) -> Decimal:
        with localcontext(self.up):
            return Decimal(f.numerator) / Decimal(f.denominator)

    def frac_interval(self, f: Fraction) -> I:
        return I(self.Dlo(f), self.Dhi(f))

    def point(self, x: Decimal) -> I:
        return I(x, x)

    def add(self, a: I, b: I) -> I:
        with localcontext(self.down): lo = a.lo + b.lo
        with localcontext(self.up): hi = a.hi + b.hi
        return I(lo, hi)

    def neg(self, a: I) -> I:
        # Decimal unary minus applies the active context and can silently round.
        # copy_negate() is an exact sign-bit operation.
        return I(a.hi.copy_negate(), a.lo.copy_negate())

    def sub(self, a: I, b: I) -> I:
        return self.add(a, self.neg(b))

    def mul(self, a: I, b: I) -> I:
        lows=[]; highs=[]
        for x in (a.lo,a.hi):
            for y in (b.lo,b.hi):
                with localcontext(self.down): lows.append(x*y)
                with localcontext(self.up): highs.append(x*y)
        return I(min(lows), max(highs))

    def div(self, a: I, b: I) -> I:
        if b.lo <= 0 <= b.hi:
            raise ZeroDivisionError(b)
        lows=[]; highs=[]
        for x in (a.lo,a.hi):
            for y in (b.lo,b.hi):
                with localcontext(self.down): lows.append(x/y)
                with localcontext(self.up): highs.append(x/y)
        return I(min(lows), max(highs))

    def sqrt(self, a: I) -> I:
        """Outward-rounded square root with exact rational verification.

        Decimal.sqrt() is correctly rounded using ROUND_HALF_EVEN rather than
        respecting a directed-rounding context.  We therefore compute a guard-
        precision approximation, round it to the target format, and verify the
        endpoints by exact Fraction comparisons.
        """
        if a.lo < 0:
            raise ValueError(a)
        if a.hi == 0:
            z = Decimal(0)
            return I(z, z)

        work = Context(prec=self.prec + 25, rounding=ROUND_FLOOR,
                       Emin=-999999999, Emax=999999999)

        def lower_sqrt(x: Decimal) -> Decimal:
            if x == 0:
                return Decimal(0)
            r = x.sqrt(context=work)
            q = self.down.plus(r)
            # One outward ulp makes the starting enclosure independent of how
            # sqrt rounded internally; exact checks below are authoritative.
            q = q.next_minus(context=self.down)
            fx = Fraction(x)
            while Fraction(q) * Fraction(q) > fx:
                q = q.next_minus(context=self.down)
            while True:
                qn = q.next_plus(context=self.down)
                if Fraction(qn) * Fraction(qn) <= fx:
                    q = qn
                else:
                    break
            return q

        def upper_sqrt(x: Decimal) -> Decimal:
            if x == 0:
                return Decimal(0)
            r = x.sqrt(context=work)
            q = self.up.plus(r)
            q = q.next_plus(context=self.up)
            fx = Fraction(x)
            while Fraction(q) * Fraction(q) < fx:
                q = q.next_plus(context=self.up)
            while True:
                qp = q.next_minus(context=self.up)
                if Fraction(qp) * Fraction(qp) >= fx:
                    q = qp
                else:
                    break
            return q

        return I(lower_sqrt(a.lo), upper_sqrt(a.hi))

    def integer(self,n:int)->I:
        d=Decimal(n); return I(d,d)


def atan_invq_bounds(q:int, digits:int=180) -> tuple[Fraction,Fraction]:
    """Exact alternating-series bounds for atan(1/q)."""
    s=Fraction(0)
    n=0
    target=Fraction(1,10**digits)
    while True:
        term=Fraction(1,(2*n+1)*q**(2*n+1))
        s = s + term if n%2==0 else s-term
        nxt=Fraction(1,(2*(n+1)+1)*q**(2*(n+1)+1))
        if nxt < target:
            # Partial sum ending at even index is above, odd index is below.
            if n%2==0:
                return s-nxt, s
            return s, s+nxt
        n+=1


def pi_bounds(digits:int=180)->tuple[Fraction,Fraction]:
    a5=atan_invq_bounds(5,digits+5)
    a239=atan_invq_bounds(239,digits+5)
    # pi = 16*a5 - 4*a239
    lo=16*a5[0]-4*a239[1]
    hi=16*a5[1]-4*a239[0]
    return lo,hi


def asin_series_scalar(x: Decimal, dr: Directed) -> I:
    """Rigorous asin(x) for exact scalar x in [0,~0.708]."""
    if x < 0 or x > Decimal('0.708'):
        raise ValueError(x)
    X=dr.point(x)
    x2=dr.mul(X,X)
    term=X
    total=X
    n=0
    threshold=Decimal(1).scaleb(-(dr.prec+15))
    while True:
        # term_{n+1}/term_n = x^2 (2n+1)^2/[2(n+1)(2n+3)]
        rat=Fraction((2*n+1)**2,2*(n+1)*(2*n+3))
        term_next=dr.mul(term,dr.mul(x2,dr.frac_interval(rat)))
        total=dr.add(total,term_next)
        n+=1
        # Remaining ratios are < x^2, so tail after current term <= next/(1-x^2).
        rat2=Fraction((2*n+1)**2,2*(n+1)*(2*n+3))
        nxt=dr.mul(term_next,dr.mul(x2,dr.frac_interval(rat2)))
        if nxt.hi < threshold:
            denom=dr.sub(dr.integer(1),x2)
            tail=dr.div(I(Decimal(0),nxt.hi),denom)
            return I(total.lo,dr.add(total,tail).hi)
        term=term_next
        if n>2000:
            raise RuntimeError('asin series failed')


def asin_scalar(x: Decimal, dr: Directed, pi: I) -> I:
    if x == 0: return I(Decimal(0),Decimal(0))
    if x < 0:
        return dr.neg(asin_scalar(x.copy_negate(),dr,pi))
    if x > 1: raise ValueError(x)
    if x == 1:
        return dr.div(pi,dr.integer(2))
    if x <= Decimal('0.7'):
        return asin_series_scalar(x,dr)
    # asin(x)=pi/2 - 2 asin(sqrt((1-x)/2)); transformed argument <= sqrt(.15).
    y2=dr.div(dr.sub(dr.integer(1),dr.point(x)),dr.integer(2))
    y=dr.sqrt(y2)
    ay_lo=asin_series_scalar(y.lo,dr)
    ay_hi=asin_series_scalar(y.hi,dr)
    ay=I(ay_lo.lo,ay_hi.hi)
    return dr.sub(dr.div(pi,dr.integer(2)),dr.mul(dr.integer(2),ay))


def kappa_scalar(x: Decimal, dr: Directed, pi: I) -> I:
    if x < -1 or x > 1: raise ValueError(x)
    if x == -1: return I(Decimal(0),Decimal(0))
    if x == 1: return I(Decimal(1),Decimal(1))
    X=dr.point(x)
    root=dr.sqrt(dr.sub(dr.integer(1),dr.mul(X,X)))
    ax=asin_scalar(x,dr,pi)
    angle=dr.add(dr.div(pi,dr.integer(2)),ax)
    numerator=dr.add(root,dr.mul(angle,X))
    return dr.div(numerator,pi)


def kappa_interval(x:I,dr:Directed,pi:I)->I:
    # kappa is monotone, so endpoint evaluation is sufficient.
    lo=kappa_scalar(x.lo,dr,pi).lo
    hi=kappa_scalar(x.hi,dr,pi).hi
    return I(lo,hi)


def kappa_pair_scalar(x: Decimal, dr: Directed, pi: I) -> tuple[I,I]:
    if x < -1 or x > 1: raise ValueError(x)
    if x == -1:
        return I(Decimal(0),Decimal(0)), I(Decimal(0),Decimal(0))
    if x == 1:
        return I(Decimal(1),Decimal(1)), I(Decimal(1),Decimal(1))
    X=dr.point(x)
    ax=asin_scalar(x,dr,pi)
    kp=dr.add(dr.frac_interval(Fraction(1,2)),dr.div(ax,pi))
    root=dr.sqrt(dr.sub(dr.integer(1),dr.mul(X,X)))
    angle=dr.add(dr.div(pi,dr.integer(2)),ax)
    numerator=dr.add(root,dr.mul(angle,X))
    return dr.div(numerator,pi),kp

def kappa_pair_interval(x:I,dr:Directed,pi:I)->tuple[I,I]:
    klo,kplo=kappa_pair_scalar(x.lo,dr,pi)
    khi,kphi=kappa_pair_scalar(x.hi,dr,pi)
    return I(klo.lo,khi.hi), I(kplo.lo,kphi.hi)

def kappa_prime_scalar(x: Decimal, dr: Directed, pi: I) -> I:
    return kappa_pair_scalar(x,dr,pi)[1]

def kappa_prime_interval(x:I,dr:Directed,pi:I)->I:
    return kappa_pair_interval(x,dr,pi)[1]

def deep_kernel_and_prime_fraction(x:Fraction,depth:int,dr:Directed,pi:I)->tuple[I,I]:
    z=dr.frac_interval(x)
    p=dr.integer(1)
    for _ in range(depth):
        z,kp=kappa_pair_interval(z,dr,pi)
        p=dr.mul(p,kp)
    return z,p

def deep_kernel_fraction(x:Fraction,depth:int,dr:Directed,pi:I)->I:
    z=dr.frac_interval(x)
    for _ in range(depth):
        z=kappa_interval(z,dr,pi)
    return z


def gegenbauer_fraction_polynomials(d:int,L:int):
    G=[[Fraction(1)]]
    if L==0:return G
    G.append([Fraction(0),Fraction(1)])
    for l in range(1,L):
        A=Fraction(2*l+d-2,l+d-2)
        B=Fraction(l,l+d-2)
        nxt=[Fraction(0)]*(l+2)
        for k,v in enumerate(G[l]):nxt[k+1]+=A*v
        for k,v in enumerate(G[l-1]):nxt[k]-=B*v
        G.append(nxt)
    return G


def auxiliary_monomials(coeff_file:str,d:int):
    """Load the exact rational Gegenbauer witness and convert to monomials.

    The formal proof never reads floating-point coefficient files.  The JSON
    witness stores each coefficient as an exact Fraction string.
    """
    obj=json.loads(Path(coeff_file).read_text())
    if int(obj.get('dimension', d)) != d:
        raise ValueError('coefficient dimension mismatch')
    coeff=[Fraction(x) for x in obj['coefficients']]
    G=gegenbauer_fraction_polynomials(d,len(coeff)-1)
    h=[Fraction(0)]*len(coeff)
    for c,g in zip(coeff,G):
        for k,v in enumerate(g):h[k]+=c*v
    return coeff,h


def polyval_fraction(c:list[Fraction],x:Fraction)->Fraction:
    y=Fraction(0)
    for v in reversed(c):y=y*x+v
    return y


def derivative(c):
    return [Fraction(k)*c[k] for k in range(1,len(c))]


def bernstein_on_minus1_1(power:list[Fraction])->list[Fraction]:
    """Bernstein coefficients of p(2y-1), y in [0,1]."""
    n=len(power)-1
    q=[Fraction(0)]*(n+1)
    for k,a in enumerate(power):
        for j in range(k+1):
            q[j]+=a*comb(k,j)*2**j*(-1)**(k-j)
    b=[]
    for i in range(n+1):
        b.append(sum(q[j]*Fraction(comb(i,j),comb(n,j)) for j in range(i+1)))
    return b


def polynomial_interval_decimal(power_intervals:list[I], x:I, dr:Directed)->I:
    y=I(Decimal(0),Decimal(0))
    for c in reversed(power_intervals):
        y=dr.add(dr.mul(y,x),c)
    return y

def polynomial_bernstein_range_on_interval(power:list[Fraction],a:Fraction,b:Fraction)->tuple[Fraction,Fraction]:
    """Exact Bernstein enclosure of a power-basis polynomial on [a,b]."""
    n=len(power)-1
    width=b-a
    q=[Fraction(0)]*(n+1)
    for k,pk in enumerate(power):
        for j in range(k+1):
            q[j]+=pk*comb(k,j)*(a**(k-j))*(width**j)
    beta=[]
    for i in range(n+1):
        beta.append(sum(q[j]*Fraction(comb(i,j),comb(n,j)) for j in range(i+1)))
    return min(beta),max(beta)

@dataclass
class ProofResult:
    dimension:int
    depth:int
    coefficient_file:str
    decimal_precision:int
    pi_lower:str
    pi_upper:str
    coefficient_nonnegative:bool
    derivative_bernstein_coefficients:list[str]
    polynomial_strictly_increasing:bool
    intervals_processed:int
    intervals_accepted:int
    maximum_depth:int
    narrowest_interval:str
    largest_certified_upper_bound:str
    largest_interval_left:str
    largest_interval_right:str
    passed:bool


def certify(coeff_file:str,d:int,depth:int,prec:int=80,initial_parts:int=32,max_depth:int=90):
    dr=Directed(prec)
    plo,phi=pi_bounds(prec+40)
    pi=I(dr.Dlo(plo),dr.Dhi(phi))
    coeff,h=auxiliary_monomials(coeff_file,d)
    hp=derivative(h)
    bern=bernstein_on_minus1_1(hp)
    increasing=all(x>0 for x in bern)
    if not increasing: raise RuntimeError('h derivative positivity not established')
    if not all(x>=0 for x in coeff[1:]): raise RuntimeError('negative Gegenbauer coefficient')

    @lru_cache(maxsize=None)
    def state(x:Fraction)->tuple[I,I]:
        return deep_kernel_and_prime_fraction(x,depth,dr,pi)

    @lru_cache(maxsize=None)
    def h_value(x:Fraction)->Fraction:
        return polyval_fraction(h,x)

    stack=[]
    for i in range(initial_parts):
        stack.append((Fraction(-1)+Fraction(2*i,initial_parts),Fraction(-1)+Fraction(2*(i+1),initial_parts),0))
    processed=accepted=0
    max_seen_depth=0
    min_width=Fraction(2)
    largest=None
    derivative_positive=derivative_negative=endpoint_boxes=0
    while stack:
        a,b,lev=stack.pop(); processed+=1; max_seen_depth=max(max_seen_depth,lev)
        (Ka,Kpa)=state(a); (Kb,Kpb)=state(b)
        hp_lo,hp_hi=polynomial_bernstein_range_on_interval(hp,a,b)
        hp_lo_d=dr.Dlo(hp_lo); hp_hi_d=dr.Dhi(hp_hi)
        # K' is increasing. Classify the derivative of g=h-K.
        if hp_lo_d > Kpb.hi:
            # g strictly increases: its maximum is g(b).
            with localcontext(dr.up): ub=dr.Dhi(h_value(b))-Kb.lo
            derivative_positive+=1
        elif hp_hi_d < Kpa.lo:
            # g strictly decreases: its maximum is g(a).
            with localcontext(dr.up): ub=dr.Dhi(h_value(a))-Ka.lo
            derivative_negative+=1
        else:
            # Dependency-safe enclosure, using monotonicity of h and K.
            with localcontext(dr.up): ub=dr.Dhi(h_value(b))-Ka.lo
            endpoint_boxes+=1
        if largest is None or ub>largest[0]: largest=(ub,a,b)
        if ub < 0:
            accepted+=1; min_width=min(min_width,b-a); continue
        if lev>=max_depth:
            raise RuntimeError(f'failed interval [{a},{b}], ub={ub}, depth={lev}')
        m=(a+b)/2
        stack.append((a,m,lev+1));stack.append((m,b,lev+1))
    assert largest is not None
    result=ProofResult(
        dimension=d,depth=depth,coefficient_file=coeff_file,decimal_precision=prec,
        pi_lower=str(pi.lo),pi_upper=str(pi.hi),coefficient_nonnegative=all(x>=0 for x in coeff[1:]),
        derivative_bernstein_coefficients=[str(x) for x in bern],polynomial_strictly_increasing=increasing,
        intervals_processed=processed,intervals_accepted=accepted,maximum_depth=max_seen_depth,
        narrowest_interval=str(min_width),largest_certified_upper_bound=str(largest[0]),
        largest_interval_left=str(largest[1]),largest_interval_right=str(largest[2]),passed=True)
    # Attach diagnostic counters without changing dataclass schema.
    result.__dict__['derivative_positive_intervals']=derivative_positive
    result.__dict__['derivative_negative_intervals']=derivative_negative
    result.__dict__['endpoint_box_intervals']=endpoint_boxes
    result.__dict__['unique_kernel_endpoints']=state.cache_info().currsize
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('coeff_file')
    p.add_argument('--dimension',type=int,default=256)
    p.add_argument('--depth',type=int,default=32)
    p.add_argument('--precision',type=int,default=120)
    p.add_argument('--initial-parts',type=int,default=64)
    p.add_argument('--max-depth',type=int,default=80)
    p.add_argument('--output',default='results/formal_interval_certificate_d256_L32.json')
    a=p.parse_args()
    r=certify(a.coeff_file,a.dimension,a.depth,a.precision,a.initial_parts,a.max_depth)
    Path(a.output).write_text(json.dumps(asdict(r),indent=2))
    print(json.dumps(asdict(r),indent=2))

if __name__=='__main__':main()
