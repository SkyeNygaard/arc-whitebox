from __future__ import annotations
import json
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from formal_interval_certificate import (
    Directed,I,pi_bounds,auxiliary_monomials,derivative,bernstein_on_minus1_1,
    deep_kernel_and_prime_fraction,polynomial_interval_decimal
)
from formal_meanvalue_certificate import deep_second_interval

F=lambda x: Fraction(str(x))
GP_RADII=[F('0.005'),F('0.01'),F('0.0005'),F('0.0002'),F('0.0005')]
GPP_CENTERS=list(map(F,[-0.9266064320828018,-0.49813713491761513,-0.03381931654866477,0.03784807384516498]))
GPP_RADII=[F('0.02'),F('0.02'),F('0.003'),F('0.003')]
EXPECTED_GPP=['negative','positive','positive','negative','negative','positive','positive','negative','negative']
RIGHT_CUT=F('0.74')

class Problem:
    def __init__(self,base:Path,prec=55,d=256,depth=32):
        self.base=Path(base);self.prec=prec;self.d=d;self.depth=depth
        self.dr=Directed(prec)
        pl,ph=pi_bounds(prec+30);self.pi=I(self.dr.Dlo(pl),self.dr.Dhi(ph))
        self.coeff,self.h=auxiliary_monomials(str(self.base/'auxiliary_coefficients_d256_L32_deg5.json'),d)
        self.hp=derivative(self.h);self.hpp=derivative(self.hp)
        self.hI=[self.dr.frac_interval(x) for x in self.h]
        self.hpI=[self.dr.frac_interval(x) for x in self.hp]
        self.hppI=[self.dr.frac_interval(x) for x in self.hpp]
        roots=json.loads((self.base/'stationary_point_hints_d256_L32.json').read_text())['roots']
        self.gp_centers=[(Fraction(r['left'])+Fraction(r['right']))/2 for r in roots]
        self.gp_boxes=[(c-r,c+r) for c,r in zip(self.gp_centers,GP_RADII)]
        self.gpp_boxes=[(c-r,c+r) for c,r in zip(GPP_CENTERS,GPP_RADII)]
        self._state=lru_cache(maxsize=None)(self._state_raw)
    def _state_raw(self,x):return deep_kernel_and_prime_fraction(x,self.depth,self.dr,self.pi)
    def state(self,x):return self._state(x)
    def pval(self,polyI,x):return polynomial_interval_decimal(polyI,I(self.dr.Dlo(x),self.dr.Dhi(x)),self.dr)
    def prange(self,polyI,a,b):return polynomial_interval_decimal(polyI,I(self.dr.Dlo(a),self.dr.Dhi(b)),self.dr)
    def gval(self,x):K,_=self.state(x);return self.dr.sub(self.pval(self.hI,x),K)
    def gp_range(self,a,b):
        H=self.prange(self.hpI,a,b);_,ka=self.state(a);_,kb=self.state(b)
        return self.dr.sub(H,I(ka.lo,kb.hi))
    def gpp_range(self,a,b):
        H=self.prange(self.hppI,a,b);K=deep_second_interval(a,b,self.depth,self.dr,self.pi)
        return self.dr.sub(H,K)
    def mesh(self,max_width=F('0.0015')):
        objects=[]
        for i in range(4):
            objects.append(self.gp_boxes[i]);objects.append(self.gpp_boxes[i])
        objects.append(self.gp_boxes[4]);objects.append((RIGHT_CUT,RIGHT_CUT))
        regions=[]
        for ((_,b),(a,_),sgn) in zip(objects[:-1],objects[1:],EXPECTED_GPP):regions.append((b,a,sgn))
        out=[]
        for rid,(a,b,sgn) in enumerate(regions):
            length=b-a
            parts=max(1,(length.numerator*max_width.denominator+length.denominator*max_width.numerator-1)//(length.denominator*max_width.numerator))
            for j in range(parts):out.append((a+(b-a)*j/parts,a+(b-a)*(j+1)/parts,sgn,rid,j,parts))
        return out
