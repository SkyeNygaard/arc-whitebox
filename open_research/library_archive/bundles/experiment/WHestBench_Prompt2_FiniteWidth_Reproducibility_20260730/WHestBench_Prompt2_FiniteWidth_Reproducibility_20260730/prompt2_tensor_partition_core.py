#!/usr/bin/env python3
"""Generic exact Hermite-energy engine for normalized ReLU tensor features.

For rank r>=1, computes chaos-degree energies of
  F_r(sqrt(2) G_+)/sqrt(m),
  F_r(x)=x^{tensor r}/||x||^{r-1}.

The output-index equality patterns are aggregated by integer partitions of r,
not Bell-number set partitions. All arithmetic is exact rational intervals.
"""
from __future__ import annotations
import itertools, math
from collections import Counter
from fractions import Fraction
from functools import lru_cache
import prompt2_full_hermite_core as c
import prompt2_traceless_quadratic_core as q2
M=c.M; I=c.I

def int_partitions(n:int, max_part:int|None=None):
    if n==0:
        yield (); return
    if max_part is None or max_part>n:max_part=n
    for first in range(max_part,0,-1):
        for rest in int_partitions(n-first, first):
            yield (first,)+rest

def fall(n:int,r:int)->int:
    x=1
    for j in range(r):x*=n-j
    return x

def equality_pattern_count(parts:tuple[int,...])->int:
    # Number of set partitions of labeled tensor positions having these block sizes.
    n=sum(parts); den=1
    for p in parts:den*=math.factorial(p)
    for z in Counter(parts).values():den*=math.factorial(z)
    return math.factorial(n)//den

class TensorEnergy:
    def __init__(self,rank:int):
        assert rank>=1
        self.rank=rank;self.p=rank-1
        self.patterns=tuple(int_partitions(rank))

    @lru_cache(None)
    def radial(self,k:int,J:int)->I:
        # 2^((J-p)/2) Gamma((k+J-p)/2)/Gamma((k+J)/2).
        if k<=0:return I.point(0)
        a=k+J-self.p
        if a<=0:return I.point(0)
        x=I.point(2**(J//2))
        if J%2:x*=c.SQRT2
        if self.p%2==0:
            r=self.p//2
            for j in range(r):x/=a+2*j
        else:
            r=(self.p-1)//2
            x/=c.chi_mean(a)
            for j in range(r):x/=a+1+2*j
        return x

    @lru_cache(None)
    def coefficient(self,lam:tuple[int,...],dist:tuple[tuple[int,int],...])->I:
        assert sum(p for _,p in dist)==self.rank
        counts=Counter(lam);extra=0;forced=[]
        for qq,p in dist:
            if qq==0:extra+=1
            else:
                if counts[qq]<=0:return I.point(0)
                counts[qq]-=1
                if counts[qq]==0:del counts[qq]
            forced.append(q2.shifted_sphere_poly(qq,p))
        rem=M-len(lam)-extra
        if rem<0:return I.point(0)
        qs=sorted(counts);total=I.point(0)
        def rec(pos,aps,ac,mult,inactive):
            nonlocal total
            if pos==len(qs):
                poly=q2._conv_many(forced+aps);rr=I.point(0);fc=len(forced)
                for J,coef in enumerate(poly):
                    if coef.lo==0 and coef.hi==0:continue
                    ss=I.point(0)
                    for kk in range(rem+1):
                        ad=kk+ac+fc
                        if ad:ss+=self.radial(ad,J)*math.comb(rem,kk)
                    rr+=coef*ss
                total+=rr*inactive*mult;return
            qq=qs[pos];cnt=counts[qq]
            for active in range(cnt+1):
                rec(pos+1,aps+[q2.shifted_sphere_poly(qq,0)]*active,ac+active,
                    mult*math.comb(cnt,active),
                    inactive*c.negative_half_hermite_mean(qq).pow_int(cnt-active))
        rec(0,[],0,1,I.point(1))
        return total*c.SQRT2/(2**M)

    @lru_cache(None)
    def category_counts(self,lam:tuple[int,...]):
        s=len(lam);u=M-s;cnt=Counter()
        for sizes in self.patterns:
            b=len(sizes);sp=equality_pattern_count(sizes)
            # Canonical block slots are distinct for assignment enumeration;
            # sp counts the underlying unlabeled equality patterns.
            for r in range(min(b,s)+1):
                um=fall(u,b-r)
                if not um:continue
                for sbt in itertools.combinations(range(b),r):
                    sb=set(sbt)
                    for sc in itertools.permutations(range(s),r):
                        amap=dict(zip(sbt,sc))
                        dist=tuple(sorted((lam[amap[j]] if j in sb else 0,p)
                                          for j,p in enumerate(sizes)))
                        cnt[dist]+=sp*um
        return tuple(cnt.items())

    def partition_contribution(self,lam:tuple[int,...])->I:
        x=I.point(0)
        for dist,mult in self.category_counts(lam):
            z=self.coefficient(lam,dist);x+=z.square()*mult
        return x*Fraction(c.multiindex_count(lam),c.alpha_factorial(lam))

    def energy(self,n:int,max_support:int|None=None)->I:
        x=I.point(0)
        for lam in c.partitions(n):
            if max_support is not None and len(lam)>max_support:continue
            x+=self.partition_contribution(lam)
        return x
