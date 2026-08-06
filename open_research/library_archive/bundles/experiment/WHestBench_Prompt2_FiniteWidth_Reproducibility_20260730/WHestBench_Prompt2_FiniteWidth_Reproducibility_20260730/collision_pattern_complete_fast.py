from __future__ import annotations
import itertools,math,sys
from collections import Counter
from fractions import Fraction
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
import prompt2_tensor_partition_core as pc
import prompt2_full_hermite_core as c

def fall(n,r):
 x=1
 for j in range(r):x*=n-j
 return x

def pattern_multiplicity(rank:int,nontrivial:tuple[int,...])->int:
 singles=rank-sum(nontrivial)
 assert singles>=0 and all(p>=2 for p in nontrivial)
 den=math.factorial(singles)
 for p in nontrivial:den*=math.factorial(p)
 for z in Counter(nontrivial).values():den*=math.factorial(z)
 return math.factorial(rank)//den

def energy(rank:int,k:int,nontrivial:tuple[int,...],max_support=None):
 nontrivial=tuple(sorted(nontrivial,reverse=True));sing=rank-sum(nontrivial)
 if sing<0:return c.I.point(0)
 te=pc.TensorEnergy(rank);sp=pattern_multiplicity(rank,nontrivial);u=len(nontrivial)
 total=c.I.point(0)
 for lam in c.partitions(k):
  s=len(lam)
  if max_support is not None and s>max_support:continue
  labels=tuple(range(s));part=c.I.point(0)
  # Assign distinct Hermite-support coordinates to any subset of the canonical
  # nontrivial block slots. Slots are canonical; sp handles unlabeled equality patterns.
  for rnt in range(min(u,s)+1):
   for slots in itertools.combinations(range(u),rnt):
    for support_order in itertools.permutations(labels,rnt):
     nt_assign=dict(zip(slots,support_order));used=set(support_order)
     rem=tuple(i for i in labels if i not in used)
     # Choose Hermite support coordinates assigned to singleton blocks.
     for rs in range(min(len(rem),sing)+1):
      for chosen in itertools.combinations(rem,rs):
       dist=[]
       for j,p in enumerate(nontrivial):dist.append((lam[nt_assign[j]],p) if j in nt_assign else (0,p))
       dist.extend((lam[i],1) for i in chosen)
       dist.extend([(0,1)]*(sing-rs))
       z=te.coefficient(lam,tuple(sorted(dist)))
       inactive=(u-rnt)+(sing-rs)
       count=sp*fall(sing,rs)*fall(c.M-s,inactive)
       part+=z.square()*count
  total+=part*Fraction(c.multiindex_count(lam),c.alpha_factorial(lam))
 return total/c.M
