from __future__ import annotations
from fractions import Fraction
import math,sys
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
import prompt2_tensor_partition_core as pc
import prompt2_full_hermite_core as c

def fall(n,r):
 x=1
 for j in range(r):x*=n-j
 return x

def energy(rank,k,max_support=None):
 te=pc.TensorEnergy(rank); total=c.I.point(0)
 for lam in c.partitions(k):
  s=len(lam)
  if max_support is not None and s>max_support: continue
  if s>rank:continue
  dist=tuple(sorted([(q,1) for q in lam]+[(0,1)]*(rank-s)))
  z=te.coefficient(lam,dist)
  count=fall(rank,s)*fall(c.M-s,rank-s)
  total += z.square()*count*Fraction(c.multiindex_count(lam),c.alpha_factorial(lam))
 return total/c.M
if __name__=='__main__':
 n=int(sys.argv[1]);K=int(sys.argv[2])
 for k in range(K+1):print(k,c.decimal_bounds(energy(n,k),30),flush=True)
