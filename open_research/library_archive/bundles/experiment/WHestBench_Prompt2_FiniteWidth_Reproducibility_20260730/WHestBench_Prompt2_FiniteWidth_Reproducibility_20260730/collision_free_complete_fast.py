from __future__ import annotations
import itertools,sys
from fractions import Fraction
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
import prompt2_tensor_partition_core as pc
import prompt2_full_hermite_core as c

def fall(n,r):
 x=1
 for j in range(r):x*=n-j
 return x

def energy(rank,k,max_total_support=None,max_outside_support=None):
 te=pc.TensorEnergy(rank);total=c.I.point(0)
 for lam in c.partitions(k):
  s=len(lam)
  if max_total_support is not None and s>max_total_support:continue
  part=c.I.point(0)
  for r in range(min(s,rank)+1):
   if max_outside_support is not None and s-r>max_outside_support:continue
   for inds in itertools.combinations(range(s),r):
    chosen=set(inds)
    dist=tuple(sorted([(lam[i],1) for i in inds]+[(0,1)]*(rank-r)))
    z=te.coefficient(lam,dist)
    count=fall(rank,r)*fall(c.M-s,rank-r)
    part+=z.square()*count
  total+=part*Fraction(c.multiindex_count(lam),c.alpha_factorial(lam))
 return total/c.M
if __name__=='__main__':
 n=int(sys.argv[1]);K=int(sys.argv[2]);out=int(sys.argv[3]) if len(sys.argv)>3 else None
 for k in range(K+1):print(k,c.decimal_bounds(energy(n,k,max_outside_support=out),30),flush=True)
