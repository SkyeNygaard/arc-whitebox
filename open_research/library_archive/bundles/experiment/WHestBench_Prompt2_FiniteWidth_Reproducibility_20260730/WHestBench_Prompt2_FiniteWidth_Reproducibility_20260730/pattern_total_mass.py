from fractions import Fraction
from collections import Counter
import math
import prompt2_full_hermite_core as c
import prompt2_tensor_partition_core as pc

def rising_half(p):
 x=Fraction(1)
 for j in range(p):x*=Fraction(2*j+1,2)
 return x

def fixed_component(rank,parts):
 b=len(parts);prod=Fraction(1)
 for p in parts:prod*=rising_half(p)
 z=Fraction(0)
 for r in range(c.M-b+1):
  d=b+r
  den=Fraction(1)
  for j in range(rank):den*=Fraction(d+2*j,2)
  z += Fraction(math.comb(c.M-b,r),2**(c.M-b))*Fraction(2,2**b)*d*prod/den
 return z

def mass(rank,parts):
 b=len(parts)
 return fixed_component(rank,parts)*pc.equality_pattern_count(tuple(parts))*pc.fall(c.M,b)/c.M

def deficit_patterns(n,maxdef):
 for parts in pc.int_partitions(n):
  if n-len(parts)<=maxdef:yield parts
if __name__=='__main__':
 for n in range(2,23):
  vals=[]
  for d in range(4): vals.append(sum(mass(n,p) for p in deficit_patterns(n,d)))
  print(n,*[float(x) for x in vals], 'increment',[float(vals[0]),float(vals[1]-vals[0]),float(vals[2]-vals[1]),float(vals[3]-vals[2])])
