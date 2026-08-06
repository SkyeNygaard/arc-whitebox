import sys
if hasattr(sys,'set_int_max_str_digits'): sys.set_int_max_str_digits(0)
from fractions import Fraction
import json,runpy,contextlib,io,math,os
import prompt2_full_hermite_core as c
with contextlib.redirect_stdout(io.StringIO()): ns=runpy.run_path('/mnt/data/explore_markov_dynamic.py')
C=ns['C']; decI=ns['decI']; R=ns['R']; v=ns['v']
MAX=22; states=list(range(9))
# widen matrix keys
for i in states:
 for k in range(17,MAX+1): C[(i,k)]=c.I.point(0)
# state 0 exact-ish precomputed intervals (exploration only, widened by 1e-54)
h=json.load(open('/mnt/data/full_hermite_energies_1_28.json'))
eps=Fraction(1,10**54)
for k in range(17,MAX+1):
 q=Fraction(h[str(k)]); C[(0,k)]=c.I(max(Fraction(0),q-eps),q+eps)/c.M
# state 1 exact ReLU row
def relu_power_numerator(n):
 if n<2 or n%2:return Fraction(0)
 r=n//2; odd=1
 for j in range(1,2*r-2,2):odd*=j
 return Fraction(odd*odd,math.factorial(2*r))
for k in range(17,MAX+1):
 if k%2==0:C[(1,k)]=c.I.point(relu_power_numerator(k))/c.PI
# Load any certified positive high-degree tensor-row subsets for the final transition.
import glob,re,os
for r in range(2,9):
    for f in glob.glob(f'/mnt/data/tensor{r}_degree_*_s2.json'):
        mo=re.search(r'degree_(\d+)_s2\.json$',f)
        if not mo: continue
        k=int(mo.group(1))
        if 17 <= k <= MAX:
            d=json.load(open(f)); C[(r,k)]=R(c.I(Fraction(d['lo']),Fraction(d['hi']))/c.M)
# final monomial coefficients
a={k:R(sum((v[i]*C[(i,k)] for i in states),c.I.point(0))) for k in range(MAX+1)}
G=c.gegenbauer_normalized(MAX,(c.D-2)//2)
mon={n:c.expand_in_basis([Fraction(0)]*n+[Fraction(1)],G) for n in range(MAX+1)}
kg={ell:c.I.point(0) for ell in range(1,MAX+1)}
for n in range(1,MAX+1):
 for ell in range(1,n+1):
  q=mon[n][ell]
  if q:kg[ell]+=a[n]*q
print('asum',c.decimal_bounds(sum(a.values(),c.I.point(0)),30))
for k in range(1,MAX+1): print(k,c.decimal_bounds(kg[k],35))
