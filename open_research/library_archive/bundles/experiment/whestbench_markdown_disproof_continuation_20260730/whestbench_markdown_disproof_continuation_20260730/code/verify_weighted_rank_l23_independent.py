from fractions import Fraction
from decimal import Decimal, getcontext
from math import comb
import json
from pathlib import Path

DIM=256; N=66048; L=23; ORDER=47
WEIGHTS=[Fraction(x) for x in ['0.580027924665828198', '1', '0.777074267674444097', '0.746755561505182408', '0.0065394127780017278', '0.000076223762757812727', '0.00000125400888991954673', '0.0000000275983300854912068', '0.000000000678764140775296306', '0.0000000000186171885693140567', '0.000000000000481912464439409887', '0.0000000000000257438763325270784', '0.000000000000000712501140985209107', '0.0000000000000000416613358324680716', '0.0000000000000000017927097612805915', '0.0000000000000000000909958121383644366', '0.00000000000000000000469839764896782971', '0.000000000000000000000299076202725570297', '0.0000000000000000000000181206660636651611', '0.000000000000000000000000454647083680545048', '0.0000000000000000000000000281050536570803334', '0.00000000000000000000000000639303109407317976', '0.000000000000000000000000000516840208522216336', '0.0000000000000000000000000000341082132220033759']]

def add(a,b):
 o=[Fraction(0)]*max(len(a),len(b))
 for i,x in enumerate(a):o[i]+=x
 for i,x in enumerate(b):o[i]+=x
 return trim(o)
def scale(a,c): return trim([x*c for x in a])
def mul(a,b):
 o=[Fraction(0)]*(len(a)+len(b)-1)
 for i,x in enumerate(a):
  for j,y in enumerate(b):o[i+j]+=x*y
 return trim(o)
def shift(a):return [Fraction(0)]+a
def trim(a):
 while len(a)>1 and a[-1]==0:a.pop()
 return a

def gegenbauer(n):
 # lambda=(d-2)/2=127. recurrence independent of SymPy.
 lam=Fraction(DIM-2,2)
 if n==0:return [Fraction(1)]
 C0=[Fraction(1)]; C1=[Fraction(0),2*lam]
 if n==1:C=C1
 else:
  for k in range(1,n):
   # (k+1) C_{k+1} = 2(k+lambda) t C_k - (k+2lambda-1) C_{k-1}
   C2=scale(add(scale(shift(C1),2*(k+lam)),scale(C0,-(k+2*lam-1))),Fraction(1,k+1))
   C0,C1=C1,C2
  C=C1
 val=sum(C)
 assert val!=0
 return scale(C,1/val)

def moment(p):
 if p%2:return Fraction(0)
 m=p//2
 num=1
 for j in range(1,m+1):num*=2*j-1
 den=1
 for j in range(m):den*=DIM+2*j
 return Fraction(num,den)
def expectation(p):return sum(c*moment(i) for i,c in enumerate(p))
def project(poly,r):
 g=G[r]
 return expectation(mul(poly,g))/expectation(mul(g,g))
def hdim(l):
 if l==0:return 1
 if l==1:return DIM
 return comb(DIM+l-1,l)-comb(DIM+l-3,l-2)

G=[gegenbauer(i) for i in range(2*L+1)]
dims=[hdim(i) for i in range(L+1)]
WK=[Fraction(0)]
for l in range(L+1): WK=add(WK,scale(G[l],WEIGHTS[l]*dims[l]))
B=[project(mul(WK,WK),r) for r in range(2*L+1)]
assert all(x>0 for x in B[1:])

jet=json.load(open(Path(__file__).resolve().parents[1] / 'results' / 'MPFR_KERNEL_JET.json'))
lo=[Fraction(x['lo']) for x in jet['coefficients']]
hi=[Fraction(x['hi']) for x in jet['coefficients']]
# verify enclosures nonempty and production mpmath midpoint-ish coefficients are contained by comparing to production interval strings if desired.
assert all(a<=b for a,b in zip(lo,hi))
monproj={}
Klo=[]
for r in range(2*L+1):
 s=Fraction(0)
 for p in range(r,ORDER+1):
  pr=project(([Fraction(0)]*p)+[Fraction(1)],r)
  assert pr>=0, (p,r,pr)
  monproj[(p,r)]=pr
  s += lo[p]*pr
 Klo.append(s)
ratios=[(Klo[r]/B[r],r) for r in range(1,2*L+1)]
gamma,binding=min(ratios)
# best rank-N approximation to diagonal A; eigenvalue weights repeated harmonic dimensions.
items=sorted([(WEIGHTS[l],dims[l],l) for l in range(L+1)], reverse=True, key=lambda x:x[0])
remaining=N; tail_sum=Fraction(0); tail_sq=Fraction(0); selection=[]
for w,count,l in items:
 take=min(remaining,count);remaining-=take;tail=count-take
 tail_sum+=tail*w;tail_sq+=tail*w*w;selection.append({'degree':l,'selected':take,'dimension':count})
assert remaining==0
rank_defect=tail_sq+tail_sum*tail_sum/N
floor=gamma*rank_defect
getcontext().prec=100
def dec(q):return Decimal(q.numerator)/Decimal(q.denominator)
kerdock=Decimal('2.433660357543006e-7')
result={
 'status':'PASS','engine':'direct-C MPFR interval Taylor jet + independent Python Fraction Gegenbauer recurrence (no mpmath.iv or SymPy)',
 'mpfr_precision_bits':jet['precision_bits'],'jet_order':ORDER,'all_monomial_projections_nonnegative':True,
 'all_active_coefficients_positive':True,'binding_degree':binding,
 'gamma_lower':str(dec(gamma)),'rank_defect_exact':f'{rank_defect.numerator}/{rank_defect.denominator}',
 'rank_defect':str(dec(rank_defect)),'floor_lower':str(dec(floor)),
 'fraction_kerdock':str(dec(floor)/kerdock),'improvement_cap':str(kerdock/dec(floor)),
 'selection':selection,
 'ratios':{str(r):str(dec(q)) for q,r in ratios},
 'active_coefficients':{str(r):f'{B[r].numerator}/{B[r].denominator}' for r in range(1,2*L+1)},
 'kernel_coefficient_lower':{str(r):str(dec(Klo[r])) for r in range(1,2*L+1)},
}
out=Path(__file__).resolve().parents[1] / 'results' / 'INDEPENDENT_WEIGHTED_RANK_L23_RECOMPUTED.json';out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','engine','binding_degree','gamma_lower','rank_defect','floor_lower','fraction_kerdock','improvement_cap']},indent=2))
