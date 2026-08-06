#!/usr/bin/env python3
from fractions import Fraction
from decimal import Decimal,getcontext
import json,math,time,sys
sys.set_int_max_str_digits(0)
from pathlib import Path
getcontext().prec=180
D=256;L=123;R=246;N=66048
lo=int(sys.argv[1]);hi=int(sys.argv[2])
OUT=Path(__file__).resolve().parent
cert=json.load(open(OUT/'SIGNED_RANK_DEGREE123_CERTIFICATE.json'))
cand=json.load(open(OUT/'DEGREE123_ENTRYWISE_DUAL_CANDIDATE.json'))
kdat=json.load(open(OUT/'KERNEL_FULL_UPPER511_DEGREE246.json'))
kup=[Fraction(x['full_upper_exact']) for x in kdat['coefficients']]
y=[Fraction(s)*Fraction(1001,1000) for s in cand['dual_weights']]
P=Fraction(cert['floor_exact'])
def hd(l):return math.comb(D+l-1,l)-(math.comb(D+l-3,l-2) if l>=2 else 0)
d=[hd(l) for l in range(L+1)];sel=cand['fixed_selection'];tail=[d[i]-sel[i] for i in range(L+1)]
def up(r):return Fraction(r+D-2,2*r+D-2)
def dn(r):return Fraction(r,2*r+D-2) if r else Fraction(0)
t0=time.time();mn=None;mpair=None;checked=0
for i in range(lo,hi+1):
 prev={i:Fraction(1)};curr=None
 for j in range(L+1):
  if j==0:prod=prev
  elif j==1:
   curr={i+1:up(i)}
   if i:curr[i-1]=dn(i)
   prod=curr
  else:
   q=j-1;A=Fraction(2*(q+127),q+254);B=Fraction(q,q+254);nxt={}
   for r,v in curr.items():
    nxt[r+1]=nxt.get(r+1,Fraction(0))+A*v*up(r)
    if r:nxt[r-1]=nxt.get(r-1,Fraction(0))+A*v*dn(r)
   for r,v in prev.items():nxt[r]=nxt.get(r,Fraction(0))-B*v
   prev,curr=curr,{r:v for r,v in nxt.items() if v};prod=curr
  if j<i:continue
  hij=(Fraction(tail[i],d[i]*d[i]) if i==j else Fraction(0))+Fraction(tail[i]*tail[j],N*d[i]*d[j])
  if not hij:continue
  lhs=sum((y[r-1]*v/kup[r] for r,v in prod.items() if r>=1 and y[r-1]),Fraction(0))*P
  mar=lhs-hij
  if mn is None or mar<mn:mn=mar;mpair=(i,j)
  checked+=1
assert mn is not None
out={'lo':lo,'hi':hi,'checked':checked,'minimum_exact_margin_sha256':__import__('hashlib').sha256((str(mn.numerator)+'/'+str(mn.denominator)).encode()).hexdigest(),'minimum_margin_decimal':str(Decimal(mn.numerator)/Decimal(mn.denominator)),'minimum_margin_pair':list(mpair),'positive':mn>0,'elapsed_seconds':time.time()-t0}
json.dump(out,open(OUT/f'DEGREE123_DUAL_CHUNK_{lo}_{hi}.json','w'),indent=2)
print(json.dumps(out,indent=2))
