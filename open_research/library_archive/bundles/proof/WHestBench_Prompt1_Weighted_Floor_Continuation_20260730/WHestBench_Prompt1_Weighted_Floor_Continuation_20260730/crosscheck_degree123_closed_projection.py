#!/usr/bin/env python3
from fractions import Fraction
from decimal import Decimal,getcontext
import json,math,time
from pathlib import Path
getcontext().prec=180
HERE=Path(__file__).resolve().parent;D=256;L=123;R=246;M=511;N=66048
cert=json.load(open(HERE/'SIGNED_RANK_DEGREE123_CERTIFICATE.json'));alpha=[Fraction(x['lo']) for x in json.load(open(HERE/'MPFR_KERNEL_JET_511.json'))['coefficients']];z=[Fraction(x) for x in cert['z']]
def hd(l):return math.comb(D+l-1,l)-(math.comb(D+l-3,l-2) if l>=2 else 0)
d=[hd(l) for l in range(L+1)]
def seq(r,maxp):
 vals={};v=Fraction(1)
 for l in range(1,r):v*=Fraction(l+D-2,2*l+D-2)
 p=r;vals[p]=v
 while p+2<=maxp:v*=Fraction((p+2)*(p+1),(p+2-r)*(D+p+r));p+=2;vals[p]=v
 return vals
t0=time.time();P=[seq(r,M) for r in range(R+1)];kfull=[sum((alpha[p]*v for p,v in P[r].items()),Fraction(0)) for r in range(R+1)]
G=[[Fraction(1)],[Fraction(0),Fraction(1)]]
for l in range(1,L):
 A=Fraction(2*(l+127),l+254);B=Fraction(l,l+254);o=[Fraction(0)]*(l+2)
 for p,v in enumerate(G[l]):o[p+1]+=A*v
 for p,v in enumerate(G[l-1]):o[p]-=B*v
 G.append(o)
poly=[Fraction(0)]*(L+1)
for l in range(L+1):
 for p,v in enumerate(G[l]):poly[p]+=z[l]*v
sq=[Fraction(0)]*(R+1)
for i,a in enumerate(poly):
 if a:
  for j,b in enumerate(poly):
   if b:sq[i+j]+=a*b
bcoef=[sum((sq[p]*P[r][p] for p in range(r,R+1,2)),Fraction(0)) for r in range(R+1)];rat=[kfull[r]/bcoef[r] for r in range(1,R+1)];mn=min(rat);bind=rat.index(mn)+1
Ptr=[[Fraction(0)]*(R+1) for _ in range(M+1)];Ptr[0][0]=1
for p in range(M):
 for l in range(min(p,R)+1):
  v=Ptr[p][l]
  if not v:continue
  if l+1<=R:Ptr[p+1][l+1]+=v*Fraction(l+D-2,2*l+D-2)
  if l:Ptr[p+1][l-1]+=v*Fraction(l,2*l+D-2)
ktr=[sum((alpha[p]*Ptr[p][r] for p in range(r,M+1) if Ptr[p][r]),Fraction(0)) for r in range(R+1)];assert all(kfull[r]>=ktr[r] for r in range(R+1))
c=[Fraction(cert['s']) if l<4 else z[l]/d[l] for l in range(L+1)];sel=[0]*(L+1);sel[0]=d[0];sel[1]=d[1];sel[2]=d[2];sel[3]=N-sum(sel[:3]);tail=[d[l]-sel[l] for l in range(L+1)]
rd=sum(Fraction(tail[l])*c[l]*c[l] for l in range(L+1))+sum(Fraction(tail[l])*c[l] for l in range(L+1))**2/Fraction(N);assert f'{rd.numerator}/{rd.denominator}'==cert['rank_defect_exact'];assert bind==106
out={'status':'PASS','full_projection_binding_degree':bind,'full_projection_minimum_ratio':str(Decimal(mn.numerator)/Decimal(mn.denominator)),'all_full_coefficients_at_least_conservative_coefficients':True,'rank_defect_exact_match':True,'elapsed_seconds':time.time()-t0}
json.dump(out,open(HERE/'SIGNED_RANK_DEGREE123_CLOSED_PROJECTION_RECHECK.json','w'),indent=2);print(json.dumps(out,indent=2))
