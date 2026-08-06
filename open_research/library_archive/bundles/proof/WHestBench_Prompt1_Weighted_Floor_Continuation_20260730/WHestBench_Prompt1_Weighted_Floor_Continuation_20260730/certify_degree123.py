#!/usr/bin/env python3
from fractions import Fraction
from decimal import Decimal, getcontext
import json, math, time
from pathlib import Path
getcontext().prec=180
HERE=Path(__file__).resolve().parent
D=256;L=123;R=246;N=66048;M=511
hd=lambda l: math.comb(D+l-1,l)-(math.comb(D+l-3,l-2) if l>=2 else 0)
dims=[hd(l) for l in range(L+1)]
disc=[x for x in json.load(open(HERE/'HIGHER_CUTOFF_DISCOVERY_CORRECTED.json')) if x['L']==L][-1]
shrink=Decimal('0.9999'); zfloat=disc['z']
s=Decimal(format(zfloat[0],'.28e'))*shrink
z=[Decimal(dims[l])*s if l<4 else Decimal(format(zfloat[l],'.28e'))*shrink for l in range(L+1)]
zf=[Fraction(str(x)) for x in z]; cf=[Fraction(s) if l<4 else zf[l]/dims[l] for l in range(L+1)]
assert all(x>=0 for x in cf) and all(cf[l]<cf[3] for l in range(4,L+1))
t0=time.time();P=[[Fraction(0)]*(R+1) for _ in range(M+1)];P[0][0]=1
for p in range(M):
 for l in range(min(p,R)+1):
  v=P[p][l]
  if not v:continue
  if l+1<=R:P[p+1][l+1]+=v*Fraction(l+D-2,2*l+D-2)
  if l:P[p+1][l-1]+=v*Fraction(l,2*l+D-2)
alpha=[Fraction(x['lo']) for x in json.load(open(HERE/'MPFR_KERNEL_JET_511.json'))['coefficients']]
klo=[sum((alpha[p]*P[p][r] for p in range(r,M+1) if P[p][r]),Fraction(0)) for r in range(R+1)]
G=[[Fraction(1)],[Fraction(0),Fraction(1)]]
for l in range(1,L):
 A=Fraction(2*(l+127),l+254);B=Fraction(l,l+254);out=[Fraction(0)]*(l+2)
 for p,v in enumerate(G[l]):out[p+1]+=A*v
 for p,v in enumerate(G[l-1]):out[p]-=B*v
 G.append(out)
poly=[Fraction(0)]*(L+1)
for l in range(L+1):
 for p,v in enumerate(G[l]):poly[p]+=zf[l]*v
sq=[Fraction(0)]*(R+1)
for i,a in enumerate(poly):
 if a:
  for j,b in enumerate(poly):
   if b:sq[i+j]+=a*b
bcoef=[sum((sq[p]*P[p][r] for p in range(r,R+1) if P[p][r]),Fraction(0)) for r in range(R+1)]
rat=[klo[r]/bcoef[r] for r in range(1,R+1)];mn=min(rat);bind=rat.index(mn)+1;assert mn>1
sel=[0]*(L+1);sel[0]=dims[0];sel[1]=dims[1];sel[2]=dims[2];sel[3]=N-sum(sel[:3]);tail=[dims[l]-sel[l] for l in range(L+1)]
rd=sum(Fraction(tail[l])*cf[l]*cf[l] for l in range(L+1))+sum(Fraction(tail[l])*cf[l] for l in range(L+1))**2/Fraction(N)
floorF=rd*mn
formal=json.load(open(HERE/'KERDOCK_MSE_CERTIFIED_INTERVAL.json'))['kerdock_mse_interval'];ku=Decimal(formal['upper'])
floorD=Decimal(floorF.numerator)/Decimal(floorF.denominator);frac=floorD/ku;cap=ku/floorD
assert frac>Decimal(10)/Decimal(11)
old=json.load(open(HERE/'SIGNED_RANK_DEGREE123_CERTIFICATE.json'))
assert bind==old['binding_degree'];assert f'{rd.numerator}/{rd.denominator}'==old['rank_defect_exact'];assert f'{floorF.numerator}/{floorF.denominator}'==old['floor_exact']
out={'status':'PASS','binding_degree':bind,'floor_lower':str(floorD),'fraction_kerdock_lower_rigorous':str(frac),'same_cost_improvement_cap_rigorous':str(cap),'elapsed_seconds':time.time()-t0}
json.dump(out,open(HERE/'SIGNED_RANK_DEGREE123_RECHECK.json','w'),indent=2);print(json.dumps(out,indent=2))
