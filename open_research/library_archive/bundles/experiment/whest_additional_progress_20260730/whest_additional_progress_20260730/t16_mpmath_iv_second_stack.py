#!/usr/bin/env python3
"""Independent mpmath.iv audit of T16 primal feasibility.
Reimplements the proof-critical interval steps from the Decimal/libmpdec proof:
  F''/F'<3, kappa^(6)+3 B_6,2>0, and a Krawczyk enclosure of the
  degree-5 Hermite coefficients. It deliberately does not use the earlier
  superseded B_6,2 >= -kappa^(6)/4 argument.
"""
from fractions import Fraction
from pathlib import Path
import json, hashlib
import mpmath as mp

mp.mp.dps=100; mp.iv.dps=90; iv=mp.iv
D=256; DEPTH=32; N=66048; BITS=220

def fs(q):return f'{q.numerator}/{q.denominator}' if q.denominator!=1 else str(q.numerator)
def endpoints(x):
 s=str(x); return mp.mpf(s[1:s.index(',')]), mp.mpf(s[s.index(',')+1:-1])
def pair(x):
 a,b=endpoints(x);return {'lower':mp.nstr(a,95),'upper':mp.nstr(b,95)}
def point(q):return iv.mpf([fs(q),fs(q)])
def interval(a,b):return iv.mpf([fs(a),fs(b)])
def asin_iv(x):return iv.atan2(x,iv.sqrt(1-x*x))
def kappa_point(x):
 s=iv.sqrt(1-x*x);kp=iv.mpf('0.5')+asin_iv(x)/iv.pi
 return s/iv.pi+kp*x
def kappa_range(x):
 lo=kappa_point(x.a);hi=kappa_point(x.b);return iv.mpf([lo.a,hi.b])
def kernel_prime(x,depth=DEPTH):
 p=iv.mpf(1);y=x
 for _ in range(depth):
  kp=iv.mpf('0.5')+asin_iv(y)/iv.pi;p*=kp;y=kappa_range(y)
 return y,p

def P(x):return 22102*x**3+21930*x**2-87*x-85
def root(a,b):
 fa=P(a);fb=P(b);assert fa*fb<0
 for _ in range(BITS):
  m=(a+b)/2;fm=P(m)
  if fa*fm<=0:b,fb=m,fm
  else:a,fa=m,fm
 return a,b
roots=[root(Fraction(-992278935,10**9),Fraction(-992278934,10**9)),
       root(Fraction(-62224856,10**9),Fraction(-62224855,10**9)),
       root(Fraction(62285891,10**9),Fraction(62285892,10**9))]

def padd(a,b):
 out=[Fraction(0)]*max(len(a),len(b))
 for i,x in enumerate(a):out[i]+=x
 for i,x in enumerate(b):out[i]+=x
 return out
def scale(a,c):return [c*x for x in a]
def mulx(a):return [Fraction(0)]+a
def gegen(maxd=5):
 G=[[Fraction(1)],[Fraction(0),Fraction(1)]]
 for l in range(1,maxd):
  A=Fraction(2*l+D-2,l+D-2);B=Fraction(l,l+D-2)
  G.append(padd(scale(mulx(G[l]),A),scale(G[l-1],-B)))
 return G[:maxd+1]
G=gegen();Gp=[[Fraction(k)*p[k] for k in range(1,len(p))] for p in G]
def peval(poly,x):
 y=iv.mpf(0)
 for c in reversed(poly):y=y*x+point(c)
 return y

# matrix helpers
def mm(A,B):
 n=len(A);k=len(A[0]);m=len(B[0]);out=[[iv.mpf(0) for _ in range(m)] for __ in range(n)]
 for i in range(n):
  for j in range(m):
   z=iv.mpf(0)
   for q in range(k):z+=A[i][q]*B[q][j]
   out[i][j]=z
 return out
def col(v):return [[x] for x in v]
def uncol(v):return [x[0] for x in v]
def abs_hi(x):
 lo,hi=endpoints(x);return max(abs(lo),abs(hi))

# Interval Hermite system
A=[[None]*6 for _ in range(6)];b=[None]*6
for j,(lo,hi) in enumerate(roots):
 X=interval(lo,hi);kv,kp=kernel_prime(X)
 for l in range(6):A[2*j][l]=peval(G[l],X);A[2*j+1][l]=peval(Gp[l],X)
 b[2*j]=kv;b[2*j+1]=kp
# independently reconstruct approximate inverse at exact root midpoints
mids=[mp.mpf(((a+b)/2).numerator)/((a+b)/2).denominator for a,b in roots]
def ppoint(poly,x):return sum(mp.mpf(q.numerator)/q.denominator*x**i for i,q in enumerate(poly))
Am=mp.matrix(6,6)
for j,x in enumerate(mids):
 for l in range(6):Am[2*j,l]=ppoint(G[l],x);Am[2*j+1,l]=ppoint(Gp[l],x)
Rm=Am**-1
R=[[point(Fraction(mp.nstr(Rm[i,j],90))) for j in range(6)] for i in range(6)]
x0s=[
'0.9747299751309444413666593085802870785923869068234348747228323827800535',
'0.00279647306154118416616586023526018213016938536334332686804673875449754',
'0.00243629527371522242447068060976310829567257873525442749320203262172687',
'0.00180373485519710060891233424000157672203071189874102965019266506169427',
'0.00103172848676742614815821374777678526714203832837998423416094757916937',
'0.000179898923463644585494486989098646638530471586830393993221578851751660']
x0=[point(Fraction(s)) for s in x0s]
Ax=uncol(mm(A,col(x0)));res=[b[i]-Ax[i] for i in range(6)];z=uncol(mm(R,col(res)));RA=mm(R,A)
E=[[point(Fraction(1 if i==j else 0))-RA[i][j] for j in range(6)] for i in range(6)]
rho=max(sum(abs_hi(E[i][j]) for j in range(6)) for i in range(6));zmax=max(abs_hi(q) for q in z);err=zmax/(1-rho)
coeff=[]
for s in x0s:
 c=mp.mpf(s);coeff.append({'lower':mp.nstr(c-err,95),'upper':mp.nstr(c+err,95)})
assert rho<1 and all(mp.mpf(c['lower'])>0 for c in coeff[1:])

# Outer F''/F' ratio on one full box, matching the conservative Decimal proof.
def outer_ratio(a='0',b='0.319'):
 y=iv.mpf([a,b]);p=iv.mpf(1);r=iv.mpf(0)
 for _ in range(31):
  s=iv.sqrt(1-y*y);kp=iv.mpf('0.5')+asin_iv(y)/iv.pi;kpp=1/(iv.pi*s)
  r += (kpp/kp)*p;p*=kp;y=kappa_range(y)
 return r
Rratio=outer_ratio();rlo,rhi=endpoints(Rratio);assert rlo>=0 and rhi<3

# Correct transformed positivity certificate on 20 boxes.
Hrows=[];minlo=None
for i in range(20):
 a=Fraction(-1)+Fraction(i,20);bb=Fraction(-1)+Fraction(i+1,20)
 T=interval(a,bb);c=-T
 sa=iv.sqrt(1-point(a)**2);sb=iv.sqrt(1-point(bb)**2);s=iv.mpf([sa.a,sb.b])
 # phi=pi/2+asin(t), monotone; direct endpoint construction
 philo=iv.pi/2+asin_iv(point(a));phihi=iv.pi/2+asin_iv(point(bb));phi=iv.mpf([philo.a,phihi.b])
 c2=c*c;c4=c2*c2;s2=s*s;s3=s2*s
 H=iv.pi*(3+24*c2+8*c4)-18*phi*c*(3+2*c2)*s2+(15+40*c2)*s3
 lo,hi=endpoints(H);assert lo>0
 if minlo is None or lo<minlo:minlo=lo
 Hrows.append({'interval':[fs(a),fs(bb)],'H':pair(H)})

# Compare with Decimal certificate.
decpath=Path(__file__).with_name('sources')/'T16_DECIMAL_PRIMAL_DUAL_CERTIFICATE.json'
dec=json.loads(decpath.read_text())
dec_r=mp.mpf(dec['outer_log_derivative_certificate']['max_upper']);dec_h=mp.mpf(dec['kappa_sixth_combination_certificate']['minimum_lower'])
assert abs(rhi-dec_r)<mp.mpf('1e-70') and abs(minlo-dec_h)<mp.mpf('1e-70')
for i,c in enumerate(coeff):
 dlo=mp.mpf(dec['hermite_coefficient_certificate']['coefficient_intervals'][i][0]);dhi=mp.mpf(dec['hermite_coefficient_certificate']['coefficient_intervals'][i][1])
 assert mp.mpf(c['lower'])<=dhi and mp.mpf(c['upper'])>=dlo

out={
 'title':'Independent mpmath.iv audit of T16 primal feasibility',
 'status':'PASSED',
 'stack':{'python':mp.python_version if hasattr(mp,'python_version') else None,'mpmath':mp.__version__,'interval_dps':90},
 'roots':[[fs(a),fs(b)] for a,b in roots],
 'krawczyk':{'contraction_norm_upper':mp.nstr(rho,95),'coefficient_error_upper':mp.nstr(err,95),'coefficient_intervals':coeff,'all_nonconstant_positive':True},
 'outer_log_derivative':{'interval':pair(Rratio),'comparison':'upper < 3','matches_decimal_to_70_digits':True},
 'kappa6_plus_3B62':{'boxes':20,'minimum_H_lower':mp.nstr(minlo,95),'identity':'pi^2*(1-t^2)^(9/2)*(kappa^(6)+3B_6,2)=3H(t)','matches_decimal_to_70_digits':True},
 'analytic_bridge':['other Bell terms B_6,3 through B_6,6 are positive under the published analytic lemmas','if B_6,2<0, F_second/F_prime<3 and kappa6+3B_6,2>0 imply K32 sixth derivative positivity','Hermite remainder gives global primal feasibility'],
 'scope':'independent interval reimplementation of numerical primal-feasibility ingredients; exact reduced-cost and primal-dual algebra are audited separately',
}
raw=json.dumps(out,sort_keys=True,separators=(',',':')).encode();out['certificate_sha256']=hashlib.sha256(raw).hexdigest()
p=Path(__file__).with_name('T16_MPMATH_IV_SECOND_STACK_AUDIT.json');p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':'PASSED','rho':out['krawczyk']['contraction_norm_upper'],'outer_upper':out['outer_log_derivative']['interval']['upper'],'H_min':out['kappa6_plus_3B62']['minimum_H_lower'],'smallest_coefficient_lower':coeff[5]['lower']},indent=2))
