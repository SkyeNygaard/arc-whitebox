from fractions import Fraction
import math,json
import prompt2_full_hermite_core as c
R=lambda x:c.round_interval_outward(x,70)
def decI(lo,hi):
 def f(s):
  if '/' in s:
   a,b=s.split('/'); return Fraction(int(a),int(b))
  return Fraction(s)
 return c.I(f(lo),f(hi))
D=json.load(open('/mnt/data/prompt2_full_hermite_degree16_certificate.json'))
h={int(k):decI(v[0],v[1]) for k,v in D['full_hermite_energies'].items()}
b2=decI(*D['architecture']['beta_256']).square();h[0]=R(b2*c.M)
T=json.load(open('/mnt/data/prompt2_traceless_energies_1_12.json'))
e={int(k):decI(v['lo'],v['hi']) for k,v in T.items()};e[0]=decI('25.61062291215508639923335271375541626648884597837034','25.61062291215508639923335271375541626648884597837035')
for n in [13,14,15,16]:
 rr=json.load(open(f'/mnt/data/trace_degree_{n}_s2.json'));e[n]=decI(rr['lo'],rr['hi'])
f3={}
for n in range(17):
 fn=f'/mnt/data/tensor3_degree_{n}.json' if n<=8 else f'/mnt/data/tensor3_degree_{n}_s2.json'
 rr=json.load(open(fn));f3[n]=decI(rr['lo'],rr['hi'])
C={}
for k in range(17):C[(0,k)]=R(h[k]/c.M)
C[(1,0)]=R(c.I.point(1)/c.PI);C[(1,1)]=c.I.point(Fraction(1,2))
def relu_q(n):
 if n<2 or n%2:return Fraction(0)
 r=n//2;odd=1
 for j in range(1,2*r-2,2):odd*=j
 return Fraction(odd*odd,math.factorial(2*r))
for k in range(2,17):C[(1,k)]=R(c.I.point(relu_q(k))/c.PI) if k%2==0 else c.I.point(0)
for k in range(17):C[(2,k)]=R(e[k]/c.M+h[k]/(c.M*c.M))
for k in range(17):C[(3,k)]=R(f3[k]/c.M)
states=[0,1,2,3];v=[c.I.point(0),c.I.point(1),c.I.point(0),c.I.point(0)]
for _ in range(31):v=[R(sum((v[i]*C[(i,j)] for i in states),c.I.point(0))) for j in states]
print('v',[c.decimal_bounds(x,30) for x in v], 'sum',c.decimal_bounds(sum(v,c.I.point(0)),30))
a={k:R(sum((v[i]*C[(i,k)] for i in states),c.I.point(0))) for k in range(17)}
G=c.gegenbauer_normalized(16,(c.D-2)//2);mon={n:c.expand_in_basis([Fraction(0)]*n+[Fraction(1)],G) for n in range(17)}
kg={ell:c.I.point(0) for ell in range(1,17)}
for n in range(1,17):
 for ell in range(1,n+1):
  q=mon[n][ell]
  if q:kg[ell]+=a[n]*q
print('a')
for k in range(17):print(k,c.decimal_bounds(a[k],30))
print('g')
for k in range(1,17):print(k,c.decimal_bounds(kg[k],30))
# current weights gamma
WEIGHTS=[Fraction(47486,10**10),Fraction(10990,10**10),Fraction(1),Fraction(9998603,10**7),Fraction(241228,10**8),Fraction(59804,10**9),Fraction(19425,10**11),Fraction(17147,10**12),Fraction(46232,10**15)]
lp=[Fraction(0)]
for ell,w in enumerate(WEIGHTS):lp=c.poly_add(lp,c.poly_scale(G[ell],w*c.harmonic_dim(ell)))
B=c.expand_in_basis(c.poly_mul(lp,lp),G)
rat=sorted((kg[n].lo/B[n],n) for n in range(1,17) if B[n]>0)
F=c.rank_floor(WEIGHTS)
print('ratios',[(n,float(r)) for r,n in rat[:10]])
print('floor',float(F*rat[0][0]),'gamma',float(rat[0][0]),'F',float(F))
