from fractions import Fraction
import json,runpy,contextlib,io
import prompt2_full_hermite_core as c
with contextlib.redirect_stdout(io.StringIO()):ns=runpy.run_path('/mnt/data/explore_markov01234.py')
C=ns['C'];decI=ns['decI'];R=ns['R'];G=ns['G'];mon=ns['mon']
for k in range(6):
 r=json.load(open(f'/mnt/data/tensor5_degree_{k}.json'));C[(5,k)]=R(decI(r['lo'],r['hi'])/c.M)
for k in range(6,17):C[(5,k)]=c.I.point(0)
states=list(range(6));v=[c.I.point(0),c.I.point(1)]+[c.I.point(0)]*4
for _ in range(31):v=[R(sum((v[i]*C[(i,j)] for i in states),c.I.point(0))) for j in states]
a={k:R(sum((v[i]*C[(i,k)] for i in states),c.I.point(0))) for k in range(17)}
kg={ell:c.I.point(0) for ell in range(1,17)}
for n in range(1,17):
 for ell in range(1,n+1):
  q=mon[n][ell]
  if q:kg[ell]+=a[n]*q
print('v',[c.decimal_bounds(x,30) for x in v],c.decimal_bounds(sum(v,c.I.point(0)),30))
print('g')
for k in range(1,17):print(k,c.decimal_bounds(kg[k],30))
WEIGHTS=[Fraction(47486,10**10),Fraction(10990,10**10),Fraction(1),Fraction(9998603,10**7),Fraction(241228,10**8),Fraction(59804,10**9),Fraction(19425,10**11),Fraction(17147,10**12),Fraction(46232,10**15)]
lp=[Fraction(0)]
for ell,w in enumerate(WEIGHTS):lp=c.poly_add(lp,c.poly_scale(G[ell],w*c.harmonic_dim(ell)))
B=c.expand_in_basis(c.poly_mul(lp,lp),G);rat=sorted((kg[n].lo/B[n],n) for n in range(1,17) if B[n]>0);F=c.rank_floor(WEIGHTS)
print('ratio',[(n,float(r)) for r,n in rat[:8]],'floor',float(F*rat[0][0]))
