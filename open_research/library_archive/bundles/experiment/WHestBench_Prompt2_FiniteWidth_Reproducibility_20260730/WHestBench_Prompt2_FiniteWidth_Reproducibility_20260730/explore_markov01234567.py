from fractions import Fraction
import json,runpy,contextlib,io
import prompt2_full_hermite_core as c
with contextlib.redirect_stdout(io.StringIO()):ns=runpy.run_path('/mnt/data/explore_markov0123456.py')
C=ns['C'];decI=ns['decI'];R=ns['R'];G=ns['G'];mon=ns['mon']
for k in range(5):
 r=json.load(open(f'/mnt/data/tensor7_degree_{k}.json'));C[(7,k)]=R(decI(r['lo'],r['hi'])/c.M)
for k in [5,6,7]:
 r=json.load(open(f'/mnt/data/tensor7_degree_{k}_s2.json'));C[(7,k)]=R(decI(r['lo'],r['hi'])/c.M)
for k in range(8,17):C[(7,k)]=c.I.point(0)
states=list(range(8));v=[c.I.point(0),c.I.point(1)]+[c.I.point(0)]*6
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
