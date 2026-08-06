import sys
if hasattr(sys,'set_int_max_str_digits'): sys.set_int_max_str_digits(0)
from fractions import Fraction
import json,runpy,contextlib,io,re,glob,os
import prompt2_full_hermite_core as c
with contextlib.redirect_stdout(io.StringIO()): ns=runpy.run_path('/mnt/data/explore_markov01234567.py')
C=ns['C']; decI=ns['decI']; R=ns['R']; G=ns['G']; mon=ns['mon']
max_state=int(os.environ.get('MAX_STATE','8'))
# ensure rows exist keys
for i in range(max_state+1):
    for k in range(17): C.setdefault((i,k),c.I.point(0))
# override any generic tensor files, preferring full over support-restricted
for r in range(3,max_state+1):
    records={}
    for f in glob.glob(f'/mnt/data/tensor{r}_degree_*.json'):
        mo=re.search(r'degree_(\d+)(?:_s(\d+))?\.json$',f)
        if not mo: continue
        deg=int(mo.group(1)); sup=int(mo.group(2)) if mo.group(2) else None
        d=json.load(open(f)); rec=(sup,d,f)
        if deg not in records or sup is None: records[deg]=rec
    for deg,(sup,d,f) in records.items():
        C[(r,deg)]=R(decI(d['lo'],d['hi'])/c.M)
# state 8 may not have existed in base
states=list(range(max_state+1)); v=[c.I.point(0),c.I.point(1)]+[c.I.point(0)]*(max_state-1)
for _ in range(31): v=[R(sum((v[i]*C[(i,j)] for i in states),c.I.point(0))) for j in states]
a={k:R(sum((v[i]*C[(i,k)] for i in states),c.I.point(0))) for k in range(17)}
kg={ell:c.I.point(0) for ell in range(1,17)}
for n in range(1,17):
    for ell in range(1,n+1):
        q=mon[n][ell]
        if q: kg[ell]+=a[n]*q
print('vsum',c.decimal_bounds(sum(v,c.I.point(0)),30))
print('v',[c.decimal_bounds(x,20) for x in v])
print('asum',c.decimal_bounds(sum(a.values(),c.I.point(0)),30))
print('g')
for k in range(1,17): print(k,c.decimal_bounds(kg[k],30))
