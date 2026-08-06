from __future__ import annotations
import sys,os,glob,re,json,runpy,contextlib,io,math
from fractions import Fraction
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
import prompt2_full_hermite_core as c
MAX=28; STATES=list(range(MAX+1))
# start from 23-state matrix entries
with contextlib.redirect_stdout(io.StringIO()): ns=runpy.run_path('/mnt/data/explore_markov_23state.py')
D={(i,k):Fraction(0) for i in STATES for k in STATES}
source={}
def pf(s):
 if isinstance(s,(int,Fraction)):return Fraction(s)
 if '/' in s:
  a,b=s.split('/');return Fraction(int(a),int(b))
 return Fraction(s)
def offer(i,k,lo,label):
 q=pf(lo)
 if q>D[(i,k)]:D[(i,k)]=q;source[(i,k)]=label
for i in range(23):
 for k in range(23):offer(i,k,ns['D'][(i,k)],'23base')
# state0 exact energies through28
h=json.load(open('/mnt/data/full_hermite_energies_1_28.json'))
eps=Fraction(1,10**54)
for k in range(23,MAX+1):
 q=Fraction(h[str(k)])/c.M;offer(0,k,max(Fraction(0),q-eps),'state0')
# state1 exact ReLU row
def relu_num(n):
 if n<2 or n%2:return Fraction(0)
 r=n//2; odd=1
 for j in range(1,2*r-2,2):odd*=j
 return Fraction(odd*odd,math.factorial(2*r))/c.PI.lo # not rational; skip interval below
for k in range(23,MAX+1):
 if k%2==0:
  r=k//2;odd=1
  for j in range(1,2*r-2,2):odd*=j
  # conservative rational lower using PI upper
  offer(1,k,Fraction(odd*odd,math.factorial(2*r))/c.PI.hi,'state1')
# generic records, normalized
for i in range(2,MAX+1):
 for f in glob.glob(f'/mnt/data/tensor{i}_degree_*.json'):
  mo=re.search(r'degree_(\d+)(?:_s(\d+))?\.json$',f)
  if not mo:continue
  k=int(mo.group(1));
  if k>MAX:continue
  d=json.load(open(f));offer(i,k,pf(d['lo'])/c.M,os.path.basename(f))
# normalized subcomponents
for i in range(1,MAX+1):
 for pref in ['cf_full','cf_complete','onepair','pattern_3','pattern_2x2','pattern_4','pattern_3x2','pattern_2x2x2']:
  for f in glob.glob(f'/mnt/data/{pref}_tensor{i}_degree_*.json'):
   mo=re.search(r'degree_(\d+)\.json$',f)
   if not mo:continue
   k=int(mo.group(1));
   if k>MAX:continue
   # components must be summed, not max individually; handled below for collision classes
   if pref=='cf_full':offer(i,k,pf(json.load(open(f))['lo']),os.path.basename(f))
 for k in STATES:
  paths=[f'/mnt/data/cf_complete_tensor{i}_degree_{k}.json',f'/mnt/data/onepair_tensor{i}_degree_{k}.json',f'/mnt/data/pattern_3_tensor{i}_degree_{k}.json',f'/mnt/data/pattern_2x2_tensor{i}_degree_{k}.json',f'/mnt/data/pattern_4_tensor{i}_degree_{k}.json',f'/mnt/data/pattern_3x2_tensor{i}_degree_{k}.json',f'/mnt/data/pattern_2x2x2_tensor{i}_degree_{k}.json']
  qs=[pf(json.load(open(f))['lo']) for f in paths if os.path.exists(f)]
  if qs:offer(i,k,sum(qs,Fraction(0)),'patterns')
for i in STATES:
 assert sum(D[(i,k)] for k in STATES)<=1,(i,float(sum(D[(i,k)] for k in STATES)))
import numpy as np
A=np.array([[float(D[(i,j)]) for j in STATES] for i in STATES])
v=np.zeros(MAX+1);v[1]=1
for _ in range(32):v=v@A
G=c.gegenbauer_normalized(MAX,(c.D-2)//2)
mon={n:c.expand_in_basis([Fraction(0)]*n+[Fraction(1)],G) for n in STATES}
kg=np.zeros(MAX+1)
for n in range(1,MAX+1):
 for ell in range(1,n+1):kg[ell]+=v[n]*float(mon[n][ell])
print('mass',v.sum());print('coeffs');
for i in range(1,MAX+1):print(i,kg[i])
