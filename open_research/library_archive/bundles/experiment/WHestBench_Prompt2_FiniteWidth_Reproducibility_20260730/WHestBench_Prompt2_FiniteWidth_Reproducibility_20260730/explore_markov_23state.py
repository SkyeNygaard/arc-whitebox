from __future__ import annotations
import sys,os,glob,re,json,runpy,contextlib,io
from fractions import Fraction
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
import prompt2_full_hermite_core as c
MAX=22; STATES=list(range(MAX+1))
# Base exact/selected rows 0..8.
with contextlib.redirect_stdout(io.StringIO()): ns=runpy.run_path('/mnt/data/explore_markov_dynamic22_high.py')
base=ns['C']
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
for i in range(9):
 for k in STATES:
  z=base.get((i,k))
  if z is not None:offer(i,k,z.lo,'base')
# All generic exact/full or support-restricted tensor records.
for i in range(2,MAX+1):
 for f in glob.glob(f'/mnt/data/tensor{i}_degree_*.json'):
  mo=re.search(r'degree_(\d+)(?:_s(\d+))?\.json$',f)
  if not mo:continue
  k=int(mo.group(1))
  if k>MAX:continue
  d=json.load(open(f));offer(i,k,d['lo']/c.M if isinstance(d['lo'],Fraction) else pf(d['lo'])/c.M,os.path.basename(f))
# Complete collision-free rows (all selected-coordinate Hermite supports).
for i in range(1,MAX+1):
 for f in glob.glob(f'/mnt/data/cf_full_tensor{i}_degree_*.json'):
  mo=re.search(r'degree_(\d+)\.json$',f)
  if not mo:continue
  k=int(mo.group(1))
  if k>MAX:continue
  d=json.load(open(f));offer(i,k,d['lo'],os.path.basename(f))
# Sum disjoint output-index equality classes: all-singletons, one pair,
# one triple, and two pairs. These are orthogonal tensor-output components.
for i in range(2,MAX+1):
 for k in STATES:
  paths=[f'/mnt/data/cf_complete_tensor{i}_degree_{k}.json',
         f'/mnt/data/onepair_tensor{i}_degree_{k}.json',
         f'/mnt/data/pattern_3_tensor{i}_degree_{k}.json',
         f'/mnt/data/pattern_2x2_tensor{i}_degree_{k}.json',
         f'/mnt/data/pattern_4_tensor{i}_degree_{k}.json',
         f'/mnt/data/pattern_3x2_tensor{i}_degree_{k}.json',
         f'/mnt/data/pattern_2x2x2_tensor{i}_degree_{k}.json']
  qs=[]
  for f in paths:
   if os.path.exists(f): qs.append(pf(json.load(open(f))['lo']))
  if qs: offer(i,k,sum(qs,Fraction(0)),'collision-deficit<=3')
# Collision-free selected-support records are already normalized transition energies.
for i in range(1,MAX+1):
 for f in glob.glob(f'/mnt/data/cf_tensor{i}_degree_*_s*.json'):
  mo=re.search(r'degree_(\d+)_s(\d+)\.json$',f)
  if not mo:continue
  k=int(mo.group(1))
  if k>MAX:continue
  d=json.load(open(f));offer(i,k,d['lo'],os.path.basename(f))
for i in STATES:
 s=sum(D[(i,k)] for k in STATES)
 assert s<=1,(i,float(s))
# Floating exploration of exact lower-endpoint sub-Markov matrix.
import numpy as np
A=np.array([[float(D[(i,j)]) for j in STATES] for i in STATES])
v=np.zeros(MAX+1);v[1]=1.0
for _ in range(32): v=v@A
# normalized Gegenbauer coefficients
G=c.gegenbauer_normalized(MAX,(c.D-2)//2)
mon={n:c.expand_in_basis([Fraction(0)]*n+[Fraction(1)],G) for n in STATES}
kg=np.zeros(MAX+1)
for n in range(1,MAX+1):
 for ell in range(1,n+1):
  kg[ell]+=v[n]*float(mon[n][ell])
print('mass',float(v.sum()), 'tailstate mass by n')
print([(i,float(x)) for i,x in enumerate(v) if float(x)>1e-9])
print('coeffs')
for i in range(1,MAX+1):print(i,float(kg[i]))
