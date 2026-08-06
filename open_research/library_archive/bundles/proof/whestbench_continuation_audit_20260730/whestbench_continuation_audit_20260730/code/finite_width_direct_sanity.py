from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
rng=np.random.default_rng(20260730)
nsim=120000; d=4; w=8; depth=3; batch=2000
# points with inner products 1,0,1/sqrt(d)=.5 relative to x=e1
x=np.array([1.,0,0,0]); y0=np.array([0.,1,0,0]); yc=np.array([.5,math.sqrt(.75),0,0])
pts=np.stack([x,-x,y0,-y0,yc,-yc])
sums=np.zeros(3); sums2=np.zeros(3); n=0
for st in range(0,nsim,batch):
 b=min(batch,nsim-st)
 W1=rng.standard_normal((b,d,w))*math.sqrt(2/d)
 W2=rng.standard_normal((b,w,w))*math.sqrt(2/w)
 W3=rng.standard_normal((b,w,w))*math.sqrt(2/w)
 # b,p,d @ b,d,w -> b,p,w
 A=np.maximum(np.einsum('pd,bdh->bph',pts,W1),0)
 A=np.maximum(np.einsum('bpi,bij->bpj',A,W2),0)
 A=np.maximum(np.einsum('bpi,bij->bpj',A,W3),0)
 Sx=(A[:,0]+A[:,1])/2; S0=(A[:,2]+A[:,3])/2; Sc=(A[:,4]+A[:,5])/2
 vals=np.stack([np.mean(Sx*Sx,axis=1),np.mean(Sx*S0,axis=1),np.mean(Sx*Sc,axis=1)],axis=1)
 sums+=vals.sum(0); sums2+=(vals*vals).sum(0); n+=b
mean=sums/n; se=np.sqrt(np.maximum(sums2/n-mean*mean,0)/n)
A,O,C=mean; a=A-O; bb=O-C; margin=a+d*bb
# Enumerate counts over 2 bases for every budget <=8 and calculate optimum association risk term.
def h(r): return 0.0 if r==0 else r/(a+bb*r)
alloc={}
for P in range(1,2*d+1):
 cand=[]
 for r1 in range(d+1):
  r2=P-r1
  if 0<=r2<=d:
   H=h(r1)+h(r2); term=1/H
   cand.append((term,r1,r2))
 cand.sort()
 best=cand[0]
 q,s=divmod(P,d)
 expected=tuple(sorted(([d]*q+([s] if s else [])+[0]*(2-q-(1 if s else 0))),reverse=True))
 got=tuple(sorted(best[1:],reverse=True))
 alloc[P]={'best_counts':got,'expected_counts':expected,'pass':got==expected,'risk_term':best[0],'runner_up_gap':cand[1][0]-cand[0][0] if len(cand)>1 else None}
out={'architecture':{'d':d,'width':w,'depth':depth,'post_relu_output':True},'simulations':n,
'association':{'A':A,'O':O,'C':C,'se_A':se[0],'se_O':se[1],'se_C':se[2],'A_minus_O':a,'O_minus_C':bb,'margin':margin,
'sign_pass':bool(a>0 and bb<0 and margin>0)},'allocation':alloc,'all_budget_allocations_pass':all(v['pass'] for v in alloc.values())}
P=Path('/mnt/data/whestbench_continuation_20260730/local_verification/finite_width_direct_sanity.json');P.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
