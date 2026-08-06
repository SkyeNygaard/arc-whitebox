#!/usr/bin/env python3
from fractions import Fraction
from decimal import Decimal
import math,json,time
import numpy as np
from scipy.optimize import linprog
from pathlib import Path
D=256;N=66048;ORDER=320;ROOT=Path(__file__).resolve().parent;KUP=float(Decimal('2.433660357543006E-7'))
def hdim(l):
 if l==0:return 1
 if l==1:return D
 return math.comb(D+l-1,l)-math.comb(D+l-3,l-2)
HD=[hdim(i) for i in range(200)]
def gegenbauer(maxd):
 lam=Fraction(D-2,2);C=[[Fraction(1)],[Fraction(0),2*lam]]
 for n in range(1,maxd):
  a=2*(Fraction(n)+lam);b=Fraction(n)+2*lam-1
  sh=[Fraction(0)]+[a*z for z in C[n]];pr=[b*z for z in C[n-1]]+[Fraction(0)]*(len(sh)-len(C[n-1]))
  C.append([(sh[i]-pr[i])/Fraction(n+1) for i in range(len(sh))])
 return [[z/Fraction(math.comb(n+D-3,n)) for z in p] for n,p in enumerate(C)]
def product(a,b):
 out=[Fraction(0)]*(len(a)+len(b)-1)
 for i,x in enumerate(a):
  if x:
   for j,y in enumerate(b):
    if y:out[i+j]+=x*y
 return out
def decompose(poly,G):
 w=list(poly)+[Fraction(0)]*(ORDER+1-len(poly));o=[Fraction(0)]*(ORDER+1)
 for l in range(min(ORDER,len(poly)-1),-1,-1):
  if w[l]:
   q=w[l]/G[l][l];o[l]=q
   for j,x in enumerate(G[l]):w[j]-=q*x
 assert not any(w);return o

def main():
 t=time.time();G=gegenbauer(ORDER)
 jet=json.load(open(ROOT/'K32_MACLAURIN_INTERVALS_ORDER320.json'));k=np.zeros(ORDER+1)
 mono=[]
 for n in range(ORDER+1):
  p=[Fraction(0)]*(n+1);p[n]=1;mono.append(decompose(p,G))
  a=float(Decimal(jet['maclaurin_intervals'][n][0]))
  for l,c in enumerate(mono[-1]):
   if c:k[l]+=a*float(c)
 print('kernel',time.time()-t,flush=True)
 max_harm=140
 pair={}
 for i in range(3,max_harm+1):
  for j in range(i,max_harm+1):
   if i+j<=ORDER and j-i<=9:
    pair[i,j]=np.fromiter((float(x) for x in decompose(product(G[i],G[j]),G)),float,count=ORDER+1)
 print('pairs',len(pair),time.time()-t,flush=True)
 cols=[];meta=[]
 def profile_col(s,aa,kind):
  aa=np.asarray(aa,dtype=float)
  aa=aa/max(aa)
  degs=np.arange(s,s+len(aa));dims=np.asarray([float(HD[i]) for i in degs])
  T=np.dot(dims,aa);S2=np.dot(dims,aa*aa);bound=T*T/N-S2
  if not np.isfinite(bound) or bound<=0:return None
  raw=np.zeros(ORDER+1)
  for u,i in enumerate(degs):
   for v in range(u,len(degs)):
    j=degs[v];key=(int(min(i,j)),int(max(i,j)))
    if key not in pair:return None
    fac=aa[u]*aa[v]*dims[u]*dims[v]*(1 if u==v else 2);raw+=fac*pair[key]
  col=np.maximum(raw[1:]/bound,0)
  return col
 # existing active columns, reconstruct exact profiles
 cert=json.load(open(ROOT/'SIGNED_NEAR_OPTIMALITY_CERTIFICATE_BLOCKTRACE_ORDER320.json'))
 seen=set()
 for row in cert['components']:
  key=(int(row['s']),row['r'])
  if key in seen:continue
  seen.add(key);r=float(Decimal(row['r']));c=profile_col(int(row['s']),[1,r],'adjacent_active')
  if c is not None:cols.append(c);meta.append({'s':int(row['s']),'a':[1,r],'kind':'adjacent_active'})
 # richer candidates
 rgrid=np.unique(np.concatenate((np.logspace(-3,3,49),np.linspace(.05,1,20))))
 rng=np.random.default_rng(20260730)
 for length in range(3,10):
  for s in range(3,max_harm-length+2):
   if 2*(s+length-1)>280:continue
   profiles=[]
   # Geometric both orientations are covered by r in wide range after normalization.
   for r in rgrid: profiles.append(([r**j for j in range(length)],f'geom{length}'))
   profiles += [([1]*length,f'flat{length}'),
                ([math.comb(length-1,j) for j in range(length)],f'binom{length}'),
                ([math.comb(length-1,j)**0.5 for j in range(length)],f'sqrtbinom{length}')]
   # discrete Gaussian bumps and ramps
   xs=np.arange(length)
   for center in [0,(length-1)/4,(length-1)/2,3*(length-1)/4,length-1]:
    for sig in [0.7,1.2,2.0,3.5]: profiles.append((np.exp(-0.5*((xs-center)/sig)**2).tolist(),f'gauss{length}'))
   profiles += [((xs+1).tolist(),f'rampup{length}'),((length-xs).tolist(),f'rampdown{length}')]
   # few deterministic random positive profiles, lognormal to span conditionings
   for scale in [0.5,1.0,2.0]:
    for _ in range(2):profiles.append((np.exp(scale*rng.standard_normal(length)).tolist(),f'rand{length}'))
   for aa,kind in profiles:
    c=profile_col(s,aa,kind)
    if c is not None:cols.append(c);meta.append({'s':s,'a':(np.asarray(aa)/max(aa)).tolist(),'kind':kind})
  print('length',length,'cols',len(cols),'time',time.time()-t,flush=True)
 print('built',len(cols),time.time()-t,flush=True)
 A=np.array(cols).T;As=A*(KUP/k[1:])[:,None];As=np.maximum(As,0)
 finite=np.all(np.isfinite(As),axis=0)&(As.max(0)<1e15)
 As=As[:,finite];meta=[m for m,z in zip(meta,finite) if z]
 # screen with current blocktrace dual
 base=json.load(open(ROOT/'EXTENDED_SIGNED_CERTIFICATE_DISCOVERY_320_BLOCKTRACE.json'));pdual=np.asarray(base['normalized_dual_prices'])
 prices=pdual@As
 adj=np.array([m['kind']=='adjacent_active' for m in meta])
 mids=np.where(~adj)[0]
 print('best multi dual price',float(prices[mids].min()),'improving',int((prices[mids]<1-1e-9).sum()),flush=True)
 # Retain all active adjacent and best 5000 multiblock candidates
 ids=list(np.where(adj)[0]); mids=mids[np.argsort(prices[mids])[:5000]];ids+=list(mids);ids=np.asarray(ids)
 As=As[:,ids];meta=[meta[i] for i in ids];prices=prices[ids]
 cm=As.max(0);B=As/cm;obj=1/cm
 print('LP',B.shape,time.time()-t,flush=True)
 res=linprog(-obj,A_ub=B,b_ub=np.ones(ORDER),bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-9,'ipm_optimality_tolerance':1e-10})
 z=res.x/cm;frac=z.sum();resid=1-As@z;nz=np.where(z>1e-13)[0]
 rows=[{**meta[i],'z':float(z[i]),'y':float(z[i]*KUP),'dual_price_screen':float(prices[i])} for i in nz]
 from collections import Counter
 out={'success':res.success,'fraction':float(frac),'objective':float(frac*KUP),'rows':rows,'slack_min':float(resid.min()),'n_candidates':len(meta),'n_nonzero':len(rows),'kinds':dict(Counter(r['kind'] for r in rows))}
 json.dump(out,open(ROOT/'MULTIBLOCK_BLOCKTRACE_DISCOVERY_320.json','w'),indent=2)
 print(json.dumps({k:out[k] for k in ['success','fraction','objective','slack_min','n_candidates','n_nonzero','kinds']},indent=2))
 print('multi active',[r for r in rows if r['kind']!='adjacent_active'][:20])
if __name__=='__main__':main()
