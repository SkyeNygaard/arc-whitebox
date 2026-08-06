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
HD=[hdim(i) for i in range(170)]
def gegenbauer(maxd):
 lam=Fraction(D-2,2);C=[[Fraction(1)],[Fraction(0),2*lam]]
 for n in range(1,maxd):
  aa=2*(Fraction(n)+lam);bb=Fraction(n)+2*lam-1
  sh=[Fraction(0)]+[aa*z for z in C[n]];pr=[bb*z for z in C[n-1]]+[Fraction(0)]*(len(sh)-len(C[n-1]))
  C.append([(sh[i]-pr[i])/Fraction(n+1) for i in range(len(sh))])
 return [[z/Fraction(math.comb(n+D-3,n)) for z in p] for n,p in enumerate(C)]
def prod(a,b):
 o=[Fraction(0)]*(len(a)+len(b)-1)
 for i,x in enumerate(a):
  if x:
   for j,y in enumerate(b):
    if y:o[i+j]+=x*y
 return o
def dec(poly,G):
 w=list(poly)+[Fraction(0)]*(ORDER+1-len(poly));o=[Fraction(0)]*(ORDER+1)
 for l in range(min(ORDER,len(poly)-1),-1,-1):
  if w[l]:
   q=w[l]/G[l][l];o[l]=q
   for j,x in enumerate(G[l]):w[j]-=q*x
 assert not any(w);return np.fromiter((float(x) for x in o),float,count=ORDER+1)
def roots_ratio(a,b,c,d,e,f):
 ns=max(abs(a),abs(b),abs(c),1e-300); ds=max(abs(d),abs(e),abs(f),1e-300)
 a,b,c=a/ns,b/ns,c/ns; d,e,f=d/ds,e/ds,f/ds
 A=c*e-b*f; B=2*(c*d-a*f); C=b*d-a*e
 cand=[0.0,1.0]
 eps=1e-15
 if abs(A)<eps:
  if abs(B)>eps:
   r=-C/B
   if 0<r<1:cand.append(r)
 else:
  disc=B*B-4*A*C
  if disc>=0:
   sd=math.sqrt(disc)
   for r in [(-B-sd)/(2*A),(-B+sd)/(2*A)]:
    if 0<r<1:cand.append(r)
 return cand

def main():
 t=time.time();G=gegenbauer(ORDER)
 jet=json.load(open(ROOT/'K32_MACLAURIN_INTERVALS_ORDER320.json'));k=np.zeros(ORDER+1)
 for n,(lo,hi) in enumerate(jet['maclaurin_intervals']):
  p=[Fraction(0)]*(n+1);p[n]=1;dd=dec(p,G);k+=float(Decimal(lo))*dd
 print('kernel',time.time()-t,flush=True)
 parts={}
 for s in range(3,160):
  if 2*(s+1)>ORDER:break
  ds=float(HD[s]);dt=float(HD[s+1])
  ss=dec(prod(G[s],G[s]),G)[1:]*ds*ds
  st=dec(prod(G[s],G[s+1]),G)[1:]*2*ds*dt
  tt=dec(prod(G[s+1],G[s+1]),G)[1:]*dt*dt
  # denominator d+er+fr2
  d=ds*ds/N-ds;e=2*ds*dt/N;f=dt*dt/N-dt
  parts[s]=(ss,st,tt,d,e,f)
 print('parts',len(parts),time.time()-t,flush=True)
 def col(s,r):
  a,b,c,d,e,f=parts[s];den=d+e*r+f*r*r
  return np.maximum((a+b*r+c*r*r)/den * (KUP/k[1:]),0)
 cert=json.load(open(ROOT/'SIGNED_NEAR_OPTIMALITY_CERTIFICATE_BLOCKTRACE_ORDER320.json'))
 cols=[];meta=[];seen=set()
 for row in cert['components']:
  s=int(row['s']);r=float(Decimal(row['r']));key=(s,round(r,16))
  if key not in seen: seen.add(key);cols.append(col(s,r));meta.append((s,r))
 for it in range(12):
  A=np.array(cols).T;cm=A.max(0);B=A/cm;obj=1/cm
  res=linprog(-obj,A_ub=B,b_ub=np.ones(ORDER),bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10,'ipm_optimality_tolerance':1e-10})
  z=res.x/cm;frac=z.sum();dual=np.maximum(-res.ineqlin.marginals,0) # dual for B constraints
  # Convert dual price for original A columns: dualB dot (A/cm), cost=1/cm -> violation iff dualB dot A <1
  # indeed dual price p = dualB; p dot A candidate compares with 1
  p=dual
  best=[]
  for s,(aa,bb,cc,d,e,f) in parts.items():
   scale=(KUP/k[1:])
   # dual weighted numerator coefficients
   a=float(np.dot(p,aa*scale));b=float(np.dot(p,bb*scale));c=float(np.dot(p,cc*scale))
   candidates=roots_ratio(a,b,c,d,e,f)
   vals=[(a+b*r+c*r*r)/(d+e*r+f*r*r) for r in candidates]
   j=int(np.argmin(vals));best.append((vals[j],s,candidates[j]))
  best.sort()
  violations=[x for x in best if x[0]<1-1e-7]
  print('iter',it,'frac',frac,'ncol',len(cols),'best',best[:10],'viol',len(violations),flush=True)
  if not violations:break
  added=0
  for price,s,r in violations[:100]:
   key=(s,round(r,14))
   if key not in seen:
    seen.add(key);cols.append(col(s,r));meta.append((s,r));added+=1
  if not added:break
 # final report, reusing the last converged finite LP solve
 z=res.x/cm;nz=np.where(z>1e-13)[0]
 out={'success':res.success,'fraction':float(z.sum()),'objective':float(z.sum()*KUP),'n_columns':len(cols),'n_nonzero':len(nz),'rows':[{'s':meta[i][0],'r':meta[i][1],'z':float(z[i]),'y':float(z[i]*KUP)} for i in nz]}
 json.dump(out,open(ROOT/'CONTINUOUS_ADJACENT_BLOCKTRACE_DISCOVERY_320_FIXED.json','w'),indent=2)
 print(json.dumps({k:out[k] for k in ['success','fraction','objective','n_columns','n_nonzero']},indent=2))
if __name__=='__main__':main()
