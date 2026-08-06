import mpmath as mp, json, math
from pathlib import Path
mp.mp.dps=110
D=256; M=180
jet=json.load(open('/mnt/data/WHestBench_Prompt1_Weighted_Floor_Continuation_20260730/MPFR_KERNEL_JET_511.json'))
alpha=[(mp.mpf(x['lo'])+mp.mpf(x['hi']))/2 for x in jet['coefficients']]
def phi(t): return (mp.sqrt(1-t*t)+(mp.pi-mp.acos(t))*t)/mp.pi
def phip(t): return (mp.pi-mp.acos(t))/mp.pi
def Kd(t):
 y=mp.fsum(alpha[n]*t**n for n in range(len(alpha)))
 d=mp.fsum(n*alpha[n]*t**(n-1) for n in range(1,len(alpha)))
 return y,d

def conv(a,b,n=None):
 if n is None:n=len(a)+len(b)-2
 o=[mp.mpf('0')]*(n+1)
 for i,x in enumerate(a):
  for j,y in enumerate(b):
   if i+j<=n:o[i+j]+=x*y
 return o

def candidate(rs,M=M):
 # Hermite q degree5
 A=mp.matrix(6,6); b=mp.matrix(6,1)
 row=0
 for r in rs:
  k,kp=Kd(r)
  for j in range(6): A[row,j]=r**j
  b[row]=k; row+=1
  A[row,0]=0
  for j in range(1,6): A[row,j]=j*r**(j-1)
  b[row]=kp; row+=1
 q=list(mp.lu_solve(A,b))
 # P polynomial
 P=[mp.mpf(1)]
 for r in rs:P=conv(P,[-r,mp.mpf(1)])
 P2=conv(P,P)
 H=[alpha[n]-(q[n] if n<6 else 0) for n in range(M+1)]
 # divide full degree-511 H polynomial by monic P2, descending (stable)
 Hfull=[alpha[n]-(q[n] if n<6 else 0) for n in range(len(alpha))]
 rem=Hfull[:]
 Q=[mp.mpf(0)]*(len(alpha)-6)
 for n in range(len(alpha)-1,5,-1):
  coef=rem[n]  # P2 leading coefficient is 1
  Q[n-6]=coef
  for j in range(7): rem[n-6+j]-=coef*P2[j]
 S=Q[:M+1]
 # sqrt S choose positive, then L=P*sqrtS. Orientation sign chosen to match D123 p3 positive.
 U=[mp.mpf(0)]*(M+1); U[0]=mp.sqrt(S[0])
 for n in range(1,M+1):
  v=S[n]
  for i in range(1,n):v-=U[i]*U[n-i]
  U[n]=v/(2*U[0])
 L=conv(P,U,M)
 # sign orientation: D123 l0 neg, P0? 
 # current U positive, use as is; inspect
 return q,P,S,U,L
# projection P_{p,l} low l
maxl=10
Proj=[[mp.mpf(0)]*(maxl+1) for _ in range(M+1)];Proj[0][0]=1
for p in range(M):
 for l in range(min(p,maxl)+1):
  v=Proj[p][l]
  if not v:continue
  if l+1<=maxl:Proj[p+1][l+1]+=v*mp.mpf(l+D-2)/(2*l+D-2)
  if l:Proj[p+1][l-1]+=v*mp.mpf(l)/(2*l+D-2)
def dims(l):return math.comb(D+l-1,l)-(math.comb(D+l-3,l-2) if l>=2 else 0)
def coeffs(rs):
 q,P,S,U,L=candidate(rs)
 z=[mp.fsum(L[p]*Proj[p][l] for p in range(l,M+1)) for l in range(maxl+1)]
 c=[z[l]/dims(l) for l in range(maxl+1)]
 return c,(q,P,S,U,L)
rs=[mp.mpf('-0.10990557917085361'),mp.mpf('-0.002216446436085688'),mp.mpf('0.1055043021273365')]
c,data=coeffs(rs)
print('P',*[mp.nstr(x,30) for x in data[1]])
print('q',*[mp.nstr(x,30) for x in data[0]])
print('S0',mp.nstr(data[2][0],30),'min first S',min(data[2][:50]))
print('L first',*[mp.nstr(x,30) for x in data[4][:10]])
for i,x in enumerate(c):print(i,mp.nstr(x,40))
print('diffs',*[mp.nstr((c[i]-c[3])/mp.mpf('1e-8'),30) for i in range(3)])



def vecfun(a,b,c):
    cc,_=coeffs([a,b,c])
    return ((cc[0]-cc[3])/mp.mpf('1e-8'),(cc[1]-cc[3])/mp.mpf('1e-8'),(cc[2]-cc[3])/mp.mpf('1e-8'))
rs0=tuple(rs)
sol=mp.findroot(vecfun,rs0,tol=mp.mpf('1e-70'),maxsteps=20,solver='mdnewton',verify=False)
print('SOL',*[mp.nstr(x,80) for x in sol])
cc,dat=coeffs(list(sol))
print('DIFF',*[mp.nstr((cc[i]-cc[3]),70) for i in range(3)])
print('C',*[mp.nstr(x,70) for x in cc[:11]])
print('Q',*[mp.nstr(x,70) for x in dat[0]])
print('P',*[mp.nstr(x,70) for x in dat[1]])
print('Lfirst',*[mp.nstr(x,70) for x in dat[4][:12]])
json.dump({'roots':[str(x) for x in sol],'q':[str(x) for x in dat[0]],'P':[str(x) for x in dat[1]],'c':[str(x) for x in cc],'L_first':[str(x) for x in dat[4][:80]],'S_first':[str(x) for x in dat[2][:80]]},open('/tmp/infinite_candidate_hp.json','w'),indent=2)
