import mpmath as mp,json,math,time
mp.mp.dps=120
D=256;M=505
jet=json.load(open('/mnt/data/WHestBench_Prompt1_Weighted_Floor_Continuation_20260730/MPFR_KERNEL_JET_511.json'))
alpha=[(mp.mpf(x['lo'])+mp.mpf(x['hi']))/2 for x in jet['coefficients']]
sol=json.load(open('/tmp/infinite_candidate_hp.json'))
rs=[mp.mpf(x) for x in sol['roots']]
def Kd(t):
 return mp.fsum(alpha[n]*t**n for n in range(512)),mp.fsum(n*alpha[n]*t**(n-1) for n in range(1,512))
def conv(a,b,n=None):
 if n is None:n=len(a)+len(b)-2
 o=[mp.mpf(0)]*(n+1)
 for i,x in enumerate(a):
  for j,y in enumerate(b):
   if i+j<=n:o[i+j]+=x*y
 return o
A=mp.matrix(6);b=mp.matrix(6,1);row=0
for r in rs:
 k,kp=Kd(r)
 for j in range(6):A[row,j]=r**j
 b[row]=k;row+=1
 for j in range(6):A[row,j]=0 if j==0 else j*r**(j-1)
 b[row]=kp;row+=1
q=list(mp.lu_solve(A,b));P=[mp.mpf(1)]
for r in rs:P=conv(P,[-r,1])
P2=conv(P,P)
H=[alpha[n]-(q[n] if n<6 else 0) for n in range(512)]
rem=H[:];Q=[mp.mpf(0)]*506
for n in range(511,5,-1):
 coef=rem[n];Q[n-6]=coef
 for j in range(7):rem[n-6+j]-=coef*P2[j]
print('max remainder',mp.nstr(max(abs(x) for x in rem[:6]),20))
S=Q
U=[mp.mpf(0)]*(M+1);U[0]=mp.sqrt(S[0])
for n in range(1,M+1):
 U[n]=(S[n]-mp.fsum(U[i]*U[n-i] for i in range(1,n)))/(2*U[0])
L=conv(P,U,M)
print('S signs',sum(x<0 for x in S),[i for i,x in enumerate(S) if x<0][:20])
print('U signs',sum(x<0 for x in U),[i for i,x in enumerate(U) if x<0][:20])
print('L signs',sum(x<0 for x in L),[i for i,x in enumerate(L) if x<0][:40])
for n in list(range(20))+[50,100,150,200,250,300,350,400,450,500,505]:
 print('coef',n,mp.nstr(S[n],16),mp.nstr(U[n],16),mp.nstr(L[n],16))
# project to Gegenbauer up to 300 via mp recurrence P[p][l], memory float enough? use mp sparse row recurrence and accumulate z online
maxl=300
state=[mp.mpf(0)]*(maxl+1);state[0]=1
z=[mp.mpf(0)]*(maxl+1)
for p in range(M+1):
 lp=L[p]
 if lp:
  for l in range(min(p,maxl)+1):
   if state[l]:z[l]+=lp*state[l]
 if p<M:
  nxt=[mp.mpf(0)]*(maxl+1)
  for l in range(min(p,maxl)+1):
   v=state[l]
   if not v:continue
   if l+1<=maxl:nxt[l+1]+=v*mp.mpf(l+D-2)/(2*l+D-2)
   if l:nxt[l-1]+=v*mp.mpf(l)/(2*l+D-2)
  state=nxt
print('z neg',sum(x<0 for x in z),[i for i,x in enumerate(z) if x<0][:50])
for l in list(range(20))+[30,40,50,60,80,100,120,150,200,250,300]:print('z',l,mp.nstr(z[l],20))
json.dump({'roots':[str(x) for x in rs],'q':[str(x) for x in q],'P':[str(x) for x in P], 'S':[str(x) for x in S], 'U':[str(x) for x in U], 'L':[str(x) for x in L], 'z0_300':[str(x) for x in z], 'remainder':[str(x) for x in rem[:6]]},open('/tmp/infinite_series_505.json','w'),indent=2)
