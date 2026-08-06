import mpmath as mp,json
mp.mp.dps=80
N=20
# arrays coeff u^0..N
def add(a,b):return [(a[i] if i<len(a) else 0)+(b[i] if i<len(b) else 0) for i in range(N+1)]
def mul(a,b):
 o=[mp.mpf(0)]*(N+1)
 for i,x in enumerate(a):
  if not x:continue
  for j,y in enumerate(b):
   if i+j>N:break
   o[i+j]+=x*y
 return o
def sqrt_series(a):
 o=[mp.mpf(0)]*(N+1);o[0]=mp.sqrt(a[0])
 for n in range(1,N+1):o[n]=(a[n]-mp.fsum(o[i]*o[n-i] for i in range(1,n)))/(2*o[0])
 return o
def shift(a,k):return [mp.mpf(0)]*k+a[:N+1-k]
def power(a,n):
 o=[mp.mpf(0)]*(N+1);o[0]=1
 for _ in range(n):o=mul(o,a)
 return o
# F(w) delta next coefficients
f={2:mp.mpf(1),3:-2*mp.sqrt(2)/(3*mp.pi),5:-mp.sqrt(2)/(30*mp.pi),7:-3*mp.sqrt(2)/(560*mp.pi),9:-5*mp.sqrt(2)/(4032*mp.pi),11:-35*mp.sqrt(2)/(101376*mp.pi),13:-63*mp.sqrt(2)/(585728*mp.pi),15:-77*mp.sqrt(2)/(2129920*mp.pi),17:-143*mp.sqrt(2)/(11141120*mp.pi),19:-mp.mpf(6435)*mp.sqrt(2)/(1354760192*mp.pi)}
delta=[mp.mpf(0)]*(N+1);delta[2]=1
for dep in range(32):
 D=delta[2:]+[mp.mpf(0)]*2;D=D[:N+1]; w=shift(sqrt_series(D),1)
 nxt=[mp.mpf(0)]*(N+1)
 for j,cf in f.items():
  pj=power(w,j)
  for i in range(N+1):nxt[i]+=cf*pj[i]
 delta=nxt
# K=1-delta
K=[-x for x in delta];K[0]+=1
cand=json.load(open('/tmp/infinite_candidate_hp.json'));q=[mp.mpf(x) for x in cand['q']]
# t=1-u2 polynomial q
T=[mp.mpf(0)]*(N+1);T[0]=1;T[2]=-1
qser=[mp.mpf(0)]*(N+1);pw=[mp.mpf(0)]*(N+1);pw[0]=1
for j in range(6):
 if j>0:pw=mul(pw,T)
 for i in range(N+1):qser[i]+=q[j]*pw[i]
H=[K[i]-qser[i] for i in range(N+1)]
L=sqrt_series(H)
print('delta')
for i,x in enumerate(delta):
 if abs(x)>mp.mpf('1e-60'):print(i,mp.nstr(x,40))
print('L plus')
for i,x in enumerate(L):
 if abs(x)>mp.mpf('1e-60'):print(i,mp.nstr(x,50))
