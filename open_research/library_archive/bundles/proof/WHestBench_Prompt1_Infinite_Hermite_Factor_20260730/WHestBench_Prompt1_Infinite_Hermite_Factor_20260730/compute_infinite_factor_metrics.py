import mpmath as mp,json,math
mp.mp.dps=100
D=256;N=66048
cand=json.load(open('/tmp/infinite_candidate_hp.json'));q=[mp.mpf(x) for x in cand['q']];c=[mp.mpf(x) for x in cand['c']];s=c[3]
jet=json.load(open('/mnt/data/WHestBench_Prompt1_Weighted_Floor_Continuation_20260730/MPFR_KERNEL_JET_511.json'));alpha=[(mp.mpf(x['lo'])+mp.mpf(x['hi']))/2 for x in jet['coefficients']]
# monomial->G P up to5
P=[[mp.mpf(0)]*6 for _ in range(6)];P[0][0]=1
for p in range(5):
 for l in range(p+1):
  v=P[p][l]
  if v:
   P[p+1][l+1]+=v*mp.mpf(l+D-2)/(2*l+D-2)
   if l:P[p+1][l-1]+=v*mp.mpf(l)/(2*l+D-2)
qg=[mp.fsum(q[p]*P[p][r] for p in range(r,6)) for r in range(6)]
# k Gegenbauer low via alpha order511
# moments projection recurrence for r0..5
state=[mp.mpf(0)]*6;state[0]=1;kg=[mp.mpf(0)]*6
for p,a in enumerate(alpha):
 for r in range(6):kg[r]+=a*state[r]
 if p<511:
  nxt=[mp.mpf(0)]*6
  for l,v in enumerate(state):
   if not v:continue
   if l+1<6:nxt[l+1]+=v*mp.mpf(l+D-2)/(2*l+D-2)
   if l:nxt[l-1]+=v*mp.mpf(l)/(2*l+D-2)
  state=nxt
bg=[kg[r]-qg[r] for r in range(6)]
L1=mp.sqrt(1-mp.fsum(q));b0=bg[0]
F=b0-2*s*L1+L1**2/N
ku=mp.mpf('2.4336603575430052276094665026697645914811e-7');frac=F/ku
print('roots',*cand['roots'])
for r in range(6):print('r',r,'k',mp.nstr(kg[r],50),'qg',mp.nstr(qg[r],50),'b',mp.nstr(bg[r],50))
print('s',mp.nstr(s,60),'c4',mp.nstr(c[4],60),'ratio',mp.nstr(c[4]/s,30))
print('L1',mp.nstr(L1,60),'b0',mp.nstr(b0,60),'F',mp.nstr(F,60),'frac',mp.nstr(frac,60),'cap',mp.nstr(1/frac,60))
print('target20/21 gap',mp.nstr(frac-mp.mpf(20)/21,50))
# endpoint singular amplitudes
# iterate 31 from 0
phi=lambda t:(mp.sqrt(1-t*t)+(mp.pi-mp.acos(t))*t)/mp.pi
phip=lambda t:(mp.pi-mp.acos(t))/mp.pi
x=mp.mpf(0);prod=mp.mpf(1)
for _ in range(31):prod*=phip(x);x=phi(x)
a=2*mp.sqrt(2)/(3*mp.pi);Ap=32*a;Am=a*prod
Hm=x-mp.fsum(q[j]*(-1)**j for j in range(6));Hp=1-mp.fsum(q)
Cp=Ap/Hp;Cm=Am/Hm
Bp=Ap/(2*mp.sqrt(Hp));Bm=Am/(2*mp.sqrt(Hm))
print('endpoint Kminus q Hp Hm',mp.nstr(Hp,40),mp.nstr(Hm,40))
print('log singular Cp Cm ratio',mp.nstr(Cp,40),mp.nstr(Cm,40),mp.nstr(Cm/Cp,30))
print('L singular Bp Bm ratio',mp.nstr(Bp,40),mp.nstr(Bm,40),mp.nstr(Bm/Bp,30))
