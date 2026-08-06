from mpmath import mp
mp.dps=100
D=256; N=66048

def kappa(t): return (mp.sqrt(1-t*t)+(mp.pi-mp.acos(t))*t)/mp.pi
def kp(t): return (mp.pi-mp.acos(t))/mp.pi
def K(t):
    y=mp.mpf(t)
    for _ in range(32): y=kappa(y)
    return y
def Kp(t):
    y=mp.mpf(t); der=mp.mpf(1)
    for _ in range(32): der*=kp(y); y=kappa(y)
    return der
roots=mp.polyroots([22102,21930,-87,-85],maxsteps=100,error=False)
roots=sorted([mp.re(r) for r in roots])
print('roots',*[mp.nstr(r,60) for r in roots],sep='\n')
# monomial c0..c5 solve values and derivatives
A=[]; b=[]
for r in roots:
    A.append([r**k for k in range(6)]); b.append(K(r))
    A.append([0]+[k*r**(k-1) for k in range(1,6)]); b.append(Kp(r))
c=mp.lu_solve(mp.matrix(A),mp.matrix(b))
print('monomial',*[mp.nstr(x,60) for x in c],sep='\n')
def h(t): return sum(c[k]*t**k for k in range(6))
# normalized Gegenbauer polynomials monomial arrays exact-ish mp
G=[[mp.mpf(1)],[mp.mpf(0),mp.mpf(1)]]
for l in range(1,5):
    Acoef=mp.mpf(2*l+D-2)/(l+D-2); Bcoef=mp.mpf(l)/(l+D-2)
    g=[mp.mpf(0)]*(l+2)
    for k,x in enumerate(G[l]): g[k+1]+=Acoef*x
    for k,x in enumerate(G[l-1]): g[k]-=Bcoef*x
    G.append(g)
work=list(c); gc=[mp.mpf(0)]*6
for l in range(5,-1,-1):
    gc[l]=work[l]/G[l][l]
    for k,x in enumerate(G[l]): work[k]-=gc[l]*x
print('gegenbauer',*[mp.nstr(x,70) for x in gc],sep='\n')
# objective and mean coefficient separately
phi=gc[0]+(1-h(1))/N
print('Phi energy',mp.nstr(phi,80))
# kernel mean by quadrature as in release
cc=mp.gamma(mp.mpf(D)/2)/(mp.sqrt(mp.pi)*mp.gamma(mp.mpf(D-1)/2))
mean=cc*mp.quad(lambda th: K(mp.cos(th))*mp.sin(th)**(D-2),[0,mp.pi/2,mp.pi])
print('mean',mp.nstr(mean,80),'risk LB',mp.nstr(phi-mean,80))
# residual samples, excluding near roots to avoid rounding zero
mins=(None,mp.inf)
for i in range(20001):
    t=mp.mpf(-1)+mp.mpf(2)*i/20000
    r=K(t)-h(t)
    if r<mins[1]: mins=(t,r)
print('grid min',mp.nstr(mins[0],30),mp.nstr(mins[1],30))
print('end residuals',mp.nstr(K(-1)-h(-1),50),mp.nstr(K(1)-h(1),50))
for r in roots: print('contact',mp.nstr(r,30),mp.nstr(K(r)-h(r),20),mp.nstr(Kp(r)-sum(k*c[k]*r**(k-1) for k in range(1,6)),20))
