import json
from pathlib import Path
from mpmath import mp
mp.dps=180
N=320

def mul(a,b,n=N):
    out=[mp.mpf('0')]*(n+1)
    la=min(len(a)-1,n); lb=min(len(b)-1,n)
    for i in range(la+1):
        ai=a[i]
        if not ai: continue
        jmax=min(lb,n-i)
        for j in range(jmax+1):
            out[i+j]+=ai*b[j]
    return out

def sqrt_series(h,n=N):
    g=[mp.mpf('0')]*(n+1)
    g[0]=mp.sqrt(h[0])
    for k in range(1,n+1):
        s=mp.mpf('0')
        for i in range(1,k): s += g[i]*g[k-i]
        g[k]=(h[k]-s)/(2*g[0])
    return g

def inv_series(g,n=N):
    r=[mp.mpf('0')]*(n+1); r[0]=1/g[0]
    for k in range(1,n+1):
        s=mp.mpf('0')
        for i in range(1,k+1): s += g[i]*r[k-i]
        r[k]=-s/g[0]
    return r

def deriv(f,n=N):
    out=[mp.mpf('0')]*(n+1)
    for k in range(n): out[k]=(k+1)*f[k+1]
    return out

def integrate(d,c0,n=N):
    out=[mp.mpf('0')]*(n+1); out[0]=c0
    for k in range(1,n+1): out[k]=d[k-1]/k
    return out

def asin_series(f,n=N):
    f2=mul(f,f,n)
    h=[-x for x in f2]; h[0]+=1
    root=sqrt_series(h,n)
    inv=inv_series(root,n)
    yp=mul(deriv(f,n),inv,n)
    return integrate(yp,mp.asin(f[0]),n)

def kappa_scalar(x):
    return (mp.sqrt(1-x*x)+(mp.pi-mp.acos(x))*x)/mp.pi

def kappa_series(f,n=N):
    a=asin_series(f,n)
    kp=a[:]
    kp[0]+=mp.pi/2
    kp=[x/mp.pi for x in kp]
    outp=mul(kp,deriv(f,n),n)
    return integrate(outp,kappa_scalar(f[0]),n)

f=[mp.mpf('0')]*(N+1); f[1]=1
for depth in range(32):
    f=kappa_series(f,N)
    print(depth+1, mp.nstr(f[0],20), mp.nstr(f[1],20))

p=Path(__file__).resolve().parents[1]/'proof'/'signed'/'K32_MACLAURIN_INTERVALS_ORDER320.json'
d=json.loads(p.read_text())
fail=[]; margins=[]
for i,(lo_s,hi_s) in enumerate(d['maclaurin_intervals']):
    lo=mp.mpf(lo_s); hi=mp.mpf(hi_s); x=f[i]
    if not (lo <= x <= hi): fail.append((i,mp.nstr(x,80),lo_s,hi_s))
    margins.append(min(x-lo,hi-x))
print('fail count',len(fail))
for row in fail[:20]: print('FAIL',row)
print('min interval containment margin index',min(range(len(margins)),key=lambda i:margins[i]), mp.nstr(min(margins),30))
print('selected')
for i in [0,1,2,3,4,5,31,32,100,200,319,320]:
    lo,hi=map(mp.mpf,d['maclaurin_intervals'][i]); print(i,mp.nstr(f[i],80),'relpos',mp.nstr((f[i]-lo)/(hi-lo),20))
