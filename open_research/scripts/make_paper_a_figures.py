#!/usr/bin/env python3
from pathlib import Path
import json
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUTS=[ROOT/'figures',ROOT/'papers/figures']
for out in OUTS: out.mkdir(parents=True,exist_ok=True)

# Ceiling factors. These are Kerdock risk divided by the certified lower bound.
labels=['Nonnegative\nstatic','Arbitrary signed\nstatic','Motivating\ncompetition gap']
vals=[1.0002332417295004,1.0671673322143325,4.34]
fig,ax=plt.subplots(figsize=(10,5.6))
bars=ax.bar(labels,vals)
ax.axhline(1,linewidth=1)
ax.set_ylabel('Upper bound on Kerdock risk / class-optimum risk')
ax.set_title('Certified fixed-node-budget ratios versus the motivating gap')
for b,v in zip(bars,vals):
    ax.text(b.get_x()+b.get_width()/2,v+0.04 if v>1.2 else v+0.01,f'{v:.7g}x',ha='center',va='bottom')
ax.text(1,0.73,'Signed factor 1.067167x = at most 6.2940%\nreduction relative to Kerdock risk',ha='center')
fig.tight_layout()
for out in OUTS: fig.savefig(out/'paperA_improvement_ceiling.png',dpi=180)
plt.close(fig)

# Sign count from exact frozen-witness verifier. Add a few logarithmic q values.
from decimal import Decimal
from fractions import Fraction
import math
cert=json.loads((ROOT/'evidence/primary_theory/signed_replay/SIGNED_NEAR_OPTIMALITY_CERTIFICATE_BLOCKTRACE_ORDER320.json').read_text())
D=256;N=66048

def fdec(s): return Fraction(Decimal(s))
def hd(l):
    if l==0:return 1
    if l==1:return D
    return math.comb(D+l-1,l)-math.comb(D+l-3,l-2)
def floor(q):
    total=Fraction(0)
    for row in cert['components']:
        s=int(row['s']);r=Fraction(row['r']);y=fdec(row['y']);ds,dt=hd(s),hd(s+1)
        T=Fraction(ds)+r*dt; S2=Fraction(ds)+r*r*dt
        total += y*(T*T/(N-q)-S2)/(T*T/N-S2)
    return total
k=fdec(cert['certified_result']['kerdock_mse_upper_bound'])
qs=[1,2,16,64,256,1024,1072,2048,4160,8192]
gains=[float(k/floor(q)) for q in qs]
fig,ax=plt.subplots(figsize=(10,5.6))
ax.plot(qs,gains,marker='o')
ax.set_xscale('log')
ax.axhline(1.05,linestyle='--',linewidth=1)
ax.axhline(1.0,linestyle='--',linewidth=1)
ax.set_xlabel('Minimum negative-weight support count (log scale)')
ax.set_ylabel('Upper bound on Kerdock risk / rule risk')
ax.set_title('Many negative support entries become counterproductive')
ax.annotate('below 1.05x',xy=(1072,gains[qs.index(1072)]),xytext=(250,1.025),arrowprops={'arrowstyle':'->'})
ax.annotate('certified worse than Kerdock',xy=(4160,gains[qs.index(4160)]),xytext=(800,0.96),arrowprops={'arrowstyle':'->'})
fig.tight_layout()
for out in OUTS: fig.savefig(out/'paperA_sign_count.png',dpi=180)
plt.close(fig)

# Proof pipeline.
fig,ax=plt.subplots(figsize=(12,5.5)); ax.axis('off')
steps=['Deep-ReLU\nlimiting kernel','Harmonic / RKHS\nrisk decomposition','Auxiliary or inertia\nlower certificate','Exact-rational replay\n+ directed intervals']
xs=[0.02,0.27,0.52,0.77]
for x,txt in zip(xs,steps):
    ax.add_patch(plt.Rectangle((x,0.42),0.21,0.25,fill=False,linewidth=1.5))
    ax.text(x+0.105,0.545,txt,ha='center',va='center',fontsize=13,weight='bold')
for x in xs[:-1]: ax.annotate('',xy=(x+0.25,0.545),xytext=(x+0.21,0.545),arrowprops={'arrowstyle':'-|>','lw':1.5})
ax.text(0.5,0.83,'Computer-assisted proof pipeline',ha='center',fontsize=22,weight='bold')
ax.text(0.5,0.18,'Publication gates: independently reconstruct the full inherited T22/kernel interval stack,\nrun public multi-platform CI, and obtain named human mathematical review.',ha='center',fontsize=12)
fig.tight_layout()
for out in OUTS: fig.savefig(out/'paperA_proof_pipeline.png',dpi=180)
plt.close(fig)
