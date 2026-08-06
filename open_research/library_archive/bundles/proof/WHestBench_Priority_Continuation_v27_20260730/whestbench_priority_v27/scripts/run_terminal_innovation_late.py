from __future__ import annotations
import json, math, sys, time, os
from pathlib import Path
import numpy as np
from scipy.special import gammaln

OGAP=Path('/mnt/data/priority_v26_inputs/ogap/whest_experiments_oracle_gap_20260730')
OUT=Path('/mnt/data/priority_v26_work/terminal_innovation_late_complete.json')
sys.path.insert(0,str(OGAP/'src'))
import run_fresh_suite as core

D=256; N=66048; TARGET=1/4.34; ENERGY=.9939595959595959; NP=2048
SEEDS=[910079,910081,910089,910103]
ROTS=[31001,31013,31033]
TS=[1,4,8,16,24,27,29,30,31]

def mch(): return math.sqrt(2)*math.exp(gammaln(128.5)-gammaln(128))

def states(W,seed):
    rng=np.random.default_rng(seed)
    u=rng.standard_normal((NP,D)).astype(np.float32)
    u/=np.linalg.norm(u,axis=1,keepdims=True)
    x=(mch()*u).astype(np.float32)
    h=np.concatenate([x,-x],axis=0)
    out={}
    wanted=set(TS+[32])
    for i,w in enumerate(W,1):
        h=np.maximum(h@w.T,0).astype(np.float32,copy=False)
        if i in wanted:
            out[i]=.5*(h[:NP].astype(np.float64)+h[NP:].astype(np.float64))
    return out

def cv(a,b):
    a=a-a.mean(0); b=b-b.mean(0)
    return a.T@b/(len(a)-1)

def psd_pinv(a,rc=1e-8):
    a=(a+a.T)/2
    e,v=np.linalg.eigh(a)
    cutoff=max(e[-1]*rc,0.0)
    keep=e>cutoff
    if not np.any(keep): return np.zeros_like(a),0,float('inf')
    return (v[:,keep]/e[keep])@v[:,keep].T,int(keep.sum()),float(e[-1]/e[keep].min())

def source(W,r):
    _,c,_=core.forward_full(W,core.rotation(r),checkpoint_depths={31})
    h=next(a for l,a in c if l==31)
    base=h.mean(0,dtype=np.float64)
    g=h.reshape(129,512,D).mean(1,dtype=np.float64)
    y=g-g.mean(0)
    _,s,vh=np.linalg.svd(y,full_matrices=False)
    q=s[:40]**2
    k=int(np.searchsorted(np.cumsum(q)/q.sum(),ENERGY)+1)
    return vh[:k].T.copy(),k,base

rows=[]
OUT.write_text(json.dumps({'status':'RUNNING','rows':rows},indent=2))
for seed in SEEDS:
    t0=time.time(); W=core.make_weights(seed); st=states(W,seed+2210000)
    Y=st[32]; A=cv(Y,Y); pre={}
    for t in TS:
        X=st[t]; D0=cv(X,X); B=cv(Y,X); P,rank,cond=psd_pinv(D0)
        R=(A-B@P@B.T); R=(R+R.T)/2
        # Remove tiny negative eigenspace caused by covariance roundoff.
        er,vr=np.linalg.eigh(R); er=np.maximum(er,0.0); R=(vr*er)@vr.T
        pre[t]=(R,rank,cond)
    for rot in ROTS:
        U,k,base=source(W,rot)
        p=OGAP/'results'/'confirmation'/f'seed_{seed}'/f'seed{seed}_rot{rot}.npz'
        with np.load(p) as z:
            truth=z['truth'].astype(float); stored=z['baseline'].astype(float)
        assert np.max(np.abs(stored-base))<2e-12
        e=base-truth; den=float(e@e); b=U.T@e
        rstar=float((den-b@b)/den); smax=math.sqrt(TARGET)-math.sqrt(rstar)
        for t in TS:
            R,rank,cond=pre[t]
            q=max(float(np.trace(U.T@R@U)),0.0)
            S=math.sqrt((2/N)*q/den)
            lower=(math.sqrt(rstar)+S)**2
            row={'case_id':f'seed{seed}_rot{rot}','seed':seed,'rotation':rot,'rank':k,
                 'checkpoint':t,'base_error_sq':den,'oracle_ratio':rstar,'Smax':smax,
                 'terminal_innovation_trace':q,'S_terminal_lower_bound':S,
                 'score_lower_bound':lower,'pass_not_ruled_out':lower<TARGET,
                 'state_cov_rank':rank,'state_cov_condition':cond}
            rows.append(row)
        OUT.write_text(json.dumps({'status':'RUNNING','sampling_pairs':NP,'rows':rows},indent=2,sort_keys=True)+'\n')
        print(f'seed{seed}_rot{rot} k={k} t24={rows[-5]["score_lower_bound"]:.6g} t29={rows[-3]["score_lower_bound"]:.6g} t30={rows[-2]["score_lower_bound"]:.6g} t31={rows[-1]["score_lower_bound"]:.6g}',flush=True)
    print('seed seconds',seed,time.time()-t0,flush=True)

summ={}
for t in TS:
    rr=[r for r in rows if r['checkpoint']==t]
    den=sum(r['base_error_sq'] for r in rr)
    rp=sum(r['oracle_ratio']*r['base_error_sq'] for r in rr)/den
    S=math.sqrt(sum(r['base_error_sq']*r['S_terminal_lower_bound']**2 for r in rr)/den)
    score=(math.sqrt(rp)+S)**2
    summ[str(t)]={'pooled_oracle_ratio':rp,'S_rms_lower_bound':S,'score_lower_bound_proxy':score,
                  'target':TARGET,'ruled_out_pooled':score>=TARGET,
                  'cases_ruled_out':sum(not r['pass_not_ruled_out'] for r in rr),
                  'cases_not_ruled_out':sum(r['pass_not_ruled_out'] for r in rr),
                  'minimum_case_score_lower_bound':min(r['score_lower_bound'] for r in rr),
                  'maximum_case_score_lower_bound':max(r['score_lower_bound'] for r in rr),
                  'median_case_score_lower_bound':float(np.median([r['score_lower_bound'] for r in rr])),
                  'median_S_lower_bound':float(np.median([r['S_terminal_lower_bound'] for r in rr]))}
out={'status':'COMPLETE','schema_version':1,
     'theorem':'For any exact linear checkpoint telescope whose latest preterminal state is h_t, the total root variance-cost difficulty S is at least the optimal terminal innovation difficulty reported here, because all earlier nonnegative block terms are granted free.',
     'source':'Agent8 frozen adaptive direct-output basis-PCA source','source_energy_rule':ENERGY,
     'sampling':f'{NP} fixed-radius antithetic pairs per base seed','pair_cost_fraction':2/N,
     'rows':rows,'summaries':summ,'target':TARGET}
OUT.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(summ,indent=2))
