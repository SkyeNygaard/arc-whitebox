#!/usr/bin/env python3
from pathlib import Path
import sys, json, math, time
import numpy as np
ROOT=Path('/mnt/data/sole_path_inputs/ogap/whest_experiments_oracle_gap_20260730')
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,'/mnt/data/sole_path_audit')
import run_fresh_suite as r
import run_source_specific_socp_audit as a
D=a.D; DEPTH=a.DEPTH; NQ=a.NQ
PARTS={k:v for k,v in a.PARTITIONS.items() if k in ['p01_l1_final','p02_l1_4_final','p03_l1_8_final','p04_l1_16_final']}

def onb_states(ws,m,seed,checkpoints,batch=8):
    rng=np.random.default_rng(seed); acc={t:[] for t in checkpoints}; rad=r.mean_chi(D)
    for start in range(0,m,batch):
        b=min(batch,m-start); qs=[]
        for _ in range(b):
            A=rng.standard_normal((D,D)); q,rr=np.linalg.qr(A); q*=np.sign(np.diag(rr))[None,:]; qs.append((rad*q).astype(np.float32))
        pos=np.concatenate(qs,axis=0); h=np.concatenate([pos,-pos],axis=0)
        for li,W in enumerate(ws,1):
            h=np.maximum(h@W.T,0).astype(np.float32)
            if li in checkpoints:
                hp=h[:b*D].reshape(b,D,D); hn=h[b*D:].reshape(b,D,D)
                acc[li].append(.5*(hp.mean(1,dtype=np.float64)+hn.mean(1,dtype=np.float64)))
    out={}
    for t,ls in acc.items():
        z=np.concatenate(ls); z-=z.mean(0); out[t]=z
    return out

def main():
    out=Path('/mnt/data/sole_path_audit'); m=1024; seed=910079; ws=r.make_weights(seed); cps=set(sum(PARTS.values(),[])); t0=time.time()
    tr=onb_states(ws,m,92060730,cps); print(json.dumps({'stage':'train_states','seconds':time.time()-t0}),flush=True)
    va=onb_states(ws,m,93060730,cps); print(json.dumps({'stage':'valid_states','seconds':time.time()-t0}),flush=True)
    Str=a.CovCache(tr); Sva=a.CovCache(va); cases=[]
    for rot in [31001,31013,31033]:
        _,U,k,B,sr,be=a.make_source(r,ROOT,seed,rot,out); rows={}
        for name,times in PARTS.items():
            gam=[512*q/(DEPTH*NQ) for q in times[1:]]; C,v,h=a.irls(times,Str,U,gam); vv=a.edge_vars(C,times,Sva)
            rows[name]={'train_S':float(np.sum(np.sqrt(np.asarray(gam)*v/(D*B)))),'valid_S':float(np.sum(np.sqrt(np.asarray(gam)*vv/(D*B)))),'iterations':len(h)}
        rec={'case_id':f'seed{seed}_rot{rot}','k':k,'B':B,'source_ratio':sr,'partitions':rows}; cases.append(rec); print(json.dumps(rec),flush=True)
    Bbar=np.mean([x['B'] for x in cases]); agg=[]
    for p in PARTS:
        agg.append({'partition':p,'train_S':float(np.mean([math.sqrt(c['B']/Bbar)*c['partitions'][p]['train_S'] for c in cases])),'valid_S':float(np.mean([math.sqrt(c['B']/Bbar)*c['partitions'][p]['valid_S'] for c in cases]))})
    result={'sample_design':'random Haar orthonormal basis with antipodes','unit_rows':512,'train_units':m,'valid_units':m,'base_seed':seed,'cases':cases,'aggregate':agg,'best_valid':min(x['valid_S'] for x in agg),'required_S_max':math.sqrt(1/4.34)-math.sqrt(sum(c['B']*c['source_ratio'] for c in cases)/sum(c['B'] for c in cases)),'runtime_seconds':time.time()-t0}
    (out/'ONB_ESCAPE_AUDIT.json').write_text(json.dumps(result,indent=2)); print(json.dumps({'FINAL':result},indent=2),flush=True)
if __name__=='__main__': main()
