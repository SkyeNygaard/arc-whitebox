#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
OGAP_DEFAULT=Path('/mnt/data/sole_path_inputs/ogap/whest_experiments_oracle_gap_20260730')
D=256; DEPTH=32; NQ=66048; THRESH=0.9939596; TARGET=1/4.34
PARTITIONS={
'p01_l1_final':[1,32],
'p02_l1_4_final':[1,4,32],
'p03_l1_8_final':[1,8,32],
'p04_l1_16_final':[1,16,32],
'p05_l1_24_final':[1,24,32],
'p06_l1_29_final':[1,29,32],
'p07_l1_4_8_final':[1,4,8,32],
'p08_l1_4_8_16_final':[1,4,8,16,32],
'p09_l1_4_8_16_24_final':[1,4,8,16,24,32],
'p10_agent2_dense':[1,4,8,16,24,27,29,31,32],
}
class CovCache:
    def __init__(self, states):
        self.states=states; self.den=len(next(iter(states.values())))-1; self.cache={}
    def get(self,a,b):
        key=(a,b)
        if key not in self.cache:
            if (b,a) in self.cache: self.cache[key]=self.cache[b,a].T
            else: self.cache[key]=self.states[a].T@self.states[b]/self.den
        return self.cache[key]
def edge_vars(C,times,S):
    out=[]
    for j in range(1,len(times)):
        p,q=times[j-1],times[j]; A,B=C[j-1],C[j]
        v=np.trace(B.T@S.get(q,q)@B)+np.trace(A.T@S.get(p,p)@A)-2*np.trace(A.T@S.get(p,q)@B)
        out.append(max(float(v),0.0))
    return np.asarray(out)
def factor_spd(A, rel=1e-9):
    A=.5*(A+A.T); scale=max(float(np.trace(A))/D,1e-20); reg=rel*scale
    for _ in range(10):
        try: return cho_factor(A+reg*np.eye(D),lower=True,check_finite=False)
        except Exception: reg*=10
    raise RuntimeError('Cholesky failed')
def block_solve(times,S,U,lams):
    nv=len(times)-1; k=U.shape[1]
    diag=[np.zeros((D,D)) for _ in range(nv)]; upper=[None]*(nv-1); rhs=[np.zeros((D,k)) for _ in range(nv)]
    for j,lam in enumerate(lams,1):
        i=j-1; p,q=times[j-1],times[j]; diag[i]+=lam*S.get(p,p)
        if j<len(times)-1:
            diag[j]+=lam*S.get(q,q); upper[i]=-lam*S.get(p,q)
        else: rhs[i]+=lam*S.get(p,q)@U
    cfs=[]; ys=[]
    for i in range(nv):
        A=diag[i].copy(); y=rhs[i].copy()
        if i:
            X=cho_solve(cfs[i-1],upper[i-1],check_finite=False)
            Z=cho_solve(cfs[i-1],ys[i-1],check_finite=False)
            A-=upper[i-1].T@X; y-=upper[i-1].T@Z
        cfs.append(factor_spd(A)); ys.append(y)
    C=[None]*nv; C[-1]=cho_solve(cfs[-1],ys[-1],check_finite=False)
    for i in range(nv-2,-1,-1): C[i]=cho_solve(cfs[i],ys[i]-upper[i]@C[i+1],check_finite=False)
    return C+[U]
def irls(times,S,U,gammas,maxit=35,tol=1e-8):
    tf=times[-1]; C=[]
    for t in times[:-1]: C.append(cho_solve(factor_spd(S.get(t,t),1e-7),S.get(t,tf)@U,check_finite=False))
    C.append(U); a=np.sqrt(np.asarray(gammas)); v=edge_vars(C,times,S); eps=max(math.sqrt(float(v.sum()))*1e-6,1e-13); hist=[]
    for it in range(maxit):
        obj=float(np.sum(a*np.sqrt(v))); lam=a/(2*np.sqrt(v+eps*eps)); Cn=block_solve(times,S,U,lam); vn=edge_vars(Cn,times,S); on=float(np.sum(a*np.sqrt(vn)))
        step=1.0
        while on>obj*(1+1e-9) and step>1/256:
            step*=.5; Ct=[(1-step)*x+step*y for x,y in zip(C,Cn)]; vt=edge_vars(Ct,times,S); ot=float(np.sum(a*np.sqrt(vt))); Cn,vn,on=Ct,vt,ot
        hist.append(on); rel=abs(obj-on)/max(obj,1e-30); C,v=Cn,vn; eps=max(eps*.5,1e-15)
        if it>=7 and rel<tol: break
    return C,v,hist
def kkt_certificate(C,times,states,gammas):
    n=len(next(iter(states.values()))); den=math.sqrt(n-1); a=np.sqrt(np.asarray(gammas)); Y=[]; R=[]
    for j in range(1,len(times)):
        rr=(states[times[j]]@C[j]-states[times[j-1]]@C[j-1])/den; nr=np.linalg.norm(rr); R.append(rr); Y.append(a[j-1]*rr/max(nr,1e-300))
    balances=[np.linalg.norm((states[times[0]]/den).T@Y[0])]
    for i in range(1,len(times)-1): balances.append(np.linalg.norm((states[times[i]]/den).T@(Y[i-1]-Y[i])))
    primal=float(sum(a[i]*np.linalg.norm(R[i]) for i in range(len(R))))
    return {'primal_raw':primal,'max_balance_fro':float(max(balances)),'relative_balance':float(max(balances)/max(primal,1e-30)),'max_norm_violation':float(max(np.linalg.norm(Y[i])-a[i] for i in range(len(Y))))}
def make_source(r,root,seed,rot,outdir):
    cache=outdir/'source_cache'/f'seed{seed}_rot{rot}.npz'; cache.parent.mkdir(parents=True,exist_ok=True)
    ws=r.make_weights(seed)
    if cache.exists():
        z=np.load(cache); return ws,z['U'],int(z['k']),float(z['B']),float(z['source_ratio']),float(z['baseline_error'])
    q=r.rotation(rot); x=r.design(); h=np.maximum(x@(ws[0]@q.T).T,0).astype(np.float32)
    for W in ws[1:]: h=np.maximum(h@W.T,0).astype(np.float32)
    yg=h.reshape(129,512,D).mean(1,dtype=np.float64); baseline=yg.mean(0); yc=yg-baseline
    ev,V=np.linalg.eigh(yc.T@yc/129); idx=np.argsort(ev)[::-1]; ev=ev[idx]; V=V[:,idx]
    k=int(np.searchsorted(np.cumsum(ev[:40])/ev[:40].sum(),THRESH)+1); U=V[:,:k]
    split='confirmation' if seed>=910079 else 'validation' if seed>=910033 else 'development'
    z=np.load(root/'results'/split/f'seed_{seed}'/f'seed{seed}_rot{rot}.npz'); e=baseline-z['truth']; B=float(np.mean(e*e)); res=e-U@(U.T@e); ratio=float(np.mean(res*res)/B); berr=float(np.max(np.abs(baseline-z['baseline'])))
    np.savez_compressed(cache,U=U,k=k,B=B,source_ratio=ratio,baseline_error=berr)
    return ws,U,k,B,ratio,berr
def pair_states(r,ws,n,seed,checkpoints):
    rng=np.random.default_rng(seed); x=rng.standard_normal((n,D)); x/=np.linalg.norm(x,axis=1,keepdims=True); x*=r.mean_chi(D); h=np.concatenate([x,-x]).astype(np.float32); out={}
    for li,W in enumerate(ws,1):
        h=np.maximum(h@W.T,0).astype(np.float32)
        if li in checkpoints:
            a=.5*(h[:n].astype(np.float64)+h[n:].astype(np.float64)); a-=a.mean(0); out[li]=a
    return out
def solve_case(U,B,Str,Sva,states_tr,partitions):
    rows={}
    for name,times in partitions.items():
        gam=[2*q/(DEPTH*NQ) for q in times[1:]]; C,v,h=irls(times,Str,U,gam); vv=edge_vars(C,times,Sva)
        tr=float(np.sum(np.sqrt(np.asarray(gam)*v/(D*B)))); va=float(np.sum(np.sqrt(np.asarray(gam)*vv/(D*B))))
        cert=kkt_certificate(C,times,{t:states_tr[t] for t in times},gam)
        rows[name]={'times':times,'train_S':tr,'valid_S':va,'validation_inflation':va/max(tr,1e-30),'iterations':len(h),'kkt':cert,'edge_train_variances':v.tolist(),'edge_valid_variances':vv.tolist()}
    return rows
def aggregate(cases,p,field):
    Bbar=float(np.mean([c['B'] for c in cases])); return float(np.mean([math.sqrt(c['B']/Bbar)*c['partitions'][p][field] for c in cases]))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--ogap',type=Path,default=OGAP_DEFAULT); ap.add_argument('--out',type=Path,default=Path('/mnt/data/sole_path_audit')); ap.add_argument('--n-train',type=int,default=4096); ap.add_argument('--n-valid',type=int,default=4096); ap.add_argument('--quick',action='store_true'); args=ap.parse_args()
    root=args.ogap; out=args.out; out.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(root/'src')); import run_fresh_suite as r
    cfg=json.loads((root/'confirmation_config.json').read_text()); splits={'development':cfg['base_seeds_development'],'validation':cfg['base_seeds_validation'],'confirmation':cfg['base_seeds_confirmation']}; rots=cfg['primary_rotation_seeds']; partitions=dict(list(PARTITIONS.items())[:4]) if args.quick else PARTITIONS; checkpoints=sorted(set(sum(partitions.values(),[])))
    allcases=[]; started=time.time()
    for split,seeds in splits.items():
        for seed in seeds:
            base_start=time.time(); ws=r.make_weights(int(seed)); sttr=pair_states(r,ws,args.n_train,70000000+int(seed),set(checkpoints)); stva=pair_states(r,ws,args.n_valid,80000000+int(seed),set(checkpoints)); Str=CovCache(sttr); Sva=CovCache(stva)
            for rot in rots:
                (out/'cases').mkdir(exist_ok=True); case_path=out/'cases'/f'seed{seed}_rot{rot}.json'
                if case_path.exists():
                    rec=json.loads(case_path.read_text()); missing={k:v for k,v in partitions.items() if k not in rec['partitions']}
                    if not missing:
                        allcases.append(rec); print(json.dumps({'resumed':rec['case_id'],'elapsed':time.time()-started}),flush=True); continue
                    _,U,k,B,sratio,berr=make_source(r,root,int(seed),int(rot),out); extra=solve_case(U,B,Str,Sva,sttr,missing); rec['partitions'].update(extra); case_path.write_text(json.dumps(rec,indent=2)); allcases.append(rec)
                    print(json.dumps({'extended':rec['case_id'],'added':list(missing),'best_valid':min(v['valid_S'] for v in rec['partitions'].values()),'elapsed':time.time()-started}),flush=True); continue
                _,U,k,B,sratio,berr=make_source(r,root,int(seed),int(rot),out); rows=solve_case(U,B,Str,Sva,sttr,partitions)
                rec={'split':split,'seed':int(seed),'rotation':int(rot),'case_id':f'seed{seed}_rot{rot}','k':k,'B':B,'source_ratio':sratio,'baseline_reconstruction_max_abs':berr,'partitions':rows}; allcases.append(rec)
                case_path.write_text(json.dumps(rec,indent=2))
                print(json.dumps({'done':rec['case_id'],'split':split,'k':k,'source_ratio':sratio,'best_train':min(v['train_S'] for v in rows.values()),'best_valid':min(v['valid_S'] for v in rows.values()),'elapsed':time.time()-started}),flush=True)
            print(json.dumps({'base_done':int(seed),'seconds':time.time()-base_start}),flush=True)
    dev=[c for c in allcases if c['split']=='development']; val=[c for c in allcases if c['split']=='validation']; conf=[c for c in allcases if c['split']=='confirmation']; selection=[]
    for p in partitions: selection.append({'partition':p,'development_train_S':aggregate(dev,p,'train_S'),'development_valid_S':aggregate(dev,p,'valid_S'),'validation_S':aggregate(val,p,'valid_S'),'confirmation_S':aggregate(conf,p,'valid_S')})
    selected=min(selection,key=lambda x:x['development_valid_S'])['partition']; rstar=float(sum(c['B']*c['source_ratio'] for c in conf)/sum(c['B'] for c in conf)); Sconf=aggregate(conf,selected,'valid_S'); Smax=math.sqrt(TARGET)-math.sqrt(rstar); xstar=Sconf/math.sqrt(rstar); minscore=(math.sqrt(rstar)+Sconf)**2
    Bbar=float(np.mean([c['B'] for c in conf])); oracle_best=float(np.mean([math.sqrt(c['B']/Bbar)*min(v['valid_S'] for v in c['partitions'].values()) for c in conf]))
    summary={'status':'STOP' if Sconf>Smax else 'CONTINUE','protected_data_opened':False,'sample_design':'independent antithetic fixed-radius sphere pairs','covariance_fit_cost_counted':False,'optimism_note':'covariance/control construction assigned zero deployment cost','n_train_pairs':args.n_train,'n_valid_pairs':args.n_valid,'partition_frontier':selection,'selected_partition':selected,'confirmation_source_ratio':rstar,'required_S_max':Smax,'confirmation_selected_valid_S':Sconf,'confirmation_oracle_per_case_partition_valid_S':oracle_best,'unconstrained_optimal_added_compute_ratio':xstar,'unconstrained_min_adjusted_ratio':minscore,'target_adjusted_ratio':TARGET,'case_count':len(allcases),'runtime_seconds':time.time()-started,'max_source_reconstruction_error':max(c['baseline_reconstruction_max_abs'] for c in allcases),'max_kkt_relative_balance':max(v['kkt']['relative_balance'] for c in allcases for v in c['partitions'].values())}
    (out/'SOURCE_SPECIFIC_SOCP_RESULTS.json').write_text(json.dumps({'summary':summary,'cases':allcases},indent=2)); (out/'SOURCE_SPECIFIC_SOCP_SUMMARY.json').write_text(json.dumps(summary,indent=2)); print(json.dumps({'FINAL':summary},indent=2),flush=True)
if __name__=='__main__': main()
