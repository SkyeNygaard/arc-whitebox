#!/usr/bin/env python3
from pathlib import Path
import sys,json,math,time
import numpy as np
from scipy.linalg import cho_factor,cho_solve
ROOT=Path('/mnt/data/sole_path_inputs/ogap/whest_experiments_oracle_gap_20260730')
OUT=Path('/mnt/data/sole_path_audit')
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(OUT))
import run_fresh_suite as r
import run_source_specific_socp_audit as a
TIMES=[1,4,32]

def dual_project(C,states,gammas):
    n=len(next(iter(states.values()))); den=math.sqrt(n-1); X={t:states[t]/den for t in TIMES}; aa=np.sqrt(np.asarray(gammas))
    R=[];Y=[]
    for j in range(1,len(TIMES)):
        rr=X[TIMES[j]]@C[j]-X[TIMES[j-1]]@C[j-1]; nr=np.linalg.norm(rr);R.append(rr);Y.append(aa[j-1]*rr/nr)
    # A maps z=[z1,z4] to [X4 z4-X1 z1, -X4 z4].
    X1,X4=X[1],X[4]
    A00=X1.T@X1;A01=-X1.T@X4;A11=2*(X4.T@X4)
    M=np.block([[A00,A01],[A01.T,A11]])
    rhs=np.vstack([-X1.T@Y[0],X4.T@(Y[0]-Y[1])])
    M=.5*(M+M.T);scale=max(np.trace(M)/M.shape[0],1e-20);cf=cho_factor(M+1e-12*scale*np.eye(M.shape[0]),lower=True,check_finite=False);Z=cho_solve(cf,rhs,check_finite=False);z1,z4=Z[:a.D],Z[a.D:]
    corr0=X4@z4-X1@z1;corr1=-X4@z4
    Yf=[Y[0]-corr0,Y[1]-corr1]
    alpha=min(1.0,aa[0]/np.linalg.norm(Yf[0]),aa[1]/np.linalg.norm(Yf[1]));Yf=[alpha*y for y in Yf]
    bal0=np.linalg.norm(X1.T@Yf[0]);bal1=np.linalg.norm(X4.T@(Yf[0]-Yf[1]));primal=sum(aa[i]*np.linalg.norm(R[i]) for i in range(2));dual=float(np.sum(Yf[1]*(X[32]@C[2])))
    return {'primal_raw':float(primal),'dual_raw':dual,'relative_gap':float((primal-dual)/primal),'alpha':float(alpha),'balance0':float(bal0),'balance1':float(bal1),'norms':[float(np.linalg.norm(y)) for y in Yf],'bounds':aa.tolist()}

def main():
    cfg=json.loads((ROOT/'confirmation_config.json').read_text());rows=[];t0=time.time()
    for seed in cfg['base_seeds_confirmation']:
        ws=r.make_weights(seed);states=a.pair_states(r,ws,2048,70000000+seed,set(TIMES));S=a.CovCache(states)
        for rot in cfg['primary_rotation_seeds']:
            _,U,k,B,sr,be=a.make_source(r,ROOT,seed,rot,OUT);gam=[2*q/(a.DEPTH*a.NQ) for q in TIMES[1:]];C,v,h=a.irls(TIMES,S,U,gam);cert=dual_project(C,states,gam);norm=math.sqrt(a.D*B);cert['primal_S']=cert['primal_raw']/norm;cert['dual_lower_S']=cert['dual_raw']/norm
            rec={'case_id':f'seed{seed}_rot{rot}','B':B,'source_ratio':sr,'k':k,'certificate':cert};rows.append(rec);print(json.dumps({'case':rec['case_id'],'primal_S':cert['primal_S'],'dual_S':cert['dual_lower_S'],'gap':cert['relative_gap'],'balance':max(cert['balance0'],cert['balance1'])}),flush=True)
    Bbar=np.mean([x['B'] for x in rows]);aggp=np.mean([math.sqrt(x['B']/Bbar)*x['certificate']['primal_S'] for x in rows]);aggd=np.mean([math.sqrt(x['B']/Bbar)*x['certificate']['dual_lower_S'] for x in rows]);rstar=sum(x['B']*x['source_ratio'] for x in rows)/sum(x['B'] for x in rows);smax=math.sqrt(1/4.34)-math.sqrt(rstar)
    result={'partition':TIMES,'sample_design':'2048 independent antithetic fixed-radius sphere pairs','cases':rows,'aggregate_primal_S':float(aggp),'aggregate_dual_lower_S':float(aggd),'required_S_max':float(smax),'dual_closes_gate':bool(aggd>smax),'max_relative_gap':max(x['certificate']['relative_gap'] for x in rows),'max_balance':max(max(x['certificate']['balance0'],x['certificate']['balance1']) for x in rows),'runtime_seconds':time.time()-t0}
    (OUT/'SELECTED_PARTITION_DUAL_CERTIFICATE.json').write_text(json.dumps(result,indent=2));print(json.dumps({'FINAL':result},indent=2))
if __name__=='__main__':main()
