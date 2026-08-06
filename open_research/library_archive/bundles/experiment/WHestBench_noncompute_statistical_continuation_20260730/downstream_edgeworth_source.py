from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import norm

ROOT=Path('/mnt/data/whest_reopened')
A9=ROOT/'agent9_10_oracle_bundle/agent9_10_oracle_bundle'
WHITE=ROOT/'arc_code/arc_whitebox'
CEIL=ROOT/'arc_code/arc_ceiling'
sys.path[:0]=[str(A9),str(WHITE/'src'),str(CEIL),'/mnt/data/competition_relevance_20260730']
import arc_experiments as ae
import whest.gaussmath as gm
from edgeworth34_rank1_eval import moments34
from sample_edgeworth34_lowrank import cumulants, trunc

D=256
OUT=Path('/mnt/data/competition_relevance_20260730/downstream_edgeworth_source')
OUT.mkdir(parents=True,exist_ok=True)


def mc_final(w,n,seed,chunk=4096):
    rng=np.random.default_rng(seed);acc=np.zeros(D);done=0
    while done<n:
        b=min(chunk,n-done);a=rng.standard_normal((b,D),dtype=np.float32)
        for W in w:a=np.maximum(a@W,0)
        acc+=a.sum(0,dtype=np.float64);done+=b
    return acc/n


def kerdock_summary(w,rot,target_layer):
    # Retain only the target preactivation and suffix gate probabilities.
    a=ae.first_activation(w[0],rot)
    target_H=None; target_A=None; gate_probs={}
    for li,W in enumerate(w[1:],start=1):
        h=a@W
        if li==target_layer:
            target_H=h.astype(np.float64)
            target_A=np.maximum(h,0).astype(np.float32)
            a=target_A
        else:
            if li>target_layer:
                gate_probs[li]=(h>0).mean(0).astype(np.float64)
            a=np.maximum(h,0)
    return target_H,target_A,gate_probs,a.mean(0,dtype=np.float64)


def kerdock_final_mean(w,rot):
    a=ae.first_activation(w[0],rot)
    for W in w[1:]:
        a=np.maximum(a@W,0)
    return a.mean(0,dtype=np.float64)

def mse(a,b):return float(np.mean((a-b)**2))


def relu_mean(mu,var):
    sd=np.sqrt(np.maximum(var,1e-30));t=mu/sd
    return mu*norm.cdf(t)+sd*norm.pdf(t)


def source_from_layer(w,H,A_target,gate_probs,layer,kernel_rank,cov_rank,order):
    H=H.astype(np.float64,copy=False)
    mu=H.mean(0);z=H-mu;sig=z.T@z/len(z)
    c21,c31,c22=cumulants(H)
    gm0,gc0=gm.relu_cov_from_gauss(mu,sig,n_nodes=12)
    gsec=gc0+np.outer(gm0,gm0)
    c3,c4=moments34(mu,sig,c21,c31,c22,gsec,kernel_rank)
    cov_edge=c3 if order==3 else c4
    dc=cov_edge-gc0
    if cov_rank and cov_rank < D:
        dc=trunc(dc,cov_rank)
    Wnext=w[layer+1].astype(np.float64)
    var_g=np.sum((gc0@Wnext)*Wnext,axis=0)
    var_e=var_g+np.sum((dc@Wnext)*Wnext,axis=0)
    # The preactivation mean of layer+1 is determined by the same-design activation mean.
    mu_next=A_target.astype(np.float64).mean(0)@Wnext
    delta=relu_mean(mu_next,var_e)-relu_mean(mu_next,var_g)
    # Carry a signed mean perturbation through the realized suffix using empirical expected gates.
    for li in range(layer+2,len(w)):
        delta=(delta@w[li].astype(np.float64))*gate_probs[li]
    return delta,{
        'delta_var_norm':float(np.linalg.norm(var_e-var_g)),
        'delta_activation_norm':float(np.linalg.norm(relu_mean(mu_next,var_e)-relu_mean(mu_next,var_g))),
        'source_norm':float(np.linalg.norm(delta)),
        'mean_abs_skew':float(np.mean(np.abs(np.diag(c21)/np.maximum(np.diag(sig),1e-30)**1.5))),
        'mean_abs_excess':float(np.mean(np.abs(np.diag(c22)/np.maximum(np.diag(sig),1e-30)**2))),
    }


def run(seed,layers,nref,rot,kernel_ranks,cov_ranks,orders,reference_mode):
    t=time.time();w=ae.make_weights(seed)
    if reference_mode=='kerdock':
        refs=[kerdock_final_mean(w,r) for r in (101,103,107,109)]
        r1=.5*(refs[0]+refs[1]);r2=.5*(refs[2]+refs[3])
    else:
        r1=mc_final(w,nref,4_000_000+seed);r2=mc_final(w,nref,4_100_000+seed)
    truth=.5*(r1+r2);noise=mse(r1,r2)/4
    rows={}
    alphas=[-1.0,-.75,-.5,-.3,-.2,-.1,-.05,0.05,.1,.2,.3,.5,.75,1.0]
    for layer in layers:
        H,A_target,gate_probs,base=kerdock_summary(w,rot,layer);bm=mse(base,truth)
        for kr in kernel_ranks:
            for cr in cov_ranks:
                for order in orders:
                    src,diag=source_from_layer(w,H,A_target,gate_probs,layer,kr,cr,order)
                    item={'diagnostics':diag,'alpha_rows':{}}
                    for a in alphas:
                        q=base+a*src;mm=mse(q,truth)
                        item['alpha_rows'][f'{a:g}']={'mse':mm,'ratio':mm/bm}
                    # target-labeled scalar ceiling in this one-dimensional source.
                    e=base-truth
                    astar=-float(e@src)/max(float(src@src),1e-30)
                    astar_clip=float(np.clip(astar,-4,4))
                    item['oracle_alpha']=astar
                    item['oracle_alpha_clipped']=astar_clip
                    item['oracle_ratio']=mse(base+astar_clip*src,truth)/bm
                    rows[f'l{layer+1}_k{kr}_c{cr}_o{order}']=item
    out={'seed':seed,'rot':rot,'nref_each':nref,'base_mse':bm,'reference_noise':noise,'rows':rows,'seconds':time.time()-t}
    (OUT/f'seed{seed}_rot{rot}.json').write_text(json.dumps(out,indent=2));print(json.dumps({'seed':seed,'base_mse':bm,'noise':noise,'seconds':out['seconds'],'best_oracle':min(v['oracle_ratio'] for v in rows.values())}),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,nargs='+',required=True)
    ap.add_argument('--layers',type=int,nargs='+',default=[15,23]);ap.add_argument('--nref',type=int,default=32768)
    ap.add_argument('--rot',type=int,default=3);ap.add_argument('--reference-mode',choices=['mc','kerdock'],default='kerdock');ap.add_argument('--kernel-ranks',type=int,nargs='+',default=[1,4])
    ap.add_argument('--cov-ranks',type=int,nargs='+',default=[8,32]);ap.add_argument('--orders',type=int,nargs='+',default=[3,4]);a=ap.parse_args()
    for s in a.seeds:run(s,a.layers,a.nref,a.rot,a.kernel_ranks,a.cov_ranks,a.orders,a.reference_mode)
if __name__=='__main__':main()
