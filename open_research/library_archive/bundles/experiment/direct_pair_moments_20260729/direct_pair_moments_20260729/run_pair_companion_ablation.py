from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/mnt/data/whest_residual_experiment')
BUNDLE=Path('/mnt/data/exp2_inputs/unpacked_DIRECT32_DEFECT_ESTIMATOR_BUNDLE')
sys.path.insert(0,str(ROOT))
import direct32_born_qmc as b
from build_direct32_checkpoint_dataset import radial_stats, crossfit_affine
D=b.D; NB=b.N_BASES; RB=b.ROWS_PER_BASIS; TARGET=b.TARGET

def forward_primary(x, ws):
    H=None
    with torch.no_grad():
        for li,w in enumerate(ws):
            x=torch.relu(x@w)
            if li==TARGET: H=x.clone()
        Y=x
    return H,Y

def forward_to(x,ws,target=TARGET):
    with torch.no_grad():
        for li,w in enumerate(ws):
            x=torch.relu(x@w)
            if li==target:return x
    raise AssertionError

def load_target(seed):
    a=np.load(BUNDLE/'companion_validation_targets'/f'companion_val_target_{seed}_a.npz')
    c=np.load(BUNDLE/'companion_validation_targets'/f'companion_val_target_{seed}_b.npz')
    h1=a['final'].astype(np.float64); h2=c['final'].astype(np.float64)
    return .5*(h1+h2),h1,h2

def mse(x,y): return float(np.mean((x-y)**2))
def ubmse(x,a,c): return float(np.mean((x-a)*(x-c)))
def cosine(x,y):
    den=np.linalg.norm(x)*np.linalg.norm(y)
    return float(np.dot(x,y)/den) if den>0 else 0.0

def lower(mu, Mq, Mt, m, ix, R):
    mui=mu[ix]; mur=mu@R; mi=m[ix]; mr=m@R
    Mii=np.diag(Mq)[ix]; Mir=np.sum(Mt[ix]*R.T,axis=1)
    return (Mii*(mur-mr)+2*(mui-mi)*Mir+2*(mi*mi-mui*mui)*mur)/(D+1)

def run(seed:int, probes=(32,128), basis_counts=(16,129), alphas=(.05,.1,.2,.5)):
    st=time.time(); ws=b.make_weights(seed); rho=b.chi_mean(D)
    Ht,Yt=forward_primary(b.make_kerdock(3),ws)
    H=Ht.double().cpu().numpy(); Y=Yt.double().cpu().numpy(); del Ht,Yt
    m=H.mean(0); base=Y.mean(0); truth,h1,h2=load_target(seed); bm=mse(base,truth)
    # Primary observable row matrix and radially-rescaled Gaussian second moment.
    Q=b.sample_anchor_matrix(H,m,rho)
    Ms=(D/(rho*rho))*(H.T@H/len(H))
    order=np.argsort(np.linalg.norm(Q,axis=1),kind='stable')[::-1]
    # Propagate full companion once, then subset by complete bases.
    HC=forward_to(b.make_kerdock(97),ws).double().cpu().numpy().reshape(NB,RB,D)
    record={'seed':seed,'baseline_mse':bm,'baseline_unbiased_mse':ubmse(base,h1,h2),'truth_noise_mse':float(.5*np.mean((h1-h2)**2)),'probe_results':{},'runtime_seconds':None}
    for p in probes:
        ix=order[:p]
        rr=Q[ix].copy(); rr/=np.maximum(np.linalg.norm(rr,axis=1,keepdims=True),1e-30); R=rr.T; L=np.eye(D)[:,ix]
        X=b.radial_features(H,m,L,R,rho); sample=np.sum(Q[ix]*R.T,axis=1)
        ss=radial_stats(H,L,R,rho); sample_conn=ss['kappa']/(D+1)
        offset,beta=crossfit_affine(X,Y,folds=6,ridge=.1)
        # Algebra check: pointwise-centered sample anchor equals connected sample term.
        alg=float(np.max(np.abs(sample-sample_conn)))
        pres={'support':ix.tolist(),'algebra_sample_vs_connected_max_abs':alg,'basis_results':{}}
        for bc in basis_counts:
            ids=np.unique(np.rint(np.linspace(0,NB-1,bc)).astype(int))
            A=HC[ids].reshape(-1,D); mu=A.mean(0); Mc=(D/(rho*rho))*(A.T@A/len(A))
            lower_map={
                'companion_pairs': lower(mu,Mc,Mc,m,ix,R),
                'primary_pairs': lower(mu,Ms,Ms,m,ix,R),
                'companion_diag_primary_row': lower(mu,Mc,Ms,m,ix,R),
                'primary_diag_companion_row': lower(mu,Ms,Mc,m,ix,R),
            }
            methods={}
            full_corr=lower_map['companion_pairs']@beta
            primary_corr=lower_map['primary_pairs']@beta
            pair_inc=full_corr-primary_corr
            pair_diag_inc=(lower_map['companion_diag_primary_row']-lower_map['primary_pairs'])@beta
            pair_row_inc=(lower_map['primary_diag_companion_row']-lower_map['primary_pairs'])@beta
            decomp={
                'full_correction_norm2':float(np.dot(full_corr,full_corr)),
                'primary_pair_correction_norm2':float(np.dot(primary_corr,primary_corr)),
                'independent_pair_increment_norm2':float(np.dot(pair_inc,pair_inc)),
                'independent_pair_increment_over_full_norm':float(np.linalg.norm(pair_inc)/max(np.linalg.norm(full_corr),1e-30)),
                'full_vs_primary_pair_cosine':cosine(full_corr,primary_corr),
                'diag_increment_norm2':float(np.dot(pair_diag_inc,pair_diag_inc)),
                'row_increment_norm2':float(np.dot(pair_row_inc,pair_row_inc)),
            }
            for name,lw in lower_map.items():
                raw=sample_conn+lw; defect=raw-sample
                corr=defect@beta
                vals={}
                for alpha in alphas:
                    pred=base + alpha*corr  # crossfit affine at sample anchor equals primary baseline up to roundoff
                    # Also calculate exact affine form to catch any intercept discrepancy.
                    pred_aff=offset+(sample+alpha*defect)@beta
                    vals[str(alpha)]={'mse':mse(pred_aff,truth),'ratio':mse(pred_aff,truth)/bm,'unbiased_mse':ubmse(pred_aff,h1,h2),'base_form_max_abs':float(np.max(np.abs(pred-pred_aff)))}
                methods[name]={'lower':lw.tolist(),'defect':defect.tolist(),'correction_norm2':float(np.dot(corr,corr)),'alphas':vals}
            pres['basis_results'][str(bc)]={'basis_ids':ids.tolist(),'methods':methods,'decomposition':decomp}
        record['probe_results'][str(p)]=pres
    record['runtime_seconds']=time.time()-st
    return record

def aggregate(records):
    out={}
    for p in records[0]['probe_results']:
      for bc in records[0]['probe_results'][p]['basis_results']:
       for method in records[0]['probe_results'][p]['basis_results'][bc]['methods']:
        for alpha in records[0]['probe_results'][p]['basis_results'][bc]['methods'][method]['alphas']:
         key=f'p{p}_b{bc}_{method}_a{alpha}'
         ms=np.array([r['probe_results'][p]['basis_results'][bc]['methods'][method]['alphas'][alpha]['mse'] for r in records])
         bs=np.array([r['baseline_mse'] for r in records]); ratios=ms/bs
         out[key]={'pooled_ratio':float(ms.sum()/bs.sum()),'wins':int(np.sum(ms<bs)),'median':float(np.median(ratios)),'p90':float(np.quantile(ratios,.9)),'worst':float(np.max(ratios)),'per_network':ratios.tolist()}
    # Pair increment summaries independent of alpha.
    dec={}
    for p in records[0]['probe_results']:
      for bc in records[0]['probe_results'][p]['basis_results']:
       vals=[r['probe_results'][p]['basis_results'][bc]['decomposition'] for r in records]
       dec[f'p{p}_b{bc}']={k:{'median':float(np.median([v[k] for v in vals])),'mean':float(np.mean([v[k] for v in vals])),'p90':float(np.quantile([v[k] for v in vals],.9))} for k in vals[0]}
    return {'candidates':out,'pair_increment':dec}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',nargs='+',type=int,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--threads',type=int,default=8); a=ap.parse_args()
    torch.set_num_threads(a.threads); records=[]
    for seed in a.seeds:
        r=run(seed); records.append(r); payload={'config':{'seeds':a.seeds,'probes':[32,128],'basis_counts':[16,129],'alphas':[.05,.1,.2,.5],'primary_rotation':3,'companion_rotation':97},'records':records,'summary':aggregate(records)}; a.out.write_text(json.dumps(payload,indent=2))
        best=min(payload['summary']['candidates'].items(),key=lambda kv:kv[1]['pooled_ratio'])
        print(seed,'sec',round(r['runtime_seconds'],1),'best',best[0],round(best[1]['pooled_ratio'],4),flush=True)
if __name__=='__main__': main()
