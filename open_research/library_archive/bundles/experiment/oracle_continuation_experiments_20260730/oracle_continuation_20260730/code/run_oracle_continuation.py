from __future__ import annotations
import json, math, hashlib
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT=Path('/mnt/data/oracle_experiments_inputs/oracle_gap_campaign/whest_experiments_oracle_gap_20260730')
OUT=Path('/mnt/data/oracle_continuation_20260730')
OUT.mkdir(exist_ok=True)
sel=json.loads((ROOT/'frozen_selection.json').read_text())['selected']

# deterministic generator import
import sys
sys.path.insert(0,str(ROOT/'src'))
import run_fresh_suite as rfs

def load(split):
    rows=[]
    for p in sorted((ROOT/'results'/split).glob('seed_*/*.npz')):
        cj=p.parent/f'case_{p.stem}.json'
        if not cj.exists(): continue
        with np.load(p,allow_pickle=False) as z:
            d={k:z[k] for k in z.files}
        d['seed']=int(p.parent.name.split('_')[-1]); d['path']=str(p)
        d['rotation']=int(p.stem.split('rot')[-1])
        rows.append(d)
    return rows

def cosine(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/den) if den>1e-30 else 0.0

def source_matrix(row):
    names=[str(x) for x in row['candidate_names']]
    shallow=sel['shallow']['name']; si=names.index(shallow)
    shallow_corr=row['candidates'][si]-row['baseline']
    aname=sel['average_companion']['name']; alpha=float(aname.split('_a')[-1])
    ai=int(np.where(np.isclose(row['companion_alphas'],alpha))[0][0])
    comps=[row['companion_preds'][ai,j]-row['baseline'] for j in range(4)]
    return np.stack([shallow_corr]+comps,axis=1)

def summary_stats(v):
    v=np.asarray(v,dtype=float)
    m=float(v.mean()); sd=float(v.std());
    if sd>1e-30: skew=float(np.mean(((v-m)/sd)**3))
    else: skew=0.0
    return [float(np.linalg.norm(v)),m,sd,float(np.max(np.abs(v))),skew]

def feature_sets(row):
    S=source_matrix(row)
    geom=[]
    # source vector summaries and Gram geometry
    for j in range(S.shape[1]): geom += summary_stats(S[:,j])
    for i in range(S.shape[1]):
        for j in range(i+1,S.shape[1]): geom.append(cosine(S[:,i],S[:,j]))
    for arr_name in ['companion_deltas','probe_deltas','probe_final_dirs']:
        A=np.asarray(row[arr_name],dtype=float)
        for j in range(A.shape[0]): geom += summary_stats(A[j])
        for i in range(A.shape[0]):
            for j in range(i+1,A.shape[0]): geom.append(cosine(A[i],A[j]))
    for j in range(4):
        geom.append(cosine(row['companion_deltas'][j],row['probe_deltas'][j]))
        geom.append(cosine(row['probe_final_dirs'][j],S[:,j+1]))
    geom += summary_stats(row['baseline'])

    layers=[]
    for depth in [0,3,7,11,15,19,23,27,29,30,31]:
        layers += summary_stats(row['baseline_layer_means'][depth])
    for a,b in zip([0,3,7,11,15,19,23,27,29,30],[3,7,11,15,19,23,27,29,30,31]):
        x=row['baseline_layer_means'][a]; y=row['baseline_layer_means'][b]
        layers += [float(np.linalg.norm(y)/max(np.linalg.norm(x),1e-30)), cosine(x,y)]

    # Weight-only legal invariants. Same for rotations of a base, but may help base-specific calibration.
    W=rfs.make_weights(int(row['seed']))
    weights=[]
    for depth in [0,7,15,23,29,30,31]:
        w=W[depth].astype(float)
        rn=np.linalg.norm(w,axis=1); cn=np.linalg.norm(w,axis=0)
        weights += [float(np.linalg.norm(w)),float(rn.mean()),float(rn.std()),float(rn.max()),float(cn.std()),float(np.trace(w)/256.0)]
        if depth in [29,30,31]:
            sv=np.linalg.svd(w,compute_uv=False)
            weights += [float(x) for x in sv[:8]]+[float(sv[-1]),float((sv[:8]**2).sum()/(sv**2).sum())]
    return {
        'geometry':np.asarray(geom,float),
        'geometry_layers':np.asarray(geom+layers,float),
        'all':np.asarray(geom+layers+weights,float),
    }

def target_coeff(row):
    S=source_matrix(row); y=row['truth']-row['baseline']
    c=np.linalg.lstsq(S,y,rcond=1e-10)[0]
    return np.clip(c,-3.0,3.0)

def mse(row,pred): return float(np.mean((pred-row['truth'])**2))
def umse(row,pred): return float(np.mean((pred-row['truth_a'])*(pred-row['truth_b'])))

def metrics(rows,preds,seed=20260730):
    b=np.array([mse(x,x['baseline']) for x in rows]); v=np.array([mse(x,p) for x,p in zip(rows,preds)])
    bu=np.array([umse(x,x['baseline']) for x in rows]); vu=np.array([umse(x,p) for x,p in zip(rows,preds)])
    rr=v/b
    groups=sorted(set(x['seed'] for x in rows)); gi=[[i for i,x in enumerate(rows) if x['seed']==g] for g in groups]
    rng=np.random.default_rng(seed); boot=[]; bootu=[]
    for _ in range(20000):
        ids=[i for k in rng.integers(0,len(gi),len(gi)) for i in gi[k]]
        boot.append(v[ids].sum()/b[ids].sum()); bootu.append(vu[ids].sum()/bu[ids].sum())
    return {
        'n_cases':len(rows),'n_bases':len(groups),'pooled_raw_ratio':float(v.sum()/b.sum()),
        'noise_corrected_ratio':float(vu.sum()/bu.sum()),'wins':int((v<b).sum()),
        'median':float(np.median(rr)),'p90':float(np.quantile(rr,.9)),'worst':float(rr.max()),
        'grouped_bootstrap_95':[float(x) for x in np.quantile(boot,[.025,.975])],
        'noise_corrected_grouped_bootstrap_95':[float(x) for x in np.quantile(bootu,[.025,.975])],
        'case_ratios':rr.tolist(),
    }

def matrices(rows,feat_name):
    X=np.stack([feature_sets(x)[feat_name] for x in rows]); C=np.stack([target_coeff(x) for x in rows])
    return X,C

def pred_from_coeff(rows,C):
    return [row['baseline']+source_matrix(row)@c for row,c in zip(rows,C)]

dev,val,conf=load('development'),load('validation'),load('confirmation')
assert (len(dev),len(val),len(conf))==(12,12,12)

# Coherence extension across all three cohorts.
def coherence(rows):
    C=np.stack([np.concatenate([x['checkpoint_corrections'][j] for x in rows]) for j in range(6)])
    U=np.diff(np.vstack([np.zeros((1,C.shape[1])),C]),axis=0)
    G=U@U.T; den=np.sqrt(np.outer(np.diag(G),np.diag(G)))
    cos=np.divide(G,den,out=np.zeros_like(G),where=den>0)
    return {'depths':[int(x) for x in rows[0]['checkpoint_depths']], 'increment_cosine':cos.tolist(),
            'increment_energy_fraction':(np.diag(G)/np.trace(G)).tolist(),
            'max_abs_offdiag':float(np.max(np.abs(cos-np.eye(6))))}
coh={s:coherence(rows) for s,rows in [('development',dev),('validation',val),('confirmation',conf)]}
# matrix stability Frobenius/correlation
for a,b in [('development','validation'),('validation','confirmation'),('development','confirmation')]:
    A=np.asarray(coh[a]['increment_cosine']); B=np.asarray(coh[b]['increment_cosine']); mask=~np.eye(6,dtype=bool)
    coh[f'{a}_vs_{b}']={'offdiag_rmse':float(np.sqrt(np.mean((A[mask]-B[mask])**2))),
                         'offdiag_correlation':float(np.corrcoef(A[mask],B[mask])[0,1]),
                         'energy_l1':float(np.abs(np.asarray(coh[a]['increment_energy_fraction'])-np.asarray(coh[b]['increment_energy_fraction'])).sum())}

alphas=[0.001,0.01,0.1,1,10,100,1000]
configs=[]
for feat in ['geometry','geometry_layers','all']:
    X,C=matrices(dev,feat); groups=np.array([x['seed'] for x in dev]); uniq=np.unique(groups)
    for alpha in alphas:
        preds=[]; order=[]
        for g in uniq:
            tr=groups!=g; te=groups==g
            model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
            model.fit(X[tr],C[tr]); cp=np.clip(model.predict(X[te]),-3,3)
            for idx,c in zip(np.flatnonzero(te),cp): order.append(idx); preds.append((idx,c))
        Cp=np.stack([dict(preds)[i] for i in range(len(dev))])
        m=metrics(dev,pred_from_coeff(dev,Cp),seed=123)
        configs.append({'features':feat,'alpha':alpha,'dev_group_cv_raw':m['pooled_raw_ratio'],'dev_group_cv_worst':m['worst']})
# selection prioritizes raw then tail guard; no hidden tuning.
elig=[x for x in configs if x['dev_group_cv_worst']<=2.0]
best=min(elig or configs,key=lambda x:(x['dev_group_cv_raw'],x['dev_group_cv_worst']))

# Fit chosen model on dev only.
Xd,Cd=matrices(dev,best['features']); model=make_pipeline(StandardScaler(),Ridge(alpha=best['alpha'])); model.fit(Xd,Cd)
# global constant source span fitted on dev outputs (not mean oracle coefficients)
Sall=np.concatenate([source_matrix(x) for x in dev],axis=0); yall=np.concatenate([x['truth']-x['baseline'] for x in dev])
cglobal=np.clip(np.linalg.lstsq(Sall,yall,rcond=1e-10)[0],-3,3)
# nonnegative bounded global control
try:
    from scipy.optimize import lsq_linear
    cnnls=lsq_linear(Sall,yall,bounds=(0,3)).x
except Exception:
    cnnls=np.clip(cglobal,0,3)

results={'selection':best,'all_dev_cv_configs':configs,'global_coefficients':cglobal.tolist(),'global_nonnegative_coefficients':cnnls.tolist(),'coherence':coh,'splits':{}}
for name,rows in [('development',dev),('validation',val),('confirmation',conf)]:
    X,_=matrices(rows,best['features']); Cpred=np.clip(model.predict(X),-3,3)
    poracle=pred_from_coeff(rows,np.stack([target_coeff(x) for x in rows]))
    pmodel=pred_from_coeff(rows,Cpred)
    pglobal=pred_from_coeff(rows,np.tile(cglobal,(len(rows),1)))
    pnnls=pred_from_coeff(rows,np.tile(cnnls,(len(rows),1)))
    # best single frozen source coefficient fit on dev only
    results['splits'][name]={
        'feature_ridge':metrics(rows,pmodel),
        'global_linear':metrics(rows,pglobal),
        'global_nonnegative':metrics(rows,pnnls),
        'per_case_oracle_span':metrics(rows,poracle),
        'predicted_coefficients':Cpred.tolist(),
    }

# feature-dependent value over matched constants
for name in results['splits']:
    results['splits'][name]['feature_value_vs_global_raw']=results['splits'][name]['global_linear']['pooled_raw_ratio']-results['splits'][name]['feature_ridge']['pooled_raw_ratio']

(OUT/'DOWNSTREAM_COEFFICIENT_SYNTHESIS.json').write_text(json.dumps(results,indent=2))
# CSV concise
import csv
with (OUT/'DOWNSTREAM_COEFFICIENT_SYNTHESIS_SUMMARY.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['split','method','raw_ratio','noise_corrected_ratio','wins','n','p90','worst','ci_low','ci_high'])
    for split,d in results['splits'].items():
        for method,m in d.items():
            if not isinstance(m,dict) or 'pooled_raw_ratio' not in m: continue
            w.writerow([split,method,m['pooled_raw_ratio'],m['noise_corrected_ratio'],m['wins'],m['n_cases'],m['p90'],m['worst'],*m['grouped_bootstrap_95']])
print(json.dumps({'selection':best,'splits':{k:{m:{kk:v[kk] for kk in ['pooled_raw_ratio','noise_corrected_ratio','wins','worst','grouped_bootstrap_95']} for m,v in d.items() if isinstance(v,dict) and 'pooled_raw_ratio' in v} for k,d in results['splits'].items()},'coherence':coh},indent=2))
