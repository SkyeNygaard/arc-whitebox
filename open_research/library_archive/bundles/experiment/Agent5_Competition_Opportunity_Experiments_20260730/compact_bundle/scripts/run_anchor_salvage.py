#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.pipeline import make_pipeline

P=Path('/mnt/data/work/T4_legal_layer31_anchor_hedge_20260729_review/T4_legal_layer31_anchor_hedge_20260729/T41_EXACT_ROTATIONS/ROWS.csv')
OUT=Path('/mnt/data/work/new_opportunities/results'); OUT.mkdir(exist_ok=True,parents=True)
FEATURES=['feature_cos_c17_p2','feature_cos_c17_p4','feature_cos_p2_p4','feature_norm_p2_c17','feature_norm_p4_c17','feature_min_nested_cos','feature_max_loo_angle_sin','feature_cos_p32_p128','feature_norm_p32_p128']

def score(df,a):
    b=df.baseline_unbiased_mse.to_numpy(float); inn=df.inner.to_numpy(float); n=df.norm_sq.to_numpy(float)
    c=b+2*a*inn+a*a*n
    # row aggregate and network-level aggregate
    ratio=float(c.sum()/b.sum()); wins=int(np.sum(c<b)); worst=float(np.max(c/np.maximum(b,1e-30)))
    net=[]
    for gid,g in df.assign(cand=c).groupby('network_id'):
        rr=float(g.cand.sum()/g.baseline_unbiased_mse.sum()); net.append(rr)
    return {'candidate_over_base':ratio,'gain_base_over_candidate':1/ratio,'row_wins':wins,'rows':len(df),'worst_row_ratio':worst,
            'network_wins':int(np.sum(np.array(net)<1)),'networks':len(net),'median_network_ratio':float(np.median(net)),'worst_network_ratio':float(np.max(net)),
            'alphas':{'mean':float(np.mean(a)),'median':float(np.median(a)),'min':float(np.min(a)),'max':float(np.max(a))}}

def inner_cv_ridge(X,y,w,groups,cliplo,cliphi):
    alphas=[1e-4,1e-3,1e-2,1e-1,1,10,100]
    uniq=np.unique(groups); splits=min(4,len(uniq))
    cv=GroupKFold(splits)
    best=None
    for reg in alphas:
        losses=[]
        for tr,va in cv.split(X,y,groups=groups):
            m=make_pipeline(StandardScaler(),Ridge(alpha=reg))
            m.fit(X[tr],y[tr],ridge__sample_weight=w[tr])
            p=np.clip(m.predict(X[va]),cliplo,cliphi)
            losses.append(float(np.sum(w[va]*(p-y[va])**2)))
        v=sum(losses)
        if best is None or v<best[0]: best=(v,reg)
    m=make_pipeline(StandardScaler(),Ridge(alpha=best[1]))
    m.fit(X,y,ridge__sample_weight=w)
    return m,best[1]

def fit_et(X,y,w,cliplo,cliphi):
    m=ExtraTreesRegressor(n_estimators=200,max_depth=4,min_samples_leaf=3,max_features=1.0,random_state=20260730,n_jobs=1)
    m.fit(X,y,sample_weight=w)
    return m,{"depth":4,"leaf":3}

def main():
    df=pd.read_csv(P).sort_values(['network_id','rotation_index']).reset_index(drop=True)
    X=df[FEATURES].to_numpy(float); grp=df.network_id.to_numpy(); inn=df.inner.to_numpy(float); norm=df.norm_sq.to_numpy(float)
    y=-inn/np.maximum(norm,1e-30); w=norm/np.mean(norm)
    logo=LeaveOneGroupOut()
    preds={
      'none':np.zeros(len(df)),
      'original_alpha1':np.ones(len(df)),
      'oracle_clip_0_2':np.clip(y,0,2),
      'oracle_clip_m2_2':np.clip(y,-2,2),
      'global_constant_clip_0_2':np.zeros(len(df)),
      'global_constant_clip_m2_2':np.zeros(len(df)),
      'ridge_clip_0_2':np.zeros(len(df)),
      'ridge_clip_m2_2':np.zeros(len(df)),
      'extratrees_clip_0_2':np.zeros(len(df)),
      'extratrees_clip_m2_2':np.zeros(len(df)),
    }
    choices={k:[] for k in preds if k not in ('none','original_alpha1','oracle_clip_0_2','oracle_clip_m2_2')}
    for tr,te in logo.split(X,y,groups=grp):
        # Exact train-optimal constant.
        ag=float(-inn[tr].sum()/max(norm[tr].sum(),1e-30))
        preds['global_constant_clip_0_2'][te]=np.clip(ag,0,2); choices['global_constant_clip_0_2'].append(ag)
        preds['global_constant_clip_m2_2'][te]=np.clip(ag,-2,2); choices['global_constant_clip_m2_2'].append(ag)
        for lo,hi,suf in [(0,2,'0_2'),(-2,2,'m2_2')]:
            mr,hr=inner_cv_ridge(X[tr],y[tr],w[tr],grp[tr],lo,hi)
            preds[f'ridge_clip_{suf}'][te]=np.clip(mr.predict(X[te]),lo,hi); choices[f'ridge_clip_{suf}'].append(hr)
            me,he=fit_et(X[tr],y[tr],w[tr],lo,hi)
            preds[f'extratrees_clip_{suf}'][te]=np.clip(me.predict(X[te]),lo,hi); choices[f'extratrees_clip_{suf}'].append(he)
    results={k:score(df,a) for k,a in preds.items()}
    # Add exact theoretical best within per-row scalar span without clipping.
    results['oracle_unclipped']=score(df,y)
    out={'source':str(P),'features':FEATURES,'protocol':'leave-one-network-out outer CV; inner grouped CV for hyperparameters; unbiased MSE reconstructed exactly from baseline+2a inner+a^2 norm_sq',
         'results':results,'hyperparameter_choices':choices}
    (OUT/'ANCHOR_SALVAGE_RESULTS.json').write_text(json.dumps(out,indent=2,sort_keys=True))
    tab=[]
    for k,v in sorted(results.items(),key=lambda kv:kv[1]['candidate_over_base']):
        tab.append({'method':k,**{z:v[z] for z in ['candidate_over_base','gain_base_over_candidate','row_wins','network_wins','worst_network_ratio']},'alpha_mean':v['alphas']['mean'],'alpha_min':v['alphas']['min'],'alpha_max':v['alphas']['max']})
    pd.DataFrame(tab).to_csv(OUT/'ANCHOR_SALVAGE_RESULTS.csv',index=False)
    print(pd.DataFrame(tab).to_string(index=False))

if __name__=='__main__':main()
