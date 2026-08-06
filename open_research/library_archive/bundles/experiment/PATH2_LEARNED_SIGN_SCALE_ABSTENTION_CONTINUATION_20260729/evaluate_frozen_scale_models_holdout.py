from __future__ import annotations
import json,re
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parent;HP=ROOT/'fresh_scale_holdout_v1';CP=ROOT/'canonical24_quadratics_audited';DP=ROOT/'hardpanel_quadratics_audited'
TR=pd.read_csv(ROOT/'legal_features_and_labels.csv');TE=pd.read_csv(HP/'legal_features.csv')

def terms_for(df,fresh=False):
 out=[]
 for _,z in df.iterrows():
  if fresh:panel=HP
  else:panel=CP if z.domain=='canonical' else DP
  v=np.load(panel/'vectors'/f'vectors_{int(z.network_seed)}_r{int(z.rotation_seed)}_n262144.npz');b=v['base'];t=.5*(v['truth_half1']+v['truth_half2']);p=v['direct32'];e=b-t;d=p-b
  a0=float(np.mean(e*e));lin=float(2*np.mean(e*d));q=float(np.mean(d*d));out.append((a0,lin,q,float(-lin/(2*q))))
 return np.array(out)
ATR=terms_for(TR);ATE=terms_for(TE,True)
META={'network_seed','rotation_seed','baseline_mse','oracle_ratio','candidate_ratio','oracle_headroom','domain','in_hard_panel','harm','no_headroom','feature_runtime_seconds'}
NUM=[c for c in TR if c not in META and c in TE.columns and np.issubdtype(TR[c].dtype,np.number)]
sets={
 'compact':[c for c in NUM if re.match(r'l(08|16|24|28|29|30|31)_fold_rel_(mean|q50|q90|max)$',c) or c in ['anchor_effrank','anchor_frob','anchor_r90','anchor_rho','anchor_trace'] or re.match(r'w_suffix_(fro|op|std|trace)_(mean|q50|q90|max)$',c)],
 'weights':[c for c in NUM if c.startswith('w')],
 'late':[c for c in NUM if re.match(r'l(20|24|28|29|30|31)_',c) or c.startswith('anchor_')]
}
proto=json.load(open(HP/'IMMUTABLE_MODEL_EVAL_PROTOCOL.json'));preds={}
for name,cfg in proto['models'].items():
 if 'members' in cfg:continue
 cs=sets[cfg['feature_set']];m=Pipeline([('imp',SimpleImputer()),('scale',StandardScaler()),('select',SelectKBest(f_regression,k=cfg['k'])),('ridge',Ridge(alpha=cfg['ridge']))])
 w=ATR[:,2]/ATR[:,2].mean();m.fit(TR[cs],np.clip(ATR[:,3],0,1.5),ridge__sample_weight=w);preds[name]=np.clip(m.predict(TE[cs]),*cfg['clip'])
preds['bounded_ensemble']=np.mean([preds[x] for x in proto['models']['bounded_ensemble']['members']],axis=0)

def met(a):
 b=ATE[:,0];m=b+a*ATE[:,1]+a*a*ATE[:,2];r=m/b;g=TE.network_seed.to_numpy();groups=np.unique(g);gb=np.array([b[g==x].sum() for x in groups]);gm=np.array([m[g==x].sum() for x in groups]);rng=np.random.default_rng(90210);ix=rng.integers(0,len(groups),size=(100000,len(groups)));vals=gm[ix].sum(1)/gb[ix].sum(1)
 return {'pooled_ratio':float(m.sum()/b.sum()),'worst':float(r.max()),'p90':float(np.quantile(r,.9)),'wins':int((r<1).sum()),'alpha_mean':float(a.mean()),'alpha_min':float(a.min()),'alpha_max':float(a.max()),'grouped_bootstrap_ci95':[float(x) for x in np.quantile(vals,[.025,.5,.975])],'ratios':r.tolist(),'alphas':a.tolist()}
res={k:met(v) for k,v in preds.items()};res|={'fixed_05':met(np.full(len(TE),.5)),'oracle_alpha':met(ATE[:,3])}
out={'protocol':proto,'models':res,'rows':TE[['network_seed','rotation_seed']].assign(**{k:v for k,v in preds.items()}).to_dict('records')};(ROOT/'results'/'FROZEN_SCALE_MODEL_HOLDOUT_RESULTS.json').write_text(json.dumps(out,indent=2));print(pd.DataFrame([{'model':k,**{x:v[x] for x in ['pooled_ratio','worst','p90','wins','alpha_mean','alpha_min','alpha_max']}} for k,v in res.items()]).sort_values('pooled_ratio').to_string(index=False))
