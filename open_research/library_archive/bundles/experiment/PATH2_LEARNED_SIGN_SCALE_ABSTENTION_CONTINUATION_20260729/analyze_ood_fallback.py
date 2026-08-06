from __future__ import annotations
import json,re
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.linear_model import Ridge
from sklearn.covariance import LedoitWolf
ROOT=Path(__file__).resolve().parent;HP=ROOT/'fresh_scale_holdout_v1'
TR=pd.read_csv(ROOT/'legal_features_and_labels.csv');TE=pd.read_csv(HP/'legal_features.csv')
def terms(df,fresh=False):
 out=[]
 for _,z in df.iterrows():
  panel=HP if fresh else (ROOT/'canonical24_quadratics_audited' if z.domain=='canonical' else ROOT/'hardpanel_quadratics_audited')
  v=np.load(panel/'vectors'/f'vectors_{int(z.network_seed)}_r{int(z.rotation_seed)}_n262144.npz');b=v['base'];t=.5*(v['truth_half1']+v['truth_half2']);p=v['direct32'];e=b-t;d=p-b;out.append([np.mean(e*e),2*np.mean(e*d),np.mean(d*d),-np.mean(e*d)/np.mean(d*d)])
 return np.array(out)
A=terms(TR);B=terms(TE,True);META={'network_seed','rotation_seed','baseline_mse','oracle_ratio','candidate_ratio','oracle_headroom','domain','in_hard_panel','harm','no_headroom','feature_runtime_seconds'};NUM=[c for c in TR if c not in META and c in TE and np.issubdtype(TR[c].dtype,np.number)]
cs=[c for c in NUM if re.match(r'l(08|16|24|28|29|30|31)_fold_rel_(mean|q50|q90|max)$',c) or c in ['anchor_effrank','anchor_frob','anchor_r90','anchor_rho','anchor_trace'] or re.match(r'w_suffix_(fro|op|std|trace)_(mean|q50|q90|max)$',c)]
imp=SimpleImputer();sc=StandardScaler();X=sc.fit_transform(imp.fit_transform(TR[cs]));T=sc.transform(imp.transform(TE[cs]));sel=SelectKBest(f_regression,k=16).fit(X,A[:,3]);Xs=sel.transform(X);Ts=sel.transform(T);ridge=Ridge(alpha=1).fit(Xs,np.clip(A[:,3],0,1.5),sample_weight=A[:,2]/A[:,2].mean());pred=np.clip(ridge.predict(Ts),.25,.75)
lw=LedoitWolf().fit(Xs);train=lw.mahalanobis(Xs);test=lw.mahalanobis(Ts)
def met(a):
 m=B[:,0]+a*B[:,1]+a*a*B[:,2];r=m/B[:,0];return {'pooled_ratio':float(m.sum()/B[:,0].sum()),'worst':float(r.max()),'p90':float(np.quantile(r,.9)),'wins':int((r<1).sum()),'coverage_model':float(np.mean(a!=.5)),'alphas':a.tolist(),'ratios':r.tolist()}
res={'model_no_fallback':met(pred),'fixed_05':met(np.full(len(B),.5))}
for q in [.9,.95,.975,.99,1.0]:
 th=float(np.quantile(train,q));a=np.where(test>th,.5,pred);res[f'fallback_train_q{q}']={**met(a),'threshold':th,'abstentions':int((test>th).sum()),'abstained_rows':TE.loc[test>th,['network_seed','rotation_seed']].to_dict('records')}
out={'exploratory_post_holdout':True,'feature_set':'compact selected16','ood':'LedoitWolf Mahalanobis','train_score_summary':pd.Series(train).describe().to_dict(),'test_score_summary':pd.Series(test).describe().to_dict(),'policies':res}
(ROOT/'results'/'OOD_FALLBACK_RESULTS.json').write_text(json.dumps(out,indent=2));print(pd.DataFrame([{'policy':k,**{x:v[x] for x in ['pooled_ratio','worst','p90','wins']},'abstentions':v.get('abstentions',0)} for k,v in res.items()]).sort_values('pooled_ratio').to_string(index=False));print(res['fallback_train_q0.99']['abstained_rows'])
