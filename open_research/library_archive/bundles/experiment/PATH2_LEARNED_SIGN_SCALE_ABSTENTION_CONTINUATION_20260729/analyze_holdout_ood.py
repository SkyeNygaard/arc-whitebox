from __future__ import annotations
import json,re
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
ROOT=Path(__file__).resolve().parent;HP=ROOT/'fresh_scale_holdout_v1'
TR=pd.read_csv(ROOT/'legal_features_and_labels.csv');TE=pd.read_csv(HP/'legal_features.csv')
# exact development alpha targets
alpha=[]
for _,z in TR.iterrows():
 panel=ROOT/'canonical24_quadratics_audited' if z.domain=='canonical' else ROOT/'hardpanel_quadratics_audited';v=np.load(panel/'vectors'/f'vectors_{int(z.network_seed)}_r{int(z.rotation_seed)}_n262144.npz');b=v['base'];t=.5*(v['truth_half1']+v['truth_half2']);p=v['direct32'];e=b-t;d=p-b;alpha.append(-np.mean(e*d)/np.mean(d*d))
alpha=np.array(alpha)
META={'network_seed','rotation_seed','baseline_mse','oracle_ratio','candidate_ratio','oracle_headroom','domain','in_hard_panel','harm','no_headroom','feature_runtime_seconds'}
NUM=[c for c in TR if c not in META and c in TE and np.issubdtype(TR[c].dtype,np.number)]
sets={'compact':[c for c in NUM if re.match(r'l(08|16|24|28|29|30|31)_fold_rel_(mean|q50|q90|max)$',c) or c in ['anchor_effrank','anchor_frob','anchor_r90','anchor_rho','anchor_trace'] or re.match(r'w_suffix_(fro|op|std|trace)_(mean|q50|q90|max)$',c)],'weights':[c for c in NUM if c.startswith('w')],'late':[c for c in NUM if re.match(r'l(20|24|28|29|30|31)_',c) or c.startswith('anchor_')]}
res={}
for sn,cs in sets.items():
 imp=SimpleImputer();sc=StandardScaler();X=sc.fit_transform(imp.fit_transform(TR[cs]));T=sc.transform(imp.transform(TE[cs]));sel=SelectKBest(f_regression,k=min(16,X.shape[1])).fit(X,alpha);Xs=sel.transform(X);Ts=sel.transform(T)
 lw=LedoitWolf().fit(Xs);mah=lw.mahalanobis(Ts)
 iso=-IsolationForest(n_estimators=1000,contamination='auto',random_state=260729).fit(Xs).score_samples(Ts)
 pca=PCA(n_components=min(12,len(Xs)-1,Xs.shape[1])).fit(Xs);rec=np.mean((Ts-pca.inverse_transform(pca.transform(Ts)))**2,axis=1)
 res[sn]={'mahalanobis':mah.tolist(),'isolation':iso.tolist(),'pca_reconstruction':rec.tolist()}
# attach frozen model bad ratio
mod=json.load(open(ROOT/'results'/'FROZEN_SCALE_MODEL_HOLDOUT_RESULTS.json'));rat=np.array(mod['models']['bounded_ensemble']['ratios']);rows=[]
for i,z in TE[['network_seed','rotation_seed']].iterrows():
 row={'network_seed':int(z.network_seed),'rotation_seed':int(z.rotation_seed),'bounded_ratio':float(rat[i])}
 for sn in res:
  for k,v in res[sn].items():row[f'{sn}_{k}']=float(v[i])
 rows.append(row)
out={'rows':rows,'bad_index':int(np.argmax(rat)),'bad_seed':int(TE.network_seed.iloc[np.argmax(rat)]),'bad_rotation':int(TE.rotation_seed.iloc[np.argmax(rat)]),'ranks':{}}
for sn in res:
 for k,v in res[sn].items():
  order=np.argsort(v)[::-1];out['ranks'][f'{sn}_{k}']=int(np.where(order==np.argmax(rat))[0][0]+1)
(ROOT/'results'/'HOLDOUT_OOD_RESULTS.json').write_text(json.dumps(out,indent=2));print(json.dumps(out['ranks'],indent=2));print(pd.DataFrame(rows).sort_values('bounded_ratio',ascending=False).head(8).to_string(index=False))
