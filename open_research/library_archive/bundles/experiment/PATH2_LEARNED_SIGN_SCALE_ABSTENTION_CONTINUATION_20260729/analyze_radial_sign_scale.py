from __future__ import annotations
import json,re,warnings
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.linear_model import Ridge,HuberRegressor
from sklearn.ensemble import IsolationForest,ExtraTreesRegressor,RandomForestRegressor
from sklearn.covariance import LedoitWolf
warnings.filterwarnings('ignore')
ROOT=Path(__file__).resolve().parent
D=pd.read_csv(ROOT/'radial16_full_features.csv')
META={'network_seed','rotation_seed','baseline_mse','ratio_alpha1','ratio_alpha04','alpha_opt','ratio_opt','harm_alpha1','harm_alpha04','oracle_ratio','feature_runtime_seconds'}
NUM=[c for c in D if c not in META and np.issubdtype(D[c].dtype,np.number)]
sets={
 'weights':[c for c in NUM if c.startswith('w')],
 'trajectory':[c for c in NUM if re.match(r'l\d\d_',c)],
 'rotation':[c for c in NUM if ('fold_' in c or 'block_' in c or 'antipodal_' in c)],
 'anchor':[c for c in NUM if c.startswith('anchor_')],
 'all':NUM,
}
y=D.alpha_opt.to_numpy(); neg=int(np.argmin(y)); rows=[]
for sn,cs in sets.items():
 X=D[cs].to_numpy()
 for k in [4,8,16,32]:
  if k>len(cs): continue
  for a in [1,10,100,1000]:
   pred=np.zeros(len(D))
   for i in range(len(D)):
    tr=np.arange(len(D))!=i
    m=Pipeline([('imp',SimpleImputer()),('sc',StandardScaler()),('sel',SelectKBest(f_regression,k=k)),('r',Ridge(alpha=a))])
    m.fit(X[tr],y[tr]); pred[i]=m.predict(X[i:i+1])[0]
   rows.append({'family':sn,'k':k,'ridge':a,'rmse':float(np.sqrt(np.mean((pred-y)**2))),
                'mae':float(np.mean(np.abs(pred-y))),'spearman':float(spearmanr(pred,y).statistic),
                'negative_seed_prediction':float(pred[neg]),'negative_seed_true':float(y[neg]),'pred':pred.tolist()})
rows=sorted(rows,key=lambda z:z['rmse'])
# Per-feature extremeness of negative seed (unsupervised, label-free after feature selection is reported only descriptively).
ext=[]
for c in NUM:
 v=D[c].to_numpy(float); med=np.nanmedian(v); mad=np.nanmedian(np.abs(v-med))+1e-12; z=(v[neg]-med)/mad
 rank=int(np.sum(v<=v[neg])); ext.append({'feature':c,'robust_z':float(z),'rank_low_to_high':rank,'n':len(v),'spearman_alpha':float(spearmanr(v,y,nan_policy='omit').statistic)})
ext=sorted(ext,key=lambda z:abs(z['robust_z']),reverse=True)
# Honest one-class scores: for each held-out example fit covariance/isolation on the other 15, no labels.
# Reduce using predeclared compact trajectory summary to avoid p>>n pathologies.
compact=[c for c in NUM if re.match(r'l(00|04|08|12|16|20|24|28|29|30|31)_(mean|std|rms|zero|max|fold_rel_mean|fold_rel_max|block_disp_mean|antipodal_imb_mean)$',c)]
X=D[compact].to_numpy(float); scores={'mahalanobis':np.zeros(len(D)),'isoforest':np.zeros(len(D))}
for i in range(len(D)):
 tr=np.arange(len(D))!=i
 imp=SimpleImputer(); Xt=imp.fit_transform(X[tr]); Xi=imp.transform(X[i:i+1]); sc=StandardScaler(); Z=sc.fit_transform(Xt); zi=sc.transform(Xi)
 # PCA via SVD to at most 6 dimensions, predeclared.
 u,s,vt=np.linalg.svd(Z,full_matrices=False); V=vt[:min(6,len(vt))].T; P=Z@V; pi=zi@V
 lw=LedoitWolf().fit(P); scores['mahalanobis'][i]=float(lw.mahalanobis(pi)[0])
 iso=IsolationForest(n_estimators=1000,contamination='auto',random_state=20260729).fit(P); scores['isoforest'][i]=float(-iso.score_samples(pi)[0])
out={'n':len(D),'negative_alpha_seed':int(D.network_seed.iloc[neg]),'alpha_summary':D[['network_seed','alpha_opt','ratio_alpha1','ratio_alpha04','ratio_opt']].to_dict('records'),
     'best_ridge':rows[0],'ridge_suite':rows,
     'negative_case_one_class_ranks':{k:{'score':float(v[neg]),'rank_high_is_anomalous':int(np.sum(v<=v[neg])),'n':len(v),'all_scores':v.tolist()} for k,v in scores.items()},
     'most_extreme_features_negative_case':ext[:30],
     'interpretation':{'supervised_sign_classification_identifiable':False,'reason':'only one negative-alpha example; its LOO training fold contains no negative labels'}}
(ROOT/'results'/'RADIAL_SIGN_SCALE_RESULTS.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k not in ('ridge_suite','alpha_summary','most_extreme_features_negative_case')},indent=2))
print('top features')
print(pd.DataFrame(ext[:20]).to_string(index=False))
