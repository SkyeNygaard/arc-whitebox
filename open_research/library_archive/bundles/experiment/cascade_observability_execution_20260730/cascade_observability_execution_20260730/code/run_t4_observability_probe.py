from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

ROOT=Path(__file__).resolve().parents[1]
CSV=ROOT/'sources'/'T4_EXACT_ROTATIONS_ROWS.csv'
df=pd.read_csv(CSV)
features=[c for c in df.columns if c.startswith('feature_')]
targets=['geometry_c17_cos_ideal','geometry_p2_cos_ideal','geometry_p4_cos_ideal','oracle_ratio']
groups=df.network_id.to_numpy(); ug=np.unique(groups)
X=df[features].to_numpy(float)
results={}
for target in targets:
    y=df[target].to_numpy(float)
    for model_name in ['ridge','extra_trees']:
        pred=np.full(len(df),np.nan)
        chosen=[]
        for g in ug:
            tr=groups!=g; te=groups==g
            if model_name=='ridge':
                # nested group CV on training groups for alpha
                alphas=[.01,.1,1,10]
                best=(1e99,1.0)
                train_groups=np.unique(groups[tr])
                for a in alphas:
                    pp=np.full(tr.sum(),np.nan); idx=np.where(tr)[0]
                    for vg in train_groups:
                        tr2=tr & (groups!=vg); va=(groups==vg)
                        m=make_pipeline(StandardScaler(),Ridge(alpha=a))
                        m.fit(X[tr2],y[tr2]); pp[np.isin(idx,np.where(va)[0])]=m.predict(X[va])
                    mse=np.nanmean((pp-y[tr])**2)
                    if mse<best[0]: best=(mse,a)
                m=make_pipeline(StandardScaler(),Ridge(alpha=best[1])); chosen.append(best[1])
            else:
                m=ExtraTreesRegressor(n_estimators=120,min_samples_leaf=4,max_features=.8,random_state=int(g)+7,n_jobs=5)
            m.fit(X[tr],y[tr]); pred[te]=m.predict(X[te])
        corr=float(pearsonr(y,pred).statistic) if np.std(pred)>0 else 0.0
        results[f'{target}:{model_name}']={
          'n_rows':len(y),'n_groups':len(ug),'grouped_oof_r2':float(r2_score(y,pred)),
          'grouped_oof_pearson':corr,'rho_squared':corr*corr,
          'sign_accuracy_vs_zero':float(np.mean((pred>=0)==(y>=0))),
          'target_mean':float(y.mean()),'target_sd':float(y.std()),
          'prediction_mean':float(pred.mean()),'prediction_sd':float(pred.std()),
          'chosen_alphas':chosen if model_name=='ridge' else None
        }
# Also test whether legal features predict which arm has highest ideal cosine.
yarms=df[['geometry_c17_cos_ideal','geometry_p2_cos_ideal','geometry_p4_cos_ideal']].to_numpy(float)
true=np.argmax(yarms,axis=1)
# One-vs-rest ExtraTrees, strict grouped OOF
score=np.zeros_like(yarms)
for g in ug:
    tr=groups!=g; te=groups==g
    for j in range(3):
        m=ExtraTreesRegressor(n_estimators=120,min_samples_leaf=4,max_features=.8,random_state=100+j+int(g),n_jobs=5)
        m.fit(X[tr],yarms[tr,j]); score[te,j]=m.predict(X[te])
pick=np.argmax(score,axis=1)
results['arm_selection']={'accuracy':float(np.mean(pick==true)),'mean_selected_ideal_cosine':float(np.mean(yarms[np.arange(len(yarms)),pick])),'mean_oracle_arm_cosine':float(np.mean(np.max(yarms,axis=1))),'mean_fixed_best_arm_cosine':float(np.max(np.mean(yarms,axis=0))),'fixed_best_arm':int(np.argmax(np.mean(yarms,axis=0)))}
out={'scope':'T4 exact-rotation development rows only; width256; 16 networks x 3 grouped rotations; no protected cohorts opened','features':features,'results':results}
(ROOT/'results'/'TEST2_T4_OBSERVABILITY_PROBE.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
