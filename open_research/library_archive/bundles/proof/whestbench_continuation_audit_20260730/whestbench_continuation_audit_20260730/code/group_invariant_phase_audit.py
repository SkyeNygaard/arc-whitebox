from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
CSV=Path('/mnt/data/whestbench_continuation_20260730/t4/T4_legal_layer31_anchor_hedge_20260729/T41_EXACT_ROTATIONS/ROWS.csv')
df=pd.read_csv(CSV).sort_values(['network_id','rotation_index']).reset_index(drop=True)
features=['feature_cos_c17_p2','feature_cos_c17_p4','feature_cos_p2_p4','feature_norm_p2_c17','feature_norm_p4_c17','feature_min_nested_cos','feature_max_loo_angle_sin','feature_cos_p32_p128','feature_norm_p32_p128']
targets=['geometry_c17_cos_ideal','geometry_p2_cos_ideal','geometry_p4_cos_ideal']
X=df[features].to_numpy(float); groups=df.network_id.to_numpy(); unique=np.unique(groups); ng=len(unique); nr=3
models={
 'ridge':lambda:make_pipeline(StandardScaler(),Ridge(alpha=10.0)),
 'poly2_ridge':lambda:make_pipeline(StandardScaler(),PolynomialFeatures(2,include_bias=False),Ridge(alpha=100.0)),
 'extra_trees':lambda:ExtraTreesRegressor(n_estimators=80,min_samples_leaf=4,max_features=0.7,random_state=20260730,n_jobs=-1),
}
folds=list(GroupKFold(n_splits=8).split(X,groups=groups)); rng=np.random.default_rng(20260730); nperm=5000; perms=np.array([rng.permutation(ng) for _ in range(nperm)])
out={'n_rows':len(df),'n_base_networks':ng,'cv':'8-fold grouped by base network','features':features,'targets':{}}
for target in targets:
 y=df[target].to_numpy(float); yblock=y.reshape(ng,nr); tres={}
 for name,maker in models.items():
  pred=np.zeros_like(y)
  for tr,te in folds:
   m=maker();m.fit(X[tr],y[tr]);pred[te]=m.predict(X[te])
  r2=float(r2_score(y,pred)); corr=float(np.corrcoef(y,pred)[0,1]); sign=float(np.mean(np.sign(pred)==np.sign(y)))
  yp=yblock[perms].reshape(nperm,-1); pc=pred-pred.mean(); yc=yp-yp.mean(axis=1,keepdims=True); den=np.sqrt(np.sum(yc*yc,axis=1)*np.sum(pc*pc)); vals=np.abs((yc@pc)/den); obs=abs(corr); p=float((1+np.sum(vals>=obs))/(1+nperm))
  tres[name]={'oof_r2':r2,'oof_correlation':corr,'sign_accuracy':sign,'group_permutation_p_abs_corr':p,'prediction_mean':float(pred.mean()),'target_mean':float(y.mean())}
 by=yblock
 tres['diagnostics']={'positive_fraction':float(np.mean(y>0)),'networks_with_both_signs':int(np.sum((by.min(axis=1)<0)&(by.max(axis=1)>0))), 'within_network_variance_fraction':float(np.mean(np.var(by,axis=1))/np.var(y)),'best_constant_sign_accuracy':float(max(np.mean(y>0),np.mean(y<0)))}
 out['targets'][target]=tres
P=Path('/mnt/data/whestbench_continuation_20260730/local_verification/group_invariant_phase_audit.json');P.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
