import json,itertools,joblib,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import RandomForestClassifier
ROOT=Path('/mnt/data/work/T4_legal_layer31_anchor_hedge_20260729_review/T4_legal_layer31_anchor_hedge_20260729/T41_EXACT_ROTATIONS')
OUT=Path('/mnt/data/work/new_opportunities/anchor_fresh');OUT.mkdir(parents=True,exist_ok=True)
F=['feature_cos_c17_p2','feature_cos_c17_p4','feature_cos_p2_p4','feature_norm_p2_c17','feature_norm_p4_c17','feature_min_nested_cos','feature_max_loo_angle_sin','feature_cos_p32_p128','feature_norm_p32_p128']
d=pd.read_csv(ROOT/'ROWS.csv').sort_values(['network_id','rotation_index']).reset_index(drop=True);X=d[F].to_numpy();inn=d.inner.to_numpy();nn=d.norm_sq.to_numpy();b=d.baseline_unbiased_mse.to_numpy();g=d.network_id.to_numpy();lab=(inn<0).astype(int)
cl=RandomForestClassifier(n_estimators=2000,max_depth=3,min_samples_leaf=3,max_features=.8,bootstrap=True,oob_score=True,class_weight='balanced',random_state=20260730,n_jobs=1);cl.fit(X,lab);q=cl.oob_decision_function_[:,1];ag=float(np.clip(-inn.sum()/nn.sum(),0,2))
def c(a):return b+2*a*inn+a*a*nn
def worst(x):
 z=pd.DataFrame({'g':g,'b':b,'c':x});return max(v.c.sum()/v.b.sum() for _,v in z.groupby('g'))
opts=[]
for lam,pth in itertools.product([.125,.25,.375,.5,.625,.75,1.0],[.5,.55,.6,.65,.7,.75,.8,.85,.9]):
 a=np.where(q>=pth,lam*ag,0);x=c(a);opts.append((worst(x)>1.10,x.sum()/b.sum(),-int((x<b).sum()),lam,pth,worst(x),int((a>0).sum())))
opts.sort();best=opts[0];_,ratio,_,lam,pth,wst,active=best
joblib.dump(cl,OUT/'anchor_rf_classifier.joblib')
modelhash=hashlib.sha256((OUT/'anchor_rf_classifier.joblib').read_bytes()).hexdigest()
spec={'frozen_at':'2026-07-30T11:20:00-04:00','development_networks':sorted(map(int,np.unique(g))),'excluded_pre_freeze_networks':[7400],
'fresh_validation_networks':list(range(7401,7409)),'features':F,'action_policy':json.load(open(ROOT/'results/T41_POLICY_RESULTS.json'))['in_sample_best']['policy'],
'classifier':{'type':'RandomForestClassifier','n_estimators':2000,'max_depth':3,'min_samples_leaf':3,'max_features':.8,'class_weight':'balanced','random_state':20260730,'sha256':modelhash},
'global_optimal_scale':ag,'frozen_multiplier':lam,'active_scale':lam*ag,'probability_threshold':pth,
'oob_development':{'candidate_over_base':ratio,'gain':1/ratio,'worst_network_ratio':wst,'active_rows':active},
'validation_gate':{'candidate_over_base_max':.98,'bootstrap_gain_lower_min':1.0,'network_wins_min':5,'worst_network_ratio_max':1.10},
'governance':'Fresh synthetic IDs only. Sealed 6016-6031 cohorts remain unopened.'}
(OUT/'FROZEN_ANCHOR_SALVAGE_POLICY.json').write_text(json.dumps(spec,indent=2));print(json.dumps(spec,indent=2))
