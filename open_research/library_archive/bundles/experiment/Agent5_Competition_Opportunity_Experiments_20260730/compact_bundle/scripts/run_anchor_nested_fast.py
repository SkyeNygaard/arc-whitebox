#!/usr/bin/env python3
import json,itertools
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
P=Path('/mnt/data/work/T4_legal_layer31_anchor_hedge_20260729_review/T4_legal_layer31_anchor_hedge_20260729/T41_EXACT_ROTATIONS/ROWS.csv');OUT=Path('/mnt/data/work/new_opportunities/results')
F=['feature_cos_c17_p2','feature_cos_c17_p4','feature_cos_p2_p4','feature_norm_p2_c17','feature_norm_p4_c17','feature_min_nested_cos','feature_max_loo_angle_sin','feature_cos_p32_p128','feature_norm_p32_p128']
d=pd.read_csv(P).sort_values(['network_id','rotation_index']).reset_index(drop=True);X=d[F].to_numpy();g=d.network_id.to_numpy();inn=d.inner.to_numpy();nn=d.norm_sq.to_numpy();b=d.baseline_unbiased_mse.to_numpy();y=-inn/np.maximum(nn,1e-30);w=nn/nn.mean()
def cand(a,idx):return b[idx]+2*a*inn[idx]+a*a*nn[idx]
def worst(c,idx):
 z=pd.DataFrame({'g':g[idx],'b':b[idx],'c':c});return max(x.c.sum()/x.b.sum() for _,x in z.groupby('g'))
methods={k:np.zeros(len(d)) for k in ['rf_oob_shrink','rf_oob_prob_gate','rf_oob_constant_gate']};choices=[]
for oi,(tr,te) in enumerate(LeaveOneGroupOut().split(X,y,groups=g)):
 rf=RandomForestRegressor(n_estimators=250,max_depth=4,min_samples_leaf=3,max_features=.8,bootstrap=True,oob_score=True,random_state=1000+oi,n_jobs=1)
 rf.fit(X[tr],y[tr],sample_weight=w[tr]);po=rf.oob_prediction_;pt=rf.predict(X[te])
 lab=(inn[tr]<0).astype(int);cl=RandomForestClassifier(n_estimators=250,max_depth=3,min_samples_leaf=3,max_features=.8,bootstrap=True,oob_score=True,class_weight='balanced',random_state=2000+oi,n_jobs=1)
 cl.fit(X[tr],lab);qo=cl.oob_decision_function_[:,1];qt=cl.predict_proba(X[te])[:,1]
 cfg=[]
 for lam,amax,thr in itertools.product([.25,.5,.75,1],[.25,.5,1,2],[0,.05,.1,.2,.3]):
  a=np.where(np.clip(po,0,amax)>=thr,lam*np.clip(po,0,amax),0);c=cand(a,tr);cfg.append((worst(c,tr)>1.1,c.sum()/b[tr].sum(),lam,amax,thr))
 cfg.sort();_,_,lam,amax,thr=cfg[0];methods['rf_oob_shrink'][te]=np.where(np.clip(pt,0,amax)>=thr,lam*np.clip(pt,0,amax),0)
 cfg=[]
 for lam,amax,pth in itertools.product([.25,.5,.75,1],[.25,.5,1,2],[.5,.6,.7,.8,.9]):
  a=np.where(qo>=pth,lam*np.clip(po,0,amax),0);c=cand(a,tr);cfg.append((worst(c,tr)>1.1,c.sum()/b[tr].sum(),lam,amax,pth))
 cfg.sort();_,_,lam2,amax2,pth=cfg[0];methods['rf_oob_prob_gate'][te]=np.where(qt>=pth,lam2*np.clip(pt,0,amax2),0)
 ag=np.clip(-inn[tr].sum()/nn[tr].sum(),0,2);cfg=[]
 for lam,pth3 in itertools.product([.25,.5,.75,1],[.5,.6,.7,.8,.9]):
  a=np.where(qo>=pth3,lam*ag,0);c=cand(a,tr);cfg.append((worst(c,tr)>1.1,c.sum()/b[tr].sum(),lam,pth3))
 cfg.sort();_,_,lam3,pth3=cfg[0];methods['rf_oob_constant_gate'][te]=np.where(qt>=pth3,lam3*ag,0)
 choices.append({'heldout':int(g[te][0]),'shrink':[lam,amax,thr],'prob':[lam2,amax2,pth],'constant':[lam3,ag,pth3]})
def summ(a):
 c=cand(a,np.arange(len(d)));rat=[c[g==z].sum()/b[g==z].sum() for z in np.unique(g)]
 return {'candidate_over_base':float(c.sum()/b.sum()),'gain':float(b.sum()/c.sum()),'row_wins':int((c<b).sum()),'network_wins':int((np.array(rat)<1).sum()),'worst_network_ratio':float(max(rat)),'median_network_ratio':float(np.median(rat)),'alpha_mean':float(a.mean()),'alpha_max':float(a.max()),'active_rows':int((a!=0).sum())}
res={k:summ(v) for k,v in methods.items()};res['oracle_clip_0_2']=summ(np.clip(y,0,2));res['original']=summ(np.ones(len(d)))
out={'alphas':{k:v.tolist() for k,v in methods.items()},'protocol':'outer leave-one-network-out; OOB tuning inside outer training; legal archived features only','features':F,'results':res,'choices':choices};(OUT/'ANCHOR_NESTED_SALVAGE.json').write_text(json.dumps(out,indent=2));pd.DataFrame([{'method':k,**v} for k,v in res.items()]).to_csv(OUT/'ANCHOR_NESTED_SALVAGE.csv',index=False);print(pd.DataFrame([{'method':k,**v} for k,v in res.items()]).sort_values('candidate_over_base').to_string(index=False))
