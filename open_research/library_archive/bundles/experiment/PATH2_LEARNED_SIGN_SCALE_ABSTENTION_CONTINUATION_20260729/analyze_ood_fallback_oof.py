from __future__ import annotations
import json,re
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.linear_model import Ridge
from sklearn.covariance import LedoitWolf
ROOT=Path(__file__).resolve().parent;D=pd.read_csv(ROOT/'legal_features_and_labels.csv')
A=[]
for _,z in D.iterrows():
 panel=ROOT/'canonical24_quadratics_audited' if z.domain=='canonical' else ROOT/'hardpanel_quadratics_audited';v=np.load(panel/'vectors'/f'vectors_{int(z.network_seed)}_r{int(z.rotation_seed)}_n262144.npz');b=v['base'];t=.5*(v['truth_half1']+v['truth_half2']);p=v['direct32'];e=b-t;d=p-b;A.append([np.mean(e*e),2*np.mean(e*d),np.mean(d*d),-np.mean(e*d)/np.mean(d*d)])
A=np.array(A);META={'network_seed','rotation_seed','baseline_mse','oracle_ratio','candidate_ratio','oracle_headroom','domain','in_hard_panel','harm','no_headroom','feature_runtime_seconds'};NUM=[c for c in D if c not in META and np.issubdtype(D[c].dtype,np.number)];cs=[c for c in NUM if re.match(r'l(08|16|24|28|29|30|31)_fold_rel_(mean|q50|q90|max)$',c) or c in ['anchor_effrank','anchor_frob','anchor_r90','anchor_rho','anchor_trace'] or re.match(r'w_suffix_(fro|op|std|trace)_(mean|q50|q90|max)$',c)]
G=D.network_seed.to_numpy();CAN=D.domain.eq('canonical').to_numpy();HARD=D.in_hard_panel.astype(bool).to_numpy();pred=np.zeros(len(D));ood=np.zeros(len(D));ab={q:np.zeros(len(D),bool) for q in [.95,.99,1.]}
for tr,te in LeaveOneGroupOut().split(D,groups=G):
 imp=SimpleImputer();sc=StandardScaler();X=sc.fit_transform(imp.fit_transform(D.iloc[tr][cs]));T=sc.transform(imp.transform(D.iloc[te][cs]));sel=SelectKBest(f_regression,k=16).fit(X,A[tr,3]);Xs=sel.transform(X);Ts=sel.transform(T);ridge=Ridge(alpha=1).fit(Xs,np.clip(A[tr,3],0,1.5),sample_weight=A[tr,2]/A[tr,2].mean());pred[te]=np.clip(ridge.predict(Ts),.25,.75);lw=LedoitWolf().fit(Xs);train=lw.mahalanobis(Xs);ood[te]=lw.mahalanobis(Ts)
 for q in ab:ab[q][te]=ood[te]>np.quantile(train,q)
def met(a):
 m=A[:,0]+a*A[:,1]+a*a*A[:,2];r=m/A[:,0]
 def sub(mask):return {'pooled_ratio':float(m[mask].sum()/A[:,0][mask].sum()),'worst':float(r[mask].max()),'p90':float(np.quantile(r[mask],.9)),'wins':int((r[mask]<1).sum())}
 return {'canonical':sub(CAN),'hard':sub(HARD),'all':sub(np.ones(len(D),bool))}
res={'model':met(pred),'fixed05':met(np.full(len(D),.5))}
for q,x in ab.items():res[f'ood_q{q}']={**met(np.where(x,.5,pred)),'abstentions':int(x.sum())}
out={'protocol':'development leave-one-base-network-out; compact k16 ridge1 bounded [.25,.75]; OOD threshold fit only on outer training rows','policies':res,'rows':D[['network_seed','rotation_seed']].assign(pred=pred,ood=ood,**{f'abstain_q{q}':v for q,v in ab.items()}).to_dict('records')};(ROOT/'results'/'OOD_FALLBACK_OOF_RESULTS.json').write_text(json.dumps(out,indent=2));print(json.dumps(res,indent=2))
