from __future__ import annotations
import json,glob,math
from pathlib import Path
import numpy as np
OUT=Path('/mnt/data/competition_relevance_20260730/a50_radial_odd_v2')
files=sorted(glob.glob(str(OUT/'n*_r*.json')))
cases=[json.load(open(f)) for f in files]
odd=['odd_base','odd_center','odd_sens','odd_center_weighted','odd_sample_anchor']
even=['even_base_norm','even_center_norm','even_sens_norm','even_sv']
models={'constant':[], 'even':even, 'odd':odd, 'even_odd':even+odd,
        'odd_sample':['odd_sample_anchor'], 'odd_center_weighted':['odd_center_weighted']}

def matrices(case,keys):
 rows=case['features'];X=np.array([[r[k] for k in keys] for r in rows],float) if keys else np.zeros((4,0));y=np.array([r['target'] for r in rows]);return X,y

def fit_predict(train,test,keys,lam):
 Xs=[];ys=[];modes=[]
 for c in train:
  X,y=matrices(c,keys);Xs.append(X);ys.append(y);modes.extend(range(4))
 X=np.vstack(Xs);y=np.concatenate(ys);modes=np.array(modes)
 # Always include mode-specific intercepts, never a free global intercept.
 M=np.eye(4)[modes]
 if keys:
  mu=X.mean(0);sd=X.std(0);sd[sd<1e-10]=1;Z=(X-mu)/sd;A=np.column_stack([M,Z])
 else:mu=np.zeros(0);sd=np.ones(0);A=M
 pen=np.ones(A.shape[1]);pen[:4]=0
 coef=np.linalg.solve(A.T@A+lam*np.diag(pen)+1e-9*np.eye(A.shape[1]),A.T@y)
 Xt,yt=matrices(test,keys);Mt=np.eye(4)
 At=np.column_stack([Mt,(Xt-mu)/sd]) if keys else Mt
 pred=At@coef
 cap=max(np.quantile(np.abs(y),.95)*1.5,1e-8);return np.clip(pred,-cap,cap)

def case_ratio(c,pred):
 e=np.array(c['error_coeff']);outside=c['outside_error_sq'];num=outside+np.sum((e+.5*pred)**2);den=256*c['base_mse'];return float(num/den)

def nested_lambda(train,keys):
 if len({c['network_id'] for c in train})<3:return 10.0
 ls=[0.01,.1,1.,10.,100.,1000.];scores=[]
 nets=sorted({c['network_id'] for c in train})
 for lam in ls:
  rr=[]
  for n in nets:
   tr=[c for c in train if c['network_id']!=n];te=[c for c in train if c['network_id']==n]
   for c in te:rr.append(case_ratio(c,fit_predict(tr,c,keys,lam)))
  scores.append((np.mean(rr),lam))
 return min(scores)[1]

nets=sorted({c['network_id'] for c in cases});results={}
for name,keys in models.items():
 rows=[]
 for n in nets:
  tr=[c for c in cases if c['network_id']!=n];te=[c for c in cases if c['network_id']==n];lam=nested_lambda(tr,keys)
  for c in te:
   pred=fit_predict(tr,c,keys,lam);rows.append({'network':n,'rot':c['rot'],'ratio':case_ratio(c,pred),'lambda':lam,'pred':pred.tolist(),'target':[r['target'] for r in c['features']]})
 ratios=np.array([r['ratio'] for r in rows]);results[name]={'mean_ratio':float(ratios.mean()),'median_ratio':float(np.median(ratios)),'wins':int(np.sum(ratios<1)),'n':len(ratios),'worst':float(ratios.max()),'rows':rows}
# source and rank4 ceilings
for key,field in [('full_oracle','oracle_half_ratio'),('rank4_oracle','rank4_half_ratio')]:
 r=np.array([c[field] for c in cases]);results[key]={'mean_ratio':float(r.mean()),'median_ratio':float(np.median(r)),'wins':int(np.sum(r<1)),'n':len(r),'worst':float(r.max())}
# sign/correlation diagnostics for raw features pooled by modes.
diag={}
for k in odd+even:
 x=[];y=[]
 for c in cases:
  for row in c['features']:x.append(row[k]);y.append(row['target'])
 x=np.asarray(x);y=np.asarray(y);diag[k]={'corr':float(np.corrcoef(x,y)[0,1]),'sign_acc':float(np.mean(np.sign(x)==np.sign(y)))}
obj={'cases':len(cases),'networks':nets,'results':results,'feature_diagnostics':diag}
(OUT/'ANALYSIS.json').write_text(json.dumps(obj,indent=2));
print(json.dumps({k:{kk:v[kk] for kk in v if kk!='rows'} for k,v in results.items()},indent=2));print('features',json.dumps(diag,indent=2))
