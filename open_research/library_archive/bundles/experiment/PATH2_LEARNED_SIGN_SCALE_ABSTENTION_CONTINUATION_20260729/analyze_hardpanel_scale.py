from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parent
D=pd.read_csv(ROOT/'hardpanel_quadratics_audited'/'fresh_screen_rows.csv')
# Add exact raw quadratics from saved vectors.
rows=[]
for _,z in D.iterrows():
 p=Path(ROOT/'hardpanel_quadratics_audited'/'vectors'/f'vectors_{int(z.network_seed)}_r{int(z.rotation_seed)}_n262144.npz');v=np.load(p)
 base=v['base'];truth=.5*(v['truth_half1']+v['truth_half2']);pred=v['direct32'];e=base-truth;d=pred-base
 rows.append({'a0':float(np.mean(e*e)),'lin':float(2*np.mean(e*d)),'q':float(np.mean(d*d))})
Q=pd.DataFrame(rows);D=pd.concat([D.reset_index(drop=True),Q],axis=1)
def mse(alpha): return D.a0.to_numpy()+alpha*D.lin.to_numpy()+alpha*alpha*D.q.to_numpy()
def metrics(alpha):
 a=np.asarray(alpha,float);m=mse(a);b=D.a0.to_numpy();rat=m/b
 return {'pooled_ratio':float(m.sum()/b.sum()),'mean_ratio':float(rat.mean()),'median_ratio':float(np.median(rat)),'worst':float(rat.max()),'p90':float(np.quantile(rat,.9)),'wins':int(np.sum(rat<1)),'alpha_mean':float(a.mean()),'alpha_min':float(a.min()),'alpha_max':float(a.max()),'ratios':rat.tolist(),'alphas':a.tolist()}
pol={}
for a in [0,.2,.25,.3,.35,.4,.5,.6,.75,1.0]: pol[f'fixed_{a:g}']=metrics(np.full(len(D),a))
pol['oracle_alpha']=metrics(D.alpha32.to_numpy())
pol['l08_abstain']=metrics(np.where(D.frozen_gate_apply_n129,1.,0.))
# Nested leave-one-base-out fit: alpha = clip(b0 + b1*z, 0, 1.5), optimize training pooled mse with ridge on slope.
z=np.log(D.risk_n129.to_numpy());g=D.network_seed.to_numpy();pred=np.zeros(len(D));params=[]
for s in np.unique(g):
 tr=g!=s;te=g==s;zt=z[tr];mu=zt.mean();sd=zt.std()+1e-12;zz=(zt-mu)/sd
 def obj(p):
  aa=np.clip(p[0]+p[1]*zz,0,1.5);return float(np.sum(D.a0.to_numpy()[tr]+aa*D.lin.to_numpy()[tr]+aa*aa*D.q.to_numpy()[tr])/np.sum(D.a0.to_numpy()[tr])+.02*p[1]*p[1])
 res=minimize(obj,[.5,0.],method='Nelder-Mead',options={'maxiter':5000});pred[te]=np.clip(res.x[0]+res.x[1]*(z[te]-mu)/sd,0,1.5);params.append({'held_seed':int(s),'b0':float(res.x[0]),'b1':float(res.x[1]),'pred':[float(x) for x in pred[te]]})
pol['nested_l08_linear_scale']=metrics(pred);pol['nested_l08_linear_scale']['details']=params
# Predeclared risk tiers: low full, middle .5, high .25. Tune thresholds by training risk quantiles only, no labels.
for q1,q2 in [(.5,.9),(.75,.9),(.75,.95),(.9,.95)]:
 a=np.zeros(len(D))
 det=[]
 for s in np.unique(g):
  tr=g!=s;te=g==s;t1=np.quantile(D.risk_n129[tr],q1);t2=np.quantile(D.risk_n129[tr],q2);a[te]=np.where(D.risk_n129[te]>t2,.25,np.where(D.risk_n129[te]>t1,.5,1.));det.append((int(s),t1,t2,a[te].tolist()))
 pol[f'tier_q{q1}_{q2}']=metrics(a);pol[f'tier_q{q1}_{q2}']['details']=det
out={'n':len(D),'groups':int(D.network_seed.nunique()),'alpha32_summary':D.alpha32.describe().to_dict(),
 'risk_correlations':{'spearman_alpha':float(spearmanr(D.risk_n129,D.alpha32).statistic),'spearman_full_ratio':float(spearmanr(D.risk_n129,D.k32_ratio).statistic)},
 'policies':pol,'rows':D[['network_seed','rotation_seed','risk_n129','k32_ratio','alpha32']].to_dict('records')}
(ROOT/'results'/'HARDPANEL_SCALE_RESULTS.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='policies'},indent=2));
print(pd.DataFrame([{ 'policy':k,**{x:v[x] for x in ['pooled_ratio','worst','p90','wins','alpha_mean']}} for k,v in pol.items()]).sort_values(['worst','pooled_ratio']).to_string(index=False))
