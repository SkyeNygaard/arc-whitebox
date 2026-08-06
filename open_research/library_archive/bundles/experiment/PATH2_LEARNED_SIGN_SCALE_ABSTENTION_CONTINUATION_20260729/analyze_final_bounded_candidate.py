from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
TH=0.0027637032840478275

def load_dev():
 d=pd.read_csv(ROOT/'legal_features_and_labels.csv');rows=[]
 for _,z in d.iterrows():
  panel=ROOT/'canonical24_quadratics_audited' if z.domain=='canonical' else ROOT/'hardpanel_quadratics_audited'
  v=np.load(panel/'vectors'/f'vectors_{int(z.network_seed)}_r{int(z.rotation_seed)}_n262144.npz');b=v['base'];t=.5*(v['truth_half1']+v['truth_half2']);p=v['direct32'];e=b-t;dd=p-b
  rows.append({'a0':np.mean(e*e),'lin':2*np.mean(e*dd),'q':np.mean(dd*dd),'risk':z.l08_fold_rel_mean,'source':'development','domain':z.domain,'in_hard':bool(z.in_hard_panel)})
 return pd.concat([d[['network_seed','rotation_seed']].reset_index(drop=True),pd.DataFrame(rows)],axis=1)
def load_hold():
 p=ROOT/'fresh_scale_holdout_v1';d=pd.read_csv(p/'fresh_screen_rows.csv');rows=[]
 for _,z in d.iterrows():
  v=np.load(p/'vectors'/f'vectors_{int(z.network_seed)}_r{int(z.rotation_seed)}_n262144.npz');b=v['base'];t=.5*(v['truth_half1']+v['truth_half2']);pr=v['direct32'];e=b-t;dd=pr-b
  rows.append({'a0':np.mean(e*e),'lin':2*np.mean(e*dd),'q':np.mean(dd*dd),'risk':z.risk_n129,'source':'immutable_holdout','domain':'fresh','in_hard':True})
 return pd.concat([d[['network_seed','rotation_seed']].reset_index(drop=True),pd.DataFrame(rows)],axis=1)
def met(d,a,boot=False):
 b=d.a0.to_numpy();m=b+a*d['lin'].to_numpy()+a*a*d.q.to_numpy();r=m/b;o={'n':len(d),'groups':int(d.network_seed.nunique()),'pooled_ratio':float(m.sum()/b.sum()),'worst':float(r.max()),'p90':float(np.quantile(r,.9)),'wins':int((r<1).sum()),'alpha_mean':float(a.mean()),'high_risk_count':int((d.risk>TH).sum())}
 if boot:
  gs=np.unique(d.network_seed);gb=np.array([b[d.network_seed.eq(g)].sum() for g in gs]);gm=np.array([m[d.network_seed.eq(g)].sum() for g in gs]);rng=np.random.default_rng(451055);ix=rng.integers(0,len(gs),size=(100000,len(gs)));v=gm[ix].sum(1)/gb[ix].sum(1);o['grouped_ci95']=[float(x) for x in np.quantile(v,[.025,.5,.975])]
 return o
dev=load_dev();hold=load_hold();all=pd.concat([dev,hold],ignore_index=True)
def alpha(d):return np.where(d.risk.to_numpy()>TH,.45,.55)
out={'candidate':{'risk_feature':'l08 six-fold Kerdock block relative dispersion','threshold':TH,'high_risk_alpha':.45,'ordinary_alpha':.55},'development_all32':met(dev,alpha(dev),True),'development_canonical24':met(dev[dev.domain.eq('canonical')],alpha(dev[dev.domain.eq('canonical')]),True),'development_hard12':met(dev[dev.in_hard],alpha(dev[dev.in_hard]),True),'immutable_holdout36':met(hold,alpha(hold),True),'combined_unique68':met(all,alpha(all),True)}
(ROOT/'results'/'FINAL_BOUNDED_CANDIDATE_RESULTS.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
