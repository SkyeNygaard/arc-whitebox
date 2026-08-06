from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parent

def load(panel):
 d=pd.read_csv(panel/'fresh_screen_rows.csv').sort_values(['network_seed','rotation_seed']).reset_index(drop=True);rows=[]
 for _,z in d.iterrows():
  v=np.load(panel/'vectors'/f'vectors_{int(z.network_seed)}_r{int(z.rotation_seed)}_n262144.npz');b=v['base'];t=.5*(v['truth_half1']+v['truth_half2']);p=v['direct128'];e=b-t;dd=p-b
  a0=float(np.mean(e*e));lin=float(2*np.mean(e*dd));q=float(np.mean(dd*dd));rows.append({'a0':a0,'lin':lin,'q':q,'alpha':float(-lin/(2*q))})
 return pd.concat([d,pd.DataFrame(rows)],axis=1)
def met(d,a,boot=0):
 a=np.asarray(a);b=d.a0.to_numpy();m=b+a*d.lin.to_numpy()+a*a*d.q.to_numpy();r=m/b;o={'pooled_ratio':float(m.sum()/b.sum()),'worst':float(r.max()),'p90':float(np.quantile(r,.9)),'wins':int((r<1).sum()),'alpha_mean':float(a.mean())}
 if boot:
  rng=np.random.default_rng(1282026);ix=rng.integers(0,len(d),size=(50000,len(d)));vals=m[ix].sum(axis=1)/b[ix].sum(axis=1)
  o['ci95']=[float(x) for x in np.quantile(vals,[.025,.975])]
 return o
can=load(ROOT/'canonical24_quadratics_audited');hard=load(ROOT/'hardpanel_quadratics_audited')
out={'canonical':{},'hard':{},'alpha_summary_canonical':can.alpha.describe().to_dict(),'alpha_summary_hard':hard.alpha.describe().to_dict(),'risk_alpha_spearman_canonical':float(spearmanr(can.risk_n129,can.alpha).statistic)}
for a in [.2,.3,.4,.45,.5,.55,.6,.75,1.]:out['canonical'][f'fixed_{a}']=met(can,np.full(len(can),a),1);out['hard'][f'fixed_{a}']=met(hard,np.full(len(hard),a))
out['canonical']['oracle_alpha']=met(can,can.alpha.to_numpy());out['hard']['oracle_alpha']=met(hard,hard.alpha.to_numpy())
(ROOT/'results'/'K128_SCALE_RESULTS.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
