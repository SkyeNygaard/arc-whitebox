#!/usr/bin/env python3
"""Bounded signed-scale rescue for the reanchored pilot defect.

Uses only three frozen legal diagnostics: target covariance-source norm, its
late-layer growth, and inter-pilot disagreement. Hyperparameters are selected
by leave-one-network-out replay on the tuning block. The validation block is
opened once.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np


def quadratic(record):
    g=np.asarray(record['alpha_grid'],float); y=np.asarray(record['candidate_mse_grid'],float)
    ix=[int(np.where(np.isclose(g,a))[0][0]) for a in (-0.25,0.0,0.2)]
    return np.polyfit(g[ix],y[ix],2)

def features(record):
    tr=record['traces']; dc=float(np.mean(record['target_delta']['pilot_cov_fro']))
    growth=float(np.mean([tr[j][-1]['delta_cov_fro']/max(tr[j][-5]['delta_cov_fro'],1e-30) for j in range(2)]))
    dis=float(record['anchor']['pilot_disagreement'])
    return np.array([np.log(max(dc,1e-30)),np.log(max(growth,1e-30)),dis],float)

def fit_predict(X,y,x,lam):
    mu=X.mean(0); sd=X.std(0); sd=np.where(sd<1e-8,1.0,sd)
    Z=(X-mu)/sd; z=(x-mu)/sd
    A=Z.T@Z+lam*np.eye(Z.shape[1]); b=Z.T@(y-y.mean())
    return float(y.mean()+z@np.linalg.solve(A,b))

def mse_at(coef,a): return float(np.polyval(coef,a))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--records',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    recs=[json.loads(p.read_text()) for p in sorted(a.records.glob('network_*.json'))]
    tune=recs[:8]; val=recs[8:16]
    X=np.stack([features(r) for r in tune]); co=[quadratic(r) for r in tune]
    y=np.array([np.clip(-c[1]/(2*c[0]),-.2,.2) for c in co])
    base=np.array([r['baseline_mse'] for r in tune])
    candidates=[]
    for lam in (0.1,1.0,10.0,100.0):
      raw=[]
      for i in range(len(tune)):
        keep=np.arange(len(tune))!=i; raw.append(fit_predict(X[keep],y[keep],X[i],lam))
      raw=np.array(raw)
      for shrink in (0.25,0.5,0.75,1.0):
       for threshold in (0.0,0.02,0.04,0.06,0.08):
        pred=np.clip(raw*shrink,-.2,.2); pred=np.where(np.abs(pred)>=threshold,pred,0.0)
        ms=np.array([mse_at(co[i],pred[i]) for i in range(len(tune))])
        ratio=float(ms.sum()/base.sum()); worst=float(np.max(ms/base)); applied=int(np.sum(pred!=0))
        # Tail-first selection; unsafe configurations are ineligible.
        eligible=worst<=1.15 and applied>=2
        candidates.append({'lambda':lam,'shrink':shrink,'threshold':threshold,'loo_ratio':ratio,'loo_worst':worst,'loo_applied':applied,'eligible':eligible,'loo_alpha':pred.tolist(),'loo_per_network_ratio':(ms/base).tolist()})
    eligible=[x for x in candidates if x['eligible']]
    choice=min(eligible,key=lambda x:x['loo_ratio']) if eligible else min(candidates,key=lambda x:(x['loo_worst'],x['loo_ratio']))

    # Freeze on all tuning records.
    def predict_block(block):
      out=[]
      for r in block:
        p=fit_predict(X,y,features(r),choice['lambda'])*choice['shrink']
        p=float(np.clip(p,-.2,.2)); p=p if abs(p)>=choice['threshold'] else 0.0; out.append(p)
      return np.array(out)
    def summarize(block,pred):
      b=np.array([r['baseline_mse'] for r in block]); cc=[quadratic(r) for r in block]
      ms=np.array([mse_at(c,p) for c,p in zip(cc,pred)])
      return {'n':len(block),'candidate_over_base':float(ms.sum()/b.sum()),'wins':int(np.sum(ms<b)),'worst':float(np.max(ms/b)),'median':float(np.median(ms/b)),'applied':int(np.sum(pred!=0)),'predicted_alpha':pred.tolist(),'per_network_ratio':(ms/b).tolist(),'network_ids':[r['network_id'] for r in block]}
    pt=predict_block(tune); pv=predict_block(val)
    payload={'method':'three-feature ridge signed-scale with LOO tail gate','features':['log target covariance-source norm','log late covariance-source growth','inter-pilot disagreement'],'choice':choice,'tuning_fit':summarize(tune,pt),'validation':summarize(val,pv),'all_hyperparameters':candidates}
    a.out.write_text(json.dumps(payload,indent=2)); print(json.dumps({k:payload[k] for k in ('choice','tuning_fit','validation')},indent=2))
if __name__=='__main__': main()
