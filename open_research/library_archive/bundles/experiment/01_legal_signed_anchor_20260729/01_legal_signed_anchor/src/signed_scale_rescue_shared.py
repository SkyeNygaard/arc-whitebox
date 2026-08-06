#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def quadratic(r):
 g=np.asarray(r['alpha_grid']); y=np.asarray(r['candidate_mse_grid']); ids=[int(np.where(np.isclose(g,a))[0][0]) for a in (-.25,0,.2)]; return np.polyfit(g[ids],y[ids],2)
def feat(r):
 tr=r['traces'][0]; dc=r['target_delta']['pilot_cov_fro'][0]; growth=tr[-1]['delta_cov_fro']/max(tr[-5]['delta_cov_fro'],1e-30); dis=r['anchor']['pilot_disagreement']; return np.array([np.log(max(dc,1e-30)),np.log(max(growth,1e-30)),dis])
def pred(X,y,x,lam):
 mu=X.mean(0); sd=np.where(X.std(0)<1e-8,1,X.std(0)); Z=(X-mu)/sd; z=(x-mu)/sd; return float(y.mean()+z@np.linalg.solve(Z.T@Z+lam*np.eye(3),Z.T@(y-y.mean())))
def main():
 a=argparse.ArgumentParser(); a.add_argument('--records',type=Path,required=True); a.add_argument('--out',type=Path,required=True); q=a.parse_args()
 rs=[json.loads((q.records/f'network_{i}.json').read_text()) for i in range(3000,3012)]; tune=rs[:6]; val=rs[6:]
 X=np.stack([feat(r) for r in tune]); C=[quadratic(r) for r in tune]; y=np.array([np.clip(-c[1]/(2*c[0]),-.2,.2) for c in C]); b=np.array([r['baseline_mse'] for r in tune])
 configs=[]
 for lam in (.1,1,10,100):
  raw=[]
  for i in range(6):
   k=np.arange(6)!=i; raw.append(pred(X[k],y[k],X[i],lam))
  raw=np.array(raw)
  for shrink in (.25,.5,.75,1):
   for th in (0,.01,.02,.04,.06):
    al=np.clip(raw*shrink,-.2,.2); al=np.where(np.abs(al)>=th,al,0); ms=np.array([np.polyval(C[i],al[i]) for i in range(6)]); ratio=ms.sum()/b.sum(); worst=np.max(ms/b); applied=np.sum(al!=0); configs.append(dict(lam=lam,shrink=shrink,threshold=th,loo_ratio=float(ratio),loo_worst=float(worst),loo_applied=int(applied),eligible=bool(worst<=1.15 and applied>=2),loo_alpha=al.tolist()))
 ok=[c for c in configs if c['eligible']]; choice=min(ok,key=lambda c:c['loo_ratio']) if ok else min(configs,key=lambda c:(c['loo_worst'],c['loo_ratio']))
 def block(block):
  al=[]
  for r in block:
   x=pred(X,y,feat(r),choice['lam'])*choice['shrink']; x=float(np.clip(x,-.2,.2)); al.append(x if abs(x)>=choice['threshold'] else 0)
  Cb=[quadratic(r) for r in block]; bb=np.array([r['baseline_mse'] for r in block]); ms=np.array([np.polyval(c,x) for c,x in zip(Cb,al)]); return dict(n=len(block),candidate_over_base=float(ms.sum()/bb.sum()),wins=int(np.sum(ms<bb)),worst=float(np.max(ms/bb)),applied=int(np.sum(np.array(al)!=0)),predicted_alpha=al,per_network_ratio=(ms/bb).tolist(),network_ids=[r['network_id'] for r in block])
 out={'method':'compute-compliant three-feature signed-scale rescue','choice':choice,'tuning_fit':block(tune),'validation':block(val),'all_hyperparameters':configs}; q.out.write_text(json.dumps(out,indent=2)); print(json.dumps({k:out[k] for k in ('choice','tuning_fit','validation')},indent=2))
if __name__=='__main__':main()
