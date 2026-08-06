#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
RIDGES=[0,1e-10,1e-8,1e-6,1e-4,1e-2,.1,1,10]
def load(p):return [json.loads(q.read_text()) for q in sorted(p.glob('network_*.json'))]
def join(ensemble,q128):
 e={r['network_id']:r for r in load(ensemble)};q={r['network_id']:r for r in load(q128)};out=[]
 for n in sorted(set(e)&set(q)):
  r=e[n];out.append({'n':n,'base':np.array(r['baseline_pred']),'y1':np.array(r['truth_half1']),'y2':np.array(r['truth_half2']),'D':np.stack([np.array(r['methods']['0.0']['delta_output']),np.array(q[n]['delta_output'])],1)})
 return out
def fit(rs,ridge):
 X=np.concatenate([r['D'] for r in rs]);y=np.concatenate([.5*(r['y1']+r['y2'])-r['base'] for r in rs]);return np.linalg.solve(X.T@X+ridge*np.eye(2),X.T@y)
def metrics(rs,c):
 bm=[];cm=[];rat=[]
 for r in rs:
  y=.5*(r['y1']+r['y2']);b=np.mean((r['base']-y)**2);p=r['base']+r['D']@c;m=np.mean((p-y)**2);bm.append(b);cm.append(m);rat.append(m/b)
 return {'candidate_over_base':float(np.sum(cm)/np.sum(bm)),'wins':int(np.sum(np.array(rat)<1)),'worst':float(np.max(rat)),'per_network':rat}
def loo(rs,ridge):
 bm=[];cm=[];rat=[]
 for i,r in enumerate(rs):
  c=fit(rs[:i]+rs[i+1:],ridge);m=metrics([r],c);rat+=m['per_network'];y=.5*(r['y1']+r['y2']);bm.append(np.mean((r['base']-y)**2));cm.append(bm[-1]*rat[-1])
 return float(np.sum(cm)/np.sum(bm)),float(np.max(rat)),int(np.sum(np.array(rat)<1))
def main():
 p=argparse.ArgumentParser();p.add_argument('--ensemble-train',type=Path,required=True);p.add_argument('--q128-train',type=Path,required=True);p.add_argument('--ensemble-validation',type=Path);p.add_argument('--q128-validation',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();tr=join(a.ensemble_train,a.q128_train);rows=[]
 for ridge in RIDGES:
  o,w,win=loo(tr,ridge);rows.append((o,w,-win,ridge))
 safe=[x for x in rows if x[1]<=1.15];o,w,nw,ridge=min(safe or rows);c=fit(tr,ridge);z={'ridge':ridge,'coefficients':c.tolist(),'oof':{'candidate_over_base':o,'worst':w,'wins':-nw},'training':metrics(tr,c),'grid':[{'ratio':x[0],'worst':x[1],'wins':-x[2],'ridge':x[3]} for x in rows]}
 if a.ensemble_validation:z['validation']=metrics(join(a.ensemble_validation,a.q128_validation),c)
 a.out.write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2))
if __name__=='__main__':main()
