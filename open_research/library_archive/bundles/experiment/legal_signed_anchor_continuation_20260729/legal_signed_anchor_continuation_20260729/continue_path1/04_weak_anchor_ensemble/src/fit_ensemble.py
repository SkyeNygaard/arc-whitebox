#!/usr/bin/env python3
from __future__ import annotations
import argparse,itertools,json
from pathlib import Path
import numpy as np
METHODS=['0.0','0.125','0.25','0.5','0.75','1.0']
RIDGES=[0.0,1e-8,1e-6,1e-4,1e-2,1e-1,1.0,10.0]

def load(path): return [json.loads(p.read_text()) for p in sorted(path.glob('network_*.json'))]
def arrays(r,subset):
 b=np.array(r['baseline_pred']);y1=np.array(r['truth_half1']);y2=np.array(r['truth_half2']);D=np.stack([np.array(r['methods'][m]['delta_output']) for m in subset],axis=1);return b,y1,y2,D

def fit(rs,subset,ridge):
 X=[];y=[]
 for r in rs:
  b,y1,y2,D=arrays(r,subset);X.append(D);y.append(.5*(y1+y2)-b)
 X=np.concatenate(X);y=np.concatenate(y);A=X.T@X+ridge*np.eye(len(subset));return np.linalg.solve(A,X.T@y)
def metrics(rs,subset,c):
 bm=[];cm=[];ub=[];rat=[]
 for r in rs:
  b,y1,y2,D=arrays(r,subset);p=b+D@c;bb=np.mean((b-.5*(y1+y2))**2);cc=np.mean((p-.5*(y1+y2))**2);bm.append(bb);cm.append(cc);ub.append(np.mean((p-y1)*(p-y2)));rat.append(cc/bb)
 return {'candidate_over_base':float(np.sum(cm)/np.sum(bm)),'wins':int(np.sum(np.array(rat)<1)),'worst':float(np.max(rat)),'per_network':rat,'candidate_unbiased_sum':float(np.sum(ub))}
def loocv(rs,subset,ridge):
 preds=[]; bases=[]; ratios=[]
 for i in range(len(rs)):
  tr=rs[:i]+rs[i+1:];c=fit(tr,subset,ridge);m=metrics([rs[i]],subset,c);ratios+=m['per_network']
  r=rs[i];b,y1,y2,D=arrays(r,subset);p=b+D@c;bases.append(np.mean((b-.5*(y1+y2))**2));preds.append(np.mean((p-.5*(y1+y2))**2))
 return float(np.sum(preds)/np.sum(bases)),float(np.max(ratios)),int(np.sum(np.array(ratios)<1))

def main():
 p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True);p.add_argument('--validation',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();tr=load(a.train);rows=[]
 for k in [1,2,3]:
  for sub in itertools.combinations(METHODS,k):
   for ridge in RIDGES:
    o,w,win=loocv(tr,sub,ridge);rows.append((o,w,-win,sub,ridge))
 # Prefer OOF-safe candidates, then aggregate; otherwise overall best.
 safe=[x for x in rows if x[1]<=1.15]
 selected=min(safe or rows,key=lambda x:(x[0],x[1],x[2]))
 o,w,nwin,sub,ridge=selected;c=fit(tr,sub,ridge);out={'selected_methods':list(sub),'ridge':ridge,'coefficients':c.tolist(),'oof':{'candidate_over_base':o,'worst':w,'wins':-nwin},'training':metrics(tr,sub,c),'top_oof':[{'ratio':x[0],'worst':x[1],'wins':-x[2],'methods':list(x[3]),'ridge':x[4]} for x in sorted(rows)[:20]]}
 if a.validation: out['validation']=metrics(load(a.validation),sub,c)
 a.out.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
