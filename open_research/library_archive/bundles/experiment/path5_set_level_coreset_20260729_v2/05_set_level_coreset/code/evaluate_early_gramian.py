#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,pickle
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

G=1.1e-8

def load_dir(p:Path):
 rows=[]
 for f in sorted(p.glob('early_gramian_seed*.npz')):
  seed=int(f.stem.split('seed')[1])
  with np.load(f,allow_pickle=False) as z:
   rows.append({'seed':seed,'X':z['X'].astype(float),'labels':z['labels'].astype(float),'candidates':z['candidates'].astype(int),'raw':z['raw_scores'].astype(float),'names':z['score_names'].astype(str)})
 return rows

def metric(vals):
 v=np.asarray(vals,float)
 return {'pass11':int(np.sum(v<=G)),'pass22':int(np.sum(v<=2.2e-8)),'mean':float(v.mean()),'median':float(np.median(v)),'worst':float(v.max()),'values':v.tolist()}

def direct(rows):
 names=rows[0]['names']; out={}
 for j,n in enumerate(names):
  vals=[];c=[]
  for r in rows:
   k=int(np.argmin(r['raw'][:,j]));vals.append(r['labels'][k]);c.append(int(r['candidates'][k]))
  out[str(n)]={**metric(vals),'candidates':c}
 return out

def ev_pred(rows,pred):
 vals=[];cs=[]
 for r in rows:
  k=int(np.argmin(pred[r['seed']]));vals.append(r['labels'][k]);cs.append(int(r['candidates'][k]))
 return {**metric(vals),'candidates':cs}

def flatten(rows):
 return np.concatenate([r['X'] for r in rows]),np.concatenate([np.log10(r['labels']+1e-20) for r in rows]),np.concatenate([r['labels'] for r in rows])

def fit_predict(kind,model,tr,te):
 X,ylog,y=flatten(tr)
 if kind=='reg':
  m=clone(model).fit(X,ylog);return {r['seed']:m.predict(r['X']) for r in te},m
 if kind=='cls':
  m=clone(model).fit(X,(y<=G).astype(int));return {r['seed']:-m.predict_proba(r['X'])[:,1] for r in te},m
 xx=[];yy=[]
 for r in tr:
  Z=r['X'];L=r['labels']
  for i in range(8):
   for j in range(i+1,8):
    d=Z[i]-Z[j];w=int(L[i]<L[j]);xx.extend([d,-d]);yy.extend([w,1-w])
 m=clone(model).fit(np.asarray(xx),np.asarray(yy));out={}
 for r in te:
  Z=r['X'];votes=np.zeros(8);ds=[];pairs=[]
  for i in range(8):
   for j in range(i+1,8):ds.append(Z[i]-Z[j]);pairs.append((i,j))
  pr=m.predict_proba(np.asarray(ds))[:,1]
  for q,(i,j) in zip(pr,pairs):votes[i]+=q;votes[j]+=1-q
  out[r['seed']]=-votes
 return out,m

def crossval(kind,model,rows):
 p={}; idx=np.arange(len(rows));groups=np.array([r['seed'] for r in rows])
 for a,b in GroupKFold(8).split(idx,groups=groups):
  q,_=fit_predict(kind,model,[rows[i] for i in a],[rows[i] for i in b]);p.update(q)
 return ev_pred(rows,p)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--train',type=Path,required=True);ap.add_argument('--dev',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 tr=load_dir(a.train);dv=load_dir(a.dev)
 drtr=direct(tr);drdv=direct(dv)
 ranked_direct=sorted(drtr,key=lambda n:(-drtr[n]['pass11'],drtr[n]['worst'],drtr[n]['mean']))
 best_direct=ranked_direct[0]
 mods=[]
 for alpha in [.01,.1,1,10,100,1000]:mods.append((f'ridge{alpha}','reg',make_pipeline(StandardScaler(),Ridge(alpha=alpha))))
 for C in [.001,.01,.1,1,10]:
  mods.append((f'logit{C}','cls',make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=3000,class_weight='balanced'))))
  mods.append((f'pair{C}','pair',make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=3000))))
 for leaf in [2,4,8,16]:mods.append((f'extra{leaf}','cls',ExtraTreesClassifier(n_estimators=300,min_samples_leaf=leaf,max_features=.7,class_weight='balanced',random_state=17,n_jobs=-1)))
 learned=[];fits={}
 for n,k,m in mods:
  cv=crossval(k,m,tr);pd,fit=fit_predict(k,m,tr,dv);dev=ev_pred(dv,pd);learned.append({'name':n,'kind':k,'cv':cv,'dev':dev});fits[n]=fit
 ranked_learned=sorted(learned,key=lambda x:(-x['cv']['pass11'],x['cv']['worst'],x['cv']['mean']))
 frozen=ranked_learned[0]
 payload={'counts':{'train':len(tr),'dev':len(dv)},'best_direct_by_train':{'name':best_direct,'train':drtr[best_direct],'dev':drdv[best_direct]},
          'direct_train':drtr,'direct_dev':drdv,'frozen_learned_by_cv':frozen,'learned':learned,
          'oracle_top8_train':metric([r['labels'].min() for r in tr]),'oracle_top8_dev':metric([r['labels'].min() for r in dv])}
 a.out.write_text(json.dumps(payload,indent=2));pickle.dump({'name':frozen['name'],'model':fits[frozen['name']]},open(a.out.with_suffix('.pkl'),'wb'))
 print(json.dumps({'best_direct':payload['best_direct_by_train'],'best_learned':frozen,'oracle_train':payload['oracle_top8_train'],'oracle_dev':payload['oracle_top8_dev']},indent=2))
if __name__=='__main__':main()
