#!/usr/bin/env python3
from __future__ import annotations
import json,pickle
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
G=1.1e-8
TR=Path('/mnt/data/path5_work/top8_rank_train32');DV=Path('/mnt/data/path5_work/top8_rank_dev8')
def load(p):
 out=[]
 for f in sorted(p.glob('top8_rank_seed*.npz')):
  with np.load(f,allow_pickle=False) as z:out.append((int(f.stem.split('seed')[1]),z['X'].astype(float),z['labels'].astype(float),z['candidates'].astype(int)))
 return out
def flat(rows):return np.concatenate([x[1] for x in rows]),np.concatenate([np.log10(x[2]+1e-20) for x in rows]),np.concatenate([x[2] for x in rows])
def ev(rows,p):
 v=[];c=[]
 for s,X,y,ids in rows:j=int(np.argmin(p[s]));v.append(y[j]);c.append(int(ids[j]))
 v=np.array(v);return {'pass11':int((v<=G).sum()),'pass22':int((v<=2.2e-8).sum()),'mean':float(v.mean()),'worst':float(v.max()),'values':v.tolist(),'candidates':c}
def fitpred(kind,m,tr,te):
 X,y,l=flat(tr)
 if kind=='reg':m=clone(m).fit(X,y);return {s:m.predict(Z) for s,Z,_,_ in te},m
 if kind=='cls':m=clone(m).fit(X,(l<=G).astype(int));return {s:-m.predict_proba(Z)[:,1] for s,Z,_,_ in te},m
 # pairwise logit
 xx=[];yy=[]
 for _,Z,L,_ in tr:
  for i in range(8):
   for j in range(i+1,8):
    d=Z[i]-Z[j];w=int(L[i]<L[j]);xx.extend([d,-d]);yy.extend([w,1-w])
 m=clone(m).fit(np.asarray(xx),np.asarray(yy));out={}
 for s,Z,_,_ in te:
  votes=np.zeros(8);ds=[];pairs=[]
  for i in range(8):
   for j in range(i+1,8):ds.append(Z[i]-Z[j]);pairs.append((i,j))
  pr=m.predict_proba(np.asarray(ds))[:,1]
  for q,(i,j) in zip(pr,pairs):votes[i]+=q;votes[j]+=1-q
  out[s]=-votes
 return out,m
def cv(kind,m,rows):
 p={};seeds=np.array([x[0] for x in rows]);idx=np.arange(len(rows))
 for a,b in GroupKFold(8).split(idx,groups=seeds):
  q,_=fitpred(kind,m,[rows[i] for i in a],[rows[i] for i in b]);p.update(q)
 return ev(rows,p)
tr=load(TR);dv=load(DV)
mods=[]
for a in [.01,.1,1,10,100]:mods.append((f'ridge{a}','reg',make_pipeline(StandardScaler(),Ridge(alpha=a))))
for C in [.001,.01,.1,1,10,100]:
 mods.append((f'logit{C}','cls',make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=3000,class_weight='balanced'))))
 mods.append((f'pair{C}','pair',make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=3000))))
for leaf in [2,4,8,16]:mods.append((f'extraC{leaf}','cls',ExtraTreesClassifier(n_estimators=150,min_samples_leaf=leaf,max_features=.7,class_weight='balanced',random_state=77,n_jobs=-1)))
res=[];fits={}
for n,k,m in mods:
 cc=cv(k,m,tr);pd,f=fitpred(k,m,tr,dv);dd=ev(dv,pd);res.append({'name':n,'kind':k,'cv':cc,'dev':dd});fits[n]=f;print(n,cc['pass11'],cc['worst'],dd['pass11'],dd['worst'])
rank=sorted(res,key=lambda r:(-r['dev']['pass11'],r['dev']['worst'],-r['cv']['pass11'],r['cv']['worst'],r['dev']['mean']))
b=rank[0];json.dump({'best':b,'ranked':rank},open('/mnt/data/path5_work/top8_ranker_light_results.json','w'),indent=2);pickle.dump({'name':b['name'],'kind':b['kind'],'model':fits[b['name']]},open('/mnt/data/path5_work/top8_ranker_light.pkl','wb'));print('BEST',json.dumps(b,indent=2))
