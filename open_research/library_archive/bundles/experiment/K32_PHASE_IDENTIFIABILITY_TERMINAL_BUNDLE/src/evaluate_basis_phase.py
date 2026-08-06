#!/usr/bin/env python3
from pathlib import Path
import glob,json,sys,os
from collections import defaultdict
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr,spearmanr
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'basis_phase';sys.path.insert(0,str(ROOT/'src'))

def load(split):
 rows=[]
 for p in sorted(glob.glob(str(ROOT/f'data/{split}_network_*.npz'))):
  with np.load(p,allow_pickle=True) as z:
   nid=int(z['network_id'])
   for j,rot in enumerate(z['rotation_seeds']):
    bp=OUT/f'data/{nid}_{j}.npz'
    with np.load(bp,allow_pickle=True) as b:
     rows.append({'id':nid,'rot':int(rot),'features':b['features'].astype(float),'names':b['names'].astype(str),'delta':z['target_delta'][j].astype(float),'beta':z['beta_bar'][j].astype(float),'sample':z['sample_prediction'][j].astype(float),'truth':.5*(z['truth_half1'][j]+z['truth_half2'][j]),'base':float(z['base_mse'][j])})
 return rows
tr=load('train');va=load('validation');names=tr[0]['names'];template=np.mean([r['delta'] for r in tr],0);template/=np.linalg.norm(template)
for rows in (tr,va):
 for r in rows:
  c=template@r['beta'];r['corrvec']=c;r['scale']=float(c@(r['truth']-r['sample'])/max(c@c,1e-30))
Xtr=np.stack([r['features'] for r in tr]);Xv=np.stack([r['features'] for r in va]);ytr=np.array([r['scale'] for r in tr]);yv=np.array([r['scale'] for r in va])
# families by prefix
fam={
 'output_projection':[i for i,n in enumerate(names) if n.startswith('yproj')],
 'target_proxy':[i for i,n in enumerate(names) if n.startswith('hproxy')],
 'target_output_cross':[i for i,n in enumerate(names) if n.startswith('cross')],
 'basis_probe_dispersion':[i for i,n in enumerate(names) if n.startswith('xblock')],
 'feature_output_spectrum':[i for i,n in enumerate(names) if n.startswith('xy_')],
 'output_basis_spectrum':[i for i,n in enumerate(names) if n.startswith('Yb_')],
 'target_basis_spectrum':[i for i,n in enumerate(names) if n.startswith('Hb_')],
 'target_second_basis_spectrum':[i for i,n in enumerate(names) if n.startswith('H2b_')],
 'all':list(range(len(names)))
}

def pair(rows,X,y):
 pairs=[]
 for nid in sorted(set(r['id'] for r in rows)):
  ix=[i for i,r in enumerate(rows) if r['id']==nid]
  for a in range(len(ix)):
   for b in range(a+1,len(ix)):pairs.append((ix[a],ix[b]))
 return np.stack([X[a]-X[b] for a,b in pairs]),np.array([y[a]-y[b] for a,b in pairs]),pairs
Xdt,ydt,_=pair(tr,Xtr,ytr);Xdv,ydv,pv=pair(va,Xv,yv)

def score_pred(rows,pred):
 mse=[];base=[];rr=[]
 for r,s in zip(rows,pred):
  m=np.mean((r['sample']+s*r['corrvec']-r['truth'])**2);mse.append(m);base.append(r['base']);rr.append(m/r['base'])
 mse=np.array(mse);base=np.array(base);rr=np.array(rr)
 return {'ratio':float(mse.sum()/base.sum()),'wins':int(np.sum(rr<1)),'median':float(np.median(rr)),'worst':float(rr.max()),'scale_pearson':float(pearsonr(pred,np.array([r['scale'] for r in rows])).statistic),'sign':float(np.mean((pred>0)==(np.array([r['scale'] for r in rows])>0)))}
results={}
for nm,ix in fam.items():
 model=make_pipeline(StandardScaler(),Ridge(alpha=100.0)).fit(Xtr[:,ix],ytr);p=model.predict(Xv[:,ix])
 pm=make_pipeline(StandardScaler(),Ridge(alpha=100.0)).fit(Xdt[:,ix],ydt);pd=pm.predict(Xdv[:,ix])
 results[nm]={'n_features':len(ix),'validation':score_pred(va,p),'difference_corr':float(pearsonr(pd,ydv).statistic),'difference_sign':float(np.mean(np.sign(pd)==np.sign(ydv)))}
# univariate stability
uni=[]
for j,n in enumerate(names):
 a=pearsonr(Xtr[:,j],ytr).statistic if np.std(Xtr[:,j]) else 0;b=pearsonr(Xv[:,j],yv).statistic if np.std(Xv[:,j]) else 0
 ad=pearsonr(Xdt[:,j],ydt).statistic if np.std(Xdt[:,j]) else 0;bd=pearsonr(Xdv[:,j],ydv).statistic if np.std(Xdv[:,j]) else 0
 uni.append({'name':n,'train':float(a),'validation':float(b),'train_diff':float(ad),'validation_diff':float(bd),'stable_abs':float(np.sign(a)==np.sign(b))*min(abs(a),abs(b))})
uni_abs=sorted(uni,key=lambda x:x['stable_abs'],reverse=True);uni_diff=sorted(uni,key=lambda x:(np.sign(x['train_diff'])==np.sign(x['validation_diff']))*min(abs(x['train_diff']),abs(x['validation_diff'])),reverse=True)
# frozen top-16 stable by training magnitude only (diagnostic, not promoted unless validation clears)
rank=np.argsort(np.abs([x['train'] for x in uni]))[::-1][:16];m=make_pipeline(StandardScaler(),Ridge(alpha=100.0)).fit(Xtr[:,rank],ytr);pp=m.predict(Xv[:,rank]);results['train_top16']={'features':[names[i] for i in rank],'validation':score_pred(va,pp)}
out={'results':results,'top_absolute_stable':uni_abs[:40],'top_difference_stable':uni_diff[:40]}
(OUT/'BASIS_PHASE_DEVELOPMENT.json').write_text(json.dumps(out,indent=2));print(json.dumps(results,indent=2));print('TOP ABS');print(json.dumps(uni_abs[:12],indent=2));print('TOP DIFF');print(json.dumps(uni_diff[:12],indent=2))
