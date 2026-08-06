#!/usr/bin/env python3
from __future__ import annotations
import glob,json,hashlib,sys,math
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));import radial_core as rc
OUT=ROOT/'hierarchical_phase_flux'/'offset_rescue';OUT.mkdir(parents=True,exist_ok=True)
M=np.load(ROOT/'hierarchical_phase_flux/hierarchical_phase_flux_model.npz');template=M['template'];xm=M['x_mean'];xs=M['x_std'];pc=M['phase_coef']

def weight_features(nid:int):
 ws,_,_=rc.make_weights(nid);W=[x.numpy().astype(np.float64) for x in ws];f=[]
 # fixed deterministic sketches shared across networks
 rng=np.random.default_rng(20260729)
 R=rng.choice([-1.0,1.0],size=(rc.D,8))/math.sqrt(rc.D)
 fs=np.ones(rc.D);fa=np.ones(rc.D);fq=np.ones(rc.D);fwd=[]
 for l,A in enumerate(W):
  rn=np.linalg.norm(A,axis=1);cn=np.linalg.norm(A,axis=0)
  q=np.quantile(rn,[0,.1,.25,.5,.75,.9,1]);qc=np.quantile(cn,[0,.1,.25,.5,.75,.9,1])
  AR=A@R;ATR=A.T@R
  sketch=np.r_[np.linalg.norm(AR,axis=0),np.linalg.norm(ATR,axis=0),np.mean(AR,axis=0),np.mean(ATR,axis=0)]
  # four deterministic power iterations for spectral norm proxy
  v=R[:,0].copy()
  for _ in range(4):
   v=A.T@(A@v);v/=max(np.linalg.norm(v),1e-12)
  spec=math.sqrt(max(float(v@(A.T@(A@v))),0.0))
  sym_norm=math.sqrt(.5*(np.sum(A*A)+np.sum(A*A.T)))
  skew_norm=math.sqrt(max(.5*(np.sum(A*A)-np.sum(A*A.T)),0.0))
  f.extend([A.mean(),A.std(),np.mean(np.abs(A)),np.max(np.abs(A)),np.mean(A**3),np.mean(A**4),np.sum(A*A),np.trace(A),np.sum(A*A.T),spec,sym_norm,skew_norm,*q,*qc,*sketch])
  fs=A.T@fs;fa=np.abs(A).T@fa;fq=(A*A).T@fq
  for v in (fs,fa,fq):v/=max(np.sqrt(np.mean(v*v)),1e-12)
  fwd.append((fs.copy(),fa.copy(),fq.copy()))
  for v in (fs,fa,fq):f.extend([v.mean(),v.std(),np.min(v),np.max(v),np.mean(np.abs(v)),np.mean(v**3),np.mean(v**4)])
  f.extend([np.corrcoef(fs,fa)[0,1],np.corrcoef(fs,fq)[0,1],np.corrcoef(fa,fq)[0,1]])
 bs=np.ones(rc.D);ba=np.ones(rc.D);bq=np.ones(rc.D);bwd=[None]*len(W)
 for l in range(len(W)-1,-1,-1):
  A=W[l];bs=A@bs;ba=np.abs(A)@ba;bq=(A*A)@bq
  for v in (bs,ba,bq):v/=max(np.sqrt(np.mean(v*v)),1e-12)
  bwd[l]=(bs.copy(),ba.copy(),bq.copy())
  for v in (bs,ba,bq):f.extend([v.mean(),v.std(),np.min(v),np.max(v),np.mean(np.abs(v)),np.mean(v**3),np.mean(v**4)])
  f.extend([np.corrcoef(bs,ba)[0,1],np.corrcoef(bs,bq)[0,1],np.corrcoef(ba,bq)[0,1]])
 for l in range(len(W)):
  for a,b in zip(fwd[l],bwd[l]):f.extend([np.dot(a,b)/rc.D,np.dot(np.abs(a),np.abs(b))/rc.D,np.dot(a*a,b*b)/rc.D])
 return np.asarray(f,dtype=np.float64)

def load(patterns):
 rows=[]
 for pat in patterns:
  for p in sorted(glob.glob(str(pat))):
   with np.load(p,allow_pickle=True) as z:
    for j,rot in enumerate(z['rotation_seeds']):
     x=np.r_[z['sample_prediction'][j],z['baseline_prediction'][j],z['sample_prediction'][j]-z['baseline_prediction'][j]].astype(float)
     d=z['target_delta'][j].astype(float);beta=z['beta_bar'][j].astype(float);corr=template@beta;truth=.5*(z['truth_half1'][j]+z['truth_half2'][j]);sample=z['sample_prediction'][j].astype(float);scale=float(corr@(truth-sample)/max(corr@corr,1e-30))
     rows.append({'id':int(z['network_id']),'rot':int(rot),'x':x,'corr':corr,'truth':truth,'sample':sample,'base':float(z['base_mse'][j]),'scale':scale,'source':p})
 return rows
train=load([ROOT/'data/train_network_*.npz']);val=load([ROOT/'data/validation_network_*.npz']);term1=load([ROOT/'final_test_data/test_network_*.npz']);term2=load([ROOT/'rescue/final_test_data/test_network_*.npz'])
# features cached
allids=sorted(set(r['id'] for r in train+val+term1+term2));cache=OUT/'weight_invariants.npz'
if cache.exists():
 z=np.load(cache);cids=z['ids'];FX0=z['features'];mp={int(i):FX0[k] for k,i in enumerate(cids)};FX=np.stack([mp[i] for i in allids])
else:
 FX=np.stack([weight_features(i) for i in allids]);np.savez_compressed(cache,ids=np.asarray(allids),features=FX)
id2f={i:FX[k] for k,i in enumerate(allids)}
# phase raw prediction is fixed pairwise ridge
for rows in (train,val,term1,term2):
 for r in rows:r['phase']=float(((r['x']-xm)/xs)@pc)
# base-level target = mean scale - mean phase. Train only.
def base_table(rows):
 g=defaultdict(list)
 for r in rows:g[r['id']].append(r)
 ids=sorted(g);X=np.stack([id2f[i] for i in ids]);y=np.array([np.mean([r['scale']-r['phase'] for r in g[i]]) for i in ids]);return ids,X,y,g
trids,Xtr,ytr,gtr=base_table(train);vids,Xv,yv,gv=base_table(val)
# frozen alpha based on dimension/sample heuristic, no sweep
alpha=float(Xtr.shape[1]/len(Xtr))
model=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='lsqr')).fit(Xtr,ytr)
# Eight-fold base-group calibration of combined prediction.
cv_pred=[];cv_y=[]
folds=np.array_split(np.arange(len(trids)),8)
for hold in folds:
 keep=np.ones(len(trids),dtype=bool);keep[hold]=False
 m=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='lsqr')).fit(Xtr[keep],ytr[keep])
 off=m.predict(Xtr[hold])
 for j,h in enumerate(hold):
  for r in gtr[trids[h]]:cv_pred.append(float(off[j]+r['phase']));cv_y.append(r['scale'])
cv_pred=np.array(cv_pred);cv_y=np.array(cv_y);cal=float(np.clip(cv_pred@cv_y/max(cv_pred@cv_pred,1e-30),0,1))

def evaluate(rows):
 by=defaultdict(list)
 for r in rows:by[r['id']].append(r)
 ids=sorted(by);offs=model.predict(np.stack([id2f[i] for i in ids]));omap=dict(zip(ids,offs));pred=np.array([cal*(omap[r['id']]+r['phase']) for r in rows]);true=np.array([r['scale'] for r in rows]);mse=[];base=[];rr=[]
 for r,s in zip(rows,pred):
  m=float(np.mean((r['sample']+s*r['corr']-r['truth'])**2));mse.append(m);base.append(r['base']);rr.append(m/r['base'])
 mse=np.array(mse);base=np.array(base);rr=np.array(rr)
 return {'n':len(rows),'aggregate':float(mse.sum()/base.sum()),'wins':int(np.sum(rr<1)),'median':float(np.median(rr)),'worst':float(rr.max()),'scale_corr':float(pearsonr(pred,true).statistic),'sign':float(np.mean((pred>0)==(true>0))),'pred_positive':float(np.mean(pred>0))},pred
vres,vp=evaluate(val);t1,_=evaluate(term1);t2,_=evaluate(term2)
np.savez_compressed(OUT/'model.npz',template=template,x_mean=xm,x_std=xs,phase_coef=pc,train_ids=np.asarray(trids),feature_mean=model[0].mean_,feature_scale=model[0].scale_,offset_coef=model[1].coef_,offset_intercept=np.asarray(model[1].intercept_),alpha=np.asarray(alpha),calibration=np.asarray(cal),feature_cache_sha256=np.asarray(hashlib.sha256(cache.read_bytes()).hexdigest()))
hashv=hashlib.sha256((OUT/'model.npz').read_bytes()).hexdigest()
out={'design':{'label':'base-average oracle scalar minus pairwise phase flux','features':'exact weight spectra, symmetry/nonnormality, forward/backward signed/absolute/squared path contractions','ridge_alpha':alpha,'calibration':'eight-fold base-group scalar clipped [0,1]','n_features':int(Xtr.shape[1])},'model_sha256':hashv,'calibration':cal,'validation':vres,'exposed_primary_terminal':t1,'exposed_rescue_terminal':t2}
(OUT/'development_result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
