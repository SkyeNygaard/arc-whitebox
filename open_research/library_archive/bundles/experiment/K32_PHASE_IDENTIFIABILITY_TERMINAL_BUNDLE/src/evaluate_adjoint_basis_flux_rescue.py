#!/usr/bin/env python3
from pathlib import Path
import glob,json,sys
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression,Ridge
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'basis_phase';
# fixed template from training
trs=[]
for p in sorted(glob.glob(str(ROOT/'data/train_network_*.npz'))):
 with np.load(p) as z:
  for j in range(len(z['rotation_seeds'])):trs.append(z['target_delta'][j].astype(float))
template=np.mean(trs,0);template/=np.linalg.norm(template)

def load(split):
 rows=[]
 for p in sorted(glob.glob(str(ROOT/f'data/{split}_network_*.npz'))):
  with np.load(p) as z:
   nid=int(z['network_id'])
   for j,rot in enumerate(z['rotation_seeds']):
    with np.load(OUT/f'data/{nid}_{j}.npz') as b:
     dx=b['dx'].astype(float);yb=b['yb'].astype(float)
    beta=z['beta_bar'][j].astype(float);corr=template@beta;cn=np.linalg.norm(corr);yc=yb-yb.mean(0);zz=dx@template;ww=yc@(corr/max(cn,1e-30))
    # complete-basis non-marginal flux features, all O(B*P + B*D)
    zc=zz-zz.mean();wc=ww-ww.mean();sz=max(zc.std(),1e-30);sw=max(wc.std(),1e-30)
    f=np.array([
      np.mean(zc*wc),
      np.mean((zc/sz)**2*(wc/sw)),
      np.mean((zc/sz)*(wc/sw)**2),
      np.mean((zc/sz)**3*(wc/sw)),
      np.mean((zc/sz)*(wc/sw)**3),
      np.mean(zc*wc)/max(sz*sw,1e-30),
      sz,sw,
      np.mean((zc/sz)**3),np.mean((wc/sw)**3),
    ])
    truth=.5*(z['truth_half1'][j]+z['truth_half2'][j]);sample=z['sample_prediction'][j].astype(float);scale=float(corr@(truth-sample)/max(corr@corr,1e-30))
    rows.append({'id':nid,'rot':int(rot),'f':f,'scale':scale,'corr':corr,'truth':truth,'sample':sample,'base':float(z['base_mse'][j])})
 return rows
tr=load('train');va=load('validation');X=np.stack([r['f'] for r in tr]);y=np.array([r['scale'] for r in tr]);Xv=np.stack([r['f'] for r in va]);yv=np.array([r['scale'] for r in va])
names=['cov_zw','z2w','zw2','z3w','zw3','corr_zw','std_z','std_w','skew_z','skew_w']
# audit correlations
cors=[]
for j,n in enumerate(names):cors.append({'name':n,'train':float(pearsonr(X[:,j],y).statistic),'validation':float(pearsonr(Xv[:,j],yv).statistic)})
# mechanistic rescue: preserve signed amplitude in complete-basis covariance
j=0
# group 8-fold calibration of affine model
ids=sorted(set(r['id'] for r in tr));folds=np.array_split(np.arange(len(ids)),8);pred=np.zeros(len(tr))
for hold in folds:
 keepids={ids[k] for k in range(len(ids)) if k not in set(hold)};testids={ids[k] for k in hold};ki=np.array([r['id'] in keepids for r in tr]);hi=np.array([r['id'] in testids for r in tr]);m=LinearRegression().fit(X[ki,j:j+1],y[ki]);pred[hi]=m.predict(X[hi,j:j+1])
# shrink to final replay objective using out-of-fold predictions
# candidate direction differs per row, so directly search analytic least-squares scalar lambda on OOF replay residual
num=den=0.0
for r,p in zip(tr,pred):
 e=r['truth']-r['sample'];d=p*r['corr'];num+=d@e;den+=d@d
lam=float(np.clip(num/max(den,1e-30),0,1))
model=LinearRegression().fit(X[:,j:j+1],y);pv=lam*model.predict(Xv[:,j:j+1])

def metrics(rows,p):
 mse=[];base=[];rr=[]
 for r,s in zip(rows,p):
  m=np.mean((r['sample']+s*r['corr']-r['truth'])**2);mse.append(m);base.append(r['base']);rr.append(m/r['base'])
 mse=np.array(mse);base=np.array(base);rr=np.array(rr)
 rng=np.random.default_rng(20260729);uids=sorted(set(r['id'] for r in rows));groups=[[i for i,r in enumerate(rows) if r['id']==u] for u in uids];vals=[]
 for _ in range(20000):
  draw=rng.integers(0,len(groups),len(groups));ix=np.concatenate([groups[k] for k in draw]);vals.append(mse[ix].sum()/base[ix].sum())
 return {'ratio':float(mse.sum()/base.sum()),'bootstrap95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],'wins':int(np.sum(rr<1)),'median':float(np.median(rr)),'worst':float(rr.max()),'scale_corr':float(pearsonr(p,np.array([r['scale'] for r in rows])).statistic),'sign':float(np.mean((p>0)==(np.array([r['scale'] for r in rows])>0))),'positive_fraction':float(np.mean(p>0))}
out={'feature_correlations':cors,'primary':{'statistic':'cov_zw = E_b[((X_b-q)·template)((Y_b-Ybar)·normalized(template·beta))]' ,'linear_coef':float(model.coef_[0]),'intercept':float(model.intercept_),'oof_replay_shrink':lam,'validation':metrics(va,pv)},'incremental_ops_estimate':129*(32+256+8)}
(OUT/'ADJOINT_BASIS_FLUX_AMPLITUDE_RESCUE.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
