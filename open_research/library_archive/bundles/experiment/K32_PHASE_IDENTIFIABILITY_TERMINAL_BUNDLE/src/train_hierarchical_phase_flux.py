#!/usr/bin/env python3
from __future__ import annotations
import glob,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'hierarchical_phase_flux';OUT.mkdir(exist_ok=True)

def load(pattern):
 rows=[]
 for p in sorted(glob.glob(str(pattern))):
  with np.load(p,allow_pickle=True) as z:
   for j,rot in enumerate(z['rotation_seeds']):
    rows.append({
      'id':int(z['network_id']),'rot':int(rot),'global':z['global_features'][j].astype(np.float64),
      'x':np.r_[z['sample_prediction'][j],z['baseline_prediction'][j],z['sample_prediction'][j]-z['baseline_prediction'][j]].astype(np.float64),
      'delta':z['target_delta'][j].astype(np.float64),'beta':z['beta_bar'][j].astype(np.float64),
      'sample':z['sample_prediction'][j].astype(np.float64),'truth':.5*(z['truth_half1'][j]+z['truth_half2'][j]),
      'base_mse':float(z['base_mse'][j]),'oracle':z['oracle_prediction'][j].astype(np.float64),'source':p})
 return rows
train=load(ROOT/'data/train_network_*.npz')
val=load(ROOT/'data/validation_network_*.npz')
template=np.mean([r['delta'] for r in train],axis=0);template/=np.linalg.norm(template)

def annotate(rows):
 for r in rows:
  c=template@r['beta'];e=r['truth']-r['sample'];r['corr']=c;r['scale']=float(c@e/max(c@c,1e-30))
annotate(train);annotate(val)
# standardize same-cloud final vector
X=np.stack([r['x'] for r in train]);mu=X.mean(0);sd=X.std(0);sd=np.maximum(sd,1e-12);Xs=(X-mu)/sd
# one difference per training base (frozen rotations 3,11)
ids=sorted(set(r['id'] for r in train));pairs=[]
for nid in ids:
 ix=[i for i,r in enumerate(train) if r['id']==nid]
 if len(ix)!=2: raise RuntimeError((nid,len(ix)))
 pairs.append(ix)
Xd=np.stack([Xs[a]-Xs[b] for a,b in pairs]);yd=np.array([train[a]['scale']-train[b]['scale'] for a,b in pairs])
phase=Ridge(alpha=100.0,fit_intercept=False).fit(Xd,yd)
bpred=phase.predict(Xs)
# per-base absolute offset target, then weight-global ridge
G=[];off=[]
for a,b in pairs:
 G.append(train[a]['global'])
 off.append(.5*(train[a]['scale']+train[b]['scale'])-.5*(bpred[a]+bpred[b]))
G=np.stack(G);off=np.array(off);gmu=G.mean(0);gsd=np.maximum(G.std(0),1e-12)
offset=Ridge(alpha=100.0).fit((G-gmu)/gsd,off)
# Freeze a scale calibration using leave-one-base-out predictions only, avoiding in-sample shrink.
loo=[];yt=[]
for hold,nid in enumerate(ids):
 keep=np.array([i!=hold for i in range(len(ids))])
 ph=Ridge(alpha=100.0,fit_intercept=False).fit(Xd[keep],yd[keep])
 bp=ph.predict(Xs)
 go=Ridge(alpha=100.0).fit((G[keep]-gmu)/gsd,off[keep])
 for idx in pairs[hold]:
  loo.append(float(bp[idx]+go.predict(((train[idx]['global']-gmu)/gsd)[None])[0]));yt.append(train[idx]['scale'])
loo=np.array(loo);yt=np.array(yt)
cal=float(np.dot(loo,yt)/max(np.dot(loo,loo),1e-30));cal=float(np.clip(cal,0.0,1.0))

def predict(rows):
 X=np.stack([r['x'] for r in rows]);B=phase.predict((X-mu)/sd);Gx=np.stack([r['global'] for r in rows]);A=offset.predict((Gx-gmu)/gsd);return cal*(A+B),A,B

def evaluate(rows,pred):
 mse=[];base=[];rat=[];sc=[]
 for r,s in zip(rows,pred):
  yp=r['sample']+s*r['corr'];m=float(np.mean((yp-r['truth'])**2));mse.append(m);base.append(r['base_mse']);rat.append(m/r['base_mse']);sc.append(r['scale'])
 mse=np.array(mse);base=np.array(base);rat=np.array(rat);sc=np.array(sc)
 rng=np.random.default_rng(20260729);ids0=sorted(set(r['id'] for r in rows));by=[[i for i,r in enumerate(rows) if r['id']==nid] for nid in ids0];draw=rng.integers(0,len(by),size=(20000,len(by)));bs=[]
 for d in draw:
  ix=np.concatenate([by[k] for k in d]);bs.append(mse[ix].sum()/base[ix].sum())
 return {'aggregate_ratio':float(mse.sum()/base.sum()),'bootstrap_95':[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],'wins':int(np.sum(rat<1)),'n':len(rows),'median':float(np.median(rat)),'worst':float(rat.max()),'scale_pearson':float(pearsonr(pred,sc).statistic),'scale_sign_accuracy':float(np.mean((pred>0)==(sc>0))),'pred_positive_fraction':float(np.mean(pred>0))}
vp,va,vb=predict(val);vres=evaluate(val,vp)
# diagnostics exposed terminal cohorts only, no selection
term1=load(ROOT/'final_test_data/test_network_*.npz');annotate(term1);tp,_,_=predict(term1);tres=evaluate(term1,tp)
term2=load(ROOT/'rescue/final_test_data/test_network_*.npz');annotate(term2);rp,_,_=predict(term2);rres=evaluate(term2,rp)
# save model arrays
np.savez_compressed(OUT/'hierarchical_phase_flux_model.npz',template=template,x_mean=mu,x_std=sd,phase_coef=phase.coef_,global_mean=gmu,global_std=gsd,offset_coef=offset.coef_,offset_intercept=np.asarray(offset.intercept_),calibration=np.asarray(cal))
model_hash=hashlib.sha256((OUT/'hierarchical_phase_flux_model.npz').read_bytes()).hexdigest()
out={'design':{'phase':'pairwise-difference ridge over [sample output, baseline output, difference], alpha=100','absolute_offset':'weight-global ridge over per-base residual mean, alpha=100','calibration':'leave-one-base-out scalar clipped to [0,1]','incremental_ops_estimate':int(3*256+512)},'model_sha256':model_hash,'calibration':cal,'validation':vres,'exposed_primary_terminal_diagnostic':tres,'exposed_rescue_terminal_diagnostic':rres}
(OUT/'development_result.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
